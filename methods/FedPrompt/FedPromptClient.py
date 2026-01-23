import numpy as np
import torch, os
from sklearn.metrics import roc_curve
from tqdm import tqdm
from utils.core_util import binary_matthews_corrcoef, multiclass_matthews_corrcoef, binary_auprc, multiclass_auprc
from methods.Client import AgentBase
from methods.prompter import Prompter
from sklearn.preprocessing import label_binarize
from utils.trainer_util import RAdam
from copy import deepcopy
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, average_precision_score


class FedPromptAgent(AgentBase):
    def __init__(self, args, global_model, logger, MIL_pool):
        super().__init__(args, global_model, logger, MIL_pool)
        self.num_coarse_stain_classes = 1
        if args.use_stain:
            stain_proto_path = f'./data/pre_extracted_color_feature/{self.args.task}'
            self.stain_prototype = torch.load('%s/Train/prototype.pt' % stain_proto_path).to(self.device)
            self.num_coarse_stain_classes = self.stain_prototype.size(0)
            print(f'========={self.num_coarse_stain_classes} stain prototype loaded=========')
            self.logger.info(f'========={self.num_coarse_stain_classes} stain prototype loaded=========')

    def init_prompt(self, args):
        if args.heter_extractor:
            feature_type = self.train_dataset.feature_type
            prompt_size = 1024 if feature_type in ['R50_features', 'UNI_features'] else 512
        else:
            prompt_size = self.local_model.size[0]
        dfp_dict = {'init': args.prompt_initialisation,
                    'number_prompts': args.number_prompts,
                    'prompt_aggregation': args.prompt_aggregation,
                    'prompt_size': prompt_size}
        self.prompter = Prompter(dfp_dict)

    def get_optim(self, weight, local_lr):
        self.prompter_gather, self.prompter_params_gather = [], []
        for i in range(self.num_coarse_stain_classes):
            self.prompter_gather.append(
                deepcopy(self.prompter)
            )
            # self.args.prompt_lr * weight if self.args.adaptive_prompt_lr else self.args.prompt_lr,
            local_lr = local_lr if local_lr is not None else self.args.prompt_lr
            self.prompter_params_gather.append(
                {'params': self.prompter_gather[i].parameters(),
                 'lr':local_lr,
                 'weight_decay':self.args.prompt_reg}
            )
        self.prompter_params_gather.append(
            {'params': filter(lambda p: p.requires_grad, self.local_model.parameters()),
             'lr': self.args.lr,
             'weight_decay': self.args.reg}
        )

        if self.args.opt == "adam":
            optimizer = optim.Adam(self.prompter_params_gather)
        elif self.args.opt == 'adamw':
            optimizer = optim.AdamW(self.prompter_params_gather)
        elif self.args.opt == 'sgd':
            for i in range(len(self.prompter_params_gather)):
                self.prompter_params_gather[i]['momentum'] = 0.9
            optimizer = optim.SGD(self.prompter_params_gather)
        elif self.args.opt == 'radam':
            optimizer = RAdam(self.prompter_params_gather)
        else:
            raise NotImplementedError
        return optimizer

    def get_prompted_ft_based_on_stain(self, h, h_stain, prompter_gather=None):
        prompted_image = []
        reform = False
        if len(h.size()) > 2:
            b, n, _ = h.size()
            for i in range(h.size(0)):
                prompted_image_batch = []
                h_i = h[i]
                h_stain_i = h_stain[i]
                unique_idx = torch.unique(h_stain_i).to(torch.long)
                for i in unique_idx:
                    idx_h = torch.where(h_stain_i == i)[0].to(torch.long)
                    prompted_image_batch.append(
                        prompter_gather[i](h_i[idx_h])
                    )
                prompted_image.append(torch.cat(prompted_image_batch, dim=0))
            prompted_image = torch.cat(prompted_image, dim=0)
            prompted_image = prompted_image.view(b, n, -1)
        else:
            indices = h_stain
            unique_idx = torch.unique(indices).to(torch.long)
            for i in unique_idx:
                idx_h = torch.where(indices == i)[0].to(torch.long)
                prompted_image.append(
                    prompter_gather[0](h[idx_h])
                )
            prompted_image = torch.cat(prompted_image, dim=0)
        return prompted_image

    def get_prompted_ft(self, h, prompter_gather=None):
        reform = False
        if len(h.size()) > 2:
            reform = True
            b, n, _ = h.size()
            h = h.view(-1, h.size(-1))
        h = prompter_gather[0](h)
        if reform:
            h = h.view(b, n, -1)
        return h

    def update_prompt(self, prompt):
        self.prompter_gather[0] = deepcopy(prompt)

    def local_train(self, agent_idx, agent_weight, epoch=None, local_model_train=True, local_lr=None):
        if local_model_train:
            self.turn_on_training()
        else:
            self.turn_off_training()
        optimizer = self.get_optim(agent_weight, local_lr)
        epoch_loss = 0.
        best_prompter = deepcopy(self.prompter_gather)
        epoch = epoch if epoch is not None else self.args.local_epochs
        #customise tqdm information
        pbar = tqdm(range(epoch), desc=f'Client {agent_idx} Training')
        for _ in pbar:
            batch_loss = 0.
            batch_error = 0.
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                prompted_data = self.get_prompted_ft(images, self.prompter_gather)
                loss, error, y_prob = self.mil_run(self.local_model, prompted_data, labels, self.mil_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        best_prompter = deepcopy(self.prompter_gather)
        return self.local_model.state_dict(), best_prompter

    def local_test(self, **kwargs):
        self.local_model.eval()
        total_error = 0.
        total_loss = 0.
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data_ft, label = data
                data_ft, label = data_ft.to(self.device), label.to(self.device)
                prompted_data = self.get_prompted_ft(data_ft, self.prompter_gather)
                loss, error, y_prob = self.mil_run(self.local_model, prompted_data, label, self.mil_loss)
                total_loss += loss.item()
                total_error += error
                probs = y_prob.detach().cpu().numpy()
                
                # Store probabilities and labels
                all_probs.append(probs)
                all_labels.append(label.item())
            
            # Convert to numpy arrays
            all_probs = np.array(all_probs).squeeze()
            all_labels = np.array(all_labels)
            
            total_loss /= len(self.test_loader)
            total_error /= len(self.test_loader)
            
            if self.args.n_classes == 2:
                # Ensure we have 1D array of probabilities for binary classification
                if all_probs.ndim > 1 and all_probs.shape[1] == 2:
                    # If we have probabilities for both classes, use the positive class
                    probs_positive = all_probs[:, 1]
                else:
                    # If we have a single probability, use it as the positive class probability
                    probs_positive = all_probs if all_probs.ndim == 1 else all_probs.ravel()
                
                # Calculate metrics using positive class probabilities
                fpr, tpr, thresholds = roc_curve(all_labels, probs_positive)
                auc = roc_auc_score(all_labels, probs_positive)
                auprc = average_precision_score(all_labels, probs_positive)
                
                # Hard predictions for positive class
                y_pred_pos = (probs_positive >= 0.5).astype(int)
                mcc = matthews_corrcoef(all_labels, y_pred_pos)
  
                # Per-class metrics
                cm = confusion_matrix(all_labels, y_pred_pos, labels=[0, 1])
                per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
                
                # For binary classification, we can calculate per-class metrics directly
                # In binary case, we only need to calculate metrics once for the positive class
                # and the negative class metrics are symmetric
                per_class_auc = [
                    roc_auc_score(1 - all_labels, 1 - probs_positive),  # Class 0
                    roc_auc_score(all_labels, probs_positive)           # Class 1
                ]
                per_class_auprc = [
                    average_precision_score(1 - all_labels, 1 - probs_positive),  # Class 0
                    average_precision_score(all_labels, probs_positive)           # Class 1
                ]
                per_class_mcc = [
                    matthews_corrcoef(1 - all_labels, 1 - y_pred_pos),  # Class 0
                    matthews_corrcoef(all_labels, y_pred_pos)           # Class 1
                ]

                per_class_auc = np.array(per_class_auc)
                per_class_auprc = np.array(per_class_auprc)
                per_class_mcc = np.array(per_class_mcc)

            else:
                fpr = dict()
                tpr = dict()
                y_true_bin = label_binarize(all_labels, classes=list(range(self.args.n_classes)))
                for i in range(y_true_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
                # auprc = multiclass_auprc(all_labels, all_probs)
                # mcc = multiclass_matthews_corrcoef(all_labels, all_probs)
                # auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
                            # global multiclass ROC AUC (ovo) and macro AUPRC
                # one-vs-rest binarization
                classes = list(range(self.args.n_classes))
                y_true_bin = label_binarize(all_labels, classes=classes)  # (N, K)

                # per-class ROC AUC & AUPRC
                per_class_auc = []
                per_class_auprc = []
                for c in range(self.args.n_classes):
                    per_class_auc.append(
                        roc_auc_score(y_true_bin[:, c], all_probs[:, c])
                    )
                    per_class_auprc.append(
                        average_precision_score(y_true_bin[:, c], all_probs[:, c])
                    )
                per_class_auc = np.array(per_class_auc)
                per_class_auprc = np.array(per_class_auprc)
                # macro metrics as simple average
                auc = per_class_auc.mean()
                auprc = per_class_auprc.mean()

                # hard predictions (argmax) for confusion matrix and MCC
                y_pred = np.argmax(all_probs, axis=1)
                cm = confusion_matrix(all_labels, y_pred, labels=classes)
                per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12)

                # global multiclass MCC (single scalar)
                mcc = matthews_corrcoef(all_labels, y_pred)

                # optional: per-class MCC as one-vs-rest (like above)
                per_class_mcc = []
                for c in range(self.args.n_classes):
                    y_true_c = y_true_bin[:, c]
                    y_pred_c = (y_pred == c).astype(int)
                    per_class_mcc.append(
                        matthews_corrcoef(y_true_c, y_pred_c)
                    )
                per_class_mcc = np.array(per_class_mcc)
        results = {'loss': total_loss, 
        'error': total_error, 
        'fpr': fpr, 
        'tpr': tpr, 
        'auc': auc, 
        'auprc': auprc, 
        'mcc': mcc, 
        'per_class_acc': per_class_acc, 
        'per_class_auc': per_class_auc, 
        'per_class_auprc': per_class_auprc, 
        'per_class_mcc': per_class_mcc, 
        'cm': cm,
        'y_true': all_labels,
        'y_prob': all_probs}
        return results