from copy import deepcopy
import numpy as np
import torch
from methods.prompter import Prompter
from methods.Server import ServerBase
from methods.FedPrompt.FedPromptClient import FedPromptAgent
import os, time


class FedPrompt(ServerBase):
    def __init__(self, args, logger):
        super(FedPrompt, self).__init__(args, logger)
        self.setup_clients()
        self.get_data_weight()
        if args.heter_extractor:
            feature_type = self.clients[0].train_dataset.feature_type
            prompt_size = 1024 if feature_type in ['R50_features', 'UNI_features'] else 512
        else:
            prompt_size = self.global_model.size[0]
        dfp_dict = {'init': args.prompt_initialisation,
                    'number_prompts': args.number_prompts,
                    'prompt_aggregation': args.prompt_aggregation,
                    'prompt_size': prompt_size}
        self.global_prompt = Prompter(dfp_dict)

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        self.weight_list = weight_list / np.sum(weight_list) * n_clnt

    def setup_clients(self):
        MIL_pool = []
        init_model = deepcopy(self.global_model)
        self.client_init_model = []
        if self.args.heter_model:
            MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']  # more will be added
            init_model = None
        for idx in self.n_clients:
            curr_client = FedPromptAgent(self.args, init_model, self.logger, MIL_pool)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            curr_client.init_prompt(self.args)
            self.clients.append(curr_client)
            print(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            self.logger.info(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            if self.args.heter_model and len(MIL_pool) > 0:
                MIL_pool.remove(curr_client.local_model_name)
            if len(MIL_pool) == 0:
                MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']
            self.client_init_model.append(deepcopy(curr_client.local_model))

    def aggregate_prompt(self, prompt_weight, method='average', weighted=False, noise_level=0):
        if method == 'average':
            for client_idx, client_prompt in enumerate(prompt_weight):
                if client_idx == 0:
                    self.global_prompt.prompt_embeddings.data = client_prompt[0].prompt_embeddings.data * self.weight_list[client_idx] if weighted else client_prompt[0].prompt_embeddings.data
                else:
                    self.global_prompt.prompt_embeddings.data += client_prompt[0].prompt_embeddings.data * self.weight_list[client_idx] if weighted else client_prompt[0].prompt_embeddings.data
        elif method == 'DP_average':
            for client_idx, client_prompt in enumerate(prompt_weight):
                noise = noise_level * torch.empty(client_prompt[0].prompt_embeddings.data.size()).normal_(mean=0, 
                                                                                                  std=client_prompt[0].prompt_embeddings.data.reshape(-1).float().std())
                if client_idx == 0:
                    self.global_prompt.prompt_embeddings.data = client_prompt[0].prompt_embeddings.data * self.weight_list[client_idx] + noise.to(self.device) if weighted else client_prompt[0].prompt_embeddings.data + noise.to(self.device)
                else:
                    self.global_prompt.prompt_embeddings.data += client_prompt[0].prompt_embeddings.data * self.weight_list[client_idx] + noise.to(self.device) if weighted else client_prompt[0].prompt_embeddings.data + noise.to(self.device)
        else:
            raise NotImplementedError

    def run(self, iter):
        # include training and testing
        train_loss = []
        best_accuracy = 0.
        train_acc_wt = 0.
        best_accuracy_per_agent = [0.] * len(self.n_clients)
        normalised_weight_list = self.weight_list / np.sum(self.weight_list)
        print('Global epochs ', self.args.global_epochs)
        for epoch in range(self.args.global_epochs):
            global_info = f'===========Global epoch {epoch}===========' if epoch > 0 else f'===========Global epoch {epoch} [Init Prompt]==========='
            print(global_info)
            self.logger.info(global_info)

            local_weights, local_prompt = [],[]
            self.logger.info(f'| Training local model + prompt |')
            # each client trains their local model+local prompt
            local_model_train = True
            for idx in self.n_clients:
                local_time = time.time()
                local_lr = self.args.prompt_lr
                w, prompt_w,= self.clients[idx].local_train(idx,
                                                            normalised_weight_list[idx],
                                                            epoch=None,
                                                            local_model_train=local_model_train,
                                                            local_lr=local_lr)
                end_time = time.time()
                local_weights.append(deepcopy(w))
                local_prompt.append(deepcopy(prompt_w))
                logging_info = f'Agent {idx} local train time: {end_time - local_time}'
                self.logger.info(logging_info)

                # logging_info = f'Agent: {idx}, Test Acc: {agent_acc}, Test AUC: {agent_auc}, Test AUPRC: {agent_auprc}, Test MCC: {agent_mcc}'
            # server aggregates the local prompt
            if self.args.dp_average:
                self.aggregate_prompt(local_prompt, method='DP_average', weighted=True, noise_level=self.args.dp_noise)
            else:
                self.aggregate_prompt(local_prompt, method='average', weighted=True)
            
            list_acc_wt = [0] * len(self.n_clients)
            for idx in self.n_clients:
                self.clients[idx].update_prompt(self.global_prompt)
                local_result = self.clients[idx].local_test()
                if 1 - local_result['error'] > best_accuracy_per_agent[idx]:
                    logging_info = f'Agent {idx} local Test Acc Improved from {best_accuracy_per_agent[idx]} to {1 - local_result["error"]}'
                    self.logger.info(logging_info)
                    best_accuracy_per_agent[idx] = 1 - local_result['error']
                    best_prompter = deepcopy(self.clients[idx].prompter_gather)
                    best_prompt_save_pth = os.path.join(self.args.results_dir, "best_prompt_%d.pt" % iter)
                    torch.save(best_prompter, best_prompt_save_pth)
                    best_model = deepcopy(self.clients[idx].local_model)
                    best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
                    torch.save(best_model.state_dict(), best_model_save_pth)

                list_acc_wt[idx] = (1 - local_result['error']) * self.weight_list[idx]
            train_acc_wt = sum(list_acc_wt)
        best_accuracy = sum(best_accuracy_per_agent) / len(best_accuracy_per_agent)
        return best_accuracy, train_acc_wt, best_accuracy_per_agent