from methods.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
import os, torch

class FedBaseAgent(AgentBase):
    def __init__(self, args, global_model, logger, MIL_pool):
        super().__init__(args, global_model, logger, MIL_pool)

    def local_train(self, agent_idx):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                loss, error, y_prob = self.mil_run(self.local_model, images, labels, self.mil_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
            print(f'Agent: {agent_idx}, Iter: {iter},Loss: {batch_loss}')

            # results = self.local_test()
            # local_acc = 1 - results['error']
            # self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Loss: {batch_loss}, ACC: {local_acc}')
            # # if (agent_idx == 0 and local_acc > 0.7 and local_acc < 0.89) or (agent_idx == 1 and local_acc > 0.7 and local_acc < 0.84):
            # if local_acc > 0.5:
            #     self.logger.info(f'Agent: {agent_idx}, Acc: {local_acc} Save!!')
            #     local_acc_str = str(local_acc).replace('.','')
            #     mcc_str = str(round(results['mcc'], 3)).replace('.','')
            #     auc_str = str(round(results['auc'], 3)).replace('.','')
            #     auprc_str = str(round(results['auprc'], 3)).replace('.','')
            #     acc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/acc/")
            #     os.makedirs(acc_save_pth, exist_ok=True)
            #     if len(os.listdir(acc_save_pth)) > 10:
            #         exist_acc = {acc:int(acc.split('.')[0].split('_')[1]) for acc in os.listdir(acc_save_pth) if acc.startswith('acc_')}
            #         sorted_accs = sorted(exist_acc.items(), key=lambda x: x[1])
            #         if sorted_accs and sorted_accs[0][1] < int(local_acc_str):
            #             os.remove(os.path.join(acc_save_pth, sorted_accs[0][0]))
            #             acc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/acc/")
            #             acc_fold_save_pth = os.path.join(acc_save_pth, f"acc_{local_acc_str}.pt")
            #             torch.save(self.local_model.state_dict(), acc_fold_save_pth)
            #     else:
            #         acc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/acc/")
            #         acc_fold_save_pth = os.path.join(acc_save_pth, f"acc_{local_acc_str}.pt")
            #         torch.save(self.local_model.state_dict(), acc_fold_save_pth)
                
            #     # check before save, save only the top 5 auc under this folder
            #     auc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/auc/")
            #     os.makedirs(auc_save_pth, exist_ok=True)
            #     if len(os.listdir(auc_save_pth)) > 5:
            #         exist_aucs = {auc:int(auc.split('.')[0].split('_')[1]) for auc in os.listdir(auc_save_pth) if auc.startswith('auc_')}
            #         # sort exist_aucs by value and get the lowest AUC
            #         sorted_aucs = sorted(exist_aucs.items(), key=lambda x: x[1])
            #         # print(exist_aucs, sorted_aucs)
            #         if sorted_aucs and sorted_aucs[0][1] < int(auc_str):
            #             os.remove(os.path.join(auc_save_pth, sorted_aucs[0][0]))
            #             auc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/auc/")
            #             auc_fold_save_pth = os.path.join(auc_save_pth, f"auc_{auc_str}.pt")
            #             torch.save(self.local_model.state_dict(), auc_fold_save_pth)
            #     else:
            #         auc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/auc/")
            #         auc_fold_save_pth = os.path.join(auc_save_pth, f"auc_{auc_str}.pt")
            #         torch.save(self.local_model.state_dict(), auc_fold_save_pth)


            

                # mcc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/mcc/{mcc_str}")
                # os.makedirs(mcc_save_pth, exist_ok=True)
                # mcc_fold_save_pth = os.path.join(mcc_save_pth, f"mcc_{mcc_str}.pt")
                # torch.save(self.local_model.state_dict(), mcc_fold_save_pth)
                
                # auprc_save_pth = os.path.join(self.args.results_dir, f"fold_{self.args.fold_idx}/client_{agent_idx}/auprc/{auprc_str}")
                # os.makedirs(auprc_save_pth, exist_ok=True)
                # auprc_fold_save_pth = os.path.join(auprc_save_pth, f"auprc_{auprc_str}.pt")
                # torch.save(self.local_model.state_dict(), auprc_fold_save_pth)

        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs
