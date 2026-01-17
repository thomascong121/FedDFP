from copy import deepcopy
import numpy as np
import torch
from methods.Server import ServerBase
from methods.FedBase.FedBaseClient import FedBaseAgent
import os
import time


class FedBase(ServerBase):
    def __init__(self, args, logger):
        super(FedBase, self).__init__(args, logger)
        self.setup_clients()
        self.get_data_weight()

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        self.weight_list = weight_list / np.sum(weight_list) * n_clnt

    def setup_clients(self):
        print('==============Setting up clients==============')
        MIL_pool = []
        init_model = deepcopy(self.global_model)
        if self.args.heter_model:
            MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']  # more will be added
            init_model = None
        for idx in self.n_clients:
            curr_client = FedBaseAgent(self.args, init_model, self.logger, MIL_pool)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)
            print(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            self.logger.info(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            if self.args.heter_model and len(MIL_pool) > 0:
                MIL_pool.remove(curr_client.local_model_name)
            if len(MIL_pool) == 0:
                MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']


    def run(self, iter):
        # include training and testing
        best_accuracy = 0.
        train_acc_wt = 0.
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        local_weights = []
        self.logger.info(f'| Local Training Round |')
        local_model_train = True
        for idx in self.n_clients:
            start_time = time.time()
            w, agent_loss = self.clients[idx].local_train(idx)
            local_weights.append(deepcopy(w))
            end_time = time.time()
            print(f'Agent {idx} local test time: {end_time - start_time}')

        local_weights, local_acc = [], []
        list_acc_wt = [0] * len(self.n_clients)
        for idx in self.n_clients:
            results = self.clients[idx].local_test()
            agent_error = results['error']
            agent_auc = results['auc']
            agent_auprc = results['auprc']
            agent_mcc = results['mcc']
            local_acc.append(1-agent_error)
            logging_info = f'Agent: {idx}, Test Acc: {1-agent_error}, Test AUC: {agent_auc}, Test AUPRC: {agent_auprc}, Test MCC: {agent_mcc}'
            self.logger.info(logging_info)
            best_model = deepcopy(self.clients[idx].local_model)
            best_model_save_pth = os.path.join(self.args.results_dir, f"best_model_client_{idx}_iter_{iter}.pt")
            torch.save(best_model.state_dict(), best_model_save_pth)
            list_acc_wt[idx] = local_acc[idx] * self.weight_list[idx]
        train_acc_wt = sum(list_acc_wt)
        best_accuracy = sum(local_acc)/len(local_acc)
        return best_accuracy, train_acc_wt, local_acc