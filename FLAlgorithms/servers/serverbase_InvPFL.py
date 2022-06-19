# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:51:31 2022

@author: tangx
"""

import torch
import os
import numpy as np
import h5py
import json
from utils.model_utils import Metrics
#from FLAlgorithms.servers.kmeans_tx import kmeans
import copy

class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, gm_Ks, local_epochs, optimizer, num_users, irm_penalty_anneal_iters, irm_penalty_weight, groupdro_eta, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.model_type = model[2]
        self.num_glob_iters = num_glob_iters
        self.gm_Ks = gm_Ks
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        #self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_per_train_acc_per, self.rs_per_acc_per = [], []
        self.rs_per_train, self.rs_per_test = [], []
        self.times = times
        self.glob_model = {'extractor': copy.deepcopy(model[0]), 'classifier': copy.deepcopy(model[1])}
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
        # Initialize multiple global models in the server
#        self.glob_models = []
#        g_model = {'extractor': copy.deepcopy(model[0]), 'classifier': copy.deepcopy(model[1])}
#        for idx_gm in range(self.gm_Ks):
#            self.glob_models.append(g_model)
            
        #self.belongs_to
        self.km_iterations = 0
            
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for key in self.glob_model:
            for param in self.glob_model[key].parameters():
                param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for key_m, key_g in zip(self.glob_model, user_grad):
            for idx, param in enumerate(self.glob_model[key_m].parameters()):
                param.grad = param.grad + user_grad[key_g][idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.glob_model)
#    def send_parameters(self):
#        assert (self.users is not None and len(self.users) > 0)
#        for user in self.users:
#            #print(int(user.id[5:7]))
#            user.set_parameters(self.glob_models[int(self.belongs_to[int(user.id[5:7])])])

    def add_parameters(self, user, ratio):
        model = {'extractor': self.glob_model['extractor'].parameters(), 'classifier': self.glob_model['classifier'].parameters()}
        for key_g, key_u in zip(self.glob_model, user.get_parameters()):
            for server_param, user_param in zip(self.glob_model[key_g].parameters(), user.get_parameters()[key_u]):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for key in self.glob_model:
            for param in self.glob_model[key].parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.glob_model, os.path.join(model_path, "server_model_" + ".pt"))

#    def save_model(self):
#        model_path = os.path.join("models", self.dataset)
#        if not os.path.exists(model_path):
#            os.makedirs(model_path)
#        for glob_idx in range(self.gm_Ks):
#            torch.save(self.glob_models[glob_idx], os.path.join(model_path, "server_model_" + str(glob_idx) + ".pt"))
        

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.glob_model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def personalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for key_g, key_l in zip(self.glob_model, user.local_weight_updated):
            for server_param, user_param in zip(self.glob_model[key_g].parameters(), user.local_weight_updated[key_l]):
                server_param.data = server_param.data + user_param.data.clone() * ratio


    def personalized_aggregate_parameters(self, ratio=None):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = {'extractor': copy.deepcopy(list(self.glob_model['extractor'].parameters())), 'classifier': copy.deepcopy(list(self.glob_model['classifier'].parameters()))}
        for key in self.glob_model:
            for param in self.glob_model[key].parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        
        user_idx = 0
        for user in self.selected_users:
            if ratio is None:
                self.add_parameters(user, user.train_samples / total_train)
                #self.add_parameters(user, 1 / len(self.selected_users))
            else:
                self.add_parameters(user, ratio[user_idx])
            user_idx += 1

        # aaggregate avergage model with previous model using parameter beta 
        for key_pre, key in zip(previous_param, self.glob_model):
            for pre_param, param in zip(previous_param[key_pre], self.glob_model[key].parameters()):
                param.data = (1 - self.beta)*pre_param.data + self.beta*param.data


#    def persionalized_aggregate_parameters(self):
#        assert (self.users is not None and len(self.users) > 0)
#        
#        # store previous parameters
#        previous_param = []
#        for idx_gm in range(self.gm_Ks):
#            gm_param = copy.deepcopy(list(self.glob_models[idx_gm].parameters()))
#            previous_param.append(gm_param)
#        for idx_gm in range(self.gm_Ks):
#            for param in self.glob_models[idx_gm].parameters():
#                param.data = torch.zeros_like(param.data)
#
#        users_param = []
#        users_ratio = []
#        # users_idx = []
#        for user in self.users:
#            user_param = user.get_parameters()
#            users_param.append(user_param)
#            users_ratio.append(user.train_samples)
#            # users_idx.append(user.id)
#        # print('users_id:')
#        # print(users_idx)
#        #temp_gm_param, self.belongs_to, km_iterations = kmeans(users_param, self.gm_Ks)
#        print('start k-means')
#        #params = list(users_param[0])
#        #print('list:')
#        #print(len(params))
#        #print(params)
#        #print(self.model_type)
#        # for i in range(len(self.users)):
#        #     self.belongs_to[i] = int(i/20)
#        _, self.belongs_to, self.km_iterations = kmeans(users_param, users_ratio, previous_param, self.model_type)
#        # print('km_iterations:')
#        # print(self.km_iterations)
#        print('belongs_to:')
#        print(self.belongs_to)
#        
#        total_train = np.zeros(self.gm_Ks)
#        for user in self.selected_users:
#            total_train[int(self.belongs_to[int(user.id[5:7])])] += user.train_samples
#
#        for user in self.selected_users:
#            self.add_parameters(user, user.train_samples / total_train[int(self.belongs_to[int(user.id[5:7])])])
#            #self.add_parameters(user, 1 / len(self.selected_users))
#            #self.add_parameters(user, self.gm_Ks / len(self.selected_users))
#
#        # aaggregate avergage model with previous model using parameter beta
#        for glob_idx in range(self.gm_Ks):
#            for pre_param, param in zip(previous_param[glob_idx], self.glob_models[glob_idx].parameters()):
#                param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + "beta" + str(self.beta) + "_" + "lam" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" \
            + "T" + str(self.num_glob_iters) + "_R" + str(self.local_epochs) + "_irm-at" + str(self.irm_iters) + "_irm-w" + str(self.irm_w) + "_dro-eta" + str(self.groupdro_eta)
        # if(self.algorithm == "InvPFL" or self.algorithm == "InvPFL_p"):
        #     alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg + "_" + str(self.learning_rate) + "_" + "beta" + str(self.beta) + "_" + "lam" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" \
            + "T" + str(self.num_glob_iters) + "_R" + str(self.local_epochs) + "_irm-at" + str(self.irm_iters) + "_irm-w" + str(self.irm_w) + "_dro-eta" + str(self.groupdro_eta)
        # if(self.algorithm == "InvPFL" or self.algorithm == "InvPFL_p"):
        alg = alg + "_K" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()
        
        per_results_path = "./results/"+alg+'_per.json'
        dir_path = os.path.dirname(per_results_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        rs_personalization = {'rs_per_train':[], 'rs_per_test':[]}
        rs_personalization['rs_per_train'] = self.rs_per_train
        rs_personalization['rs_per_test'] = self.rs_per_test
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with open(per_results_path,'w') as outfile:
                json.dump(rs_personalization, outfile)

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            # print("user:", c.id)
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self, glob_iter=0):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss(glob_iter) 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self, glob_iter=0):
        stats = self.test()  
        stats_train = self.train_error_and_loss(glob_iter)
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        # train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        self.rs_per_train = stats_train
        self.rs_per_test = stats
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        per_acc = np.divide(stats[2], stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        per_train_acc = np.divide(stats_train[2], stats_train[1])
        per_train_ratio = stats_train[1]/np.sum(stats_train[1])
        train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        # train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_per_acc_per.append(per_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_per_train_acc_per.append(per_train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Personal Accurancy: ", per_acc)
        print("Average Personal Accurancy: ", glob_acc)
        print("Personal Trainning Accurancy: ", per_train_acc)
        print("Personal Trainning Ratio: ", per_train_ratio)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        # train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
