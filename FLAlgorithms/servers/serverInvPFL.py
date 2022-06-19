# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:43:44 2022

@author: tangx
"""

import torch
import os

from FLAlgorithms.users.userInvPFL import UserInvPFL
from FLAlgorithms.servers.serverbase_InvPFL import Server
from utils.model_utils import read_data, read_user_data, prepare_confounder_data

import numpy as np
from numpy import linalg as LA
import h5py
 
# Implementation for pFedMe Server

class InvPFL(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, irm_penalty_anneal_iters, irm_penalty_weight, groupdro_eta, times):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks,
                         local_epochs, optimizer, num_users, irm_penalty_anneal_iters, irm_penalty_weight, groupdro_eta, times)

        # Initialize data for all  users
        if dataset in ["CMnist", "Mnist", "WaterBird"]:
            data = read_data(dataset)
        total_users = len(data[0])
        self.model = model
        self.model_type = model[2]
        #print('model name:')
        #print(self.model[1])
        self.algorithm = algorithm
        self.irm_iters = irm_penalty_anneal_iters
        self.irm_w = irm_penalty_weight
        # for global DRO
        self.DRO_q = ((1/num_users)*torch.ones(num_users)).to(device)
        self.groupdro_eta = groupdro_eta
        self.K = K
        self.times = times
        self.personal_learning_rate = personal_learning_rate
        self.lamda_ps = lamda*np.ones(total_users)
        self.K_ps = self.K*np.ones(total_users)
        # for i in range(11):
            # self.lamda_ps[i+8] = lamda + 5
        # self.lamda_ps[0] = lamda + 1
        # self.lamda_ps[1] = lamda + 1
        # self.lamda_ps[2] = lamda + 1
        # self.lamda_ps[3] = lamda + 1
        # self.lamda_ps[4] = lamda + 1
        # self.lamda_ps[5] = lamda + 1
        # self.lamda_ps[6] = lamda + 2
        # self.lamda_ps[7] = lamda + 1
        # self.lamda_ps[8] = lamda + 1.75
        # self.lamda_ps[9] = lamda + 1
        # self.lamda_ps[total_users-2] = lamda - 2
        # self.K_ps[0] = self.K - 2
        # # self.K_ps[total_users-2] = self.K + 9
        # lamda_var = 1.90
        # print("lamdas-Ks:")
        # print(self.lamda_ps)
        # print(self.K_ps)
        for i in range(total_users):
            if dataset in ["CMnist", "Mnist", "WaterBird"]:
                id, train , test = read_user_data(i, data, dataset)
            elif dataset in ["PACS"]:
                id = 'f_{0:05d}'.format(i)
                train = prepare_confounder_data(i, dataset, self.model_type, train=True)
                test = prepare_confounder_data(i, dataset, self.model_type, train=False)
            else:
                print("Error: not valid dataset.")
            # 0.3---15---29.7
            # lamda_p = np.random.normal(lamda, lamda_var)
            # if lamda_p < 0.1:
            #     lamda_p = 0.1
            lamda_p = self.lamda_ps[i]
            K_p = int(self.K_ps[i])
            user = UserInvPFL(device, id, dataset, algorithm, train, test, model, batch_size, learning_rate, beta, lamda_p, local_epochs, optimizer, K_p, personal_learning_rate, irm_penalty_anneal_iters, irm_penalty_weight, times)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")
        self.belongs_to = np.zeros(len(self.users))
        # for idx in range(len(self.users)):
        #     if idx < (len(self.users)/self.gm_Ks):
        #         self.belongs_to[idx] = 0
        #     else:
        #         self.belongs_to[idx] = 1
        self.km_iters = np.zeros(self.num_glob_iters)
        self.km_groups = np.zeros((len(self.users), self.num_glob_iters))
        self.models_dis = np.zeros(self.num_glob_iters)

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = {'extractor': [], 'classifier': []}
        for key in self.glob_model:
            for param in self.glob_model[key].parameters():
                if param.grad is None:
                    grads[key].append(torch.zeros_like(param.data))
                else:
                    grads[key].append(param.grad)
        for user in self.users:
            user.set_grads(grads)

#    def send_grads(self):
#        assert (self.users is not None and len(self.users) > 0)
#        for glob_idx in range(self.gm_Ks):
#            grads = []
#            for param in self.glob_models[glob_idx].parameters():
#                if param.grad is None:
#                    grads.append(torch.zeros_like(param.data))
#                else:
#                    grads.append(param.grad)
#            for user in self.users:
#                if self.belongs_to[int(user.id[5:7])] == glob_idx:
#                    user.set_grads(grads)


    def train(self):
        loss = []
#        loss_user = torch.zeros(num_users).to(device)
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",self.times, '---', glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate(glob_iter)

            # do update for all users not only selected users
            user_idx = 0
            for user in self.users:
                loss_user = user.train(self.local_epochs, glob_iter, self.DRO_q[user_idx]) #* user.train_samples
                loss.append(loss_user)
                user_idx += 1
            
#            self.models_dis[glob_iter] = self.model_distance()
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter, self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            
            # Aggregate the personalized models
            if self.algorithm == "GroupDRO":
                # global DRO
                for user_idx in range(len(self.users)):
                    self.DRO_q[user_idx] *= (self.groupdro_eta * loss[user_idx].data).exp()
                self.DRO_q /= self.DRO_q.sum()
                self.personalized_aggregate_parameters(self.DRO_q)
                # end of DRO
            else:
                # global ERM
                self.personalized_aggregate_parameters()
                # end of ERM
            


        #print(loss)
        #print(self.km_iters)
        
        #print(self.km_groups)
        self.save_results()
        self.save_model()
        # print("Distances between models:")
    
  