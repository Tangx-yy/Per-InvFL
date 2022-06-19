# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:24:25 2022

@author: tangx
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD, PFLL2Optimizer
from FLAlgorithms.users.userbase_InvPFL import User
import copy

# For Debugging
import numpy as np
from PIL import Image

# Implementation for pFeMe clients

class UserInvPFL(User):
    def __init__(self, device, numeric_id, dataset, algorithm, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate, irm_penalty_anneal_iters, irm_penalty_weight, times):
        super().__init__(device, numeric_id, dataset, algorithm, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs, irm_penalty_anneal_iters, irm_penalty_weight, times)

        if(model[2] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
            self.accuracy = self.total_accuracy
        elif(model[3] in ["CMnist", "WaterBird"]):
            self.loss = self.BCE_loss
            self.accuracy = self.total_accuracy_binary
        else:
            self.loss = nn.NLLLoss()
            self.accuracy = self.total_accuracy
        
        self.algorithm = algorithm
#        self.l2_weight = 0
        self.l2_weight = 0.0011
        self.K = K
        self.times = times
        self.learning_rate = learning_rate
        self.personal_learning_rate = personal_learning_rate
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.irm_penalty_weight = irm_penalty_weight
        if(model[2][0:6] == "resnet"):
            self.optimizer = PFLL2Optimizer([{'params': filter(lambda p: p.requires_grad, self.model['extractor'].parameters()), 'lr': self.personal_learning_rate, 'lamda': self.lamda}, \
                                              {'params': self.model['classifier'].parameters(), 'lr': self.personal_learning_rate, 'lamda': self.lamda}])
            self.inv_optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model['extractor'].parameters()), 'lr': self.learning_rate}, \
                                              {'params': self.model['classifier'].parameters(), 'lr': self.learning_rate}])
            self.per_optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model['extractor'].parameters()), 'lr': self.personal_learning_rate}, \
                                              {'params': self.model['classifier'].parameters(), 'lr': self.personal_learning_rate}])
        else:
        
            self.optimizer = PFLL2Optimizer([{'params': self.model['extractor'].parameters(), 'lr': self.personal_learning_rate, 'lamda': self.lamda}, \
                                              {'params': self.model['classifier'].parameters(), 'lr': self.personal_learning_rate, 'lamda': self.lamda}])
            self.inv_optimizer = optim.Adam([{'params': self.model['extractor'].parameters(), 'lr': self.learning_rate}, \
                                              {'params': self.model['classifier'].parameters(), 'lr': self.learning_rate}])
            self.per_optimizer = optim.Adam([{'params': self.model['extractor'].parameters(), 'lr': self.personal_learning_rate}, \
                                              {'params': self.model['classifier'].parameters(), 'lr': self.personal_learning_rate}])

    def set_grads(self, new_grads):
        for key, key_new in zip(self.model, new_grads):
            if isinstance(new_grads[key_new], nn.Parameter):
                for model_grad, new_grad in zip(self.model[key].parameters(), new_grads[key_new]):
                    model_grad.data = new_grad.data
            elif isinstance(new_grads[key_new], list):
                for idx, model_grad in enumerate(self.model[key].parameters()):
                    model_grad.data = new_grads[key_new][idx]

    def train(self, epochs, glob_iter, DRO_q):
        LOSS = -1
        self.model['extractor'].train()
        self.model['classifier'].train()
        local_R = 1 
        if self.algorithm in ["InvPFL", "IRM", "IRM-L2", "IRM-FT"] and glob_iter < self.irm_penalty_anneal_iters:
            local_R = 1
        else:
            local_R = self.local_epochs
        for epoch in range(1, self.local_epochs + 1):  # local update
#        for epoch in range(1, local_R + 1):  # local update
            
#            loss_g = torch.tensor(0.).cuda().requires_grad_()
            self.model['extractor'].train()
            self.model['classifier'].train()
            X, y = self.get_next_train_batch()
            
            # K = 30 # K is number of personalized steps
            ###### Select the personalized federated learning algorithms ###
            ### algorithms to update the personalized models ###
            if self.algorithm in ["pFedMe", "Ditto", "IRM-L2"]:
                # pFedMe and Ditto
                self.update_parameters(self.local_model)
                for i in range(self.K):
                    self.optimizer.zero_grad()
                    z = self.model['extractor'](X)
                    output = self.model['classifier'](z)
                    loss = self.loss(output, y)
                    loss.backward()
                    self.personalized_model_bar, _ = self.optimizer.step(self.local_model)
                # end of pFedMe and Ditto
            
            elif self.algorithm in ["FTFA", "Finetune", "IRM-FT"]:
                # finetuning
                self.update_parameters(self.local_model)
                for i in range(self.K):
                    self.per_optimizer.zero_grad()
                    z = self.model['extractor'](X)
                    output = self.model['classifier'](z)
                    loss = self.loss(output, y)
                    loss.backward()
                    self.per_optimizer.step()
                    for key_new, key_p in zip(self.model, self.personalized_model_bar):
                        for new_param, perweight in zip(self.model[key_new].parameters(), self.personalized_model_bar[key_p]):
                            perweight.data =  new_param.data.clone()
                # end of finetune
                
            elif self.algorithm == "InvPFL":
                # pFedInv
                for i in range(self.K):
                    self.update_parameters(self.local_model)
                    z_positive = self.model['extractor'](X).clone().detach()
                    self.update_parameters(self.personalized_model)
                    z_negative = self.model['extractor'](X).clone().detach()
                    #print("positive features:", torch.linalg.norm(z_positive, dim=1))
                    #print("negative features:", torch.linalg.norm(z_negative, dim=1))
              
                    self.update_parameters(self.local_model)
                    # self.update_parameters(self.personalized_model_bar)
                    self.per_optimizer.zero_grad()
                    z = self.model['extractor'](X)
                    output = self.model['classifier'](z)
                    per_loss = self.loss(output, y) + self.lamda * self.contrastive_loss(z, z_positive, z_negative)
                    #print("loss:", float(self.loss(output, y)))
                    #print("per_loss:", per_loss)
                    weight_norm = torch.tensor(0.).cuda()
                    for key in self.model:
                        for w in self.model[key].parameters():
                            weight_norm += w.norm().pow(2)
                    per_loss = per_loss + self.l2_weight * weight_norm
                    per_loss.backward()
                    self.per_optimizer.step()
                    for key_new, key_p in zip(self.model, self.personalized_model_bar):
                        for new_param, perweight in zip(self.model[key_new].parameters(), self.personalized_model_bar[key_p]):
                            perweight.data =  new_param.data.clone()
                # end of personalized invariant training
            elif self.algorithm in ["FedAvg", "IRM", "GroupDRO"]:
                if epoch == 0:
                    print(f"{self.algorithm} doesn't have any personalized part!")
            else:
                print(f"Error! No implementation for such algorithm: {self.algorithm}")
            
            
            ###### algorithms to update the global models ######
            if self.algorithm == "pFedMe":
                # global of pFedMe
                # update local weight after finding aproximate theta
                for key_p, key_l in zip(self.personalized_model_bar, self.local_model):
                    for new_param, localweight in zip(self.personalized_model_bar[key_p], self.local_model[key_l]):
                        localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)
                # improvement for algorithm pFedMe
                # setup the initial point of the personalized model
                self.update_parameters(self.local_model)
                # end updating
                # end of global pFedMe
                
            elif self.algorithm in ["FedAvg", "FTFA", "Ditto"]:
                # Train the global ditto and FedAvg
                self.update_parameters(self.local_model)
                self.inv_optimizer.zero_grad()
                z = self.model['extractor'](X)
                output = self.model['classifier'](z)
                loss_e = self.loss(output, y)
                loss_e.backward()
                self.inv_optimizer.step()
                for key_new, key_e in zip(self.model, self.local_model):
                    for new_param, localweight in zip(self.model[key_new].parameters(), self.local_model[key_e]):
                        localweight.data =  new_param.data.clone()
                # self.update_parameters(self.local_model)
                # end of training global ditto
            
            elif self.algorithm in ["IRM", "InvPFL", "IRM-L2", "IRM-FT"]:
                # global invariant learning
                # Using optimizer to update the local weights
                self.update_parameters(self.local_model)
                self.inv_optimizer.zero_grad()
                z = self.model['extractor'](X)
                output = self.model['classifier'](z)
                loss_g = self.loss(output, y)
                g_weight_norm = torch.tensor(0.).cuda()
                for key in self.model:
                    for w in self.model[key].parameters():
                        g_weight_norm += w.norm().pow(2)
                loss_g = loss_g + self.l2_weight * g_weight_norm
                loss_irm_penalty = self.irm_penalty(output, y)
                #print("irm penalty:", float(loss_irm_penalty))
                irm_weight = (self.irm_penalty_weight if glob_iter >= self.irm_penalty_anneal_iters else 1.0)
                loss_g_total = loss_g + irm_weight * loss_irm_penalty
                if irm_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss_g_total /= irm_weight
                
                #print("loss_g:", loss_g)
                loss_g_total.backward()
                self.inv_optimizer.step()
                for key_new, key_l in zip(self.model, self.local_model):
                    for new_param, localweight in zip(self.model[key_new].parameters(), self.local_model[key_l]):
                        localweight.data =  new_param.data.clone()
                # end of global IRM
            
            elif self.algorithm == "GroupDRO":
                # gloal DRO training
                self.update_parameters(self.local_model)
                self.inv_optimizer.zero_grad()
                z = self.model['extractor'](X)
                output = self.model['classifier'](z)
                loss_DRO_g = DRO_q * self.loss(output, y)
                loss_DRO_g.backward()
                self.inv_optimizer.step()
                for key_new, key_l in zip(self.model, self.local_model):
                    for new_param, localweight in zip(self.model[key_new].parameters(), self.local_model[key_l]):
                        localweight.data =  new_param.data.clone()
                
                LOSS = self.loss(output, y).clone().detach()
                # end of global DRO training
            
            else:
                print(f"Error! No implementation for such algorithm: {self.algorithm}")
                
            ###### Train the assistant pure local model ######
            if self.algorithm == "InvPFL":
                # Train the pure local model h_e
                self.update_parameters(self.personalized_model)
                self.inv_optimizer.zero_grad()
                z = self.model['extractor'](X)
                output = self.model['classifier'](z)
                loss_e = self.loss(output, y)
                loss_e.backward()
                self.inv_optimizer.step()
                for key_new, key_e in zip(self.model, self.personalized_model):
                    for new_param, perweight in zip(self.model[key_new].parameters(), self.personalized_model[key_e]):
                        perweight.data =  new_param.data.clone()
                self.update_parameters(self.local_model)
                # end of training for the local model h_e

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        # update the parameters of self.model for the global aggregation
        self.update_parameters(self.local_model)

        return LOSS