# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:26:49 2022

@author: tangx
"""

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, dataset, algorithm, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , lamda = 0, local_epochs = 0, irm_penalty_anneal_iters=0, irm_penalty_weight=0, glob_iters=0):

        self.device = device
        self.model = {'extractor': copy.deepcopy(model[0]), 'classifier': copy.deepcopy(model[1])}
        self.id = id  # integer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        if dataset in ["CMnist", "Mnist", "WaterBird"]:
            self.train_samples = len(train_data)
            print("number of train samples:", self.train_samples)
            self.test_samples = len(test_data)
            print("number of test samples:", self.test_samples)
            self.trainloader = DataLoader(train_data, self.batch_size)
            self.testloader =  DataLoader(test_data, self.batch_size)
            self.testloaderfull = DataLoader(test_data, self.test_samples)
            self.trainloaderfull = DataLoader(train_data, self.train_samples)
            self.iter_trainloader = iter(self.trainloader)
            self.iter_testloader = iter(self.testloader)
        elif dataset in ["PACS"]:
            loader_kwargs = {'batch_size':self.batch_size}
            self.trainloader = train_data.get_loader(train=True, reweight_groups=False, **loader_kwargs)
            self.testloader = test_data.get_loader(train=True, reweight_groups=False, **loader_kwargs)
        
        # for invariant learning
        self.glob_iters = glob_iters
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.irm_penalty_weight = irm_penalty_weight
        
        # those parameters are for persionalized federated learing.
#        self.local_model = copy.deepcopy(list(self.model.parameters()))
#        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
#        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        
        self.local_model = {'extractor': copy.deepcopy(list(self.model['extractor'].parameters())), 'classifier': copy.deepcopy(list(self.model['classifier'].parameters()))}
        self.personalized_model = {'extractor': copy.deepcopy(list(self.model['extractor'].parameters())), 'classifier': copy.deepcopy(list(self.model['classifier'].parameters()))}
        self.personalized_model_bar = {'extractor': copy.deepcopy(list(self.model['extractor'].parameters())), 'classifier': copy.deepcopy(list(self.model['classifier'].parameters()))}
        print("########## user model copy!!! ############")
    
    def BCE_loss(self, logits, y, reduction='mean'):
        # print("size of logits:", logits.size())
        # print("size of y:", y.size())
        y = y.view(logits.size())
        return nn.functional.binary_cross_entropy_with_logits(logits, y.float(), reduction=reduction)
    
    def total_accuracy_binary(self, logits, y):
        y = y.view(logits.size())
        preds = (logits > 0.).float()
        acc_total = float((torch.sum(((preds - y).abs() < 1e-2).float())).item())
        return acc_total
    
    def total_accuracy(self, logits, y):
        acc_total = float((torch.sum(torch.argmax(logits, dim=1) == y)).item())
        return acc_total
    
    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()
    
    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        # loss = self.BCE_loss(logits * scale, y)
        loss = self.loss(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def cosine_sim(self, a, b, eps=1e-08):
        '''Returns the cosine similarity between two arrays a and b
        '''
        norm_a = torch.linalg.norm(a, dim=1)
        norm_b = torch.linalg.norm(b, dim=1)
        epsilon = eps * torch.ones(norm_b.size()).cuda()
#        print("norm a:", norm_a)
#        print("norm b:", norm_b)
#        print("size of norm:", norm_a.size())
        z_dim = a.size()
#        norm_a = norm_a.view(z_dim[0], 1)
#        norm_b = norm_b.view(z_dim[0], 1)
        a = a.view(-1, 1, z_dim[1])
        b = b.view(-1, z_dim[1], 1)
        dot_product = torch.bmm(a, b)
#        print("dot_product:", dot_product)
        dot_product = dot_product.view(norm_a.size())
#        print("size of norm_a:", norm_a.size())
#        print("dot_product:", dot_product)
        return dot_product * 1.0 / torch.max(norm_a * norm_b, epsilon)
    
    def contrastive_loss(self, anchor, positive, negative):
        weight_negative = 0.2
        positive_sim = self.cosine_sim(anchor, positive)
        negative_sim = self.cosine_sim(anchor, negative)
#        print("norm of positive:", torch.linalg.norm(anchor, dim=1))
#        print("positive aim:", positive_sim)
#        print("negative aim:", negative_sim)
        contrastive_loss = -1.0 * torch.log(torch.exp(positive_sim) / (torch.exp(positive_sim) + weight_negative*torch.exp(negative_sim)))
#        print("positive sim:", positive_sim)
#        print("exp of negative sim:", torch.exp(negative_sim))
#        print("contrastive loss:", contrastive_loss)
        return contrastive_loss.mean()
    
    def set_parameters(self, model):
        for key_sm, key_m in zip(self.model, model):
            for old_param, new_param, local_param in zip(self.model[key_sm].parameters(), model[key_m].parameters(), self.local_model[key_sm]):
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for key in self.model:
            for param in self.model[key].parameters():
                param.detach()
        return {'extractor': self.model['extractor'].parameters(), 'classifier': self.model['classifier'].parameters()}
    
    def clone_model_paramenter(self, param, clone_param):
        for key, key_c in zip(param, clone_param):
            for param, clone_param in zip(param[key], clone_param[key_c]):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for key, key_new in zip(self.model, new_params):
            for param , new_param in zip(self.model[key].parameters(), new_params[key_new]):
                param.data = new_param.data.clone()

    def get_grads(self):
        grads = {'extractor': [], 'classifier': []}
        for key in self.model:
            for param in self.model[key].parameters():
                if param.grad is None:
                    grads[key].append(torch.zeros_like(param.data))
                else:
                    grads[key].append(param.grad.data)
        return grads

    def test(self):
        self.model['extractor'].eval()
        self.model['classifier'].eval()
        test_acc = 0
        for x, y in self.testloaderfull:
        # for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            z = self.model['extractor'](x)
            output = self.model['classifier'](z)

            test_acc += self.accuracy(output, y)
            #test_acc += float((torch.sum(torch.argmax(output, dim=1) == y)).item())
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def train_error_and_loss(self, glob_iter=0):
        self.model['extractor'].eval()
        self.model['classifier'].eval()
        train_acc = 0
        loss = 0
        loss_irm_penalty = 0
        num = 0
        for x, y in self.trainloaderfull:
        # for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            z = self.model['extractor'](x)
            output = self.model['classifier'](z)
            # train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # loss += self.loss(output, y)
            train_acc += self.accuracy(output, y)
            #train_acc += float((torch.sum(torch.argmax(output, dim=1) == y)).item())
            loss_y = float(self.loss(output, y))
            loss_irm_penalty = float(self.irm_penalty(output, y))
            # print("global iteration:", glob_iter)
            irm_penalty_weight = (self.irm_penalty_weight if glob_iter >= self.irm_penalty_anneal_iters else 1.0)
            loss_total = loss_y + irm_penalty_weight * loss_irm_penalty
            if irm_penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss_total /= irm_penalty_weight
            loss += loss_total
            # loss += loss_y
            num += y.shape[0]
            #print(self.id + ", Train Accuracy:", train_acc)
        print(self.id + ", Train IRM Penalty Loss:", loss_irm_penalty)
        # return train_acc, loss , self.train_samples
        return train_acc, loss , num
    
    def test_persionalized_model(self):
        self.model['extractor'].eval()
        self.model['classifier'].eval()
        test_acc = 0
        self.update_parameters(self.personalized_model_bar)
        # self.update_parameters(self.personalized_model)
        for x, y in self.testloaderfull:
        # for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            z = self.model['extractor'](x)
            output = self.model['classifier'](z)
            test_acc += self.accuracy(output, y)
            
            #test_acc += float((torch.sum(torch.argmax(output, dim=1) == y)).item())
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model['extractor'].eval()
        self.model['classifier'].eval()
        train_acc = 0
        loss = 0
        loss_con = 0
        num = 0
        # self.update_parameters(self.personalized_model)
        self.update_parameters(self.personalized_model_bar)
#        print("trainloaderfull:", self.trainloaderfull)
        for x, y in self.trainloaderfull:
        # for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            z = self.model['extractor'](x)
            output = self.model['classifier'](z)
            train_acc += self.accuracy(output, y)
            #train_acc += float((torch.sum(torch.argmax(output, dim=1) == y)).item())
            loss += float(self.loss(output, y))
            num += y.shape[0]
#            
            self.update_parameters(self.local_model)
            z_positive = self.model['extractor'](x).clone().detach()
            self.update_parameters(self.personalized_model)
            z_negative = self.model['extractor'](x).clone().detach()
            loss_con += float(self.contrastive_loss(z, z_positive, z_negative))
            # loss += float(self.loss(output, y) + self.lamda * self.contrastive_loss(z, z_positive, z_negative))
            # print(self.id + ", Train Accuracy:", train_acc)
        print(self.id + ", Train Loss:", loss/num)
        print(self.id + ", Train Contrastive Loss:", loss_con)
        self.update_parameters(self.local_model)
        # return train_acc, loss , self.train_samples
        return train_acc, loss , num
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))