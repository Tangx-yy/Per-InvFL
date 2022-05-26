# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:58:13 2022

@author: tangx
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
#from FLAlgorithms.servers.serverhypcluster import FedAvg
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedKm import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverInvPFL import InvPFL

from FLAlgorithms.servers.serverhypcluster import HypCluster

from FLAlgorithms.trainmodel.models_InvPFL import *
from FLAlgorithms.trainmodel.models_fe_resnet import resnet18, resnet34, resnet50
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks,
         local_epochs, optimizer, numusers, K, personal_learning_rate, irm_penalty_anneal_iters, irm_penalty_weight, groupdro_eta, times, gpu):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    for i in range(times):
        print("---------------Running time:------------",i)
#        seed_time = int(10000 * torch.rand(1))
#        torch.manual_seed(seed_time)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), Classifier().to(device), model, dataset
            elif(dataset == "CMnist"):
                model = Mclr_Logistic().to(device), Classifier().to(device), model, dataset
            else:
                model = Mclr_Logistic(60,10).to(device), Classifier().to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), Classifier().to(device), model, dataset
            elif(dataset == "CMnist"):
                model = Net().to(device), Classifier().to(device), model, dataset
            elif(dataset == "Cifar10"):
                model = CifarNet().to(device), Classifier().to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN(input_dim=28*28).to(device), Classifier(output_dim=10).to(device), model, dataset
            elif(dataset == "CMnist"):
                model = DNN(input_dim=2*14*14).to(device), Classifier(output_dim=1).to(device), model, dataset
            else: 
                model = DNN(60,20,10).to(device), Classifier().to(device), model
                
        if(model[0:6] == "resnet"):
            if(dataset == "WaterBird"):
                f_extractor = resnet18()
                # f_extractor = resnet18(pretrained=True)
                feature_dim = f_extractor.output_dim
                model = f_extractor.to(device), Classifier(input_dim=feature_dim, output_dim=1).to(device), model, dataset

        # select algorithm
        if(algorithm == "FedAvg0"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "pFedMe0"):
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks, local_epochs, optimizer, numusers, K, personal_learning_rate, i)

        if(algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks, local_epochs, optimizer, numusers, K, personal_learning_rate, irm_penalty_anneal_iters, irm_penalty_weight, groupdro_eta, i)
        
        if(algorithm == "HypCluster0"):
            server = HypCluster(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks, local_epochs, optimizer, numusers, i)
        
        if(algorithm in ["InvPFL", "FedAvg", "pFedMe", "Ditto", "FTFA", "IRM", "GroupDRO", "IRM-L2", "IRM-FT"]):
            server = InvPFL(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, gm_Ks, local_epochs, optimizer, numusers, K, personal_learning_rate, irm_penalty_anneal_iters, irm_penalty_weight, groupdro_eta, i)

        server.train()
        server.test()

    # Average data 
    if(algorithm == "PerAvg"):
        algorithm == "PerAvg_p"
    if(algorithm == "pFedMe0"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedMe_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)
    if(algorithm in ["InvPFL", "FedAvg", "pFedMe", "Ditto", "FTFA", "IRM", "GroupDRO", "IRM-L2", "IRM-FT"]):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda, learning_rate=learning_rate, beta = beta, algorithms=(algorithm+"_p"), batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate, irm_iters=irm_penalty_anneal_iters, irm_w=irm_penalty_weight, dro_eta=groupdro_eta, times = times)
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate, irm_iters=irm_penalty_anneal_iters, irm_w=irm_penalty_weight, dro_eta=groupdro_eta, times = times)
    print("the dim of latent features:", feature_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WaterBird", choices=["CMnist", "Mnist", "Synthetic", "Cifar10", "WaterBird"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["dnn", "mclr", "cnn", "resnet18", "resnet50"])
    parser.add_argument("--batch_size", type=int, default=200)
#    parser.add_argument("--batch_size", type=int, default=1000)
#    parser.add_argument("--learning_rate", type=float, default=0.03, help="Local learning rate")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Local learning rate")
    # parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    # parser.add_argument("--beta", type=float, default=0.001, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    # parser.add_argument("--lamda", type=float, default=0.01, help="Regularization term")
    parser.add_argument("--lamda", type=float, default=1.0, help="Regularization term")
    # parser.add_argument("--num_global_iters", type=int, default=111)
    parser.add_argument("--num_global_iters", type=int, default=201)
    # Number of global models
    parser.add_argument("--gm_Ks", type=int, default=1)
    parser.add_argument("--local_epochs", type=int, default=5)
    # parser.add_argument("--local_epochs", type=int, default=10)
    # parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FTFA",choices=["FedAvg", "pFedMe", "PerAvg", "Ditto", "FTFA", "IRM", "GroupDRO", "InvPFL", "IRM-L2", "IRM-FT"]) 
    parser.add_argument("--numusers", type=int, default=4, help="Number of Users per round")
    # parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--K", type=int, default=2, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.0001, help="Persionalized learning rate to caculate theta aproximately using K steps")
    # parser.add_argument("--personal_learning_rate", type=float, default=0.006, help="Persionalized learning rate to caculate theta aproximately using K steps")
    # parser.add_argument("--times", type=int, default=10, help="running time")
    # Hyperparameters for invariant learning
    parser.add_argument("--irm_penalty_anneal_iters", type=int, default=12)
    # parser.add_argument("--irm_penalty_anneal_iters", type=int, default=0)
    # parser.add_argument("--irm_penalty_weight", type=float, default=0)
#    parser.add_argument("--irm_penalty_weight", type=float, default=2e7)
    parser.add_argument("--irm_penalty_weight", type=float, default=1e9)
    # end of hyperparameters for invariant learning
    # hyperparameters for groupDRO
    parser.add_argument("--groupdro_eta", type=float, default=1e-2)
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of global models       : {}".format(args.gm_Ks))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        gm_Ks = args.gm_Ks,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        irm_penalty_anneal_iters = args.irm_penalty_anneal_iters,
        irm_penalty_weight = args.irm_penalty_weight,
        groupdro_eta = args.groupdro_eta,
        times = args.times,
        gpu=args.gpu
        )
