# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:58:08 2022

@author: csxtang
"""

"""Build environments."""
import attr
import numpy as np
import random
import torch
from torchvision import datasets
import json
import os

def _make_environment(images, labels, e, flags_label_noise = 0.25):
    
    # NOTE: low e indicates a spurious correlation from color to (noisy) label

    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()

    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1

    samples = dict()
    # 2x subsample for computational convenience
    subsample = 2
    images = images.reshape((-1, 28, 28))[:, ::subsample, ::subsample]
#    images = images.reshape((-1, 28, 28))[:, ::4, ::4]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    samples.update(preliminary_labels=labels)
    label_noise = torch_bernoulli(flags_label_noise, len(labels))
    labels = torch_xor(labels, label_noise)
    samples.update(final_labels=labels)
    samples.update(label_noise=label_noise)
    # Assign a color based on the label; flip the color with probability e
    color_noise = torch_bernoulli(e, len(labels))
    colors = torch_xor(labels, color_noise)
    samples.update(colors=colors)
    samples.update(color_noise=color_noise)
#    # Apply the color to the background
#    backs = 255.0*torch.ones(images.shape)
#    color_back = colors
#    backs_batch = torch.transpose(backs, 0, 2)
#    color_backs_batch = backs_batch * color_back
#    color_backs = torch.transpose(color_backs_batch, 0, 2)
#    images = torch.stack([images, color_backs], dim=1)
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    # Coloring ends
    images = (images.float() / 255.)
    labels = labels[:, None]
#    if cuda and torch.cuda.is_available():
#      images = images.cuda()
#      labels = labels.cuda()
    samples.update(images=images, labels=labels)
    return samples

def construct_dataset():
    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 4
    NUM_ENVS = NUM_USERS
    # Setup directory for train/test data
    train_path = './data/train/c_fmnist_train.json'
    test_path = './data/test/c_fmnist_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    mnist = datasets.FashionMNIST('~/datasets/fmnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    
    # Setup the parameters of environments
#    train_e = np.zeros(NUM_ENVS)
#    test_e = 1.0*np.ones(NUM_ENVS)
    if NUM_ENVS == 2:
        train_e = np.array([0.10, 0.20])
        test_e = np.array([0.90, 0.90])
    elif NUM_ENVS == 4:
        train_e = np.array([0.10, 0.15, 0.20, 0.25])
#        train_e = np.array([0.10, 0.20, 0.30, 0.40])
        test_e = np.array([0.90, 0.90, 0.90, 0.90])
    else:
        train_e = np.zeros(NUM_ENVS)
        test_e = 1.0*np.ones(NUM_ENVS)
        print("Warning: the number of users is:", NUM_ENVS)
    
    
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    data_DS = NUM_USERS
    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        
        env_train = _make_environment(mnist_train[0][i::data_DS], mnist_train[1][i::data_DS], train_e[i])
        env_test = _make_environment(mnist_val[0][i::data_DS], mnist_val[1][i::data_DS], test_e[i])
        X_train_pre = env_train['images']
        y_train_pre = env_train['labels']
        X_test_pre = env_test['images']
        y_test_pre = env_test['labels']
        
        # Rotation
        if i > (0+10*(4-NUM_USERS)):
            X_train = torch.rot90(X_train_pre, i, [2, 3])
            X_test = torch.rot90(X_test_pre, i, [2, 3])
            print("Rotation:", 90*i)
        else:
            X_train = X_train_pre
            X_test = X_test_pre
        
        X_train_final = X_train.cpu().numpy().tolist()
        y_train_final = y_train_pre.cpu().numpy().tolist()
        X_test_final = X_test.cpu().numpy().tolist()
        y_test_final = y_test_pre.cpu().numpy().tolist()
        train_len = len(y_train_final)
        test_len = len(y_test_final)
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X_train_final, 'y': y_train_final}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test_final, 'y': y_test_final}
        test_data['num_samples'].append(test_len)
        
    print("Num_samples:", train_data['num_samples'])
    print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

#    return train_data, test_data
    return NUM_USERS
#    envs = [
#        _make_environment(mnist_train[0][::data_DS], mnist_train[1][::data_DS], flags.train_env_1__color_noise),
#        _make_environment(mnist_train[0][1::data_DS], mnist_train[1][1::data_DS], flags.train_env_2__color_noise),
#        _make_environment(mnist_val[0][::int(data_DS/2)], mnist_val[1][::int(data_DS/2)], flags.test_env__color_noise)
#        ]


construct_dataset()
#train_data, test_data = construct_dataset()
#sample_X_train = train_data['user_data'][train_data['users'][0]]['x'][1234]
#sample_X_test = test_data['user_data'][test_data['users'][0]]['x'][234]
#sample_X90_train = train_data['user_data'][train_data['users'][1]]['x'][1234]
#sample_X90_test = test_data['user_data'][test_data['users'][1]]['x'][234]
#sample_X180_train = train_data['user_data'][train_data['users'][2]]['x'][1234]
#sample_X180_test = test_data['user_data'][test_data['users'][2]]['x'][234]
#sample_X270_train = train_data['user_data'][train_data['users'][3]]['x'][1234]
#sample_X270_test = test_data['user_data'][test_data['users'][3]]['x'][234]
#
#sample_y_train = train_data['user_data'][train_data['users'][0]]['y'][1234]
#sample_y_test = test_data['user_data'][test_data['users'][0]]['y'][234]
#sample_y90_train = train_data['user_data'][train_data['users'][1]]['y'][1234]
#sample_y90_test = test_data['user_data'][test_data['users'][1]]['y'][234]
#sample_y180_train = train_data['user_data'][train_data['users'][2]]['y'][1234]
#sample_y180_test = test_data['user_data'][test_data['users'][2]]['y'][234]
#sample_y270_train = train_data['user_data'][train_data['users'][3]]['y'][1234]
#sample_y270_test = test_data['user_data'][test_data['users'][3]]['y'][234]

print("Finish Generating Samples")
