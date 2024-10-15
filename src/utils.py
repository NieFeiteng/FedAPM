import copy
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets, transforms
from sampling import mnist_iid, q_label_skew,dir_label_skew, 
from torch.utils.data import ConcatDataset

import statistics

from torch.utils.model_zoo import tqdm

import pandas as pd
import sys
import os
import pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def c(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0)
        torch.nn.init.constant_(m.bias, 0)

def get_dataset(args):

    if args.dataset == 'crisis_mmd' or args.dataset == 'ku_har' or args.dataset == 'crema_d':
        combined_data = None
        user_groups = [None] * args.num_users
        return combined_data, user_groups    
  
    
    elif args.dataset == 'cifar10':
        if args.dataset == 'cifar10':
            data_dir = '../data/cifar/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
            combined_data = ConcatDataset([train_dataset, test_dataset])

        if args.iid:
            user_groups = mnist_iid(combined_data, args.num_users)
        elif args.partition == 'dir-label-skew':
            user_groups = dir_label_skew(args.dataset, combined_data, args.num_users)
        elif args.partition == 'q-label-skew':
            user_groups = q_label_skew(args.dataset, combined_data, args.num_users, args.q)


    return combined_data, user_groups

def average_loss_acc(local_model, num_users, malicious_users):
    benign_users = list(set((range(num_users))) - set(malicious_users))
    num_benign_users = len(benign_users)
    train_loss_personal_local, train_loss_global_local, test_acc_personal_local, test_acc_global_local = [], [], [], []
    test_loss_personal_local, test_loss_global_local = [], []
    train_loss_hybrid_local, test_acc_hybrid_local, test_loss_hybrid_local = [], [], []
    for idx in benign_users:
        train_loss_personal_local.append(local_model[idx].train_personal_loss)
        train_loss_global_local.append(local_model[idx].train_global_loss)
        test_acc_personal_local.append(local_model[idx].test_acc_personal)
        test_acc_global_local.append(local_model[idx].test_acc_global)

        test_loss_personal_local.append(local_model[idx].test_personal_loss)
        test_loss_global_local.append(local_model[idx].test_global_loss)

        test_acc_hybrid_local.append(max(local_model[idx].test_acc_personal, local_model[idx].test_acc_global))
        test_loss_hybrid_local.append(min(local_model[idx].test_personal_loss, local_model[idx].test_global_loss))
        train_loss_hybrid_local.append(min(local_model[idx].train_personal_loss, local_model[idx].train_global_loss))

    train_loss_personal_avg = sum(train_loss_personal_local) / num_benign_users
    train_loss_global_avg = sum(train_loss_global_local) / num_benign_users
    train_loss_hybrid_avg = sum(train_loss_hybrid_local) / num_benign_users

    test_acc_personal_avg = sum(test_acc_personal_local) / num_benign_users
    test_acc_global_avg = sum(test_acc_global_local) / num_benign_users
    test_acc_hybrid_avg = sum(test_acc_hybrid_local) / num_benign_users

    test_loss_personal_avg = sum(test_loss_personal_local) / num_benign_users
    test_loss_global_avg = sum(test_loss_global_local) / num_benign_users
    test_loss_hybrid_avg = sum(test_loss_hybrid_local) / num_benign_users

    test_acc_personal_variance = statistics.variance(test_acc_personal_local)
    test_acc_global_variance = statistics.variance(test_acc_global_local)
    test_acc_hybrid_variance = statistics.variance(test_acc_hybrid_local)
    test_loss_personal_variance = statistics.variance(test_loss_personal_local)
    test_loss_global_variance = statistics.variance(test_loss_global_local)
    test_loss_hybrid_variance = statistics.variance(test_loss_hybrid_local)
    return (train_loss_hybrid_avg,test_acc_hybrid_avg, test_loss_hybrid_avg, train_loss_global_avg, train_loss_personal_avg,
            test_acc_personal_avg, test_acc_global_avg, test_loss_personal_avg, test_loss_global_avg, test_acc_personal_variance,
            test_acc_global_variance,test_loss_personal_variance,test_loss_global_variance, test_acc_hybrid_variance, test_loss_hybrid_variance)

def average_loss_acc_centralized(local_model, num_users, malicious_users):
    train_loss_local, test_acc_local = [], []
    F1_local, AUC_local = [], []
    test_loss_local = []
    benign_users = list(set((range(num_users))) - set(malicious_users))
    num_benign_users = len(benign_users)
    for idx in benign_users:
        train_loss_local.append(local_model[idx].train_loss)
        test_acc_local.append(local_model[idx].test_acc)
        test_loss_local.append(local_model[idx].test_loss)
        
        F1_local.append(local_model[idx].f1)
        AUC_local.append(local_model[idx].auc)

    train_loss_avg = sum(train_loss_local) / num_benign_users
    test_acc_avg = sum(test_acc_local) / num_benign_users
    test_loss_avg = sum(test_loss_local) / num_benign_users
    
    test_F1_avg = sum(F1_local) / num_benign_users
    test_AUC_avg = sum(AUC_local) / num_benign_users    

    test_acc_variance = statistics.variance(test_acc_local)
    test_loss_variance = statistics.variance(test_loss_local)

    return train_loss_avg, test_acc_avg, test_loss_avg, test_acc_variance, test_loss_variance, test_F1_avg, test_AUC_avg



def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model              : {args.model}')
    print(f'    Optimizer          : {args.optimizer}')
    print(f'    Framework          : {args.framework}')
    print(f'    Client selection   : {args.strategy}')
    print(f'    Attack             : {args.corrupted}')
    print(f'    Malicious fraction : {args.num_malicious}')

    print(f'    Global Rounds    : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    dataset            : {args.dataset}')
    print(f'    Data partition     : {args.partition}')
    print(f'    Num of users       : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Learning  Rate     : {args.lr}')
    print(f'    rho                : {args.rho}')
    print(f'    mu                 : {args.mu}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}\n')
    return

