import torch
import random
import numpy as np
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from constants import feature_len_dict, num_class_dict, max_class_dict
from dataload_manager_multimodal import DataloadManager
from models import ImageTextClassifier, MMDatasetGenerator

from types import SimpleNamespace

def parse_args_for_multimodal(dataset_t, alpha_t: float=1.0):
    path_conf = {
        "data_dir": f'../data/{dataset_t}',
        "output_dir": f'../data/output'
    }

    args = SimpleNamespace()
    args.data_dir = path_conf['output_dir']
    args.acc_feat = 'acc'
    args.gyro_feat = 'gyro'
    args.text_feat = 'mobilebert'
    args.img_feat = 'mobilenet_v2'
    args.audio_feat = 'mfcc'
    args.video_feat = 'mobilenet_v2'
    args.batch_size = 100
    args.alpha = alpha_t
    args.modality = 'multimodal'
    args.dataset = dataset_t

    return args


def get_multimodal_dataloaders(dataset_t, user_id, alpha_t: float=1.0):
    args_t = parse_args_for_multimodal(dataset_t, alpha_t)      
    if dataset_t == 'hateful_memes' or dataset_t == 'crisis_mmd':  
        return get_dataloaders_for_text_img(user_id, args_t)  
    elif dataset_t == 'uci_har':
        return get_dataloaders_for_uci_har(user_id, args_t)  
    elif dataset_t == 'ku_har':
        return get_dataloaders_for_ku_har(user_id, args_t, fold_idx = 1)  
    elif dataset_t == 'crema_d':
        return get_dataloaders_for_crema_d(user_id, args_t, fold_idx = 1)      
          

def get_dataloaders_for_text_img(user_id, args):
    dm = DataloadManager(args)
    
    dm.get_client_ids()
    client_id = dm.client_ids[user_id]
    for client_id in tqdm(dm.client_ids):
        if client_id not in ['test', 'dev'] and client_id.isdigit() and int(client_id) == user_id:
            img_dict = dm.load_img_feat(client_id=client_id)
            text_dict = dm.load_text_feat(client_id=client_id)    

    dm.get_label_dist(img_dict, user_id)

    default_feat_shape_a = np.array([1, feature_len_dict["mobilenet_v2"]])
    default_feat_shape_b = np.array([32, feature_len_dict["mobilebert"]])
    
    test_img_dict = dm.load_img_feat(client_id='test')
    test_text_dict = dm.load_text_feat(client_id='test')

    train_loader = dm.set_dataloader(
        img_dict, 
        text_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )
    
    test_loader = dm.set_dataloader(
        test_img_dict, 
        test_text_dict,
        shuffle=False,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )
    

    # return train_loader
    return train_loader, test_loader

def get_dataloaders_for_uci_har(user_id, args):
    dm = DataloadManager(args)
    dm.get_client_ids()

    client_id = dm.client_ids[user_id]
    acc_dict, test_acc_dict = dm.load_acc_feat(client_id=client_id, fold_idx=1)
    gyro_dict, test_gyro_dict = dm.load_gyro_feat(client_id=client_id, fold_idx=1)
    
    dm.get_label_dist(gyro_dict, client_id)
    
    default_feat_shape_a=np.array([128, feature_len_dict['acc']]),
    default_feat_shape_b=np.array([128, feature_len_dict['gyro']]),

    train_loader = dm.set_dataloader(
        acc_dict,
        gyro_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )
    
    dm.get_label_dist(test_gyro_dict, client_id)

    test_loader = dm.set_dataloader(
        test_acc_dict,
        test_gyro_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )    
    
    return train_loader, test_loader



def get_dataloaders_for_ku_har(user_id, args, fold_idx: int=1):
    dm = DataloadManager(args)
    dm.get_client_ids()

    start_index = user_id 
    client_ids = [
        dm.client_ids[start_index],
        dm.client_ids[start_index + 19],
        dm.client_ids[start_index + 39],
        dm.client_ids[start_index + 59]
    ]
    

    acc_dict = {}
    test_acc_dict = {}
    gyro_dict = {}
    test_gyro_dict = {}

    for client_id in client_ids:
        acc_data, test_acc_data = dm.load_acc_feat(client_id=client_id, fold_idx=fold_idx)
        gyro_data, test_gyro_data = dm.load_gyro_feat(client_id=client_id, fold_idx=fold_idx)

        if len(acc_dict) != 0 or  len(test_acc_dict) != 0 or  len(gyro_dict) != 0 or  len(test_gyro_dict) != 0:       
            acc_dict = acc_dict + acc_data
            test_acc_dict = test_acc_dict + test_acc_data
            gyro_dict = gyro_dict + gyro_data
            test_gyro_dict = test_gyro_dict + test_gyro_data
        else:
            acc_dict = acc_data
            test_acc_dict = test_acc_data
            gyro_dict = gyro_data
            test_gyro_dict = test_gyro_data        
    
    default_feat_shape_a=np.array([256, feature_len_dict['acc']]),
    default_feat_shape_b=np.array([256, feature_len_dict['gyro']]),

    train_loader = dm.set_dataloader(
        acc_dict,
        gyro_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )
    

    test_loader = dm.set_dataloader(
        test_acc_dict,
        test_gyro_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )    
    return train_loader, test_loader


def get_dataloaders_for_crema_d(user_id, args, fold_idx: int=1):
    dm = DataloadManager(args)
    dm.get_client_ids(fold_idx=fold_idx)
   
   
    client_id = dm.client_ids[user_id]
    audio_dict, test_audio_dict = dm.load_audio_feat(client_id=client_id, fold_idx=fold_idx)
    video_dict, test_video_dict = dm.load_video_feat(client_id=client_id, fold_idx=fold_idx)
    
    dm.get_label_dist(audio_dict, client_id)
    

    default_feat_shape_a = np.array([600, feature_len_dict["mfcc"]]),
    default_feat_shape_b = np.array([6, feature_len_dict["mobilenet_v2"]]),

    train_loader = dm.set_dataloader(
        audio_dict, 
        video_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )
    
    test_loader = dm.set_dataloader(
        test_audio_dict, 
        test_video_dict,
        shuffle=True,
        client_sim_dict=None,
        default_feat_shape_a=default_feat_shape_a,
        default_feat_shape_b=default_feat_shape_b
    )
    
    return train_loader, test_loader
