import numpy as np
from torchvision import datasets, transforms
import torch
import math
from collections import defaultdict
from torch.utils.data import ConcatDataset

def dir_label_skew(dataset_name, dataset, num_users):
    if dataset_name == 'cifar10':
        labels_train = np.array(dataset.datasets[0].targets)
        labels_test = np.array(dataset.datasets[1].targets)
        labels = np.concatenate((labels_train, labels_test), axis=0)

    num_labels = len(np.unique(labels))
    beta = 0.5
    label_to_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    user_groups = defaultdict(list)

    for label, indices in label_to_indices.items():
        label_count = len(indices)
        label_distribution = np.random.dirichlet(np.ones(num_users) * beta)
        for i, client_probs in enumerate(label_distribution):
            sampled_indices = np.random.choice(indices, size=int(client_probs * label_count), replace=False)
            user_groups[i].extend(sampled_indices)

    for client, indices in user_groups.items():
        user_groups[client] = np.array(indices)
    return user_groups

def q_label_skew(dataset_name, dataset, num_users, q):
    if dataset_name == 'cifar10':
        labels_train = np.array(dataset.datasets[0].targets)
        labels_test = np.array(dataset.datasets[1].targets)
        labels = np.concatenate((labels_train, labels_test), axis=0)
    num_labels = len(np.unique(labels))

    num_shards = math.ceil(q * num_users / num_labels)

    indices = [np.where(labels == i)[0] for i in range(num_labels)]
    data_split_indices = []
    for i in range(num_labels):
        indices_i = np.array_split(indices[i], num_shards)
        data_split_indices.extend(indices_i)

    user_data_indices = {}
    for user_id in range(num_users):
        np.random.shuffle(data_split_indices)
        selected_indices = np.concatenate(data_split_indices[:q])
        user_data_indices[user_id] = selected_indices
        del data_split_indices[:q]

    return user_data_indices