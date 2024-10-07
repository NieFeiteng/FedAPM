import numpy as np
from torchvision import datasets, transforms
import torch
import math
from collections import defaultdict
from torch.utils.data import ConcatDataset

def add_gaussian_noise(data, mean=0, std=0.1):
    noise = torch.randn_like(data) * std + mean
    return data + noise

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def hybrid_skew(dataset_name, dataset, num_users, q):
    # 1/2 clients for q-label skew, 1/2 clients for quantity skew
    num_users_label = math.ceil(num_users/2)
    num_users_quantity = math.floor(num_users/2)
    data_indices_label = q_label_skew(dataset_name, dataset, num_users_label, q)
    data_indices_quantity = dir_quantity_skew(dataset_name, dataset, num_users_quantity)
    data_indices_label = list(data_indices_label.values())
    data_indices = data_indices_label + data_indices_quantity
    return data_indices

def dir_quantity_skew(dataset_name, dataset, num_users):
    beta = 0.5
    probabilities = np.random.dirichlet(np.ones(num_users) * beta)
    dataset_size = len(dataset)
    data_indices = []
    for i in range(num_users):
        # num_data_points = int(probabilities[i] * dataset_size)
        num_data_points = max(100, int(probabilities[i] * dataset_size))
        user_data_indices = np.random.choice(dataset_size, size=num_data_points, replace=False)
        data_indices.append(user_data_indices)

    return data_indices

def dir_label_skew(dataset_name, dataset, num_users):
    if dataset_name == 'mmnist':
        labels = np.array(dataset.targets)
    elif dataset_name == 'cifar10':
        labels_train = np.array(dataset.datasets[0].targets)
        labels_test = np.array(dataset.datasets[1].targets)
        labels = np.concatenate((labels_train, labels_test), axis=0)
    elif dataset_name == 'mnist' or dataset_name == 'fmnist':
        labels_train = dataset.datasets[0].targets.numpy()
        labels_test = dataset.datasets[1].targets.numpy()
        labels = np.concatenate((labels_train, labels_test), axis=0)
    elif dataset_name == 'femnist':
        labels = dataset.tensors[1]

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

def noise_feature_skew(dataset_name, dataset, num_users):
    if dataset_name == 'mmnist':
        labels = np.array(dataset.targets)
    elif dataset_name == 'cifar10':
        labels_train = np.array(dataset.datasets[0].targets)
        labels_test = np.array(dataset.datasets[1].targets)
        labels = np.concatenate((labels_train, labels_test), axis=0)
    else:
        labels_train = dataset.datasets[0].targets.numpy()
        labels_test = dataset.datasets[1].targets.numpy()
        labels = np.concatenate((labels_train, labels_test), axis=0)

    total_data_length = len(labels)
    data_per_user = total_data_length // num_users

    user_data_indices = []
    current_length = 0
    for i in range(num_users):
        start_index = current_length
        end_index = min(current_length + data_per_user, total_data_length)
        user_indices = list(range(start_index, end_index))

        user_data_indices.append(user_indices)
        current_length = end_index

    # noisy_data = []
    # user_id = 
    # for user_indices in user_data_indices:
    #     user_id = user_id + 1
    #     noise_scale = 0 * user_id/num_users
    #     user_data = all_data[user_indices]
    #     all_data[user_indices] = all_data[user_indices]
    #
    #
    #     all_data[user_indices] = (all_data[user_indices] + np.random.normal(scale=noise_scale, size=user_data.shape)).astype(np.float32)
    #
    #
    # noisy_targets = all_targets
    # noisy_dataset = torch.utils.data.TensorDataset(torch.tensor(all_data.astype(np.float32)), torch.tensor(noisy_targets))

    return user_data_indices

def feature_skew(dataset, num_users):
    features, labels, creators = dataset
    user_data_indices = {i: [] for i in range(num_users)}

    total_data = sum(creators)
    writers_num = len(creators)
    user_data_indices = {}
    start_data_id = 0
    start_id = 0
    # for creator_idx, creator_data_count in enumerate(creators):
    num_shards = writers_num // num_users
    for user_id in range(num_users):
        user_i_data_indices = []
        for i in range(start_id, num_shards * (1 + user_id)):
            end_data_id = start_data_id + creators[i]
            user_i_data_indices += range(start_data_id, end_data_id)
            start_data_id = end_data_id
        start_id = num_shards * (1 + user_id)
        user_data_indices[user_id] = user_i_data_indices

    return user_data_indices

def q_label_skew(dataset_name, dataset, num_users, q):
    if dataset_name == 'mmnist':
        labels = np.array(dataset.targets)
    elif dataset_name == 'cifar10':
        labels_train = np.array(dataset.datasets[0].targets)
        labels_test = np.array(dataset.datasets[1].targets)
        labels = np.concatenate((labels_train, labels_test), axis=0)
    elif dataset_name == 'femnist':
        labels = dataset.tensors[1]
    elif dataset_name == 'mnist' or dataset_name == 'fmnist':
        labels_train = dataset.datasets[0].targets.numpy()
        labels_test = dataset.datasets[1].targets.numpy()
        labels = np.concatenate((labels_train,labels_test), axis=0)
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
def covtype_noniid(labels, num_users, q):
    num_labels = len(np.unique(labels))

    num_shards = math.ceil(q * num_users / num_labels)

    indices = [np.where(labels == i)[0] for i in range(0, num_labels)]
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
def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, q):
    labels_train = np.array(dataset.datasets[0].targets)
    labels_test = np.array(dataset.datasets[1].targets)
    labels = np.concatenate((labels_train, labels_test), axis=0)
    num_labels = len(np.unique(labels))

    num_shards = q * num_users // num_labels

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

