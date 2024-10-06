import time

import torch
import copy
import random
import math
from utils import clip_norm, generate_gaussian_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import Model1, Model2, MLP
from utils import get_flat_model_params, set_flat_params_to_param_groups
import numpy as np
# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return torch.tensor(image), torch.tensor(label)
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, user_id, malicious_users, dataset_name):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.user_id = user_id
        self.malicious_users = malicious_users
        self.dataset_name = dataset_name
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.user_id in self.malicious_users:
            image, label = self.dataset[self.idxs[item]]
            if self.dataset_name == 'mnist' or self.dataset_name == 'cifar10' or self.dataset_name == 'fmnist':
                label = np.random.randint(0, 10)
            elif self.dataset_name == 'mmnist':
                label = np.random.randint(0, 5)
        else:
            image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class GlobalLocalUpdate(object):
    def __init__(self, args, global_model, dataset, idxs, logger, user_id, malicious_users):
        self.args = args
        self.logger = logger
        self.model = copy.deepcopy(global_model)
        self.weights = copy.deepcopy(self.model.state_dict())
        self.alpha = {}
        self.dataset_name = args.dataset
        self.corrupted = args.corrupted
        for key in self.weights.keys():
            self.alpha[key] = torch.zeros_like(self.weights[key]).to(self.args.device)
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs), user_id, malicious_users)



        self.criterion = nn.CrossEntropyLoss()
        self.train_loss = self.inference(self.model)
        self.test_acc, self.test_loss = test_inference(self.args, model=global_model, testloader=self.testloader, criterion=self.criterion)



    def train_val_test(self, dataset, idxs, user_id, malicious_users):
        # 定义新的训练数据和测试数据的比例
        train_ratio = 0.8
        test_ratio = 0.2

        # 划分训练数据和测试数据
        total_samples = len(idxs)
        train_size = int(train_ratio * total_samples)
        test_size = total_samples - train_size
        random.shuffle(idxs)
        idxs_train = idxs[:train_size]
        idxs_test = idxs[train_size+1:]
        dataset_name = self.dataset_name
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train, user_id, malicious_users, dataset_name), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test, user_id, malicious_users, dataset_name), batch_size=self.args.local_bs, shuffle=True)
        
        print(len(dataset))
        print(len(idxs_test))
        
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        
        return trainloader, testloader
    def update_weights(self, global_round, global_model, w, UserID, lr, malicious_users):
        start_time = time.monotonic()
        local_sum = {}
        self.model.train()
        self.model.load_state_dict(w)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        epoch_loss = []
        model_prev = copy.deepcopy(self.model.state_dict())
        alpha_prev = copy.deepcopy(self.alpha)
        if self.args.fixed == 1:
            E = self.args.local_ep
        else:
            E = random.randint(1, self.args.local_ep)
        if UserID in malicious_users and self.corrupted == '2':
            flat_wi = get_flat_model_params(self.weights)
            flat_wi = torch.ones_like(flat_wi) * np.random.normal(0, 0.1, 1).item()
            local_sum = set_flat_params_to_param_groups(self.weights, flat_wi)

        elif UserID in malicious_users and self.corrupted == '3':
            magnitude = abs(np.random.normal(0, 1, 1).item())
            flat_wi = get_flat_model_params(self.weights)
            flat_wi = - magnitude * flat_wi
            local_sum = set_flat_params_to_param_groups(self.weights, flat_wi)
        elif UserID in malicious_users and self.corrupted == '4':
            flat_wi = get_flat_model_params(self.weights)
            flat_wi = torch.normal(mean=0., std=1, size=flat_wi.size())
            local_sum = set_flat_params_to_param_groups(self.weights, flat_wi)
        elif UserID not in malicious_users or self.corrupted == '0' or self.corrupted == '1':  # training without SVA, SFA, or GA.
            for iter in range(E):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    if self.args.model == 'MLP' or self.args.model == 'MLR':
                        images = images.reshape(-1, 784)
                    self.model.zero_grad()
                    optimizer.zero_grad()
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels.long())
                    loss.backward()
                    model_weights_pre = copy.deepcopy(self.model.state_dict())
                    if self.args.framework == 'FedADMM':
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.grad = param.grad + (self.alpha[name] + self.args.rho * (model_weights_pre[name] - w[name]))+self.args.mu * model_weights_pre[name]
                                # param.grad = param.grad + (self.alpha[name] + self.args.rho * (model_weights_pre[name] - w[name]))

                    elif self.args.framework == 'FedAvg':
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.grad = param.grad+self.args.mu * model_weights_pre[name]
                    else:
                        continue

                    optimizer.step()

                    if self.args.verbose and (iter % 1 == 0) and (batch_idx % 10 == 0):
                        print('| Communication Round : {} | Client {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, UserID,
                            iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss))

            self.weights = self.model.state_dict()

            if self.args.framework == 'FedADMM':
                for key in self.alpha.keys():
                    self.alpha[key] = self.alpha[key] + self.args.rho * (self.weights[key]-w[key])
                    local_sum[key] = (self.weights[key] - model_prev[key]) + (1/self.args.rho) * (self.alpha[key]-alpha_prev[key])
                    # local_sum[key] = (self.weights[key] + (1 / self.args.rho) * self.alpha[key])
            elif self.args.framework == 'FedAvg':
                for key in self.alpha.keys():
                    local_sum[key] = self.weights[key] - model_prev[key]






        self.test_acc, self.test_loss = test_inference(self.args, global_model, self.testloader, self.criterion)
        # self.train_loss = epoch_loss[-1]
        self.train_loss = self.inference(global_model)
        end_time = time.monotonic()
        # print(f"\x1b[{32}m{'Test accuracy of global model on Client {} is {:.2f}%'.format(UserID, 100*self.test_acc)}\x1b[0m")
        print(f"\x1b[{32}m{'Round: {} | Client {} | Test accuracy: {:.2f}% | Test loss: {:.2f} | Time: {:.2f}s'.format(global_round, UserID, 100*self.test_acc, self.test_loss, end_time-start_time)}\x1b[0m")

        return local_sum, self.test_acc, self.train_loss

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            if self.args.model == 'MLP' or self.args.model == 'MLR':
                images = images.reshape(-1, 784)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels.long())
            loss += batch_loss.item()
        return loss


def test_inference(args, model, testloader, criterion):
    # model.eval()
    # loss, total, correct = 0.0, 0.0, 0.0
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.CrossEntropyLoss().to(args.device)
    # # testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    #
    # for batch_idx, (images, labels) in enumerate(testloader):
    #     images, labels = images.to(args.device), labels.to(args.device)
    #     if args.model == 'MLP' or args.model == 'MLR':
    #         images = images.reshape(-1, 784)
    #     outputs = model(images)
    #     # batch_loss = criterion(outputs, labels)
    #     # loss += batch_loss.item()
    #     _, pred_labels = torch.max(outputs, 1)
    #     pred_labels = pred_labels.view(-1)
    #     correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #     total += len(labels)
    #
    # accuracy = correct/total



    model.eval()
    correct = 0
    total = 0
    total_test_loss=0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(args.device), labels.to(args.device)
            if args.model == 'MLP' or args.model == 'MLR':
                images = images.reshape(-1, 784)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels.long())
            total_test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    return accuracy, total_test_loss
