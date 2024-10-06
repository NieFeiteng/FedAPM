import time

import torch
import copy
import random
import math

from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import Model1, Model2, MLP, MLR, CNN, SVM, CNN1, CNN2, ResNet18
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
from utils import get_flat_model_params, set_flat_params_to_param_groups

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, user_id, malicious_users, dataset_name, partition, noise_scale, num_users, corrupted):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.user_id = user_id
        self.malicious_users = malicious_users
        self.dataset_name = dataset_name
        self.partition = partition
        self.noise_scale = noise_scale
        self.num_users = num_users
        self.corrupted = corrupted
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.user_id in self.malicious_users:
            image, label = self.dataset[self.idxs[item]]
            if self.corrupted == '1' and (self.dataset_name == 'mnist' or self.dataset_name == 'cifar10' or self.dataset_name == 'fmnist'):
                label = np.random.randint(0, 10)
            elif self.corrupted == '1' and self.dataset_name == 'mmnist':
                label = np.random.randint(0, 5)
        else:
            image, label = self.dataset[self.idxs[item]]

        if self.partition == 'quality-skew':
            image = image + torch.randn_like(image) * self.user_id * self.noise_scale / self.num_users


        return torch.tensor(image), torch.tensor(label)


class PerLocalUpdate(object):
    def __init__(self, args, global_model, dataset, idxs, logger, w, Proj, user_id, malicious_users):
        self.args = args
        self.logger = logger
        self.local_dataset_size = len(idxs)
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            input_size = 784
            num_classes = 10
        self.model = copy.deepcopy(global_model)
        if args.framework == 'ditto':
            self.local_model = copy.deepcopy(global_model)
        self.corrupted = args.corrupted
        # if self.args.model == 'Model1':
        #     self.model = Model1(args=args)
        # elif self.args.model == 'Model2':
        #     self.model = Model2(args=args)
        # elif self.args.model == 'MLP':
        #     self.model = MLP(input_size,num_classes)
        # elif self.args.model == 'MLR':
        #     self.model = MLR(args=args)
        # elif self.args.model == 'CNN':
        #     self.model = CNN()
        # elif self.args.model == 'SVM':
        #     self.model = SVM()
        # elif self.args.model == 'CNN1':
        #     self.model = CNN1()
        # elif self.args.model == 'CNN2':
        #     self.model = CNN2()
        # else:
        #     exit('Error: unrecognized model')
        self.dataset_name = args.dataset
        self.model.to(self.args.device)
        self.model.train()
        self.Proj = Proj
        self.wi = w
        self.alpha = {}
        if self.args.framework == 'FLAME' or self.args.framework == 'pFedMe' or self.args.framework == 'ditto':
            self.wi = copy.deepcopy(global_model.state_dict())
            for key in self.wi.keys():
                self.alpha[key] = torch.zeros_like(self.wi[key]).to(self.args.device)
        else:
            self.alpha = torch.zeros_like(self.wi).to(self.args.device)
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs), user_id, malicious_users)
        if self.args.model == 'SVM':
            self.criterion = nn.MultiMarginLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.train_personal_loss, self.train_global_loss = self.inference(self.model, global_model)
        self.test_acc_personal, self.test_acc_global, self.test_personal_loss, self.test_global_loss = test_inference(self.args, self.model, global_model, testloader=self.testloader, criterion=self.criterion)
        # self.test_acc_global = test_inference(self.args, model=global_model, testloader=self.testloader)




    def calculate_gradient_l2_norm(self, w):
        self.model.train()
        data_iterator = iter(self.trainloader)
        batch_data, batch_labels = next(data_iterator)
        if self.args.model == 'MLP' or self.args.model == 'MLR':
            batch_data = batch_data.reshape(-1, 784)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        self.model.zero_grad()
        output = self.model(batch_data)
        loss = self.criterion(output, batch_labels)
        loss.backward()
        total_norm = 0
        weights = copy.deepcopy(self.model.state_dict())

        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                param_norm = (param.grad + self.args.Lambda * (weights[name] - w[name])+self.args.mu*weights[name]).data.norm(2)
                # param_norm = (param.grad).data.norm(2)

                total_norm += param_norm.item() ** 2
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         # +(self.args.Lambda * (theta_pre - self.wi))
        #         param_norm = (param.grad).data.norm(2)
        #         total_norm += param_norm.item() ** 2
        return total_norm

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
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train, user_id, malicious_users, dataset_name, self.args.partition, self.args.noise_scale, self.args.num_users, self.corrupted), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test, user_id, malicious_users, dataset_name, self.args.partition, self.args.noise_scale, self.args.num_users, self.corrupted), batch_size=self.args.local_bs, shuffle=True)
        return trainloader, testloader

    def update_weights(self, global_round, global_model, w, UserID, lr, malicious_users):

        start_time = time.monotonic()




        local_sum = {}
        self.model.train()
        hpy_lambda = self.args.Lambda/(self.args.num_users)

        if self.args.framework == 'pFedMe' or self.args.framework == 'FLAME' or self.args.framework == 'lp-proj-2' or self.args.framework == 'FLAME-lp-proj-2' or self.args.framework == 'ditto':
            self.wi = copy.deepcopy(w)

        # if self.args.framework == 'pFedMe' or self.args.framework == 'lp-proj-2' or self.args.framework == 'FLAME-lp-proj-2' or self.args.framework == 'ditto':
        #     self.wi = copy.deepcopy(w)




        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.args.momentum)
        epoch_loss = []
        alpha_prev = copy.deepcopy(self.alpha)
        wi_prev = copy.deepcopy(self.wi)
        if self.args.fixed == 1:
            E = self.args.local_ep
        else:
            E = random.randint(1, self.args.local_ep)
        if UserID in malicious_users and self.corrupted == '2':
            flat_wi = get_flat_model_params(self.wi)
            flat_wi = torch.ones_like(flat_wi) * np.random.normal(0, 0.01, 1).item()
            self.wi = set_flat_params_to_param_groups(self.wi, flat_wi)

        elif UserID in malicious_users and self.corrupted == '3':
            magnitude = abs(np.random.normal(0, 1, 1).item())
            flat_wi = get_flat_model_params(self.wi)
            flat_wi = - magnitude * flat_wi
            self.wi = set_flat_params_to_param_groups(self.wi, flat_wi)
        elif UserID in malicious_users and self.corrupted == '4':
            flat_wi = get_flat_model_params(self.wi)
            flat_wi = torch.normal(mean=0., std=1, size=flat_wi.size())
            self.wi = set_flat_params_to_param_groups(self.wi, flat_wi)
        elif UserID not in malicious_users or self.corrupted == '0' or self.corrupted == '1':  # training without SVA, SFA, or GA.
            for iter in range(E):
                batch_loss = []
                global_loss = 0
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    if self.args.model == 'MLP' or self.args.model == 'MLR':
                        images = images.reshape(-1, 784)
                    self.model.zero_grad()
                    optimizer.zero_grad()
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels.long())
                    loss.backward()
                    theta_pre = copy.deepcopy(self.model.state_dict())

                    log_probs_global = global_model(images)
                    global_loss_batch = self.criterion(log_probs_global, labels.long())
                    global_loss += global_loss_batch.item()

                    if self.args.framework == 'FLAME' or self.args.framework == 'pFedMe':
                        for name, param in self.model.named_parameters():
                            if param.requires_grad == True:
                                param.grad = param.grad + \
                                             self.args.Lambda * (theta_pre[name]-self.wi[name]) + \
                                             self.args.mu * theta_pre[name]
                    # elif self.args.framework == 'ditto':
                    #     for name, param in self.model.named_parameters():
                    #         if param.requires_grad == True:
                    #             param.grad = param.grad + \
                    #                          self.args.Lambda * (theta_pre[name]-w[name]) + \
                    #                          self.args.mu * theta_pre[name]
                    elif self.args.framework == 'ditto':
                        for name, param in self.model.named_parameters():
                            if param.requires_grad == True:
                                param.grad = param.grad + \
                                             self.args.Lambda * (theta_pre[name]-self.wi[name]) + \
                                             self.args.mu * theta_pre[name]
                    elif self.args.framework == 'lp-proj-2' or self.args.framework == 'FLAME-lp-proj-2':
                        flat_theta = get_flat_model_params(theta_pre)
                        # flat_wi = get_flat_model_params(self.wi)

                        regularizer_gradient = self.args.Lambda * torch.matmul(self.Proj.t(), (torch.matmul(self.Proj, flat_theta)-self.wi))
                        regularizer_gradient = set_flat_params_to_param_groups(theta_pre, regularizer_gradient)
                        for name, param in self.model.named_parameters():
                            if param.requires_grad == True:
                                param.grad = param.grad + \
                                             regularizer_gradient[name] + \
                                             self.args.mu * theta_pre[name]
                                # param.grad = param.grad + \
                                #              self.args.mu * theta_pre[name]
                    # elif self.args.framework == 'ditto':
                    else:
                        continue
                    optimizer.step()
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss))
                    # if self.args.verbose and (iter % 1 == 0) and (batch_idx % 10 == 0):
                    #     print('| >>> Round: {} | Client {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #         global_round, UserID,
                    #         iter, batch_idx * len(images),
                    #         len(self.trainloader.dataset),
                    #         100. * batch_idx / len(self.trainloader), loss.item()))


                # if self.args.framework == 'ditto':
                #
                #     self.local_model.zero_grad()
                #     optimizer_local.zero_grad()
                #     log_probs_local = self.local_model(images)
                #     loss_local = self.criterion(log_probs_local, labels.long())
                #     loss_local.backward()
                #     optimizer_local.step()
                #     self.wi = copy.deepcopy(self.local_model.state_dict())




                weights = self.model.state_dict()

                if self.args.framework == 'FLAME':
                    alpha_prev = copy.deepcopy(self.alpha)

                    for key in self.alpha.keys():
                        self.wi[key] = 1 / (hpy_lambda + self.args.rho) * (hpy_lambda * weights[key] + self.args.rho * w[key] - alpha_prev[key])
                        self.alpha[key] = self.alpha[key] + self.args.rho * (self.wi[key] - w[key])

                        # local_sum[key] = (self.wi[key] - wi_prev[key]) + (1 / self.args.rho) * (self.alpha[key] - alpha_prev[key])
                elif self.args.framework == 'pFedMe':
                    for key in self.alpha.keys():
                        # self.wi[key] = copy.deepcopy(weights[key])
                        self.wi[key] = self.wi[key] - self.args.eta2 * self.args.Lambda * (self.wi[key] - weights[key])
                        local_sum[key] = self.wi[key]

                elif self.args.framework == 'ditto':
                    self.local_model.train()
                    optimizer_local = torch.optim.SGD(self.local_model.parameters(), lr=self.args.eta, momentum=self.args.momentum)
                    for iter in range(E):
                        images, labels = random.choice(list(self.trainloader))

                    # for batch_idx, (images, labels) in enumerate(self.trainloader):
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        if self.args.model == 'MLP' or self.args.model == 'MLR':
                            images = images.reshape(-1, 784)
                        self.local_model.zero_grad()
                        optimizer_local.zero_grad()
                        log_probs_local = self.local_model(images)
                        loss_local = self.criterion(log_probs_local, labels.long())
                        loss_local.backward()
                        optimizer_local.step()
                    self.wi = copy.deepcopy(self.local_model.state_dict())



            # weights = self.model.state_dict()
            # if self.args.framework == 'FLAME':
            #     for key in self.alpha.keys():
            #         self.wi[key] = (hpy_lambda * weights[key] + self.args.rho * w[key] - alpha_prev[key]) / (
            #                 hpy_lambda + self.args.rho)
            #         self.alpha[key] = self.alpha[key] + self.args.rho * (self.wi[key] - w[key])
            #
            # if self.args.framework == 'pFedMe':
            #     for key in self.alpha.keys():
            #         self.wi[key] = self.wi[key] - self.args.eta2 * self.args.Lambda * (self.wi[key] - weights[key])
        # if self.args.framework == 'ditto':
        #     for batch_idx, (images, labels) in enumerate(self.trainloader):
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         if self.args.model == 'MLP' or self.args.model == 'MLR':
        #             images = images.reshape(-1, 784)
        #         self.local_model.zero_grad()
        #         optimizer_local.zero_grad()
        #         log_probs_local = self.local_model(images)
        #         loss_local = self.criterion(log_probs_local, labels.long())
        #         loss_local.backward()
        #         optimizer_local.step()
        #         self.wi = copy.deepcopy(self.local_model.state_dict())
        #
        # weights = self.model.state_dict()
        # if self.args.framework == 'FLAME':
        #     for key in self.alpha.keys():
        #         # self.wi[key] = 1/(hpy_lambda + self.args.rho) * \
        #         #                (hpy_lambda * weights[key] + self.args.rho*w[key]-alpha_prev[key])
        #         self.alpha[key] = self.alpha[key] + self.args.rho * (self.wi[key] - w[key])
        #
        #
        # elif self.args.framework == 'pFedMe':
        #     for key in self.alpha.keys():
        #         self.wi[key] = self.wi[key] - self.args.eta2 * self.args.Lambda * (self.wi[key] - weights[key])





        # elif self.args.framework == 'lp-proj-2':
        #     # self.wi[key] = copy.deepcopy(weights[key])
        #     flat_theta = get_flat_model_params(weights)
        #     self.wi = self.wi - self.args.eta2 * self.args.Lambda * (self.wi - torch.matmul(self.Proj, flat_theta))
        #     # local_sum = (self.wi)
        #     # local_sum = (self.wi - wi_prev)
        #
        # elif self.args.framework == 'FLAME-lp-proj-2':
        #     flat_theta = get_flat_model_params(weights)
        #
        #     self.wi = 1 / (hpy_lambda + self.args.rho) * \
        #                    (hpy_lambda * torch.matmul(self.Proj, flat_theta) + self.args.rho * w - alpha_prev)
        #     self.alpha = self.alpha + self.args.rho * (self.wi - w)

            # local_sum = (self.wi) + (1 / self.args.rho) * (self.alpha)
            # local_sum = (self.wi - wi_prev) + (1 / self.args.rho) * (self.alpha - alpha_prev)

            self.test_acc_personal, self.test_acc_global, self.test_personal_loss, self.test_global_loss = test_inference(self.args, self.model, global_model, self.testloader, self.criterion)
             # self.test_acc_global = test_inference(self.args, global_model, self.testloader)
            # self.train_global_loss = self.inference(global_model)
            self.train_global_loss = global_loss
            self.train_personal_loss = epoch_loss[-1]

            # self.train_global_loss = 0
            #
            # self.train_personal_loss = 0
            end_time = time.monotonic()
            running_time = end_time - start_time
            print(f"\x1b[{32}m{'Round: {} | Client {} | Personal model Test accuracy: {:.2f}% | Global model Test accuracy: {:.2f}% | Personalized model Test loss: {:.2f} | Global model Test loss: {:.2f} | Time: {:.2f}s'.format(global_round, UserID, 100*self.test_acc_personal, 100*self.test_acc_global, self.test_personal_loss, self.test_global_loss, running_time)}\x1b[0m")

            # return self.test_acc_personal, self.train_personal_loss, self.test_acc_global,

    def inference(self, personalized_model, global_model):
        personalized_model.eval()
        global_model.eval()
        personalized_loss, global_loss, total, correct = 0.0, 0.0, 0.0, 0.0

        for images, labels in self.trainloader:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            if self.args.model == 'MLP' or self.args.model == 'MLR':
                images = images.reshape(-1, 784)
            personalized_outputs = personalized_model(images)
            global_output = global_model(images)
            personalized_batch_loss = self.criterion(personalized_outputs, labels.long())
            global_batch_loss = self.criterion(global_output, labels.long())
            personalized_loss += personalized_batch_loss.item()
            global_loss += global_batch_loss.item()

        return personalized_loss, global_loss


def test_inference(args, personal_model, global_model, testloader, criterion):

    # loss, total, personal_correct, global_correct = 0.0, 0.0, 0.0,0.0
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    #
    # for batch_idx, (images, labels) in enumerate(testloader):
    #     images, labels = images.to(args.device), labels.to(args.device)
    #     if args.model == 'MLP' or args.model == 'MLR':
    #         images = images.reshape(-1, 784)
    #     personal_outputs = personal_model(images)
    #     global_outputs = global_model(images)
    #     # batch_loss = criterion(outputs, labels)
    #     # loss += batch_loss.item()
    #     _, personal_pred_labels = torch.max(personal_outputs, 1)
    #     _, global_pred_labels = torch.max(global_outputs, 1)
    #
    #     personal_pred_labels = personal_pred_labels.view(-1)
    #     personal_correct += torch.sum(torch.eq(personal_pred_labels, labels)).item()
    #
    #     global_pred_labels = global_pred_labels.view(-1)
    #     global_correct += torch.sum(torch.eq(global_pred_labels, labels)).item()
    #
    #     total += len(labels)
    #
    # personal_accuracy = personal_correct/total
    # global_accuracy = global_correct/total
    personal_model.eval()
    global_model.eval()
    personal_correct, global_correct = 0, 0
    total = 0
    total_personalized_test_loss, total_global_test_loss = 0.0, 0.0
    with torch.no_grad():  
        for images, labels in testloader:
            images, labels = images.to(args.device), labels.to(args.device)
            if args.model == 'MLP' or args.model == 'MLR':
                images = images.reshape(-1, 784)
            personal_outputs = personal_model(images)
            global_outputs = global_model(images)
            _, personal_predicted = torch.max(personal_outputs.data, 1)
            _, global_predicted = torch.max(global_outputs.data, 1)

            personal_loss = criterion(personal_outputs, labels.long())
            global_loss = criterion(global_outputs, labels.long())
            total_personalized_test_loss += personal_loss.item()
            total_global_test_loss += global_loss.item()



            total += labels.size(0)
            personal_correct += (personal_predicted == labels).sum().item()
            global_correct += (global_predicted == labels).sum().item()
    personal_accuracy = personal_correct / total
    global_accuracy = global_correct / total

    return personal_accuracy, global_accuracy, total_personalized_test_loss, total_global_test_loss
