import time

import torch
import copy
import random
import math
import torchmetrics

from utils import clip_norm, generate_gaussian_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import Model1, Model2, MLP, ResNet18
from utils import get_flat_model_params, set_flat_params_to_param_groups
import numpy as np
from get_multimodal import get_multimodal_dataloaders
import torchmetrics

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
        self.is_multimodal = args.is_multimodal
        self.corrupted = args.corrupted 
        for key in self.weights.keys():
            self.alpha[key] = torch.zeros_like(self.weights[key]).to(self.args.device)

        if self.is_multimodal:
            self.trainloader, self.testloader = get_multimodal_dataloaders(args.dataset,user_id,args.alpha)
        else:        
            self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs), user_id, malicious_users)
    
        self.criterion = nn.CrossEntropyLoss()        
        self.train_loss = self.inference(self.model)
        self.test_acc, self.test_loss, self.f1, self.auc  = test_inference(self.args, model=global_model, testloader=self.testloader, criterion=self.criterion)
    
    def train_val_test(self, dataset, idxs, user_id, malicious_users):
        train_ratio = 0.8
        test_ratio = 0.2
        
        total_samples = len(idxs)
        train_size = int(train_ratio * total_samples)
        test_size = total_samples - train_size
        random.shuffle(idxs)
        idxs_train = idxs[:train_size]
        idxs_test = idxs[train_size+1:]
        dataset_name = self.dataset_name
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train, user_id, malicious_users, dataset_name), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test, user_id, malicious_users, dataset_name), batch_size=self.args.local_bs, shuffle=True)
        return trainloader, testloader    
    def update_weights(self, global_round, global_model, w, UserID, lr, malicious_users):
        start_time = time.monotonic()
        local_sum = {}
        self.model.train()
        num_layers = self.args.layer_num
        
        self.model.load_remaining_layers(w, num_layers)
        all_params = list(self.model.named_parameters())
        
        
        if self.args.model == 'MLP' or self.args.model == 'CNN': 
            remaining_layers_names = [name for name, param in all_params[:num_layers]]
            remaining_layers_params = [param for name, param in all_params[:num_layers]]
            first_n_layers_params = [param for name, param in all_params[num_layers:]]            
        else:   
            remaining_layers_names = [name for name, param in all_params[num_layers:]]
            remaining_layers_params = [param for name, param in all_params[num_layers:]]
            first_n_layers_params = [param for name, param in all_params[:num_layers]]
        heterogeneous_param =  0 
        for key in remaining_layers_names:
            heterogeneous_param += torch.norm(self.weights[key]-w[key])
        optimizer_step1 = torch.optim.SGD(list(first_n_layers_params), lr=lr)
        optimizer_step2 = torch.optim.SGD(list(remaining_layers_params), lr=lr) 
        
        epoch_loss = []
        model_prev = copy.deepcopy(self.model.state_dict()) 
        alpha_prev = copy.deepcopy(self.alpha)  

        if self.args.fixed == 1:
            E = self.args.local_ep
        else:
            E = random.randint(1, self.args.local_ep)

        v_i_t =  copy.deepcopy(self.model.state_dict())
        sigma_i_prox = 0.1
        sigma_i = 0.01

        def framework_common_step(optimizer, params1, params2, framework, PHASE_FLAG: bool=False):
            for iter in range(E):    
                batch_loss = []
                for batch_idx, batch_data in enumerate(self.trainloader):
                    if self.is_multimodal:
                        x_a, x_b, l_a, l_b, y = batch_data
                        x_a, x_b, y = x_a.to(self.args.device), x_b.to(self.args.device), y.to(self.args.device)
                        l_a, l_b = l_a.to(self.args.device), l_b.to(self.args.device)                    
                    else:
                        images, labels = batch_data
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        if self.args.model == 'MLP' or self.args.model == 'MLR':
                            images = images.reshape(-1, 784)
    
                    for param in params1:
                        param.requires_grad = True

                    for param in params2:
                        param.requires_grad = False      
                        
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    if self.is_multimodal:
                        log_probs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                        loss = self.criterion(log_probs, y)
                    else:
                        log_probs = self.model(images)
                        loss = self.criterion(log_probs, labels.long())
                    loss.backward()
                    model_weights_pre = copy.deepcopy(self.model.state_dict())

                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            if framework == 'FedAPM':
                                if PHASE_FLAG:
                                    param.grad = (param.grad   + self.args.mu * model_weights_pre[name] ) / self.args.num_users  +  (self.alpha[name] + self.args.rho * (model_weights_pre[name] - w[name]))        
                                else:
                                    param.grad = (param.grad + self.args.mu * model_weights_pre[name]) / self.args.num_users + (model_weights_pre[name] - v_i_t[name]) * sigma_i
                            elif framework == 'FedAlt' or framework == 'FedAvg':
                                param.grad = (param.grad + self.args.mu * model_weights_pre[name]) / self.args.num_users
                            elif framework == 'FedProx':
                                param.grad = (param.grad + self.args.mu * model_weights_pre[name]) / self.args.num_users + (model_weights_pre[name] - v_i_t[name]) * sigma_i_prox
                            elif framework == 'FedSim': 
                                 param.grad = (param.grad  + self.args.mu * model_weights_pre[name]) / self.args.num_users
                                
                    optimizer.step()
                    if framework != 'FedSim':
                        batch_loss.append(loss.item())
                    else:
                        for param in params1:
                            param.requires_grad = False
                        for param in params2:
                            param.requires_grad = True  
                              
                        self.model.zero_grad()
                        optimizer.zero_grad()                            
                        if self.is_multimodal:
                            log_probs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                            loss = self.criterion(log_probs, y)
                        else:
                            log_probs = self.model(images)
                            loss = self.criterion(log_probs, labels.long())
                        loss.backward()
                        model_weights_pre = copy.deepcopy(self.model.state_dict())                            
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.grad = (param.grad  + self.args.mu * model_weights_pre[name]) / self.args.num_users
                        optimizer_step2.step()
                        batch_loss.append(loss.item())                     
                             
                if PHASE_FLAG:
                    epoch_loss.append(sum(batch_loss))

        def execute_epoch(E, optimizer_step1, optimizer_step2, framework):
            if framework == 'FedAlt' or framework == 'FedAPM':
                framework_common_step(optimizer_step1, first_n_layers_params, remaining_layers_params, framework, False)
            if framework != 'FedSim':
                framework_common_step(optimizer_step2, remaining_layers_params, first_n_layers_params, framework, True)
            if framework == 'FedSim':
                framework_common_step(optimizer_step1, remaining_layers_params, first_n_layers_params, framework, True)
            # epoch_loss.append(sum(batch_loss))
            
            
        execute_epoch(E, optimizer_step1, optimizer_step2, self.args.framework)

        self.weights = self.model.state_dict()

        if self.args.framework == 'FedAPM':
            for key in self.alpha.keys():
                self.alpha[key] = self.alpha[key] + self.args.rho * (self.weights[key]-w[key])
                # local_sum[key] = (self.weights[key] - model_prev[key]) + (1/self.args.rho) * (self.alpha[key]-alpha_prev[key])
                local_sum[key] = (self.weights[key] + (1 / self.args.rho) * self.alpha[key])
                                    
        elif self.args.framework == 'FedAvg' or self.args.framework == 'FedProx' or self.args.framework == 'FedAlt' or self.args.framework == 'FedSim':  
            for key in self.alpha.keys():
                # local_sum[key] = self.weights[key] - model_prev[key]
                local_sum[key] = self.weights[key] 
        heterogeneous_param =  0 
        for key in remaining_layers_names:
            heterogeneous_param += torch.norm(self.weights[key]-w[key])

        self.test_acc, self.test_loss, self.f1, self.auc = test_inference(self.args, self.model, self.testloader, self.criterion)
        self.train_loss = self.inference(self.model)
        

        end_time = time.monotonic()
        print(f"\x1b[{32}m{'Round: {} | Client {} | Test accuracy: {:.2f}% | Test loss: {:.2f} |  F1 Score: {:.2f} |  AUC: {:.2f} | Time: {:.2f}s'.format(global_round, UserID, 100*self.test_acc, self.test_loss, self.f1, self.auc, end_time-start_time)}\x1b[0m")

        return local_sum, self.test_acc, self.train_loss, self.f1, self.auc, heterogeneous_param.item()

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, batch_data in enumerate(self.trainloader):
            if self.is_multimodal:
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.args.device), x_b.to(self.args.device), y.to(self.args.device)
                l_a, l_b = l_a.to(self.args.device), l_b.to(self.args.device)
                
                # forward
                outputs, _ = self.model(
                    x_a.float(), x_b.float(), l_a, l_b
                )    
                # outputs = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, y)         
            else:
                images, labels = batch_data
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.model == 'MLP' or self.args.model == 'MLR':
                    images = images.reshape(-1, 784)
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels.long())
                loss += batch_loss.item()
        return loss

def test_inference(args, model, testloader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_test_loss = 0.0
    
    if args.model in ['MLP', 'ResNet18', 'CNN']:
        num_classes = args.num_classes
    else:
        num_classes = args.num_classes[0]
        
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average='weighted').to(args.device) if num_classes > 2 else torchmetrics.F1Score(task="binary", average='weighted').to(args.device)
    auc_metric = torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(args.device) if num_classes > 2 else torchmetrics.AUROC(task="binary").to(args.device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data_temp in testloader:
            if args.is_multimodal:
                x_a, x_b, l_a, l_b, y = data_temp
                x_a, x_b, y = x_a.to(args.device), x_b.to(args.device), y.to(args.device)
                l_a, l_b = l_a.to(args.device), l_b.to(args.device)

                outputs, _ = model(x_a.float(), x_b.float(), l_a, l_b)

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs.data, 1)

                loss = criterion(outputs, y)
                total_test_loss += loss.item()
                total += y.size(0)
                correct += (predicted == y).sum().item()

                all_probs.append(probs)
                all_labels.append(y)

            else:
                images, labels = data_temp
                images, labels = images.to(args.device), labels.to(args.device)

                if args.model in ['MLP', 'MLR']:
                    images = images.reshape(-1, 784)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs.data, 1)

                loss = criterion(outputs, labels.long())
                total_test_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_probs.append(probs)
                all_labels.append(labels)

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if num_classes <= 2:
            all_probs = all_probs[:, 1]

        unique_labels = torch.unique(all_labels)
        f1 = f1_metric(all_probs, all_labels)

        if len(unique_labels) > 1:
            auc = auc_metric(all_probs, all_labels)
        else:
            auc = 0. 

        accuracy = correct / total

    return accuracy, total_test_loss, f1, auc

