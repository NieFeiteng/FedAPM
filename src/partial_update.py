import time
import copy
import torch
import random
import numpy as np
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, Dataset
from get_multimodal import get_multimodal_dataloaders
from utils import get_flat_model_params, set_flat_params_to_param_groups


class DatasetSplit(Dataset):
    """
    Initialize an instance of the DatasetSplit class.

    Args:
        dataset (torch.utils.data.Dataset): The complete dataset.
        idxs (list): A list of data indices for the current user.
        user_id (int): The ID of the current user.
        malicious_users (list): A list of IDs of malicious users.
        dataset_name (str): The name of the dataset.
    """
    def __init__(self, dataset, idxs, user_id, malicious_users, dataset_name):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.user_id = user_id
        self.malicious_users = malicious_users
        self.dataset_name = dataset_name
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # Check if the current user is a malicious user
        if self.user_id in self.malicious_users:
            image, label = self.dataset[self.idxs[item]]
            if self.dataset_name == 'mnist' or self.dataset_name == 'cifar10' or self.dataset_name == 'fmnist':
                label = np.random.randint(0, 10)
            elif self.dataset_name == 'mmnist':
                label = np.random.randint(0, 5)
        else:
            # If the user is not malicious, simply get the image and label from the dataset
            image, label = self.dataset[self.idxs[item]]

        return torch.tensor(image), torch.tensor(label)

class GlobalLocalUpdate(object):
    def __init__(self, args, global_model, dataset, idxs, logger, user_id, malicious_users):
        """
        Initialize a GlobalLocalUpdate object for a specific user in a federated learning setting.

        Args:
            args (argparse.Namespace): Command-line arguments containing various configuration parameters.
            global_model (torch.nn.Module): The global model shared across all users.
            dataset (torch.utils.data.Dataset): The entire dataset used for training and testing.
            idxs (list): Indices of the data samples assigned to the current user.
            logger (logging.Logger): Logger object for logging training and testing information.
            user_id (int): Unique identifier for the current user.
            malicious_users (list): List of user IDs that are considered malicious.
        """
        self.args = args
        self.logger = logger
        # Create a deep copy of the global model for the local user
        self.model = copy.deepcopy(global_model)
        # Store the initial weights of the local model
        self.weights = copy.deepcopy(self.model.state_dict())
        # Initialize the dual variable for FedAPM
        self.alpha = {}
        # Store the name of the dataset
        self.dataset_name = args.dataset
        # Flag indicating whether the dataset is multimodal
        self.is_multimodal = args.is_multimodal
        # Flag indicating whether the user is corrupted
        self.corrupted = args.corrupted  # no
        # Initialize the alpha parameters to zero tensors (if not using FedAPM, alpha will be zero for all layers)
        for key in self.weights.keys():
            self.alpha[key] = torch.zeros_like(self.weights[key]).to(self.args.device)

        # If the dataset is multimodal, use a specialized function to get the data loaders
        if self.is_multimodal:
            self.trainloader, self.testloader = get_multimodal_dataloaders(args.dataset,user_id,args.alpha)
        # Otherwise, split the data into training and testing sets
        else:        
            self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs), user_id, malicious_users)
    
        # Define the loss function as cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()        
        # Calculate the initial training loss
        self.train_loss = self.inference(self.model)
        # Calculate the initial test accuracy, test loss, F1 score, and AUC score
        self.test_acc, self.test_loss, self.f1, self.auc  = test_inference(self.args, model=global_model, testloader=self.testloader, criterion=self.criterion)
    
    def train_val_test(self, dataset, idxs, user_id, malicious_users):
        """
        Split the dataset into training and test sets and create corresponding DataLoader objects.

        Args:
            dataset (torch.utils.data.Dataset): The complete dataset.
            idxs (list): A list of data indices for the current user.
            user_id (int): The ID of the current user.
            malicious_users (list): A list of IDs of malicious users.

        Returns:
            tuple: A tuple containing the training set DataLoader and the test set DataLoader.
        """
        train_ratio = 0.8
        total_samples = len(idxs)
        train_size = int(train_ratio * total_samples)
        random.shuffle(idxs)
        idxs_train = idxs[:train_size]
        idxs_test = idxs[train_size+1:]
        dataset_name = self.dataset_name
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train, user_id, malicious_users, dataset_name), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test, user_id, malicious_users, dataset_name), batch_size=self.args.local_bs, shuffle=True)
        return trainloader, testloader    
    
    def update_weights(self, global_round, global_model, w, UserID, lr, malicious_users):
        """
        Update the local model weights.

        Args:
            global_round (int): The current global communication round.
            global_model (torch.nn.Module): The global model shared across all users.
            w (dict): The state dictionary of the global model's weights.
            UserID (int): The unique identifier of the current user.
            lr (float): The learning rate for the optimizer.
            malicious_users (list): A list of user IDs that are considered malicious.

        Returns:
            tuple: A tuple containing the local sum of model updates, test accuracy, train loss, F1 score, AUC score, and heterogeneous parameter.
        """
        start_time = time.monotonic()
        local_sum = {}
        self.model.train()
        num_layers = self.args.layer_num
        # Load the shared layers of the global model into the local model
        if self.args.framework == 'FedAPM' or self.args.framework == 'FedAlt' or self.args.framework == 'FedSim':
            self.model.load_remaining_layers(w, num_layers)
        elif self.args.framework == 'FedProx' or self.args.framework == 'FedAvg' or self.args.framework == 'FedADMM':
            self.model.load_state_dict(w)

        # Adjust the parameters of the personalized and shared layers
        all_params = list(self.model.named_parameters())
        remaining_layers_names = [name for name, param in all_params[num_layers:]]
        first_n_layers_params = [param for name, param in all_params[:num_layers]]
        remaining_layers_params = [param for name, param in all_params[num_layers:]]
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
            v_i_t =  copy.deepcopy(self.model.state_dict())
            sigma_i_prox = 0.1
            sigma_i = 0.01

            if self.args.framework == 'FedAlt' or self.args.framework == 'FedAPM': 
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
    
                        for param in first_n_layers_params:
                            param.requires_grad = True

                        for param in remaining_layers_params:
                            param.requires_grad = False                          
                            
                        self.model.zero_grad()
                        optimizer_step1.zero_grad()
                        
                        if self.is_multimodal:
                            log_probs1, _ = self.model(x_a.float(), x_b.float(), l_a, l_b) 
                            loss1 = self.criterion(log_probs1, y)                                                                       
                        else:
                            log_probs1 = self.model(images)
                            loss1 = self.criterion(log_probs1, labels.long())

                        loss1.backward()
                        model_weights_pre = copy.deepcopy(self.model.state_dict())
                        if self.args.framework == 'FedAPM':
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.grad = (param.grad  + self.args.mu * model_weights_pre[name])  + (model_weights_pre[name] - v_i_t[name]) * sigma_i    
                        elif self.args.framework == 'FedAlt':  
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.grad = param.grad  + self.args.mu * model_weights_pre[name]
                        optimizer_step1.step()
            
            if self.args.framework != 'FedSim':
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
                        if self.args.framework == 'FedAlt' or self.args.framework == 'FedAPM':
                            for param in first_n_layers_params:
                                param.requires_grad = False    
                            for param in remaining_layers_params:
                                param.requires_grad = True              
                        self.model.zero_grad()
                        optimizer_step2.zero_grad()
                        if self.is_multimodal:
                            log_probs2, _  = self.model(x_a.float(), x_b.float(), l_a, l_b) 
                            loss2 = self.criterion(log_probs2, y)         
                        else:
                            log_probs2 = self.model(images)
                            loss2 = self.criterion(log_probs2, labels.long())
                        loss2.backward()
                        model_weights_pre = copy.deepcopy(self.model.state_dict())

                        if self.args.framework == 'FedAPM':
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.grad = (param.grad   + self.args.mu * model_weights_pre[name] )  +  (self.alpha[name] + self.args.rho * (model_weights_pre[name] - w[name]))

                        elif self.args.framework == 'FedAvg' or self.args.framework == 'FedAlt':
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.grad = (param.grad  + self.args.mu * model_weights_pre[name])
                        elif self.args.framework == 'FedProx': 
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.grad = (param.grad  + self.args.mu * model_weights_pre[name]) + (model_weights_pre[name] - v_i_t[name]) * sigma_i_prox
                        elif self.args.framework == 'FedADMM':
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.grad = (param.grad   + self.args.mu * model_weights_pre[name] ) +  (self.alpha[name] + self.args.rho * (model_weights_pre[name] - w[name]))
                        optimizer_step2.step()
                        self.logger.add_scalar('loss', loss2.item())
                        batch_loss.append(loss2.item())
                    epoch_loss.append(sum(batch_loss))

            if self.args.framework == 'FedSim':
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
                                
                        for param in first_n_layers_params:
                            param.requires_grad = True

                        for param in remaining_layers_params:
                            param.requires_grad = False                          
                            
                        self.model.zero_grad()
                        optimizer_step1.zero_grad()
                        
                        if self.is_multimodal:
                            log_probs1, _ = self.model(x_a.float(), x_b.float(), l_a, l_b) 
                            loss1 = self.criterion(log_probs1, y)                                                                       
                        else:
                            log_probs1 = self.model(images)
                            loss1 = self.criterion(log_probs1, labels.long())

                        loss1.backward()
                        model_weights_pre = copy.deepcopy(self.model.state_dict())
          
                                        
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.grad = param.grad  + self.args.mu * model_weights_pre[name]
                        optimizer_step1.step()                                
                                
                        for param in first_n_layers_params:
                            param.requires_grad = False    
                        for param in remaining_layers_params:
                            param.requires_grad = True              
                                        
                        self.model.zero_grad()
                        optimizer_step2.zero_grad()
                        if self.is_multimodal:
                            log_probs2, _  = self.model(x_a.float(), x_b.float(), l_a, l_b) 
                            loss2 = self.criterion(log_probs2, y)         
                        else:
                            log_probs2 = self.model(images)
                            loss2 = self.criterion(log_probs2, labels.long())
                        loss2.backward()
                        model_weights_pre = copy.deepcopy(self.model.state_dict())
   
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.grad = (param.grad  + self.args.mu * model_weights_pre[name]) 
                        optimizer_step2.step()
                        self.logger.add_scalar('loss', loss2.item())
                        batch_loss.append(loss2.item())
                    epoch_loss.append(sum(batch_loss))


            self.weights = self.model.state_dict()

            if self.args.framework == 'FedAPM' or self.args.framework == 'FedADMM':
                for key in self.alpha.keys():
                    self.alpha[key] = self.alpha[key] + self.args.rho * (self.weights[key]-w[key])
                    local_sum[key] = (self.weights[key] + (1 / self.args.rho) * self.alpha[key])
                                      
            elif self.args.framework == 'FedAvg' or self.args.framework == 'FedProx' or self.args.framework == 'FedAlt' or self.args.framework == 'FedSim' or self.args.framework == 'FedALA':  
                for key in self.alpha.keys():
                    local_sum[key] = self.weights[key] 

            # Calculate the client drift. Here, 'weights' and 'w' are the same except for the shared model, so we can directly subtract them.
            heterogeneous_param =  0 
            for key in remaining_layers_names:
                heterogeneous_param += torch.norm(self.weights[key]-w[key])

        self.test_acc, self.test_loss, self.f1, self.auc = test_inference(self.args, self.model, self.testloader, self.criterion)
        self.train_loss = self.inference(self.model)        
        end_time = time.monotonic()
        print(f"\x1b[{32}m{'Round: {} | Client {} | Test accuracy: {:.2f}% | Test loss: {:.2f} |  F1 Score: {:.2f} |  AUC: {:.2f} | Time: {:.2f}s'.format(global_round, UserID, 100*self.test_acc, self.test_loss, self.f1, self.auc, end_time-start_time)}\x1b[0m")
        return local_sum, self.test_acc, self.train_loss, self.f1, self.auc, heterogeneous_param.item()

    def inference(self, model):
        """
        Perform inference on the given model and calculate the loss on the training set.

        Args:
            model (torch.nn.Module): The model to be used for inference.

        Returns:
            float: The total loss on the training set.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, batch_data in enumerate(self.trainloader):
            if self.is_multimodal:
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.args.device), x_b.to(self.args.device), y.to(self.args.device)
                l_a, l_b = l_a.to(self.args.device), l_b.to(self.args.device)
                outputs, _ = self.model(
                    x_a.float(), x_b.float(), l_a, l_b
                )    
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
    """
    Perform inference on the test set and calculate the model's accuracy, total test loss, F1 score, and AUC score.

    Args:
        args (argparse.Namespace): Command-line arguments containing information such as model type and number of classes.
        model (torch.nn.Module): The model to perform inference on.
        testloader (torch.utils.data.DataLoader): The data loader for the test set.
        criterion (torch.nn.Module): The loss function used to calculate the test loss.

    Returns:
        tuple: A tuple containing the accuracy, total test loss, F1 score, and AUC score.
    """

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

