import os
import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from partial_update import GlobalLocalUpdate, test_inference
from models import Model1, Model2, MLP, MLR, CNN, SVM, CNN1, CNN2, ResNet18, BasicBlock, ImageTextClassifier, HARClassifier, MMActionClassifier, ECGClassifier
from utils import get_dataset, average_weights, exp_details, setup_seed, average_loss_acc_centralized, load_model
from constants import feature_len_dict, num_class_dict, max_class_dict




if __name__ == '__main__':
    seeds = [10]

    for seed in seeds:
        start_time = time.time()
        path_project = os.path.abspath('.')
        logger = SummaryWriter('../logs')
        args = args_parser()
        exp_details(args)  
        setup_seed(seed)    
        
        print('random seed =', seed)
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, user_groups = get_dataset(args)
        local_model, model = [], []
        global_model = load_model(args)
        global_model.to(args.device)
        global_model.train()
        global_weights = global_model.state_dict()
        w = copy.deepcopy(global_weights)
        train_loss, test_acc, test_loss = [], [], []
        f1_score, auc = [], []
        test_loss_variances, test_acc_variances = [], []
        train_loss_personal_local, train_loss_global_local = 0, 0
        malicious_users = []
        # Check if the attack mode is enabled. If args.corrupted is '1', it means the attack mode is enabled.
        if args.corrupted == '1':
            # Randomly select a portion of users as malicious users.
            # The number of malicious users is calculated by args.num_malicious * args.num_users, and the maximum value is taken to ensure that there is at least one malicious user.
            # Use the np.random.choice function to randomly select a specified number of users from all users as malicious users, and do not allow repeated selection.
            malicious_users = np.random.choice(range(args.num_users), max(int(args.num_malicious * args.num_users), 1), replace=False)

        # Iterate through all users, create a GlobalLocalUpdate object for each user, and add it to the local_model list.
        # Each object represents a local model of a user and contains user-specific data and configurations.
        for idx in range(args.num_users):
            local_model.append(GlobalLocalUpdate(args=args, global_model=global_model, dataset=dataset, idxs=user_groups[idx], logger=logger, user_id=idx, malicious_users=malicious_users))
        # Calculate the average training loss, test accuracy, test loss, test accuracy variance, test loss variance, F1 score, and AUC score
        # across all local models using the average_loss_acc_centralized function.
        train_loss_avg, test_acc_avg, test_loss_avg, test_acc_variance, test_loss_variance, f1_avg, auc_avg= average_loss_acc_centralized(local_model, args.num_users, malicious_users)
        train_loss.append(train_loss_avg)
        test_acc.append(test_acc_avg)
        test_loss.append(test_loss_avg)
        test_loss_variances.append(test_loss_variance)
        test_acc_variances.append(test_acc_variance)
        f1_score.append(f1_avg)
        auc.append(auc_avg)   
        norm_params = [] 

        for epoch in range(args.epochs):
            gradients_L2, local_sum, local_train_losses_personal, local_test_accuracies_personal, \
            local_test_accuracies_global, local_train_losses_global = [], [], [], [], [], []
            heterogeneous_param_list = []
            global_model.train()
            # Determine the client selection strategy based on the 'args.strategy' hyperparameter.
            m = max(int(args.frac_clients * args.num_users), 1)
            if args.strategy == 'random': idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            elif args.strategy == 'full': idxs_users = range(args.num_users)
            else: exit('Error: unrecognized client selection strategy.')
            print(f"\n \x1b[{35}m{'The IDs of selected clients: ' + ', '.join(map(str, np.sort(idxs_users)))}\x1b[0m")
            # Iterate through each selected client.
            for idx in idxs_users:
                # Update the local model's weights for the current client in the current global round.
                lsum, local_test_acc_personal, local_train_loss_personal, local_f1_personal, local_auc_personal, heterogeneous_param = local_model[idx].update_weights(global_round=epoch, global_model=global_model, w=w, UserID=idx, lr=args.lr, malicious_users=malicious_users)
                local_test_accuracies_personal.append(copy.deepcopy(local_test_acc_personal))
                heterogeneous_param_list.append(heterogeneous_param)
            # Update global model weights based on the selected clients' local model updates.
            for key in w.keys():
                w[key] = w[key].float() 
                w[key] = torch.zeros_like(w[key])
                for i in range(0, len(local_model)):
                    w[key] += (local_model[i].weights[key] + (1 / args.rho) * local_model[i].alpha[key]) * 1.0 / args.num_users

            norm_param = sum(heterogeneous_param_list) / len(heterogeneous_param_list)
            global_model.load_state_dict(w)
            train_loss_avg, test_acc_avg, test_loss_avg, test_acc_variance, test_loss_variance, f1_avg, auc_avg= average_loss_acc_centralized(local_model, args.num_users, malicious_users)
            train_loss.append(train_loss_avg)
            test_acc.append(test_acc_avg)
            test_acc_variances.append(test_acc_variance)
            test_loss_variances.append(test_loss_variance)
            f1_score.append(f1_avg)
            auc.append(auc_avg)                
            norm_params.append(norm_param)
            print(f"\n\x1b[{34}m{'>>> Round: {} / Test accuracy: {:.2f}% / Training loss: {:.4f} / F1 Score: {:.4f} / AUC: {:.4f}% / Param Norm: {:.4f} / Test accuracy variance: {:.6f}'.format(epoch, 100 * test_acc_avg, train_loss_avg, f1_avg, 100 * auc_avg, norm_param, test_acc_variance)}\x1b[0m")


        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
        
        output = {}
        output['dataset'] = args.dataset
        output['framework'] = args.framework
        output['num_users'] = args.num_users
        output['seed'] = args.seed
        output['local_ep'] = args.local_ep
        output['training_loss'] = train_loss
        output['test_acc'] = test_acc
        output['test_loss'] = test_loss
        output['test_acc_variances'] = test_acc_variances
        output['f1_score'] = f1_score
        output['auc'] = auc
        output['norm'] = norm_params

        data_file = '../save/{}_{}_random_seed_{}_users_{}_rho_{}_lambda_{}_epoch_{}_partition_{}_q_{}_attack_{}_num_malicious_{}_aggr_{}.json'.format(args.dataset,
                                                                                                        args.model,
                                                                                                        args.framework,
                                                                                                        seed,
                                                                                                        args.num_users,
                                                                                                        args.rho,
                                                                                                        args.Lambda,
                                                                                                        args.local_ep,
                                                                                                        args.partition,
                                                                                                        args.q,
                                                                                                        args.corrupted,
                                                                                                        args.num_malicious,
                                                                                                        args.aggr)
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist() 
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()} 
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]  
            else:
                return obj  
        
        output_serializable = convert_to_serializable(output)

        with open(data_file, "w") as dataf:
            json.dump(output_serializable, dataf)        
            

            


