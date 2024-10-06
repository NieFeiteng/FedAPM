import os
import copy
import time
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from partial_update import GlobalLocalUpdate
from models import Model1, Model2, MLP, MLR, CNN, SVM, CNN1, CNN2, ResNet18, BasicBlock, ImageTextClassifier, HARClassifier, MMActionClassifier
from utils import get_dataset, exp_details, setup_seed, average_loss_acc_centralized
from constants import feature_len_dict, num_class_dict


if __name__ == '__main__':

    start_time = time.time()

    path_project = os.path.abspath('.')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)   
    setup_seed(args.seed)  
    
    print('random seed =', args.seed)
    torch.cuda.set_device(1)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, user_groups = get_dataset(args)

    local_model, model = [], []
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        input_size = 784
        num_classes = 10
        args.num_classes = num_classes
    if args.model == 'Model1':
        global_model = Model1(args=args)
    elif args.model == 'Model2':
        global_model = Model2(args=args)
    elif args.model == 'MLP':
        input_size = 784
        num_classes = 10
        global_model = MLP(input_size, num_classes)
    elif args.model == 'MLR':
        global_model = MLR(args=args)
    elif args.model == 'CNN':
        num_classes = 10
        args.num_classes = num_classes
        global_model = CNN()
    elif args.model == 'SVM':
        global_model = SVM()
    elif args.model == 'CNN1':
        global_model = CNN1()
    elif args.model == 'CNN2':
        global_model = CNN2()
    elif args.model == 'ResNet18':
        num_classes = 10
        args.num_classes = num_classes
        global_model = ResNet18(BasicBlock, [2,2,2,2], num_classes)   
    elif args.model == 'ImageTextClassifier': 
        args.num_classes = num_class_dict[args.dataset],
        global_model = ImageTextClassifier(
            num_classes=num_class_dict[args.dataset],
            img_input_dim=feature_len_dict["mobilenet_v2"], 
            text_input_dim=feature_len_dict["mobilebert"],
            d_hid=args.hid_size,
            en_att=True,
            att_name=args.att_name,
            is_adapter = False
        )
    elif args.model == 'HARClassifier': 
        args.num_classes = num_class_dict[args.dataset],
        global_model = HARClassifier(
            num_classes=num_class_dict[args.dataset],         # Number of classes 
            acc_input_dim=feature_len_dict['acc'],    # Acc data input dim
            gyro_input_dim=feature_len_dict['gyro'],  # Gyro data input dim
            en_att=True,                                            # Enable self attention or not
            d_hid=args.hid_size,
            att_name=args.att_name,
            is_adapter = False
        )                
    elif args.model == 'MMActionClassifier': 
        args.num_classes = num_class_dict[args.dataset],
        global_model = MMActionClassifier(
            num_classes=num_class_dict[args.dataset],         # Number of classes 
            audio_input_dim=feature_len_dict["mfcc"],    # Acc data input dim
            video_input_dim=feature_len_dict["mobilenet_v2"],  # Gyro data input dim
            d_hid=args.hid_size,
            en_att=True,                  
            att_name=args.att_name,
            is_adapter = False
        )                
    else:
        exit('Error: unrecognized model')

    global_model.to(args.device)
    global_model.train()
    # print(global_model)
    global_weights = global_model.state_dict()
    w = copy.deepcopy(global_weights)

    train_loss, test_acc, test_loss = [], [], []
    f1_score, auc = [], []
    test_loss_variances, test_acc_variances = [], []
    train_loss_personal_local, train_loss_global_local = 0, 0
    malicious_users = []
    if args.corrupted == '1':
        malicious_users = np.random.choice(range(args.num_users), max(int(args.num_malicious * args.num_users), 1), replace=False)


    for idx in range(args.num_users):
        local_model.append(GlobalLocalUpdate(args=args, global_model=global_model, dataset=dataset, idxs=user_groups[idx], logger=logger, user_id=idx, malicious_users=malicious_users))

    
    train_loss_avg, test_acc_avg, test_loss_avg, test_acc_variance, test_loss_variance, f1_avg, auc_avg= average_loss_acc_centralized(local_model, args.num_users, malicious_users)
    
    train_loss.append(train_loss_avg)
    test_acc.append(test_acc_avg)
    test_loss.append(test_loss_avg)
    test_loss_variances.append(test_loss_variance)
    test_acc_variances.append(test_acc_variance)
    
    f1_score.append(f1_avg)
    auc.append(auc_avg)   
    norm_params = [] 
    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(args.epochs):
        #  不用看
        gradients_L2, local_sum, local_train_losses_personal, local_test_accuracies_personal, \
        local_test_accuracies_global, local_train_losses_global = [], [], [], [], [], []
        heterogeneous_param_list = []
        m1 = max(int(args.frac_candidates * args.num_users), 1)
        m2 = max(int(args.frac * args.num_users), 1)

        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        idxs_candidates_users = np.random.choice(range(args.num_users), m1, replace=False)

        # Client selection:
        if args.strategy == 'biased':
            for idx in idxs_candidates_users:
                gradient_L2 = local_model[idx].calculate_gradient_l2_norm(w)
                gradients_L2.append((idx, gradient_L2))

                # gradients_norm = [item[1] for item in gradients_L2]
                # item0_gradients_L2 = [item[0] for item in gradients_L2]
                # total_gradient = sum(gradients_norm)
                # probabilities = [gradient / total_gradient for gradient in gradients_norm]
                # idxs_users = np.random.choice(item0_gradients_L2, size=m2, p=probabilities, replace=False)

                sorted_norms = sorted(gradients_L2, key=lambda x: x[1], reverse=True)
            idxs_users = [x[0] for x in sorted_norms[:m2]]
        elif args.strategy == 'random':
            idxs_users = np.random.choice(idxs_candidates_users, m2, replace=False)
        elif args.strategy == 'full':
            idxs_users = range(args.num_users)

        else:
            exit('Error: unrecognized client selection strategy.')

        #  不用看
        

        print(f"\n \x1b[{35}m{'The IDs of selected clients:{}'.format(np.sort(idxs_users))}\x1b[0m")
        # Update local and central models


        lr = args.lr
        for idx in idxs_users:
            lsum, local_test_acc_personal, local_train_loss_personal, local_f1_personal, local_auc_personal, heterogeneous_param = local_model[idx].update_weights(global_round=epoch, global_model=global_model, w=w, UserID=idx, lr=lr, malicious_users=malicious_users)

            local_test_accuracies_personal.append(copy.deepcopy(local_test_acc_personal))
            heterogeneous_param_list.append(heterogeneous_param)
            # local_sum.append(copy.deepcopy(lsum))

        # update_msg = average_weights(local_sum, args.num_users, args.aggr, args.num_malicious)  # update ui

        for key in w.keys():
            w[key] = w[key].float() 
            w[key] = torch.zeros_like(w[key])
            for i in range(0, len(local_model)):
                w[key] += (local_model[i].weights[key] + (1 / args.rho) * local_model[i].alpha[key]) * 1.0 / args.num_users

        # for key in w.keys():                   
        #     w[key] = w[key] + update_msg[key]
            # w[key] = copy.deepcopy(update_msg[key])
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
        print(
            f"\n\x1b[{34}m{'>>> Round: {} / Test accuracy: {:.2f}% / Training loss: {:.4f} / F1 Score: {:.4f} / AUC: {:.4f}% / Param Norm: {:.4f} / Test accuracy variance: {:.6f}'.format(epoch, 100 * test_acc_avg, train_loss_avg, f1_avg, 100 * auc_avg, norm_param, test_acc_variance)}\x1b[0m")
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

    data_file = '../../save/{}_{}_{}_{}_lr_{}_frac_{}_seed_{}_users_{}_rho_{}_lambda_{}_epoch_{}_partition_{}_q_{}_attack_{}_num_malicious_{}_aggr_{}.json'.format(args.dataset,
                                                                                                        args.model,
                                                                                                        args.framework,
                                                                                                        args.strategy,
                                                                                                        args.lr,
                                                                                                        args.frac_candidates,
                                                                                                        args.seed,
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
            return obj.tolist()  # 将 Tensor 转为 list
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}  # 递归处理字典
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]  # 递归处理列表
        else:
            return obj  # 其他类型保持不变
    
    output_serializable = convert_to_serializable(output)

    # 写入 JSON 文件
    with open(data_file, "w") as dataf:
        json.dump(output_serializable, dataf)        
        
    # with open(data_file, "w") as dataf:
    #     json.dump(output, dataf)

        


