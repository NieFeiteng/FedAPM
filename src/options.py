import argparse


# crisis_mmd ImageTextClassifier 20  
# ku_har HARClassifier 20
# crema_d MMActionClassifier 72


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default='FedAvg', choices=['FLAME', 'pFedMe', 'ditto', 'FedAPM', 'FedAvg','lp-proj-2','FLAME-lp-proj-2', 'FedAlt', 'FedSim', 'FedProx'], help="federated learning framework")
    parser.add_argument('--partition', type=str, default='dir-label-skew', choices=['hybrid-skew', 'dir-label-skew', 'q-label-skew', 'quality-skew', 'dir-quantity-skew','feature-skew', 'VQA-partition'], help="type of partitioning data")
    parser.add_argument('--num_users', type=int, default=20, help="number of users, must be a multiple of 5")
    parser.add_argument('--noise_scale', type=float, default=0.1, help='when partition=noise-feature-skew, use this param')
    parser.add_argument('--q', type=int, default=2, help='number of labels in each client')
    parser.add_argument('--corrupted', type=str, default='0', choices=['0', '1', '2', '3', '4'], help='0: no corrupt, 1: label poison, 2: SVA, 3: SFA, 4: GA')
    parser.add_argument('--malicious_scale', type=float, default=0.1, help='when corrupted=2,3,4, use this param')
    parser.add_argument('--num_malicious', type=int, default=0, help='fraction of malicious clients')
    parser.add_argument('--aggr', type=str, default='regular', help='robust aggregation method', choices=['regular', 'mkrum'])
    parser.add_argument('--model', type=str, default='CNN', choices=['ResNet18', 'Model1', 'Model2', 'MLP', 'MLR', 'CNN', 'SVM', 'CNN1', 'CNN2', 'VQAModel','ImageTextClassifier', 'HARClassifier', 'MMActionClassifier'], help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', "fmnist", 'mmnist', "cifar10", 'femnist', 'synthetic', 'VQA', 'hateful_memes', 'crisis_mmd', 'uci-har', 'ku_har', 'crema_d', 'ptb-xl'], help="name of dataset")
    parser.add_argument('--strategy', type=str, default='random', choices=['biased', 'random', 'full'], help="client selection strategy")
    parser.add_argument('--frac_candidates', type=float, default=0.3, help='fraction of clients candidates: S')
    parser.add_argument('--frac', type=float, default=0.3, help='fraction of clients: C')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--momentum', type=float, default=0.01, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=300, help='total communication rounds')
    parser.add_argument('--local_ep', type=int, default=3, help='number of local epochs: E')
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--mu', type=float, default=0.01, help='hpy in regularization term')
    parser.add_argument('--Lambda', type=float, default=1, help='hpy in Moreau Envelope')
    parser.add_argument('--rho', type=float, default=0.01, help='hyp in Penalty term')
    parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--fixed', type=int, default=1, help='fixed local epochs, 1 for fixed')
    parser.add_argument('--eta', type=float, default=0.1, help='learning rate of global model phase 2')
    parser.add_argument('--eta2', type=float, default=0.1, help="learning rate of global model phase 2")

    parser.add_argument("--alpha", type=float, default=1.0, help="alpha in direchlet distribution")  
    parser.add_argument('--hid_size', type=int, default=128, help='RNN hidden size dim')
    parser.add_argument('--att', type=bool, default=True,help='self attention applied or not')
    parser.add_argument('--att_name', type=str, default='fuse_base', help='attention name')
     
    parser.add_argument('--layer_num', type=int, default='4', help='attention name') 
    parser.add_argument('--is_multimodal', type=bool, default=False, help='Whether the dataset is multimodal')

    args = parser.parse_args()

    if args.dataset in ['crisis_mmd', 'ku_har', 'crema_d']:
        args.is_multimodal = True
    else:
        args.is_multimodal = False    
    return args

