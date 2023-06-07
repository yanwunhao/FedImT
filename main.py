import torch
from torchvision import datasets, transforms
import numpy as np

from warnings import simplefilter
import copy

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

import matplotlib
matplotlib.use('Agg')

from utils.sampling import mnist_nonidd, get_auxiliary_data
from utils.options import args_parser
from models.networks import LeNet5
from models.federated import ground_truth_composition
from models.update import LocalUpdate

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

print(args)

if args.dataset == "mnist":
    # 0.1307 and 0.3081 were provided by officials
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_for_train = datasets.MNIST('./data/', train=True, download=True, transform=trans_mnist)
    dataset_for_test = datasets.MNIST('./data/', train=False, download=True, transform=trans_mnist)

    img_size = dataset_for_train[0][0].shape

    if args.iid:
        # this work only considers non-iid scenario
        pass
    else:
        dict_clients = mnist_nonidd(dataset_for_train, args)
        # print(dict_clients)
else:
    exit("ERROR: NO RECOGNIZED DATASET")

# build the deep model
if args.model == "cnn":
    net_glob = LeNet5().to(args.device)
elif args.model == "mlp":
    pass
else:
    exit('Error: unrecognized model')

print(net_glob)

# start federated learning workflow
net_glob.train()

# copy weights
w_glob = net_glob.state_dict()

# training
loss_train = []
ratio = None

# get auxiliary data for ratio estimation
dict_classes, num_classes = get_auxiliary_data(dataset_for_train, args)

for round in range(args.epochs):
    w_locals, loss_locals, ac_locals, num_samples = [], [], [], []

    # select clients for federated training
    m = max(int(args.frac * args.num_users), 1)
    selected_clients = np.random.choice(range(args.num_users), m, replace=False)
    selected_clients_composition = ground_truth_composition(dict_clients, selected_clients, num_classes, dataset_for_train.targets)
    print("The ground truth composition of each class is ", selected_clients_composition)

    for client in selected_clients:
        client_side = LocalUpdate(args=args, dataset=dataset_for_train, idxs=dict_clients[client], alpha=ratio, size_average=False)
        client_side.train(model=copy.deepcopy(net_glob).to(args.device))
        break
    break

