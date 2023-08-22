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
from models.federated import ground_truth_composition, FedAvg, outlier_detect, imba_aware_monitoring
from models.update import LocalUpdate
from models.evaluation import evaluate_model
from models.evaluation import cosine_similarity

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# args.device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
args.gpu = -1

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

autoregressive_quantity_observation = []
autoregressive_ratio_observation = []

global_ground_truth_composition_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0,
                                                   1.0, 1.0, 1.0, 1.0, 1.0])

Tj_buffer = []
TG_buffer = []

for g_round in range(args.rounds):
    w_locals, loss_locals, ac_locals, num_samples = [], [], [], []

    # select clients for federated training
    m = max(int(args.frac * args.num_users), 1)
    selected_clients = np.random.choice(range(args.num_users), m, replace=False)
    selected_clients_composition = ground_truth_composition(dict_clients, selected_clients, num_classes,
                                                            dataset_for_train.targets)
    print("The ground truth composition of each class is ", selected_clients_composition)

    if g_round == 0:
        autoregressive_quantity_observation = [np.sum(selected_clients_composition) / num_classes for i in range(num_classes)]
        autoregressive_ratio_observation = autoregressive_quantity_observation / np.sum(selected_clients_composition)
    else:
        autoregressive_quantity_observation = autoregressive_ratio_observation * np.sum(selected_clients_composition)

    for client in selected_clients:
        client_side = LocalUpdate(args=args, dataset=dataset_for_train, idxs=dict_clients[client], alpha=ratio,
                                  size_average=False, observation=autoregressive_quantity_observation)
        w, loss, ac = client_side.train(model=copy.deepcopy(net_glob).to(args.device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        ac_locals.append(copy.deepcopy(ac))
        num_samples.append(len(dict_clients[client]))

    # select "important" connections
    imt_model, imt_loss = [], []
    auxiliary_classes = [i for i in range(len(dict_classes))]
    for i in auxiliary_classes:
        aux_client = LocalUpdate(args=args, dataset=dataset_for_train, idxs=dict_classes[i], alpha=ratio,
                                 size_average=False, observation=autoregressive_quantity_observation)
        imt_w, imt_loss, _ = aux_client.train(model=copy.deepcopy(net_glob).to(args.device))
        imt_model.append(copy.deepcopy(imt_w))

    pos = outlier_detect(w_glob, imt_model)

    # aggregation
    w_glob_last = copy.deepcopy(w_glob)
    w_glob = FedAvg(w_locals)

    # imbalance aware monitoring
    total_samples = np.sum(num_samples)

    pro_res_1, pro_res_2 = imba_aware_monitoring(imt_model, pos, w_glob_last, w_glob, num_classes, m, total_samples, args)

    estimated_total_samples = np.sum(pro_res_1)
    new_quantity_observation = pro_res_1.tolist()
    new_ratio_observation = new_quantity_observation / estimated_total_samples
    autoregressive_ratio_observation = autoregressive_quantity_observation / total_samples
    for i in range(len(autoregressive_quantity_observation)):
        autoregressive_ratio_observation[i] = (1 - args.frac) * autoregressive_ratio_observation[i] + args.frac * new_ratio_observation[i]


    net_glob.load_state_dict(w_glob)


    # record round loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    ac_avg = sum(ac_locals) / len(ac_locals)
    print('Round {:3d}, Average loss {:.3f}, Accuracy {:.3f}\n'.format(g_round, loss_avg, ac_avg))
    loss_train.append(loss_avg)

    # evaluation
    net_glob.eval()
    acc_test, loss_test = evaluate_model(net_glob, dataset_for_test, args)

    # ratio estimation evaluation
    T_j = cosine_similarity(np.array(new_ratio_observation), selected_clients_composition)
    T_G = cosine_similarity(np.array(autoregressive_ratio_observation), global_ground_truth_composition_vector)

    print("T_j Value: ", T_j)
    print("T_G Value: ", T_G)

f_Tj = open("./Tj.txt", "w")
f_TG = open("./TG.txt", "w")

f_Tj.writelines(Tj_buffer.tolist())
f_TG.writelines(TG_buffer.tolist())

f_Tj.close()
f_TG.close()
