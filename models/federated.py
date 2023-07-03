import copy

import numpy as np
import torch


def ground_truth_composition(dict_clients, selected_clients, num_classes, labels):
    res = [0 for i in range(num_classes)]
    classes = np.unique(labels)
    for idx in selected_clients:
        for i in dict_clients[idx]:
            for j in classes:
                i_label = int(labels[i].numpy())
                j_label = int(j)
                if i_label == j_label:
                    res[j_label] += 1
    return res


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
