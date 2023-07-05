import copy

import numpy as np
import torch


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


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


def outlier_detect(w_global, w_class):
    last_fc_weight_global = w_global['fc3.weight'].cpu().numpy()
    delta_weights = []
    for i in range(len(last_fc_weight_global)):
        delta_w = (w_class[i]['fc3.weight'].cpu().numpy() - last_fc_weight_global) * 100
        delta_weights.append(delta_w)

    res = calculate_neuron_importance(delta_weights)
    return res


def calculate_neuron_importance(delta_weights):
    delta_weights = np.array(delta_weights)

    output_nodes_num = delta_weights.shape[1]
    last_layer_nodes_num = delta_weights.shape[2]

    pos_res = np.zeros((len(delta_weights), output_nodes_num, last_layer_nodes_num))

    for i in range(output_nodes_num):
        for j in range(last_layer_nodes_num):
            delta_weight_for_this_connection = []
            for m in range(len(delta_weights)):
                delta_weight_for_this_connection.append(delta_weights[m, i, j])
            delta_weight_for_this_connection = np.array(np.abs(delta_weight_for_this_connection))
            max_index = np.argmax(delta_weight_for_this_connection)
            max_value = delta_weight_for_this_connection[max_index]
            outlier = np.where((delta_weight_for_this_connection / max_value) > 0.75)
            if len(outlier[0]) < 2:
                pos_res[max_index, i, j] = 1
    return pos_res


def imba_aware_monitoring(cc_net, pos, w_glob_last, w_glob, num_class, num_users, num_samples, args):
    return "a", "b"
