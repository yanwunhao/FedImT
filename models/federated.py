import numpy as np

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
