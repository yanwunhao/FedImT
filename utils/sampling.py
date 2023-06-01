import numpy as np


def mnist_nonidd(dataset, args):
    # spilt samples into shards
    number_of_each_shard = 100
    num_shards = int(len(dataset) / number_of_each_shard)

    # create dictionary mapping samples to users
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([], dtype='int64') for i in range(args.num_users)}

    # calculate valid samples index
    idxs = np.arange(num_shards * number_of_each_shard)
    labels = dataset.targets.numpy()[:len(idxs)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(args.num_users):
        rand_shards = set(np.random.choice(idx_shard, int(len(idx_shard) / args.num_users), replace=False))
        # remove the selected shards from sample pool
        idx_shard = list(set(idx_shard) - rand_shards)
        for shard in rand_shards:
            dict_clients[i] = np.concatenate(
                (dict_clients[i], idxs[shard * number_of_each_shard:(shard + 1) * number_of_each_shard]), axis=0)
    return dict_clients


def get_auxiliary_data(dataset, args):
    labels = dataset.targets
    classes = np.unique(labels)
    num_classes = len(classes)
    dict_classes = {i: np.array([], dtype='int64') for i in classes}

    for selected_label in classes:
        idx_temp = np.where(labels == selected_label)
        dict_classes[selected_label] = np.concatenate((dict_classes[selected_label], idx_temp[0][0:args.bs]), axis=0)

    return dict_classes, num_classes
