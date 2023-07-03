import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn import metrics
from sklearn.preprocessing import label_binarize


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def evaluate_model(model, datatest, args):
    model.eval()
    test_loss = 0
    correct = 0
    auc_final = []
    auc_final_new = []

    data_loader = DataLoader(DatasetSplit(datatest, idxs=[i for i in range(len(datatest))]), batch_size=args.bs)
    l = len(data_loader)

    classes = np.unique(datatest.targets)
    num_classes = len(classes)
    false = np.zeros((num_classes, 1))
    false_all = np.zeros((num_classes, 1))

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # cuda acceleration
            data, target = data.cuda(), target.cuda()
        data, target = data.to(args.device, dtype=torch.float), target.to(args.device, dtype=torch.long)
        predicts = model(data)

        # calculate batch loss, reduction="sum"
        test_loss += F.cross_entropy(predicts, target, reduction='sum').item()

        pred_index = predicts.data.max(1, keepdim=True)[1]
        correct += pred_index.eq(target.data.view_as(pred_index)).long().cpu().sum()

        # calculate auc
        pred_one_hot = label_binarize(pred_index.data.cpu().numpy().flatten(), classes=[i for i in range(num_classes)])
        target_one_hot = label_binarize(target.data.cpu().numpy().flatten(), classes=[i for i in range(num_classes)])

        auc_temp = metrics.roc_auc_score(target_one_hot, predicts.cpu().detach().numpy(), average='micro')
        auc_temp_new = metrics.roc_auc_score(target_one_hot, pred_one_hot, average='micro')

        auc_final.append(auc_temp)
        auc_final_new.append(auc_temp_new)

        for i in range(torch.numel(pred_index)):
            false_all[target.data.view_as(pred_index)[i][0], 0] += 1
            if pred_index.eq(target.data.view_as(pred_index)).long().cpu()[i] == 0:
                false[target.data.view_as(pred_index)[i][0], 0] += 1

    test_loss = test_loss/len(data_loader.dataset)
    auc_res = np.mean(auc_final)
    auc_res_new = np.mean(auc_final_new)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    for i in range(len(false)):
        ac_temp = (false_all[i, 0] - false[i, 0]) / false_all[i, 0]
        print('{:.4f}'.format(ac_temp))
    print('AUC Score: {:.6f}, AUC Score New: {:.6f}'.format(auc_res, auc_res_new))
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\nAUC Score: {:.4f}'.format(
            test_loss, correct, len(data_loader.dataset), accuracy, auc_res))

    return accuracy, test_loss
