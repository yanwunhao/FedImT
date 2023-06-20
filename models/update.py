import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        print(self.dataset, self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, alpha, size_average):
        self.args = args
        if self.args.loss == "cross-entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.bs, shuffle=True)

    def accuracy_calculation(self, pred, label):
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
        pred_number = np.argmax(pred, 1)
        count = 0
        for i in range(len(pred_number)):
            if pred_number[i] == label[i]:
                count += 1
        return count, len(pred_number)


    def train(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        epoch_loss = []
        epoch_ac = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_ac = []
            batch_whole = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # images = images.unsqueeze(1)
                print(images.shape)
                images, labels = images.to(self.args.device, dtype=torch.float), labels.to(self.args.device,
                                                                                           dtype=torch.long)
                model.zero_grad()
                log_form_probs = model(images)
                ac, whole = self.accuracy_calculation(log_form_probs, labels)
                # print(ac, whole)

                loss = self.loss_fn(log_form_probs, labels)
                loss.backward()
                optimizer.step()
                break



