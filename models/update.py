import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from balanced_loss import Loss

from .ghm import GHMC_Loss


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, alpha, size_average, observation):
        self.args = args
        if self.args.loss == "cross-entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif self.args.loss == "focal":
            self.loss_fn = Loss(loss_type="focal_loss")
        elif self.args.loss == "balanced-cross-entropy":
            self.loss_fn = Loss(
                loss_type="cross_entropy",
                samples_per_class=observation,
                class_balanced=True
            )
        elif self.args.loss == 'ghm':
            self.loss_fn = GHMC_Loss(bins=10, alpha=0.75)
        else:
            exit("No loss function specified")
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
                # print(images.shape)
                images, labels = images.to(self.args.device, dtype=torch.float), labels.to(self.args.device,
                                                                                           dtype=torch.long)
                model.zero_grad()
                prediction = model(images)
                ac, whole = self.accuracy_calculation(prediction, labels)
                # print(ac, whole)

                loss = self.loss_fn(prediction, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, batch_idx * len(images),
                                                                                    len(self.ldr_train.dataset),
                                                                                    100. * batch_idx / len(
                                                                                        self.ldr_train), loss.item()))

                batch_loss.append(loss.item())
                batch_ac.append(ac)
                batch_whole.append(whole)

            epoch_ac.append(sum(batch_ac) / sum(batch_whole))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_ac) / len(epoch_ac)
