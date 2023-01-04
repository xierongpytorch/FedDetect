import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
torch.backends.cudnn.benchmark=True

import torch.utils.data as Data

import pandas as pd
from sklearn.model_selection import train_test_split

# load dataset from a csv file
df = pd.read_csv('AllData.csv')

# drop the userID
df = df.drop(columns=['UserId', ])
df.tail()

anomalies = df[df["IsStealer"] == 1]
normal = df[df["IsStealer"] == 0]

anomalies.shape, normal.shape

DF_train, DF_test = train_test_split(df, test_size = 0.4, random_state = 66)
DF_train.shape, DF_test.shape

DF_train_y = DF_train["IsStealer"]
DF_train_X = DF_train.drop(columns=['IsStealer'])

DF_test_y = DF_test["IsStealer"]
DF_test_X = DF_test.drop(columns=['IsStealer'])

train_y = np.array(DF_train_y).reshape(DF_train_y.shape[0] , 1)
train_X = np.array(DF_train_X).reshape(DF_train_X.shape[0] , 1, DF_train_X.shape[1])

test_y = np.array(DF_test_y).reshape(DF_test_y.shape[0] , 1)
test_X = np.array(DF_test_X).reshape(DF_test_X.shape[0] , 1, DF_test_X.shape[1])

print("training dataset shapes: train_X: %s and train_y: %s" % (train_X.shape, train_y.shape))
print("testing dataset shapes: test_X: %s and test_y: %s" % (test_X.shape, test_y.shape))

train_y_ts = torch.from_numpy(train_y).float()
train_X_ts = torch.from_numpy(train_X).float()

test_y_ts = torch.from_numpy(test_y).float()
test_X_ts = torch.from_numpy(test_X).float()

train_set = Data.TensorDataset(train_X_ts, train_y_ts)
test_set = Data.TensorDataset(test_X_ts, test_y_ts)

num_clients = 10
num_selected = 10
num_rounds = 29
epochs = 1
batch_size = 64

traindata_split = torch.utils.data.random_split(train_set, [int(train_X.shape[0] / num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.conv_1 = nn.Conv1d(1, 64, kernel_size=2, dilation=1, padding=((2-1) * 1))
        self.conv_2 = nn.Conv1d(64, 32, kernel_size=4, dilation=2, padding=((4-1) * 2))
        self.conv_3 = nn.Conv1d(32, 16, kernel_size=8, dilation=4, padding=((8-1) * 4))
        self.conv_4 = nn.Conv1d(16, 8, kernel_size=16, dilation=8, padding=((16-1) * 8))
        self.conv_5 = nn.Conv1d(8, 4, kernel_size=32, dilation=16, padding=((32-1) * 16))
        self.dense_1 = nn.Linear(1035*4, 128)
        self.dense_2 = nn.Linear(128, 2)

        self.lossfunction = nn.CrossEntropyLoss()

    def forward(self, x):

        x = self.conv_1(x)
        x = x[:, :, :-self.conv_1.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_2(x)
        x = x[:, :, :-self.conv_2.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_3(x)
        x = x[:, :, :-self.conv_3.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_4(x)
        x = x[:, :, :-self.conv_4.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_5(x)
        x = x[:, :, :-self.conv_5.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = x.view(-1, 1035*4)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        #output = F.log_softmax(x, dim=1)
        return x


def client_update(client_model, optimizer, train_loader, epoch=5):

    model.train()

    epoch_loss = []
    for e in range(epoch):

        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)

            labels = target.squeeze()
            labels = labels.long()
            loss = client_model.lossfunction(output, labels)

            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / (len(batch_loss) * batch_size))

    return sum(epoch_loss) / len(epoch_loss)


def server_aggregate(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        w0 = global_dict[k]
        x1 = w0 - client_models[0].state_dict()[k].float()
        x2 = x1 * 1024
        x3 = torch.clamp(x2, min=-32768, max=32767)
        x4 = torch.div(x3, 1)
        x5 = x4
        for i in range(1, len(client_models)):
            x1 = w0 - client_models[i].state_dict()[k].float()
            x2 = x1 * 1024
            x3 = torch.clamp(x2, min=-32768, max=32767)
            x4 = torch.div(x3, 1)
            x5 = x5 + x4
        x6 = torch.div(x5, 1024 * len(client_models))
        x7 = w0 - x6
        global_dict[k] = x7
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def test(global_model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)

            labels = target.squeeze().long()
            loss = global_model.lossfunction(output, labels)
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred).long()).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc


global_model = TCN().cuda()


client_models = [ TCN().cuda() for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())


opt = [optim.Adam(model.parameters(), lr=0.001) for model in client_models]

losses_train = []
losses_test = []
acc_train = []
acc_test = []


for r in range(num_rounds):
    client_idx = np.random.permutation(num_clients)[:num_selected]
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)

    losses_train.append(loss / num_selected)
    server_aggregate(global_model, client_models)

    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.4g | test loss %0.4g | test acc: %0.4f' % (loss / num_selected, test_loss, acc))