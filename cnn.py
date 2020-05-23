import numpy as np
import json, os, sys, random, time
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

t0 = time.time()

# 参数
max_len = 500
batch_size = 64
label_size = 8
embed_size, kernel_sizes, nums_channels = 300, [2, 3, 4], [256, 256, 256]
pool_kernel, pool_stride = 2, 2
hidden_size, drop_out = 500, 0.8
lr, lr_decay, wei_decay = 0.001, 0, 0.0005
num_epochs = 10

# Data
train_x = np.load('./dataset/train.embed.npy')[:2000]
train_y = np.load('./dataset/train.score.npy')[:2000]
print('shape of train_x, train_y:', np.shape(train_x), np.shape(train_y))
train_dataset = Data.TensorDataset(torch.tensor(train_x).type('torch.FloatTensor'),
                                   torch.tensor(train_y).type('torch.FloatTensor'))
loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_x = np.load('./dataset/train.embed.npy')[2000:]
test_y = np.load('./dataset/train.score.npy')[2000:]
print('shape of test_x, test_y:', np.shape(test_x), np.shape(test_y))
test_dataset = Data.TensorDataset(torch.tensor(test_x).type('torch.FloatTensor'),
                                  torch.tensor(test_y).type('torch.FloatTensor'))
loader_test = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# test_x = np.load('./dataset/test.embed.npy')[:500]
# test_y = np.load('./dataset/test.score.npy')[:500]
# print('shape of test_x, test_y:', np.shape(test_x), np.shape(test_y))
# test_dataset = Data.TensorDataset(torch.tensor(test_x).type('torch.FloatTensor'),
#                                   torch.tensor(test_y).type('torch.FloatTensor'))
# loader_test = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# CNN
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(drop_out)
        self.decoder = nn.Linear(sum(nums_channels), label_size)
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        self.loss_func = nn.CrossEntropyLoss()
        for c, k in zip(nums_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embed_size, out_channels=c, kernel_size=k))

    def forward(self, inputs, glabels=None):
        embeddings = inputs.permute(0, 2, 1)
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        labels = F.softmax(outputs, dim=1)
        loss = self.loss_func(labels, torch.argmax(glabels, dim=1))
        return loss, labels.detach()


model = CNN()
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wei_decay)
optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=wei_decay)

t1 = time.time() - t0
print('Begin training CNN. Time={}m{}s'.format(int(t1//60), int(t1 % 60)))

for epoch in range(num_epochs):
    model.train()
    for step, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        loss, logit = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            t1 = time.time() - t0
            print('Epoch[{}] | loss={:.4f} | Time={}m{}s'.
                  format(epoch + 1, loss, int(t1 // 60), int(t1 % 60)))

model.eval()
corr_test = 0
total_test = 0
for step, (batch_x, batch_y) in enumerate(loader_test):
    loss, logit = model(batch_x, batch_y)
    labels = torch.max(batch_y, 1)[1]
    result = torch.max(logit, 1)[1].view(labels.size())
    corr_test += (result.data == labels.data).sum()
    total_test += batch_size
acc_test = corr_test * 100.0 / total_test
t1 = time.time() - t0
print('Test accuracy={:.3f}% | Time={}m{}s'.
      format(acc_test, int(t1//60), int(t1 % 60)))


    # model.eval()
    # corr = 0
    # total = 0
    # for step, (batch_x, batch_y) in enumerate(loader):
    #     loss, logit = model(batch_x, batch_y)
    #     labels = torch.max(batch_y, 1)[1]
    #     result = torch.max(logit, 1)[1].view(labels.size())
    #     corr += (result.data == labels.data).sum()
    #     total += batch_size
    # acc_train = corr * 100.0 / total
    # corr_test = 0
    # total_test = 0
    # for step, (batch_x, batch_y) in enumerate(loader_test):
    #     loss, logit = model(batch_x, batch_y)
    #     labels = torch.max(batch_y, 1)[1]
    #     result = torch.max(logit, 1)[1].view(labels.size())
    #     corr_test += (result.data == labels.data).sum()
    #     total_test += batch_size
    # acc_test = corr_test * 100.0 / total_test
    # t1 = time.time() - t0
    # print('Epoch[{}] | Train accuracy={:.3f}% | Test accuracy={:.3f}% | Time={}m{}s'.
    #       format(epoch+1, acc_train, acc_test, int(t1//60), int(t1 % 60)))

