import numpy as np
import time
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
embed_size = 300
hidden_size, drop_out = 512, 0.5
lr, lr_decay, wei_decay = 0.01, 0, 0.0001
num_epochs = 30

# Train Data
train_x = np.load('./dataset/train.embed.npy')[:2000]
train_y = np.load('./dataset/train.score.npy')[:2000]
print('shape of train_x, train_y:', np.shape(train_x), np.shape(train_y))
train_dataset = Data.TensorDataset(torch.tensor(train_x).type('torch.FloatTensor'),
                                   torch.tensor(train_y).type('torch.FloatTensor'))
loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Validation Data
test_x = np.load('./dataset/train.embed.npy')[2000:]
test_y = np.load('./dataset/train.score.npy')[2000:]
print('shape of test_x, test_y:', np.shape(test_x), np.shape(test_y))
test_dataset = Data.TensorDataset(torch.tensor(test_x).type('torch.FloatTensor'),
                                  torch.tensor(test_y).type('torch.FloatTensor'))
loader_test = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Test Data
# test_x = np.load('./dataset/test.embed.npy')[:1000]
# test_y = np.load('./dataset/test.score.npy')[:1000]
# print('shape of test_x, test_y:', np.shape(test_x), np.shape(test_y))
# test_dataset = Data.TensorDataset(torch.tensor(test_x).type('torch.FloatTensor'),
#                                   torch.tensor(test_y).type('torch.FloatTensor'))
# loader_test = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(max_len * embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, label_size)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, glabels=None):
        inputs = inputs.view(inputs.size(0), -1)
        outputs = self.mlp(inputs)
        labels = F.softmax(outputs, dim=1)
        loss = self.loss_func(outputs, torch.argmax(glabels, dim=1))
        return loss, labels.detach()


model = MLP()
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wei_decay)
optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=wei_decay)
t1 = time.time() - t0
print('Begin training MLP. Time={}m {}s'.format(int(t1//60), int(t1 % 60)))

for epoch in range(num_epochs):
    model.train()
    for step, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        loss, logit = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
    model.eval()
    corr = 0
    total = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        loss, logit = model(batch_x, batch_y)
        labels = torch.max(batch_y, 1)[1]
        result = torch.max(logit, 1)[1].view(labels.size())
        corr += (result.data == labels.data).sum()
        total += batch_size
    acc_train = corr * 100.0 / total
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
    print('Epoch[{}] | Train accuracy={:.3f}% | Test accuracy={:.3f}% | Time={}m {}s'.
          format(epoch+1, acc_train, acc_test, int(t1//60), int(t1 % 60)))

