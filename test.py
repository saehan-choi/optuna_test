import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import os

layers = []
in_features, out_features = 784, 118
layers.append(nn.Linear(in_features, out_features))
layers.append(nn.ReLU())
layers.append(nn.Dropout(0.29682944214431034))
layers.append(nn.Linear(out_features, 10))
layers.append(nn.LogSoftmax(dim=1))


DEVICE = torch.device("cuda")
BATCHSIZE = 512
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 100
LOG_INTERVAL = 10
# N_TRAIN_EXAMPLES = BATCHSIZE * 30
# N_VALID_EXAMPLES = BATCHSIZE * 10

def get_mnist():
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader

model = nn.Sequential(*layers).cuda()

# Get the FashionMNIST dataset.
train_loader, valid_loader = get_mnist()
optimizer = optim.RMSprop(params=model.parameters(), lr=0.005732091924927557)
print(model)
# Training of the model.
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Limiting training data for faster epochs.

        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            # Limiting validation data.
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            output = model(data)
            # Get the index of the max log-probability.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print(valid_loader.dataset)
    accuracy = correct / len(valid_loader.dataset)
    
    print(f'acc:{accuracy}')
