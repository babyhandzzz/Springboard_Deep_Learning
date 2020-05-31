import torch.nn as nn
import torch.nn.functional as F
import torch
from sequence_data_loader import *
import torch.optim as optim
print("PyTorch Version: ",torch.__version__)

path = '/Users/babyhandzzz/Desktop/ELEPH@NT/Datasets/IMDB Dataset.csv'
dataset = Sequences(path)
print('Dataset is {} entries long'.format(len(dataset)))
print('Loading dataset')
train_loader = DataLoader(dataset, batch_size=64)

# DEFINE FORWARD PROPAGATION

class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, hidden1, hidden2):
        super(BagOfWordsClassifier, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, inputs):
        # squeeze - Remove single-dimensional entries from the shape of an array.
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = BagOfWordsClassifier(len(dataset.token2idx),128,64)
print('Model parameters:')
print(model)

# DEFINE BACK PROPAGATION

criterion = nn.BCEWithLogitsLoss()
# what's up with the model parameters variable
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

model.train()
train_losses = []

for epoch in range(10):
    for inputs, target in enumerate(train_loader):
        print('Epoch: {}'.format(epoch))
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), target.float())
        loss.backward()
        print('Loss: {}'.format(loss))
        optimizer.step()
        train_losses.append(loss.item())
