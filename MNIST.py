# dataloading imports
import torch 
import torchvision
import torch.utils.data  # wtf does this thing do ? 
import torchvision.transforms as transforms

# modeling imports
import torch.nn as nn
import torch.functional as F

"""1. DOWNLOADING THE DATASETS"""
# define the pipeline-transformer futher usend in dataloader
transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))])

# This thing loads the data into a local folder and applies the transforamtions
train_set = torchvision.datasets.MNIST('/Users/babyhandzzz/Desktop/ELEPH@NT/Datasets/Mnist_Pytorch',train=True,
transform=transformations,download=True)

test_set = torchvision.datasets.MNIST('/Users/babyhandzzz/Desktop/ELEPH@NT/Datasets/Mnist_Pytorch', train=False, 
transform=transformations, download=True)

"""2. DATALOADERS (BATCH UP THE DATASET TO FEED IT TO THE MODEL)"""
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=1, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=1, batch_size=32)

"""3. DEFINING THE MODEL"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28*1,200) # we are not sure if we can put ReLU here
        self.fc2 = nn.Linear(200,10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

"""4. DECALRING THE MODEL AND DEFINING LOSS AND OPTIMIZER"""

model = Net()
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) #derived from torch library itslef
criterion = nn.CrossEntropyLoss()
"""5. TRAINING THE MODEL"""

for epoch in range(10):
    for batch_idx, data_target in enumerate(train_loader):
        
        data = data_target[0]
        target = data_target[1]


        data = data.view(-1, 28 * 28)
        optimizer.zero_grad()
        
        output = model(data)

        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
