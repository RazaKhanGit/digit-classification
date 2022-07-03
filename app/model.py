import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Setting Hyper-parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Loading the dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))]) # Mean=0.1307, SD=0.3081 for MNIST

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Setting up model
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28*28, 128) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 10)  
    
    def forward(self, x):
        x = self.flatten(x)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NNet()

# Optimizers & Loss Function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Forwardprop
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if batch %100 == 0:
        loss, current = loss.item(), batch*len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
      for X, y in dataloader:
          pred = model(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Running epochs
for t in range(num_epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_loader, model, loss_fn, optimizer)
  test(test_loader, model, loss_fn)

torch.save(model.state_dict(), "app/mnist.pth")