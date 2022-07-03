import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import io

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Setting Hyper-parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Loading the dataset


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
model.load_state_dict(torch.load("mnist.pth"))
model.eval()

# Tranform image to normalized 28x28 grayscale 
def tranform_img(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(img):
    pred = model(img)
    return pred.argmax(1)