import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from model import binaryImageClassification


data_path = "data\PetImages"
batch_size = 32

# check if cuda is available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# All data transforms from this, imageSize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#create the ImageFolder dataset
train_dataset = datasets.ImageFolder(root = data_path, transform = transform)

#create the data loader
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = True)

#import model and move to device
model = binaryImageClassification()
model.to(device)





classes = ('cat', 'dog')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
