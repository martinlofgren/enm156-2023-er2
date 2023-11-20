import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from model import BinaryImageClassification
from tqdm import tqdm
import PIL
from PIL import Image
import os

#training loop
num_epochs = 10
data_path = "data/PetImages"
batch_size = 128
image_size = 224
folder_paths = [
    "data/PetImages/Cat",
    "data/PetImages/Dog"
]

#removes files that are not readable, remove last 2 lines to remove files manually instead
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        try:
            image = Image.open(os.path.join(folder_path, filename))
        except PIL.UnidentifiedImageError as e:
            print(f"Error in file {filename}: {e}")
            os.remove(os.path.join(folder_path, filename))
            print(f"Removed file {filename}")

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
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

#create the ImageFolder dataset
train_dataset = datasets.ImageFolder(root = data_path, transform = transform)

#create the data loader
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = True)

#import model and move to device
model = BinaryImageClassification()
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



for epoch in range(num_epochs):
    model.train()

    # Use tqdm to create a progress bar for the training loader
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
        for inputs, labels in t:
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # Update the progress bar with the current loss
                t.set_postfix(loss=loss.item())
            except Exception as e:
                print(f"Error: {e}")

torch.save(model.state_dict(), "data/model")