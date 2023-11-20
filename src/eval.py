import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import BinaryImageClassification
import matplotlib.pyplot as plt
import numpy as np

test_dataset_path = "data\Tests"
batch_size = 4
image_size = 224

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

eval_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

class_mapping = {0: 'cat', 1: 'dog'}

model = BinaryImageClassification()
model.load_state_dict(torch.load('data\model'))
model.eval()

test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Append true and predicted labels
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Display the last 4 images on the same plot
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

for i, ax in enumerate(axes):
    true_class_name = class_mapping[true_labels[-4 + i].item()]
    predicted_class_name = class_mapping[predicted_labels[-4 + i].item()]

    img = np.transpose(inputs[-4 + i].numpy(), (1, 2, 0))

    ax.imshow(img)
    ax.set_title(f'True: {true_class_name}\nPredicted: {predicted_class_name}')
    ax.axis('off')

plt.show()

# Calculate overall accuracy
accuracy = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred]) / len(true_labels)
print(f'Test Accuracy: {accuracy}')
