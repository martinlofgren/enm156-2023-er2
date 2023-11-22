import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import BinaryImageClassification
from tqdm import tqdm

test_dataset_path = "data\Tests"
batch_size = 64
image_size = 224

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

eval_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

class_mapping = {0: 'cat', 1: 'dog'}

model = BinaryImageClassification()
model.load_state_dict(torch.load('data\model.pth'))
model.eval()

test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []
with tqdm(total=len(test_loader), desc="Processing", unit="batch") as progress_bar:
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Append true and predicted labels
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())
            progress_bar.update(1)

# Calculate overall accuracy
accuracy = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred]) / len(true_labels)
print(f'Test Accuracy: {accuracy}')
