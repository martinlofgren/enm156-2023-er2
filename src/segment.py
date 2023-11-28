import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the image
image_path = "C:/Users/nilsa/Pictures/ID_2_2021-05-24_13-43-29.jpg"
image = Image.open(image_path).convert('RGB')

# Define the transformation
transform = T.Compose([T.ToTensor()])

# Preprocess the image
input_image = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    prediction = model(input_image)

# Display the results
image = Image.open(image_path).convert('RGB')
draw = ImageDraw.Draw(image)

# Set a threshold for confidence to filter out weak detections
threshold = 0.4

# Iterate through the predictions
for score, label, box in zip(prediction[0]['scores'], prediction[0]['labels'], prediction[0]['boxes']):
    if score > threshold:
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"Class {label.item()}, {round(score.item(), 3)}", fill="red")

# Display the results
image.show()
