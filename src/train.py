from ultralytics import YOLO
import multiprocessing
import shutil

def train():
    model = YOLO("yolov8n.pt")
    results = model.train(data = "bees.yaml", epochs = 25, imgsz = 640)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()