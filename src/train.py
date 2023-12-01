from ultralytics import YOLO
import multiprocessing

def train():
    model = YOLO("yolov8n.pt")
    results = model.train(data = "bees.yaml", epochs = 1, imgsz = 640)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()