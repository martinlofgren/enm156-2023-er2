from ultralytics import YOLO, checks
import multiprocessing

def train():
    print(checks())
    model = YOLO("yolov8x.pt")
    model.train(data = "bees.yaml", epochs = 1, imgsz = 640)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()