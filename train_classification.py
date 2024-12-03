from ultralytics import YOLO
import os

# write classificaiton model by yolo using images in yolo/not_labeled_objects/classes
model = YOLO("yolov8n-cls.pt")
results = model.train(data="cifar10", epochs=100, imgsz=32)