from ultralytics import YOLO

model = YOLO("models/PEPSI_yolov8n_960_14926/weights/best.pt")
model.export(format='coreml', nms=True, name='pepsi',  )
