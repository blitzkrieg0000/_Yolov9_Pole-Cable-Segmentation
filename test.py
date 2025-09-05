from ultralytics import YOLO

model = YOLO("yolov9c-seg.yaml", task="segment")