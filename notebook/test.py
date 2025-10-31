from ultralytics import YOLO

model=YOLO('../runs/detect/train16/weights/best.pt')
results=model(source=0,show=True,conf=0.6,save=True)