from ultralytics import YOLO

model = YOLO("best.pt")

print(model.model.nc)