from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)


#check config.yaml file
model.train(data='config.yaml', epochs=1, batch=16, imgsz=740,device=0,workers=0)


