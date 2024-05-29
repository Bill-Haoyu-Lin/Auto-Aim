from ultralytics import YOLO
model = YOLO('best.pt')  # load an official model  # load a custom trained model

# Export the model
model.export(format='engine')  