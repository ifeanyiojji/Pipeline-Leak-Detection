from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Export to ONNX format
model.export(format="onnx")  # This will create 'best.onnx'
