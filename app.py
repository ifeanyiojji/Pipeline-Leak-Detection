import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load model (change to .onnx if you exported)
MODEL_PATH = "best.pt"   # or "best.onnx"
model = YOLO(MODEL_PATH)

# Load class names from data.yaml
class_names = ['no leak', 'leak', 'crack', 'water', 'other']  # replace with your YAML classes

st.title("Pipeline Leak Detection App")
st.write("Upload an image to detect pipeline leaks with bounding boxes.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def draw_boxes(image, results):
    """
    Draw bounding boxes and labels on the image using OpenCV.
    """
    for box in results[0].boxes:
        # Extract box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # Label text
        label = f"{class_names[cls_id]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - th - 5), (x1 + tw, y1), (255, 255, 255), -1)
        # Put label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    return image

if uploaded_file is not None:
    # Read uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # Run inference
    results = model(img_array, conf=0.1)

    # Draw boxes
    annotated_img = draw_boxes(img_array.copy(), results)

    # Display result
    st.image(annotated_img, caption="Detection Results", use_container_width=True)

    # Show detected labels separately
    detected_labels = [class_names[int(box.cls[0])] for box in results[0].boxes]
    if detected_labels:
        st.write("### Detected Objects:")
        st.write(", ".join(detected_labels))
    else:
        st.write("⚠️ No objects detected. Try lowering confidence threshold.")
