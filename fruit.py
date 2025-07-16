#pip install torch torchvision ultralytics pyttsx3 opencv-python pillow

import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import os
import pyttsx3  # Text to speech for audio feedback

# Define a generic classifier model architecture
class FruitClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FruitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load classifier models for each fruit
def load_model(model_path):
    model = FruitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Paths to classifier models
model_paths = { 
    "apple": r"C:\Users\Admin\Desktop\epics\fruit\Scripts\apple_classifier.pth",
    "banana": r"C:\Users\Admin\Desktop\epics\fruit\Scripts\banana_classifier.pth",
    "orange": r"C:\Users\Admin\Desktop\epics\fruit\Scripts\orange_classifier.pth"
}


# Check if model paths exist
for fruit, path in model_paths.items():
    if not os.path.exists(path):
        print(f"Model file for {fruit} not found at path: {path}")
    else:
        print(f"Model file for {fruit} found at path: {path}")

# Labels for the classifiers
labels = ["fresh", "rotten"]

# Initialize the device
device = torch.device("cpu")
print(f"Using device: {device}")

# Load each fruit-specific model
models = {
    fruit: load_model(path) for fruit, path in model_paths.items()
}
yolo_model_path = r"C:\Users\Admin\Desktop\epics\fruit\Scripts\yolov8n.pt"
# Initialize YOLOv8 model for object detection
yolo_model = YOLO(yolo_model_path)  # Pre-trained on COCO dataset

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define COCO class IDs for the fruits
coco_class_ids = {
    47: 'apple',
    46: 'banana',
    49: 'orange',
}

# Initialize pyttsx3 for audio output
engine = pyttsx3.init()

# Function to speak the output text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize the camera
cap = cv2.VideoCapture(0)  # Camera 0 is usually the default camera on Raspberry Pi

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 detection on the frame
    results = yolo_model(frame)

    # Process detection results for each image
    for result in results:
        for det in result.boxes:
            class_id = int(det.cls[0])  # Class ID from YOLO detection

            # Check if the detected class ID is one of the fruits
            if class_id in coco_class_ids:
                fruit_name = coco_class_ids[class_id]

                x1, y1, x2, y2 = map(int, det.xyxy[0])  # Bounding box coordinates
                # Crop the detected fruit
                fruit_crop = frame[y1:y2, x1:x2]

                # Convert crop to PIL Image and transform
                fruit_image = Image.fromarray(cv2.cvtColor(fruit_crop, cv2.COLOR_BGR2RGB))
                fruit_transformed = transform(fruit_image).unsqueeze(0).to(device)

                # Select the appropriate classifier model
                model = models[fruit_name]

                # Predict using the fruit-specific classifier
                with torch.no_grad():
                    outputs = model(fruit_transformed)
                    _, predicted = torch.max(outputs, 1)

                # Get classification result
                classification_label = labels[predicted.item()]
                result_text = f"{classification_label} {fruit_name}"

                # Print the result
                print(result_text)

                # Speak the result
                speak(result_text)

    # Display the resulting frame (optional for debugging)
    cv2.imshow('Fruit Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()

