from pathlib import Path

import cv2
import time
from PIL import Image, ImageTk
import tkinter as tk

import torch
from torch import nn
from torchvision import models, transforms

from predict_utils import load_model, predict_image

MODEL_PATH = Path("sport_classifier.pth")
IMAGE_SIZE = 224
INTERVAL_SECONDS = 1

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def list_cameras():
    """List all available cameras."""
    available_cameras = []
    print("Checking for available cameras...")
    
    # Check cameras 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if available_cameras:
        print(f"Available cameras: {available_cameras}")
    else:
        print("No cameras found.")
    
    return available_cameras

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, classes = load_model(device)

    # List available cameras
    cameras = list_cameras()
    
    # Use first available camera, or default to 0
    camera_index = cameras[0] if cameras else 0
    print(f"Using camera: {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    last_pred_time = 0

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break

            # frame = cv2.flip(frame, 1)

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            now = time.time()

            elapsed = now - last_pred_time
            if elapsed >= INTERVAL_SECONDS:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame_rgb)
                
                label, confidence = predict_image(img, model, classes, device)

                print(f"Prediction: {label}, Confidence: {confidence:.2f}")
                last_pred_time = now


    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    main()