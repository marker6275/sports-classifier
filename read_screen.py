import time
import mss
from PIL import Image
import torch

from predict_utils import load_model, predict_image

INTERVAL_SECONDS = 1

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        return img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(device)

    print(f"Using device: {device}")
    print(f"Classes: {classes}")

    try:
        while True:
            start = time.time()

            img = capture_screen()
            
            label, confidence = predict_image(img, model, classes, device)

            print(f"Prediction: {label}, Confidence: {confidence:.2f}")

            elapsed = time.time() - start
            time.sleep(max(0, INTERVAL_SECONDS - elapsed))
    except KeyboardInterrupt:
        print("\nStopping...")
        return

if __name__ == "__main__":
    main()