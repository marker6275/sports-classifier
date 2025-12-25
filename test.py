import sys
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

IMAGE_SIZE = 224
MODEL_PATH = Path("sport_classifier_best.pth")
TESTING_DIR = Path("testing")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    classes = checkpoint["classes"]

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, classes

def predict_image(image_path, model, classes):
    if not image_path.exists():
        print(f"Error: Image file {image_path} does not exist.")
        return None

    img = Image.open(image_path).convert("RGB")

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)[0]

    pred_index = probabilities.argmax().item()
    pred_class = classes[pred_index]
    confidence = probabilities[pred_index].item()

    print(f"Image: {image_path.name}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    return pred_class, confidence

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"Error: Model file {MODEL_PATH} does not exist.")
        sys.exit(1)

    if not TESTING_DIR.exists():
        print(f"Error: Testing directory {TESTING_DIR} does not exist.")
        sys.exit(1)

    # Load model once
    print("Loading model...")
    model, classes = load_model()
    print(f"Model loaded. Classes: {classes}")
    print("=" * 50)

    # Get all image files in testing directory
    image_files = sorted(TESTING_DIR.glob("*.png"))
    
    if not image_files:
        print(f"No PNG files found in {TESTING_DIR}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to test\n")

    correct_amt = 0
    # Test each image
    for image_path in image_files:
        prediction, confidence = predict_image(image_path, model, classes)
        answer = image_path.stem.split("_")[0]
        is_correct = prediction == answer
        if is_correct:
            correct_amt += 1
        print("Correct: ", is_correct)
        print("-" * 50)

    print(f"Correct amount: {correct_amt}/{len(image_files)}")
    print(f"Accuracy: {(correct_amt / len(image_files)) * 100:.1f}%")