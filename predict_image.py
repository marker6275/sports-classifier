import sys
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

IMAGE_SIZE = 224
MODEL_PATH = Path("sport_classifier.pth")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    classes = checkpoint["classes"]

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    in_features = model.fc.in_features
    
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, classes

def predict_image(image_path):
    if not image_path.exists():
        print(f"Error: Image file {image_path} does not exist.")
        return None

    model, classes = load_model()

    img = Image.open(image_path).convert("RGB")

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)[0]

    pred_index = probabilities.argmax().item()
    pred_class = classes[pred_index]
    confidence = probabilities[pred_index].item()

    print(f"Image: {image_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    return pred_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    predict_image(image_path)