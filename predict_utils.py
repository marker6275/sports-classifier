import time
from PIL import Image
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models

IMAGE_SIZE = 224
MODEL_PATH = Path("sport_classifier.pth")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, classes

def predict_image(img, model, classes, device):
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)[0]

    index = probabilities.argmax().item()

    return classes[index], probabilities[index].item()

def predict_image_from_file(file_path, model, classes, device):
    img = Image.open(file_path).convert("RGB")
    return predict_image(img, model, classes, device)
