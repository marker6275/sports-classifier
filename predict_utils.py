from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models

IMAGE_SIZE = 224
MODEL_PATH = Path("sport_classifier_best.pth")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
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

def predict_image_all(img, model, classes, device):
    """
    Predict image and return all class predictions with their confidences.
    
    Returns:
        tuple: (top_label, top_confidence, all_predictions)
        where all_predictions is a list of (class_name, confidence) tuples sorted by confidence
    """
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Get all predictions sorted by confidence
    all_probs = probabilities.cpu().numpy()
    sorted_indices = all_probs.argsort()[::-1]  # Sort descending
    
    all_predictions = [(classes[i], float(all_probs[i])) for i in sorted_indices]
    
    # Top prediction
    top_index = probabilities.argmax().item()
    top_label = classes[top_index]
    top_confidence = probabilities[top_index].item()

    return top_label, top_confidence, all_predictions