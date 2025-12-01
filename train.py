from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

DATA_DIR = Path("data")
IMAGE_SIZE = 224
BATCH_SIZE = 8
LEARNING_RATE = 0.0003
EPOCHS = 30

# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# DATASETS
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

print("Train classes:", train_ds.classes)
print("Number of train images:", len(train_ds))
print("Number of validation images:", len(val_ds))

# DATALOADERS
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Number of batches in train loader:", len(train_dl))
print("Number of batches in validation loader:", len(val_dl))

# PRINT BATCH
for images, labels in train_dl:
    print("Images shape:", images.shape)
    print("Labels:", labels)
    break

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MODEL
num_classes = len(train_ds.classes)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# partial unfreeze - freeze everything, then unfreeze layer4 and fc
for name, param in model.named_parameters():
    param.requires_grad = False
    if name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

# LOSS FUNCTION
criterion = nn.CrossEntropyLoss()

# OPTIMIZER
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=0.0001)

# TRAIN + VALIDATE
best_val_acc = 0
best_state = None

for epoch in range(EPOCHS):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)

        train_loss += loss.item() * images.size(0)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total
        
    # VALIDATION
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)

            val_loss += loss.item() * images.size(0)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}")

# SAVE MODEL
if best_state is not None:
    model.load_state_dict(best_state)

torch.save(
    {
        "state_dict": model.state_dict(),
        "classes": train_ds.classes,
    },
    "sport_classifier_best.pth",
)
print(f"Saved model to sport_classifier_best.pth (best validation accuracy: {best_val_acc:.4f})")