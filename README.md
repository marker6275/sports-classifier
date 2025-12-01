# Sports Identifier

A deep learning-based sports classification system that can identify sports (basketball, football) and commercials from images. The whole point of this project was so I could tell when a game was on commercial or live.

## Project Structure

```
sports-identifier/
├── data/
│   ├── train/          # Training images organized by class
│   │   ├── basketball/
│   │   ├── football/
│   │   └── commercial/
│   └── val/            # Validation images organized by class
│       ├── basketball/
│       ├── football/
│       └── commercial/
├── testing/            # Test images for evaluation
├── train.py            # Model training script
├── test.py             # Batch testing script
├── predict_image.py    # Single image prediction script
├── read_camera.py      # Real-time camera prediction
├── read_screen.py      # Screen capture prediction
├── predict_utils.py    # Utility functions for predictions
├── camera_test.py      # Camera test utility
└── sport_classifier_best.pth  # Trained model checkpoint
```

## Usage

### Training

Train the model on your dataset:

```bash
python train.py
```

The script will:

- Load images from `data/train/` and `data/val/`
- Train a ResNet18 model with transfer learning
- Save the best model to `sport_classifier_best.pth`

**Training Configuration:**

- Image size: 224x224
- Batch size: 8
- Learning rate: 0.0003
- Epochs: 30
- Optimizer: Adam with weight decay
- Data augmentation: Random resized crop, horizontal flip, color jitter

### Testing

Test the model on images in the `testing/` directory:

```bash
python test.py
```

The script expects test images to be named with the format `{class}_{number}.png` (e.g., `basketball_1.png`, `football_2.png`) and will report accuracy metrics.

### Single Image Prediction

Predict a single image:

```bash
python predict_image.py <path_to_image>
```

### Real-time Camera Prediction

Run live prediction from your webcam:

```bash
python read_camera.py
```

- Press 'q' to quit
- Predictions are made every second
- The script automatically detects available cameras

### Screen Capture Prediction

Classify content from screen captures:

```bash
python read_screen.py
```

- Press Ctrl+C to stop
- Captures the primary monitor every second
- Useful for classifying sports content displayed on screen

## Model Architecture

The model uses:

- **Base**: ResNet18 pre-trained on ImageNet
- **Transfer Learning**: Only the last layer (layer4) and fully connected layer are fine-tuned
- **Output**: 3 classes (basketball, football, commercial)
- **Input**: 224x224 RGB images

## Model Files

- `sport_classifier_best.pth`: Best model checkpoint (saved during training)
- `sport_classifier.pth`: Alternative model checkpoint

Both files contain:

- `state_dict`: Model weights
- `classes`: List of class names
