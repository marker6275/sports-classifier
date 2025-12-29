# Sports Identifier

A deep learning-based sports classification system that can identify sports (basketball, football) and commercials from images. The whole point of this project was so I could tell when a game was on commercial or live.

## Project Structure

```
sports-identifier
├── display/           # Electron display app for showing classification status
│   ├── electron/      # Electron main process
│   ├── src/           # React frontend
│   └── package.json   # Node dependencies
├── testing/           # Test images for evaluation
├── train.py           # Model training script
├── test.py            # Batch testing script
├── predict_image.py   # Single image prediction script
├── read_camera.py     # Real-time camera prediction (main function)
├── read_screen.py     # Screen capture prediction
├── predict_utils.py   # Utility functions for predictions
├── label_images.py    # Interactive image labeling tool (saves directly to train/val)
├── get_images.py      # Image collection utility
├── utils.py           # General utility functions
├── websocket.py       # WebSocket server for real-time status updates
├── sport_classifier_best.pth  # Best trained model
├── sport_classifier.pth       # Alternative model
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
- **Screen Detection**: Uses YOLO + basic edge detection to find and track TV screens
- **Bounding Box Updates**: Detects screen position every second, updates bounding box every 5 seconds
- **Screenshot Capture**: By default, screenshots are automatically saved to `screenshots/` every second for later labeling
- **WebSocket Server**: Sends real-time classification status to display app (port 8765)

### Data Collection and Labeling

The system includes tools for continuous data collection and manual validation:

1. **Capture Screenshots** (automatic):

   - `read_camera.py` automatically saves screenshots to `screenshots/` every second
   - Set `CAPTURE_SCREENSHOTS = False` in `read_camera.py` to disable

2. **Label Images**:
   ```bash
   python label_images.py
   ```
   - Interactive GUI for reviewing and labeling captured screenshots from `screenshots/`
   - **Automatic Train/Val Split**: Labeled images are automatically moved to `data/train/{class}/` or `data/val/{class}/` with a 65/35 random split
   - Images are removed from `screenshots/` after labeling
   - Ready for training with `train.py`

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
