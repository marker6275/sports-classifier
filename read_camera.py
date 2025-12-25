from pathlib import Path

import cv2
import time
from PIL import Image

import torch
from torchvision import transforms
import numpy as np
from ultralytics import YOLO

from predict_utils import load_model, predict_image_all
from websocket import WebSocketServer

IMAGE_SIZE = 224
INTERVAL_SECONDS = 1.0

transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def refine_box_to_aspect_ratio(x1, y1, x2, y2, target_aspect_ratio, frame_width, frame_height):

    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    current_aspect = width / height if height > 0 else 1.0

    if current_aspect > target_aspect_ratio:

        new_height = width / target_aspect_ratio
        new_width = width
    else:

        new_width = height * target_aspect_ratio
        new_height = height

    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(frame_width, new_x2)
    new_y2 = min(frame_height, new_y2)

    return new_x1, new_y1, new_x2, new_y2

def refine_yolo_detection_with_brightness(frame, yolo_box, target_aspect_ratio,
                                         brightness_threshold_percentile=50, step_size=30, expand_ratio=0.2):

    if yolo_box is None:
        return None

    h, w = frame.shape[:2]
    yx1, yy1, yx2, yy2, yconf = yolo_box

    box_width = yx2 - yx1
    box_height = yy2 - yy1
    expand_x = int(box_width * expand_ratio)
    expand_y = int(box_height * expand_ratio)

    search_x1 = max(0, yx1 - expand_x)
    search_y1 = max(0, yy1 - expand_y)
    search_x2 = min(w, yx2 + expand_x)
    search_y2 = min(h, yy2 + expand_y)

    search_region = frame[search_y1:search_y2, search_x1:search_x2]
    if search_region.size == 0:
        return None

    try:
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    except:
        gray = search_region if len(search_region.shape) == 2 else cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

    brightness_threshold = np.percentile(gray, brightness_threshold_percentile)

    search_h, search_w = gray.shape[:2]
    search_area = search_h * search_w

    min_width = max(10, int(box_width * 0.5))
    max_width = min(search_w, int(search_h * target_aspect_ratio))

    if max_width < min_width:
        return None

    best_box = None
    best_score = 0

    sizes_to_try = []
    current_width = max_width
    iterations = 0
    max_iterations = 6
    while current_width >= min_width and iterations < max_iterations:
        width = int(current_width)
        height = int(width / target_aspect_ratio)
        if height <= search_h and width <= search_w and height > 0 and width > 0:
            sizes_to_try.append((width, height))
        current_width *= 0.85
        iterations += 1

    sizes_to_try.sort(key=lambda s: s[0] * s[1], reverse=True)

    for width, height in sizes_to_try[:3]:
        max_y = max(1, search_h - height + 1)
        max_x = max(1, search_w - width + 1)

        for y in range(0, max_y, step_size):
            for x in range(0, max_x, step_size):
                try:
                    region = gray[y:y+height, x:x+width]
                    if region.size == 0:
                        continue
                except:
                    continue

                mean_brightness = np.mean(region)
                bright_pixel_ratio = np.sum(region > brightness_threshold) / region.size

                area = width * height
                area_normalized = min(area / search_area, 1.0)
                size_weight = area_normalized ** 2
                brightness_score = (mean_brightness / 255.0) * (1 + bright_pixel_ratio)
                score = brightness_score * (0.3 + 0.7 * size_weight) * area

                if score > best_score:
                    best_score = score

                    frame_x1 = search_x1 + x
                    frame_y1 = search_y1 + y
                    frame_x2 = search_x1 + x + width
                    frame_y2 = search_y1 + y + height

                    combined_conf = (yconf * 0.6 + brightness_score * 0.4)
                    best_box = (frame_x1, frame_y1, frame_x2, frame_y2, combined_conf)

    return best_box

def find_brightest_16_9_region(frame, target_aspect_ratio, min_size_ratio=0.1, max_size_ratio=0.85,
                                brightness_threshold_percentile=50, step_size=20):

    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    if h < 10 or w < 10:
        return None

    frame_area = h * w

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:

        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness_threshold = np.percentile(gray, brightness_threshold_percentile)

    min_area = frame_area * min_size_ratio * min_size_ratio
    max_area = frame_area * max_size_ratio * max_size_ratio

    best_box = None
    best_score = 0

    max_width = min(w, int(h * target_aspect_ratio))

    min_width = max(10, int(np.sqrt(min_area * target_aspect_ratio)))

    if max_width < min_width:

        return None

    sizes_to_try = []
    current_width = max_width
    iterations = 0
    max_iterations = 12
    while current_width >= min_width and iterations < max_iterations:
        width = int(current_width)
        height = int(width / target_aspect_ratio)
        if height <= h and width <= w and height > 0 and width > 0:
            sizes_to_try.append((width, height))
        current_width *= 0.90
        iterations += 1

    sizes_to_try.sort(key=lambda s: s[0] * s[1], reverse=True)

    if not sizes_to_try:
        return None

    for width, height in sizes_to_try:
        area = width * height
        if area < min_area or area > max_area:
            continue

        max_y = max(1, h - height + 1)
        max_x = max(1, w - width + 1)

        for y in range(0, max_y, step_size):
            for x in range(0, max_x, step_size):

                try:
                    region = gray[y:y+height, x:x+width]
                    if region.size == 0:
                        continue
                except:
                    continue

                mean_brightness = np.mean(region)
                bright_pixel_ratio = np.sum(region > brightness_threshold) / region.size

                area_normalized = min(area / max_area, 1.0)

                size_weight = area_normalized ** 2

                brightness_score = (mean_brightness / 255.0) * (1 + bright_pixel_ratio)

                score = brightness_score * (0.3 + 0.7 * size_weight) * area

                if score > best_score:
                    best_score = score
                    best_box = (x, y, x + width, y + height, min(brightness_score, 1.0))

    return best_box

def list_cameras():

    available_cameras = []
    print("Checking for available cameras...")

    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():

                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
            continue

    if available_cameras:
        print(f"Available cameras: {available_cameras}")
    else:
        print("No cameras found.")
        print("Troubleshooting tips:")
        print("  - Make sure your camera is connected")
        print("  - Check if another application is using the camera")
        print("  - On macOS, grant camera permissions in System Settings > Privacy & Security")

    return available_cameras

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, classes = load_model(device)

    print("Loading YOLO model for screen detection...")
    yolo_model = YOLO("yolov8n.pt")
    print("YOLO model loaded.")

    KNOWN_SCREEN_WIDTH = 1920
    KNOWN_SCREEN_HEIGHT = 1080
    KNOWN_ASPECT_RATIO = KNOWN_SCREEN_WIDTH / KNOWN_SCREEN_HEIGHT if KNOWN_SCREEN_HEIGHT > 0 else None

    YOLO_CONFIDENCE_THRESHOLD = 0.3
    BRIGHTNESS_THRESHOLD_PERCENTILE = 50
    SCAN_STEP_SIZE = 60
    BRIGHTNESS_REFINE_STEP_SIZE = 50
    DARKEN_FACTOR = 0.2
    DETECTION_SKIP_FRAMES = 10
    USE_YOLO_FIRST = True

    print(f"Using hybrid detection (YOLO + brightness) for {KNOWN_SCREEN_WIDTH}x{KNOWN_SCREEN_HEIGHT} screen (aspect ratio: {KNOWN_ASPECT_RATIO:.3f})")
    print(f"YOLO confidence threshold: {YOLO_CONFIDENCE_THRESHOLD}")
    print(f"Brightness scan step size: {SCAN_STEP_SIZE} pixels")
    print(f"Darkening non-screen regions to {DARKEN_FACTOR*100:.0f}% brightness")

    MIN_SCREEN_RATIO = 0.3
    MAX_SCREEN_RATIO = 0.95

    UPDATE_EVERY_N_FRAMES = 3
    AVERAGE_WEIGHT_NEW = 0.3
    AVERAGE_WEIGHT_OLD = 0.7
    MAX_DETECTION_DISTANCE = 0.3

    DETECTION_HISTORY_SIZE = 10
    CONFIDENCE_WEIGHT_POWER = 2.0
    RECENCY_WEIGHT_DECAY = 0.9

    CONFIDENCE_TRACKING_ENABLED = True
    CONFIDENCE_EMA_ALPHA = 0.3
    CONFIDENCE_THRESHOLD = 0.5

    MIN_CONSECUTIVE_FRAMES = {
        "commercial": 6,
        "basketball": 3,
        "football": 3
    }
    DEFAULT_MIN_CONSECUTIVE_FRAMES = 4

    WEBSOCKET_ENABLED = True
    WEBSOCKET_HOST = 'localhost'
    WEBSOCKET_PORT = 8765

    ws_server = None
    if WEBSOCKET_ENABLED:
        ws_server = WebSocketServer(host=WEBSOCKET_HOST, port=WEBSOCKET_PORT)
        ws_server.start()

    cameras = list_cameras()

    if not cameras:
        print("Error: No cameras found. Please check your camera connection.")
        return

    camera_index = cameras[0]

    print(f"Using camera: {camera_index}")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Trying AVFoundation backend for macOS...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        print("Available cameras were:", cameras)
        print("Please check:")
        print("  1. Camera is connected and not being used by another application")
        print("  2. Camera permissions are granted in System Settings")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Camera opened successfully")
    print(f"Target FPS: 30")
    print(f"Actual FPS: {actual_fps:.1f}")
    print(f"Prediction interval: {INTERVAL_SECONDS}s ({1/INTERVAL_SECONDS:.1f} predictions/sec)")
    print(f"Detection runs every {DETECTION_SKIP_FRAMES} frame(s) (~{30/DETECTION_SKIP_FRAMES:.1f} detections/sec at 30 FPS)")

    last_pred_time = 0

    smoothed_box = None
    frame_counter = 0
    detection_frame_counter = 0

    detection_history = []

    class_confidence_tracker = {}
    current_tracked_label = None
    consecutive_high_confidence = {}
    total_frames_per_label = {}

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame")
                break

            h, w = frame.shape[:2]
            frame_area = h * w

            best_screen = None
            detection_frame_counter += 1
            if detection_frame_counter >= DETECTION_SKIP_FRAMES:
                detection_frame_counter = 0

                if USE_YOLO_FIRST:

                    yolo_results = yolo_model(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

                    best_yolo_box = None
                    best_yolo_score = 0

                    for result in yolo_results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls == 62 or cls == 72:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                conf = float(box.conf[0])

                                area = (x2 - x1) * (y2 - y1)
                                score = area * conf

                                if score > best_yolo_score:
                                    best_yolo_score = score
                                    best_yolo_box = (x1, y1, x2, y2, conf)

                    if best_yolo_box:
                        refined_box = refine_yolo_detection_with_brightness(
                            frame,
                            best_yolo_box,
                            KNOWN_ASPECT_RATIO,
                            brightness_threshold_percentile=BRIGHTNESS_THRESHOLD_PERCENTILE,
                            step_size=BRIGHTNESS_REFINE_STEP_SIZE
                        )
                        best_screen = refined_box if refined_box else best_yolo_box

                    else:

                        best_screen = find_brightest_16_9_region(
                            frame,
                            KNOWN_ASPECT_RATIO,
                            min_size_ratio=MIN_SCREEN_RATIO,
                            max_size_ratio=MAX_SCREEN_RATIO,
                            brightness_threshold_percentile=BRIGHTNESS_THRESHOLD_PERCENTILE,
                            step_size=SCAN_STEP_SIZE
                        )
                else:

                    best_screen = find_brightest_16_9_region(
                        frame,
                        KNOWN_ASPECT_RATIO,
                        min_size_ratio=MIN_SCREEN_RATIO,
                        max_size_ratio=MAX_SCREEN_RATIO,
                        brightness_threshold_percentile=BRIGHTNESS_THRESHOLD_PERCENTILE,
                        step_size=SCAN_STEP_SIZE
                    )

                if best_screen and detection_frame_counter % 20 == 0:
                    bx1, by1, bx2, by2, bconf = best_screen

            if best_screen:
                x1, y1, x2, y2, conf = best_screen

                if smoothed_box is None:

                    if KNOWN_ASPECT_RATIO is not None:
                        refined_x1, refined_y1, refined_x2, refined_y2 = refine_box_to_aspect_ratio(
                            x1, y1, x2, y2, KNOWN_ASPECT_RATIO, w, h
                        )
                        smoothed_box = (float(refined_x1), float(refined_y1), float(refined_x2), float(refined_y2), conf)
                        detection_history = [(float(refined_x1), float(refined_y1), float(refined_x2), float(refined_y2), conf, frame_counter)]
                    else:
                        smoothed_box = (float(x1), float(y1), float(x2), float(y2), conf)
                        detection_history = [(float(x1), float(y1), float(x2), float(y2), conf, frame_counter)]
                    frame_counter = 0
                else:

                    old_x1, old_y1, old_x2, old_y2, old_conf = smoothed_box
                    old_center_x = (old_x1 + old_x2) / 2
                    old_center_y = (old_y1 + old_y2) / 2

                    new_center_x = (x1 + x2) / 2
                    new_center_y = (y1 + y2) / 2

                    center_distance_x = abs(new_center_x - old_center_x) / w
                    center_distance_y = abs(new_center_y - old_center_y) / h
                    max_distance = max(center_distance_x, center_distance_y)

                    frame_counter += 1

                    should_update = (max_distance < MAX_DETECTION_DISTANCE and
                                    frame_counter >= UPDATE_EVERY_N_FRAMES)

                    if should_update:

                        detection_history.append((float(x1), float(y1), float(x2), float(y2), conf, frame_counter))

                        if len(detection_history) > DETECTION_HISTORY_SIZE:
                            detection_history.pop(0)

                        total_weight = 0.0
                        weighted_x1 = 0.0
                        weighted_y1 = 0.0
                        weighted_x2 = 0.0
                        weighted_y2 = 0.0
                        weighted_conf = 0.0

                        for i, (hx1, hy1, hx2, hy2, hconf, hframe) in enumerate(detection_history):

                            recency_weight = RECENCY_WEIGHT_DECAY ** (len(detection_history) - 1 - i)

                            conf_weight = hconf ** CONFIDENCE_WEIGHT_POWER

                            weight = recency_weight * conf_weight

                            weighted_x1 += hx1 * weight
                            weighted_y1 += hy1 * weight
                            weighted_x2 += hx2 * weight
                            weighted_y2 += hy2 * weight
                            weighted_conf += hconf * weight
                            total_weight += weight

                        if total_weight > 0:
                            new_x1 = weighted_x1 / total_weight
                            new_y1 = weighted_y1 / total_weight
                            new_x2 = weighted_x2 / total_weight
                            new_y2 = weighted_y2 / total_weight
                            new_conf = weighted_conf / total_weight
                        else:

                            new_x1 = AVERAGE_WEIGHT_OLD * old_x1 + AVERAGE_WEIGHT_NEW * x1
                            new_y1 = AVERAGE_WEIGHT_OLD * old_y1 + AVERAGE_WEIGHT_NEW * y1
                            new_x2 = AVERAGE_WEIGHT_OLD * old_x2 + AVERAGE_WEIGHT_NEW * x2
                            new_y2 = AVERAGE_WEIGHT_OLD * old_y2 + AVERAGE_WEIGHT_NEW * y2
                            new_conf = AVERAGE_WEIGHT_OLD * old_conf + AVERAGE_WEIGHT_NEW * conf

                        if KNOWN_ASPECT_RATIO is not None:
                            new_x1, new_y1, new_x2, new_y2 = refine_box_to_aspect_ratio(
                                int(new_x1), int(new_y1), int(new_x2), int(new_y2),
                                KNOWN_ASPECT_RATIO, w, h
                            )
                            new_x1, new_y1, new_x2, new_y2 = float(new_x1), float(new_y1), float(new_x2), float(new_y2)

                        smoothed_box = (new_x1, new_y1, new_x2, new_y2, new_conf)
                        frame_counter = 0

            display_box = smoothed_box if smoothed_box is not None else best_screen

            if display_box:
                sx1, sy1, sx2, sy2, sconf = display_box
                sx1, sy1, sx2, sy2 = int(sx1), int(sy1), int(sx2), int(sy2)

                display_frame = frame.copy()

                mask = np.ones((h, w, 3), dtype=np.float32) * DARKEN_FACTOR
                mask[sy1:sy2, sx1:sx2] = 1.0
                display_frame = (frame * mask).astype(np.uint8)

                cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)
                cv2.putText(display_frame, f"Screen {sconf:.2f}", (sx1, sy1 - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:

                display_frame = (frame * DARKEN_FACTOR * 0.5).astype(np.uint8)

            cv2.imshow("Camera", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            now = time.time()

            elapsed = now - last_pred_time
            if elapsed >= INTERVAL_SECONDS:

                classification_box = display_box if display_box else (best_screen if best_screen else smoothed_box)

                h_classify, w_classify = frame.shape[:2]
                frame_area_classify = h_classify * w_classify

                if classification_box:
                    x1, y1, x2, y2, screen_conf = classification_box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    shrink_factor = 0.02
                    width = x2 - x1
                    height = y2 - y1
                    shrink_x = int(width * shrink_factor)
                    shrink_y = int(height * shrink_factor)

                    x1 = max(0, x1 + shrink_x)
                    y1 = max(0, y1 + shrink_y)
                    x2 = min(w_classify, x2 - shrink_x)
                    y2 = min(h_classify, y2 - shrink_y)

                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    crop_area = crop_width * crop_height
                    relative_size = crop_area / frame_area_classify

                    if MIN_SCREEN_RATIO * MIN_SCREEN_RATIO <= relative_size <= MAX_SCREEN_RATIO * MAX_SCREEN_RATIO:
                        cropped_frame = frame[y1:y2, x1:x2]
                        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        print(f"Screen size invalid ({relative_size*100:.1f}% of frame), using full frame")
                        print(f"  Detected size: {crop_width}x{crop_height}")
                else:

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    print("No screen detected and no previous box, using full frame")
                    if smoothed_box is None:
                        print("  No previous smoothed box available")

                img = Image.fromarray(frame_rgb)

                label, confidence, all_predictions = predict_image_all(img, model, classes, device)

                if label not in total_frames_per_label:
                    total_frames_per_label[label] = 0
                total_frames_per_label[label] += 1

                if CONFIDENCE_TRACKING_ENABLED:

                    for class_name, _ in all_predictions:
                        if class_name not in class_confidence_tracker:
                            class_confidence_tracker[class_name] = 0.0
                            consecutive_high_confidence[class_name] = 0
                        if class_name not in total_frames_per_label:
                            total_frames_per_label[class_name] = 0

                    for class_name, class_conf in all_predictions:
                        old_avg = class_confidence_tracker[class_name]

                        new_avg = CONFIDENCE_EMA_ALPHA * class_conf + (1 - CONFIDENCE_EMA_ALPHA) * old_avg
                        class_confidence_tracker[class_name] = new_avg

                        if new_avg >= CONFIDENCE_THRESHOLD:
                            consecutive_high_confidence[class_name] += 1
                        else:
                            consecutive_high_confidence[class_name] = 0

                    sorted_tracker = sorted(class_confidence_tracker.items(), key=lambda x: x[1], reverse=True)
                    best_tracked_class, best_tracked_conf = sorted_tracker[0]

                    required_frames = MIN_CONSECUTIVE_FRAMES.get(
                        best_tracked_class,
                        DEFAULT_MIN_CONSECUTIVE_FRAMES
                    )

                    if (best_tracked_conf >= CONFIDENCE_THRESHOLD and
                        consecutive_high_confidence[best_tracked_class] >= required_frames):
                        if current_tracked_label is not None and current_tracked_label != best_tracked_class:
                            print(f"⚠️  Label changed: {current_tracked_label} → {best_tracked_class}")
                            if ws_server:
                                ws_server.send_update({
                                    'type': 'label_changed',
                                    'previous_label': current_tracked_label,
                                    'new_label': best_tracked_class,
                                    'confidence': best_tracked_conf,
                                    'timestamp': time.time()
                                })
                        current_tracked_label = best_tracked_class
                    elif current_tracked_label is None:
                        current_tracked_label = best_tracked_class
                        
                    if ws_server:
                        is_sport = current_tracked_label in ['basketball', 'football']
                        ws_server.send_update({
                            'type': 'status_update',
                            'label': current_tracked_label,
                            'is_sport': is_sport,
                            'is_commercial': current_tracked_label == 'commercial',
                            'confidence': class_confidence_tracker.get(current_tracked_label, 0),
                            'all_confidences': {k: float(v) for k, v in class_confidence_tracker.items()},
                            'timestamp': time.time()
                        })

                    print(f"Instant Prediction: {label}, Confidence: {confidence:.2f}")
                    print(f"Tracked Label: {current_tracked_label} (avg: {class_confidence_tracker.get(current_tracked_label, 0):.2f})")
                    print("Running Average Confidences:")
                    for class_name, avg_conf in sorted_tracker:
                        consecutive = consecutive_high_confidence.get(class_name, 0)
                        required = MIN_CONSECUTIVE_FRAMES.get(class_name, DEFAULT_MIN_CONSECUTIVE_FRAMES)
                        total_frames = total_frames_per_label.get(class_name, 0)
                        indicator = "✓" if consecutive >= required else " "
                        print(f"  {indicator} {class_name}: {avg_conf:.4f} (consecutive: {consecutive}/{required}, total: {total_frames})")
                    print("Instant Confidences:")
                    for class_name, class_conf in all_predictions:
                        print(f"  {class_name}: {class_conf:.4f}")
                else:

                    print(f"Prediction: {label}, Confidence: {confidence:.2f}")
                    print("All predictions:")
                    for class_name, class_conf in all_predictions:
                        print(f"  {class_name}: {class_conf:.4f}")

                print("-" * 50)
                last_pred_time = now

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if ws_server:
            ws_server.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    main()