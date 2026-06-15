# Trained Models

This project utilizes two trained deep learning models developed for the Real-Time Malaysian Traffic Sign Detection and Alert System.

---

## 1. YOLOv8 Traffic Sign Detection Model

### File Name
- best.pt

### Purpose
The YOLOv8 model serves as the primary object detection engine of the system. It is responsible for:

- Detecting Malaysian traffic signs from video frames.
- Generating bounding boxes around detected traffic signs.
- Displaying class labels and confidence scores.
- Supporting real-time traffic sign detection and alert generation.

### Performance

| Metric | Result |
|----------|----------|
| mAP@50 | 88.7% |
| mAP@50-95 | 67.2% |

The model was trained using a Malaysian Traffic Sign Dataset prepared in YOLO format and optimized for real-time deployment.

---

## 2. CNN Speed Limit Classification Model

### File Name
- best_speed_class_cnn.pt

### Purpose

The CNN model acts as a secondary classifier specifically for speed limit traffic signs.

After a speed sign is detected by YOLOv8:

1. The detected speed sign region is cropped.
2. The cropped image is passed to the CNN model.
3. The CNN model classifies the exact speed category.

Supported speed categories include:

- 30 km/h
- 50 km/h
- 60 km/h
- 70 km/h
- 90 km/h
- 110 km/h

### Performance

| Metric | Result |
|----------|----------|
| Classification Accuracy | 94% |

The model demonstrated strong classification performance with only minor confusion between visually similar speed limit signs.

---

## Model Availability

The trained model weight files are not included in this repository due to GitHub file size limitations.

Model files:

- best.pt
- best_speed_class_cnn.pt

The repository contains the complete source code, project report, and documentation required to reproduce the project workflow and system implementation.
