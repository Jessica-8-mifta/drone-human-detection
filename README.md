# 🚁 Drone Human Detection & Counting System

A computer vision pipeline for detecting humans and vehicles in drone/aerial imagery using YOLOv8, with object tracking via ByteTrack.

Built as part of the ANTS Antlings AI/ML Internship Technical Assessment.

---

## 📌 Project Overview

This project implements an end-to-end object detection and tracking pipeline on the [VisDrone2019 dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset) — one of the most challenging benchmarks for drone-based computer vision due to small object sizes, dense crowds, and varying altitudes.

### Objectives
- Detect humans and vehicles in aerial drone images
- Count total humans per frame
- Visualize detections with bounding boxes
- Track objects across frames using ByteTrack (bonus)

---

## 🗂️ Repository Structure

```
drone-human-detection/
│
├── notebook/
│   └── drone_detection.ipynb    # Full pipeline notebook
│
├── outputs/
│   ├── sample_images.png        # Task 01: Dataset samples
│   ├── class_distribution.png   # Task 01: Class distribution
│   ├── training_curves.png      # Task 02: Training metrics
│   ├── detection_results.png    # Task 03: Detection outputs
│   ├── tracking_results.png     # Task 04: Tracking outputs
│   ├── count_over_frames.png    # Task 04: Count over frames
│   ├── per_class_ap.png         # Task 05: Per-class AP
│   └── final_detection.png      # Task 05: Final detections
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset: VisDrone2019

| Split | Images |
|---|---|
| Train | 6,471 |
| Validation | 548 |
| Test | 1,610 |
| **Total** | **8,629** |

### Classes
| ID | Class | Type |
|---|---|---|
| 0 | pedestrian | 🔴 Human |
| 1 | people | 🔴 Human |
| 2 | bicycle | 🟢 Vehicle |
| 3 | car | 🟢 Vehicle |
| 4 | van | 🟢 Vehicle |
| 5 | truck | 🟢 Vehicle |
| 6 | tricycle | 🟢 Vehicle |
| 7 | awning-tricycle | 🟢 Vehicle |
| 8 | bus | 🟢 Vehicle |
| 9 | motor | 🟢 Vehicle |

### Key Challenges
- **Small object sizes** — humans appear as tiny pixels from drone altitude
- **Class imbalance** — cars (144,867) vs awning-tricycle (3,246)
- **Dense scenes** — heavily overlapping bounding boxes
- **Varying altitude** — inconsistent object scale across images

---

## 🧠 Model: YOLOv8n

- **Architecture:** YOLOv8 Nano (pretrained on COCO, fine-tuned on VisDrone)
- **Framework:** Ultralytics
- **Training:** 20 epochs, batch size 16, image size 640×640
- **Device:** Tesla T4 GPU (Google Colab)
- **Optimizer:** Auto (SGD), lr=0.01, momentum=0.937

### Training Configuration
```python
model = YOLO('yolov8n.pt')
model.train(
    data='visdrone.yaml',
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    patience=5
)
```

---

## 📈 Results

### Evaluation Metrics (Validation Set)

| Metric | Value |
|---|---|
| mAP@50 | 0.2697 |
| mAP@50-95 | 0.1495 |
| Precision | 0.3916 |
| Recall | 0.3026 |
| Inference Speed | 2.8ms/image |

### Discussion
Results are consistent with expected performance for a lightweight model (YOLOv8n) trained for 20 epochs on VisDrone — one of the hardest aerial detection benchmarks. State-of-the-art models trained for 100+ epochs typically achieve mAP@50 of 0.40–0.50. Key limitations include small object sizes and class imbalance, which compress recall particularly for rare classes.

---

## 🎯 Task Summary

### Task 01 — Dataset Understanding & Preprocessing
- Explored dataset structure across train/val/test splits
- Analyzed class distribution — identified heavy imbalance (car dominates)
- Visualized sample images with ground truth bounding boxes
- Identified key challenges: small objects, dense crowds, altitude variance

### Task 02 — Model Training
- Fine-tuned YOLOv8n pretrained on COCO
- Trained for 20 epochs with early stopping (patience=5)
- Applied standard augmentations: mosaic, random flip, HSV shift
- Monitored box loss, class loss, and DFL loss across epochs

### Task 03 — Human & Car Detection with Counting
- Ran inference on validation images at conf=0.25
- Separated human classes (pedestrian, people) from vehicle classes
- Drew color-coded bounding boxes: 🔴 Red = humans, 🟢 Green = vehicles
- Displayed real-time human and vehicle count overlay per image

### Task 04 — Object Tracking (Bonus)
- Implemented ByteTrack via YOLOv8's built-in tracker
- Assigned unique color-coded IDs to each tracked object
- Tracked 20 consecutive frames from validation set
- Plotted human and vehicle count trends across frames

### Task 05 — Evaluation & Visualization
- Ran formal validation on full val set (548 images)
- Computed mAP@50, mAP@50-95, precision, recall
- Generated per-class AP@50 bar chart
- Produced final detection output visualizations

---

## 🔧 Strengths & Limitations

### Strengths
- End-to-end pipeline from raw data to tracked detections
- ByteTrack integration for temporal consistency
- Fast inference — 2.8ms per image suitable for real-time use
- Clean, modular notebook structure

### Limitations
- YOLOv8n is the smallest variant — larger models (YOLOv8m/l) would improve accuracy
- 20 epochs is limited — 100+ epochs would significantly improve mAP
- No data augmentation specifically designed for small objects
- Tracking applied to static images rather than true video sequences

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open `notebook/drone_detection.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Add your Kaggle API credentials
4. Run all cells sequentially

### Option 2: Local
```bash
pip install ultralytics opencv-python matplotlib pandas pyyaml
```
Then run the notebook locally with your dataset path configured.

---

## 📦 Requirements

```
ultralytics>=8.4.0
opencv-python>=4.6.0
matplotlib>=3.3.0
numpy>=1.23.0
pandas>=1.3.0
pyyaml>=5.3.1
torch>=1.8.0
```

---

## 👩‍💻 Author

**Miftahul Jannat Jessica**
BSc Statistics (Honours), SUST
[GitHub](https://github.com/Jessica-8-mifta) | [LinkedIn](https://linkedin.com/in/miftahul-jannat-jessica)

---

## 📄 License
MIT License
