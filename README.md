# 🌴 Deep Palm: Satellite-Based Palm Tree Detection Server

![Project Banner](docs/images/banner_project.png)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv12](https://img.shields.io/badge/Model-YOLOv12-orange)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

## 📖 Overview

**Deep Palm Backend** is a high-performance AI inference server designed to revolutionize Precision Agriculture. [cite_start]This project addresses the critical challenge of low-resolution satellite imagery by integrating **State-of-the-Art Object Detection (YOLOv12)** with **Slicing Aided Hyper Inference (SAHI)**[cite: 363, 924].

[cite_start]The server acts as the central intelligence engine: processing geospatial requests, automatically retrieving satellite imagery, and returning actionable agronomic insights (Tree Count, Density, Health Status) in near real-time (~1.7s latency)[cite: 915].

### 🚀 Key Features
* [cite_start]**Satellite Intelligence:** Automatic retrieval of high-resolution imagery via Google Earth Engine / Sentinel Hub[cite: 520, 560].
* [cite_start]**Deep Tech Core:** Integrates **YOLOv12 + SAHI** to bridge the resolution gap, detecting small palm crowns that standard models miss[cite: 924].
* [cite_start]**Precision Inventory:** Returns exact GPS coordinates and canopy area ($m^2$) for every single tree (Digital Passport)[cite: 907].
* [cite_start]**Spatial Diagnosis:** Implements a "Traffic Light" system (Green/Yellow/Red) to visually identify overcrowding and optimal spacing[cite: 906].
* [cite_start]**High Performance:** Built on **FastAPI** (Asynchronous) and containerized with **Docker** for scalable deployment[cite: 557, 567].


---

## 📊 Dataset at a Glance

![Dataset Overview](docs/images/readme_dataset_overview.png)

---

## 🏗️ System Architecture

The system follows a modular end-to-end pipeline, from geospatial data acquisition to AI inference and post-processing.

![System Architecture](docs/images/system_architecture.png)
1.  [cite_start]**Input:** Receives GeoJSON Polygon from the Mobile App[cite: 519].
2.  [cite_start]**Preprocessing:** Satellite image retrieval, normalization & Tiling (Slicing)[cite: 561].
3.  [cite_start]**Inference:** YOLOv12 model scans individual slices (SAHI) to detect trees[cite: 562].
4.  [cite_start]**Post-processing:** Merging detections, Soft-NMS (Non-Maximum Suppression), and Density Calculation[cite: 372].
5.  [cite_start]**Output:** Returns JSON Statistics + Base64 Overlay Image[cite: 565].

---

## 🔬 Methodology

### 1. Image Type Taxonomy and Stratified Handling

To ensure robust generalization across heterogeneous landscapes, the collected RGB satellite tiles were categorized into representative *image types* based on plantation density, background context, and radiometric conditions. This taxonomy enables stratified sampling during training and targeted preprocessing/augmentation policies while maintaining a single unified detector.

*   **Image Type A: Dense plantation tiles.** Regular grid patterns with frequent crown-to-crown proximity and overlapping shadows. Primary challenge: crowding and partial occlusion.
*   **Image Type B: Sparse/desert-adjacent tiles.** Widely spaced palms embedded in sandy/rocky backgrounds with variable contrast. Primary challenge: small-object visibility and background confusion.
*   **Image Type C: Mixed urban–agricultural tiles.** Palms co-exist with buildings, roads, irrigation circles, and man-made structures. Primary challenge: false positives on circular roofs/water structures.
*   **Image Type D: Low-contrast/shadow/haze tiles.** Reduced visibility due to haze, long shadows, or low solar elevation. Primary challenge: blurred crown boundaries and missed detections.

Rather than training separate models per type, the dataset was stratified across these categories to balance exposure during optimization. Type-specific preprocessing and augmentation were applied through region/type-aware scheduling while preserving consistent supervision signals.

![Representative image-type taxonomy](docs/images/fig_type_taxonomy.png)
*Representative image-type taxonomy used in this study: (A) dense plantations, (B) sparse/desert-adjacent, (C) mixed urban--agricultural, and (D) low-contrast/shadow/haze conditions.*

### 2. Type-Aware Processing and Sampling Workflow

Each tile was assigned an image-type tag during ingestion. The tag is used to (i) enforce stratified splits, (ii) adjust normalization and shadow/haze routines when required, and (iii) control augmentation probabilities (e.g., stronger mosaic/copy--paste in dense scenes and stronger photometric perturbations in low-contrast scenes). This procedure avoids geographic leakage and prevents performance inflation due to over-representation of visually easy tiles.

![Type-aware workflow](docs/images/fig_type_workflow.png)
*Type-aware workflow integrated into the pipeline: image-type tagging → stratified partitioning → type-aware preprocessing/augmentation → unified model training → SAHI inference.*

### 3. Type-Specific Preprocessing and Augmentation

| Image Type | Main Failure Mode | Emphasized Policy |
| :--- | :--- | :--- |
| **A: Dense plantation** | Occlusion, clustered crowns | Mosaic/Copy--Paste (early), Soft-NMS, SAHI overlap |
| **B: Sparse/desert-adjacent** | Small objects, low contrast vs soil | Contrast normalization, mild HSV jitter, multiscale |
| **C: Mixed urban--agri** | False positives on circular man-made objects | Hard-negative sampling, stricter label rules, context mosaic |
| **D: Low-contrast/shadow/haze** | Missed detections, boundary blur | Shadow-aware adjustment, haze simulation, CLAHE-like boost |


### 📝 Model & Dataset Cards

#### Model Card
*   **Architecture:** YOLOv12m (Medium) + SAHI (Slicing Aided Hyper Inference)
*   **Input Resolution:** 640x640 (base), Slicing: 1280x1280 patches resized to 640.
*   **Hyperparameters:**
    *   Confidence Threshold: 0.4
    *   IoU Threshold: 0.5
    *   Slice Overlap: 0.2
*   **Performance:** ~1.7s latency per tile on NVIDIA T4.
*   **Limitations:** Performance may degrade in extreme haze or with circular non-palm structures (e.g., water tanks) in urban areas.

#### Dataset Card
*   **Regions:** Al-Ahsa, Qassim, Medina (Saudi Arabia).
*   **Resolution (GSD):** 0.3m - 0.5m per pixel.
*   **Labeling:** Bounding box annotations for palm crowns.
*   **Splits:** Spatially disjoint splitting to prevent geographic leakage.
    *   Train: 70% | Val: 20% | Test: 10%
*   **Bias Mitigation:** Stratified sampling across Dense (A), Sparse (B), Urban (C), and Haze (D) categories.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* [cite_start]**Framework:** FastAPI (Uvicorn/Gunicorn)[cite: 557].
* **AI/Computer Vision:** PyTorch, Ultralytics YOLOv12, SAHI, OpenCV.
* **Geospatial:** Rasterio, Shapely, PyProj, Google Earth Engine API.
* **Deployment:** Docker, Docker Compose.

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+.
* CUDA-enabled GPU (Recommended for real-time inference).
* Google Earth Engine / Sentinel Hub account credentials.

### 1. Clone the repository
```bash
git clone [https://github.com/YourOrg/deep-palm-backend.git](https://github.com/YourOrg/deep-palm-backend.git)
cd deep-palm-backend

```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
PORT=8000
# AI Model Config
MODEL_PATH=weights/yolov12_palm_best.pt
CONF_THRESHOLD=0.4
IOU_THRESHOLD=0.5

# Geospatial API Keys
GEE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
SENTINEL_CLIENT_ID=your_client_id
SENTINEL_CLIENT_SECRET=your_client_secret

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Server

```bash
# Development Mode
uvicorn app.main:app --reload

# Production Mode
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app

```

---

## � Repository Structure

```text
├── app/                # Application source code
├── tools/              # Training, evaluation, and inference scripts
├── configs/            # Configuration files (YOLO, SAHI, types)
├── docs/images/        # Documentation assets
├── weights/            # Pre-trained models
└── requirements.txt    # Python dependencies
```

## 🔁 Reproducibility

To reproduce the results reported in this study, follow these steps:

*   **Training:**
    ```bash
    python tools/train.py --config configs/exp_yolov12m.yaml --seed 42
    ```
*   **Evaluation:**
    ```bash
    python tools/eval.py --weights weights/best.pt --task val
    ```
*   **SAHI Inference:**
    ```bash
    python tools/infer_sahi.py --source data/test_images/ --config configs/sahi.yaml
    ```

---

## �🐳 Docker Deployment (Recommended)

To build and run the containerized application:

```bash
# Build the image
docker-compose build

# Run the container
docker-compose up -d

```

The server will be available at: `http://localhost:8000`.

---

## 🔌 API Documentation

### `POST /predict`

Analyzes a user-defined polygon area and returns palm tree statistics.

**Request Body (JSON):**

```json
{
  "polygon": [
    [106.660172, 10.762622],
    [106.660172, 10.862622],
    [106.760172, 10.862622],
    [106.760172, 10.762622]
  ],
  "date": "2025-10-20"
}

```

**Response:**

```json
{
  "status": "success",
  "data": {
    "total_trees": 227,
    "density": 78.4,
    "avg_canopy_area": 12.5,
    "recommendation": "Optimal Spacing",
    "overlay_image": "base64_encoded_string...",
    "inventory": [
      {"id": 1, "lat": 10.76, "lng": 106.66, "status": "Green"},
      {"id": 2, "lat": 10.77, "lng": 106.67, "status": "Red"}
    ]
  }
}

```

---

## 📊 Results & Demo

### SAHI Slicing Mechanism

*Visualizing how the system detects small objects by "slicing" the satellite map into high-resolution patches.*

### Final Application Output

*The Mobile Interface displaying detected palm trees with Health Status indicators (Green/Red).*

## 🤝 Contributors

* 
**Jikey (Nguyen Nhat Truong)** - System Architect & Fullstack Lead.


* 
**Thinh (Pham Le Duc Thinh)** - AI Research Lead (YOLOv12 & SAHI).


* 
**Kiet (Do Anh Kiet)** - Data & QA Lead.



---

## 📄 License

This project is licensed under the MIT License.
