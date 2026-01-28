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

## 🏗️ System Architecture

The system follows a modular end-to-end pipeline, from geospatial data acquisition to AI inference and post-processing.

![System Architecture](docs/images/system_architecture.png)
1.  [cite_start]**Input:** Receives GeoJSON Polygon from the Mobile App[cite: 519].
2.  [cite_start]**Preprocessing:** Satellite image retrieval, normalization & Tiling (Slicing)[cite: 561].
3.  [cite_start]**Inference:** YOLOv12 model scans individual slices (SAHI) to detect trees[cite: 562].
4.  [cite_start]**Post-processing:** Merging detections, Soft-NMS (Non-Maximum Suppression), and Density Calculation[cite: 372].
5.  [cite_start]**Output:** Returns JSON Statistics + Base64 Overlay Image[cite: 565].

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

## 🐳 Docker Deployment (Recommended)

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
