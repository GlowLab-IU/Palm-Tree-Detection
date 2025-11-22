# -*- coding: utf-8 -*-
"""
FastAPI application logic for Palm Tree Counting API (Server-side Tile Processing + Filtering)
"""
# --- Imports (giữ nguyên) ---
import io
import os
import logging
import numpy as np
import cv2
import torch
import base64
from datetime import datetime
import math
import requests
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pydantic import BaseModel, Field
from typing import List, Dict

# --- Cấu hình & Hằng số (giữ nguyên) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_PATH = os.getenv("MODEL_PATH", "./best_palmtree.pt")
SAHI_SLICE_HEIGHT = int(os.getenv("SAHI_SLICE_HEIGHT", 640))
SAHI_SLICE_WIDTH = int(os.getenv("SAHI_SLICE_WIDTH", 640))
SAHI_OVERLAP_HEIGHT_RATIO = float(os.getenv("SAHI_OVERLAP_HEIGHT_RATIO", 0.2))
SAHI_OVERLAP_WIDTH_RATIO = float(os.getenv("SAHI_OVERLAP_WIDTH_RATIO", 0.2))
SAHI_CONFIDENCE_THRESHOLD = float(os.getenv("SAHI_CONFIDENCE_THRESHOLD", 0.3))
TARGET_ZOOM = 20
TILE_SIZE = 256
GOOGLE_TILE_URL_TEMPLATE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
MAX_TILES = 225

# --- Hàm GIS & Vẽ ---
def calculate_gsd(lat, zoom):
    """
    Tính Ground Sample Distance (GSD) - độ phân giải mặt đất (m/pixel)
    Công thức Web Mercator standard:
    GSD = 156543.03392 * cos(lat) / 2^zoom
    
    Args:
        lat: Latitude (độ)
        zoom: Zoom level
    Returns:
        GSD in meters/pixel
    """
    gsd = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    return gsd

def calculate_polygon_area_m2(polygon_latlng):
    """
    Tính diện tích polygon trên mặt cầu (m²) sử dụng công thức spherical excess
    
    Args:
        polygon_latlng: List of dicts with 'latitude' and 'longitude' keys
    Returns:
        Area in square meters
    """
    EARTH_RADIUS = 6378137.0  # meters
    
    if len(polygon_latlng) < 3:
        return 0.0
    
    # Convert to radians
    coords = [(math.radians(p['latitude']), math.radians(p['longitude'])) for p in polygon_latlng]
    
    # Spherical excess formula
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += (coords[j][1] - coords[i][1]) * (2 + math.sin(coords[i][0]) + math.sin(coords[j][0]))
    
    area = abs(area * EARTH_RADIUS * EARTH_RADIUS / 2.0)
    return area

def latLngToWorldXY(lat, lng, zoom):
    # --- SỬA LỖI CÚ PHÁP: Bỏ comment JS ---
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    y = math.log((1 + siny) / (1 - siny))
    worldY = (TILE_SIZE * (1 << zoom) * (1 - y / (2 * math.pi))) / 2
    worldX = TILE_SIZE * (1 << zoom) * (lng / 360 + 0.5)
    return {"x": worldX, "y": worldY}

def getTileBoundingBox(points, zoom):
    # --- SỬA LỖI CÚ PHÁP: Bỏ comment JS ---
    minWorldX, maxWorldX = float('inf'), float('-inf')
    minWorldY, maxWorldY = float('inf'), float('-inf')
    for p in points:
        worldP = latLngToWorldXY(p["latitude"], p["longitude"], zoom)
        minWorldX = min(minWorldX, worldP["x"])
        maxWorldX = max(maxWorldX, worldP["x"])
        minWorldY = min(minWorldY, worldP["y"])
        maxWorldY = max(maxWorldY, worldP["y"])
    worldBounds = {"minWorldX": minWorldX, "maxWorldX": maxWorldX, "minWorldY": minWorldY, "maxWorldY": maxWorldY}
    minTileX = math.floor(worldBounds["minWorldX"] / TILE_SIZE)
    maxTileX = math.floor(worldBounds["maxWorldX"] / TILE_SIZE)
    minTileY = math.floor(worldBounds["minWorldY"] / TILE_SIZE)
    maxTileY = math.floor(worldBounds["maxWorldY"] / TILE_SIZE)
    tileBounds = {"minTileX": minTileX, "maxTileX": maxTileX, "minTileY": minTileY, "maxTileY": maxTileY}
    return tileBounds

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=1):
    # --- SỬA LỖI CÚ PHÁP: Bỏ comment JS ---
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    elif thickness > 0:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

# --- Tải Model (giữ nguyên) ---
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"
logging.info(f"Device: {device} | CUDA: {torch.version.cuda if use_cuda else 'N/A'} | GPU: {torch.cuda.get_device_name(0) if use_cuda else 'N/A'}")
detection_model = None
try:
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    logging.info(f"⏳ Loading model from '{MODEL_PATH}' on '{device.upper()}'...")
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=MODEL_PATH, confidence_threshold=SAHI_CONFIDENCE_THRESHOLD, device=device)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Critical error loading model: {e}")

# --- FastAPI App & Input Model (giữ nguyên) ---
app = FastAPI( title="Palm Tree Detection API (Server Tile + Filter)", version="5.1.1") # Bump version
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
class LatLng(BaseModel): latitude: float; longitude: float
class PredictRequest(BaseModel): polygon: List[LatLng]; center: LatLng | None = None; area_m2: float | None = None; area_ha: float | None = None

@app.get("/")
def read_root(): return {"message": "POST /predict with JSON polygon."}

# --- ENDPOINT /PREDICT (giữ nguyên logic) ---
@app.post("/predict")
async def predict_from_polygon(data: PredictRequest = Body(...)):
    if detection_model is None: raise HTTPException(status_code=503, detail="Error: Model not loaded.")
    try:
        polygon_latlng = [p.dict() for p in data.polygon]
        if len(polygon_latlng) < 3: raise HTTPException(status_code=400, detail="Invalid polygon")
        logging.info(f"Processing polygon with {len(polygon_latlng)} points.")

        # 0. Tính toán GSD và diện tích polygon
        # Lấy latitude trung tâm của polygon để tính GSD
        avg_lat = sum(p["latitude"] for p in polygon_latlng) / len(polygon_latlng)
        gsd = calculate_gsd(avg_lat, TARGET_ZOOM)
        logging.info(f"GSD at lat {avg_lat:.6f}, zoom {TARGET_ZOOM}: {gsd:.6f} m/pixel")
        
        # Tính diện tích polygon (nếu client không gửi)
        if data.area_m2 is None:
            area_m2 = calculate_polygon_area_m2(polygon_latlng)
            area_ha = area_m2 / 10000.0
            logging.info(f"Calculated polygon area: {area_m2:.2f} m² ({area_ha:.4f} ha)")
        else:
            area_m2 = data.area_m2
            area_ha = data.area_ha if data.area_ha else area_m2 / 10000.0
            logging.info(f"Client-provided polygon area: {area_m2:.2f} m² ({area_ha:.4f} ha)")

        # 1. Tính toán & Tải Tile
        tileBounds = getTileBoundingBox(polygon_latlng, TARGET_ZOOM)
        minTileX, maxTileX = tileBounds["minTileX"], tileBounds["maxTileX"]
        minTileY, maxTileY = tileBounds["minTileY"], tileBounds["maxTileY"]
        tiles_to_download = [(x, y) for x in range(minTileX, maxTileX + 1) for y in range(minTileY, maxTileY + 1)]
        logging.info(f"Tiles to download: {len(tiles_to_download)}")
        if len(tiles_to_download) > MAX_TILES: raise HTTPException(status_code=400, detail=f"Area too large ({len(tiles_to_download)} tiles). Max {MAX_TILES}.")
        if not tiles_to_download: raise HTTPException(status_code=400, detail="No tiles found")
        stitch_width = (maxTileX - minTileX + 1) * TILE_SIZE
        stitch_height = (maxTileY - minTileY + 1) * TILE_SIZE
        stitched_image = Image.new("RGBA", (stitch_width, stitch_height))
        stitchOriginWorldX = minTileX * TILE_SIZE
        stitchOriginWorldY = minTileY * TILE_SIZE
        session = requests.Session()
        logging.info("Downloading and stitching tiles...")
        for x, y in tiles_to_download:
            url = GOOGLE_TILE_URL_TEMPLATE.format(x=x, y=y, z=TARGET_ZOOM)
            try:
                response = session.get(url, timeout=10); response.raise_for_status()
                tile_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
                stitched_image.paste(tile_image, ((x - minTileX) * TILE_SIZE, (y - minTileY) * TILE_SIZE))
            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to download tile ({x},{y}): {e}. Filling black.")
                draw = ImageDraw.Draw(stitched_image)
                offsetX, offsetY = (x - minTileX) * TILE_SIZE, (y - minTileY) * TILE_SIZE
                draw.rectangle([(offsetX, offsetY), (offsetX + TILE_SIZE, offsetY + TILE_SIZE)], fill=(0, 0, 0, 255))
        logging.info("Stitching complete.")

        # 2. Chuyển ảnh ghép sang NumPy RGB cho SAHI
        stitched_image_rgb_pil = stitched_image.convert("RGB")
        stitched_image_rgb_np = np.array(stitched_image_rgb_pil)
        logging.info(f"Stitched image prepared for SAHI, shape: {stitched_image_rgb_np.shape}")

        # 3. CHẠY SAHI TRÊN TOÀN BỘ ẢNH GHÉP
        logging.info("Starting SAHI prediction on stitched image...")
        result = get_sliced_prediction(
            stitched_image_rgb_np,
            detection_model,
            slice_height=SAHI_SLICE_HEIGHT, slice_width=SAHI_SLICE_WIDTH,
            overlap_height_ratio=SAHI_OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=SAHI_OVERLAP_WIDTH_RATIO
        )
        all_predictions = result.object_prediction_list
        logging.info(f"🌴 SAHI found {len(all_predictions)} total objects on stitched image.")

        # 4. Tạo mặt nạ Polygon trên kích thước ảnh ghép
        polygon_px = []
        for p in polygon_latlng:
            worldP = latLngToWorldXY(p["latitude"], p["longitude"], TARGET_ZOOM)
            x = worldP["x"] - stitchOriginWorldX
            y = worldP["y"] - stitchOriginWorldY
            polygon_px.append((x, y))

        mask_pil = Image.new("L", (stitch_width, stitch_height), 0)
        ImageDraw.Draw(mask_pil).polygon(polygon_px, outline=1, fill=1)
        mask_np = np.array(mask_pil)

        # 5. LỌC CÁC DỰ ĐOÁN NẰM TRONG POLYGON MASK & TÍNH TOÁN PHENOTYPING METRICS
        filtered_predictions = []
        total_canopy_area = 0.0  # Acover - Tổng diện tích tán cây
        
        for pred in all_predictions:
            box = pred.bbox
            center_x = int((box.minx + box.maxx) / 2)
            center_y = int((box.miny + box.maxy) / 2)
            if 0 <= center_y < mask_np.shape[0] and 0 <= center_x < mask_np.shape[1] and mask_np[center_y, center_x] > 0:
                # Tính kích thước bounding box trong pixel
                wp = box.maxx - box.minx  # Chiều rộng (pixel)
                hp = box.maxy - box.miny  # Chiều cao (pixel)
                
                # Tính bán kính tán cây (meters)
                # rm = ((wp + hp) / 4) * GSD
                radius_m = ((wp + hp) / 4.0) * gsd
                
                # Tính diện tích tán cây (m²)
                # Atree = π * rm²
                canopy_area_m2 = math.pi * (radius_m ** 2)
                total_canopy_area += canopy_area_m2
                
                filtered_predictions.append({
                    "class_id": pred.category.id,
                    "class_name": pred.category.name,
                    "confidence": round(float(pred.score.value), 4),
                    "box_pixels_stitched": [int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)],
                    "canopy_width_m": round(wp * gsd, 3),
                    "canopy_height_m": round(hp * gsd, 3),
                    "canopy_radius_m": round(radius_m, 3),
                    "canopy_area_m2": round(canopy_area_m2, 3)
                })
        
        logging.info(f"Filtered down to {len(filtered_predictions)} objects inside the polygon.")
        predicted_count = len(filtered_predictions)
        
        # Tính mật độ tán cây (Cover Density)
        # ρ = (Acover / Aplot) * 100
        cover_density_percent = (total_canopy_area / area_m2 * 100.0) if area_m2 > 0 else 0.0
        
        logging.info(f"Total canopy area (Acover): {total_canopy_area:.2f} m²")
        logging.info(f"Cover density (ρ): {cover_density_percent:.2f}%")

        # 6. Cắt ảnh gốc (RGBA) theo bounding box của polygon để gửi về client
        logging.info("Cropping original stitched image for client...")
        bbox_mask = mask_pil.getbbox()
        if not bbox_mask: raise HTTPException(status_code=400, detail="Polygon outside tile area")
        cropped_image_rgba = stitched_image.crop(bbox_mask)

        # Mã hóa ảnh cắt gốc (input) sang Base64 JPEG
        input_image_rgb = cropped_image_rgba.convert("RGB")
        buffered_input = BytesIO()
        input_image_rgb.save(buffered_input, format="JPEG", quality=90)
        input_image_base64 = base64.b64encode(buffered_input.getvalue()).decode('utf-8')

        # 7. Vẽ overlay: Vẽ các box ĐÃ LỌC lên ảnh ĐÃ CẮT
        image_overlay_bgra = cv2.cvtColor(np.array(cropped_image_rgba), cv2.COLOR_RGBA2BGRA)
        crop_origin_x, crop_origin_y = bbox_mask[0], bbox_mask[1]

        for tree_data in filtered_predictions:
            box_stitched = tree_data['box_pixels_stitched']
            x_min = box_stitched[0] - crop_origin_x
            y_min = box_stitched[1] - crop_origin_y
            x_max = box_stitched[2] - crop_origin_x
            y_max = box_stitched[3] - crop_origin_y
            label = f"{tree_data['class_name']}"
            color = (0, 125, 255); thickness = 2
            font_scale = 0.6; font = cv2.FONT_HERSHEY_SIMPLEX
            draw_rounded_rectangle(image_overlay_bgra, (x_min, y_min), (x_max, y_max), color, thickness, radius=20)
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_bg_ymin = max(y_min + text_height + 10, text_height + 10)
            label_bg_xmin = max(x_min, 0)
            draw_rounded_rectangle(image_overlay_bgra, (label_bg_xmin, label_bg_ymin - text_height - 10), (label_bg_xmin + text_width + 10, label_bg_ymin), color, -1, radius=5)
            cv2.putText(image_overlay_bgra, label, (label_bg_xmin + 5, label_bg_ymin - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 8. Mã hóa ảnh overlay (BGRA) sang Base64 JPEG
        image_overlay_bgr_final = cv2.cvtColor(image_overlay_bgra, cv2.COLOR_BGRA2BGR)
        is_success, buffer = cv2.imencode(".jpg", image_overlay_bgr_final, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not is_success: raise HTTPException(status_code=500, detail="Cannot encode overlay image.")
        overlay_image_base64 = base64.b64encode(buffer).decode("utf-8")
        logging.info("Overlay Base64 image created successfully.")

        # 9. Trả kết quả về app (Bổ sung phenotyping metrics)
        return JSONResponse(content={
            "predicted_count": predicted_count,
            "input_image_base64": input_image_base64,
            "overlay_image_base64": overlay_image_base64,
            "phenotyping_metrics": {
                "gsd_m_per_pixel": round(gsd, 6),
                "plot_area_m2": round(area_m2, 2),
                "plot_area_ha": round(area_ha, 4),
                "total_canopy_area_m2": round(total_canopy_area, 2),
                "cover_density_percent": round(cover_density_percent, 2),
                "avg_canopy_area_m2": round(total_canopy_area / predicted_count, 2) if predicted_count > 0 else 0.0,
                "tree_density_per_ha": round(predicted_count / area_ha, 2) if area_ha > 0 else 0.0
            },
            "trees": filtered_predictions
        })

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logging.error(f"Critical error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

