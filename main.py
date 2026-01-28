# -*- coding: utf-8 -*-
"""
FastAPI application logic for Palm Tree Counting API 
Feature: Server-side Tile Processing + Spatial Analysis (Smart K-NN) + Agri-Intelligence + GPS Extraction
Language: English Responses
"""
import io
import os
import logging
import numpy as np
import cv2
import torch
import base64
import math
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pydantic import BaseModel
from typing import List


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("PalmTreeAPI")

MODEL_PATH = os.getenv("MODEL_PATH", "./best_palmtree.pt")
SAHI_SLICE_HEIGHT = int(os.getenv("SAHI_SLICE_HEIGHT", 640))
SAHI_SLICE_WIDTH = int(os.getenv("SAHI_SLICE_WIDTH", 640))
SAHI_OVERLAP_HEIGHT_RATIO = float(os.getenv("SAHI_OVERLAP_HEIGHT_RATIO", 0.2))
SAHI_OVERLAP_WIDTH_RATIO = float(os.getenv("SAHI_OVERLAP_WIDTH_RATIO", 0.2))
SAHI_CONFIDENCE_THRESHOLD = float(os.getenv("SAHI_CONFIDENCE_THRESHOLD", 0.35))
TARGET_ZOOM = 20
TILE_SIZE = 256
GOOGLE_TILE_URL_TEMPLATE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
MAX_TILES = 225

AGRI_SPECS = {
    "OPTIMAL_DENSITY_MIN": 100,
    "OPTIMAL_DENSITY_MAX": 125,
    "SPACING_SPARSE_THRESHOLD_M": 12.0,
    "SPACING_CROWDED_THRESHOLD_M": 6.0,
    "YIELD_AVG_PER_TREE_KG": 48.0,
    "YIELD_HIGH_PER_TREE_KG": 85.0
}


def calculate_gsd(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def calculate_polygon_area_m2(polygon_latlng):
    EARTH_RADIUS = 6378137.0
    if len(polygon_latlng) < 3: return 0.0
    coords = [(math.radians(p['latitude']), math.radians(p['longitude'])) for p in polygon_latlng]
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += (coords[j][1] - coords[i][1]) * (2 + math.sin(coords[i][0]) + math.sin(coords[j][0]))
    return abs(area * EARTH_RADIUS * EARTH_RADIUS / 2.0)

def latLngToWorldXY(lat, lng, zoom):
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    y = math.log((1 + siny) / (1 - siny))
    worldY = (TILE_SIZE * (1 << zoom) * (1 - y / (2 * math.pi))) / 2
    worldX = TILE_SIZE * (1 << zoom) * (lng / 360 + 0.5)
    return {"x": worldX, "y": worldY}

def worldXYToLatLng(worldX, worldY, zoom):
    """Chuyển đổi ngược từ World XY sang GPS LatLng"""
    n = 1 << zoom
    lng_deg = (worldX / (TILE_SIZE * n) - 0.5) * 360.0
    y_norm = 0.5 - (worldY / (TILE_SIZE * n))
    lat_rad = 2 * math.atan(math.exp(2 * math.pi * y_norm)) - math.pi / 2
    lat_deg = math.degrees(lat_rad)
    return {"latitude": lat_deg, "longitude": lng_deg}

def getTileBoundingBox(points, zoom):
    minWorldX, maxWorldX = float('inf'), float('-inf')
    minWorldY, maxWorldY = float('inf'), float('-inf')
    for p in points:
        worldP = latLngToWorldXY(p["latitude"], p["longitude"], zoom)
        minWorldX = min(minWorldX, worldP["x"])
        maxWorldX = max(maxWorldX, worldP["x"])
        minWorldY = min(minWorldY, worldP["y"])
        maxWorldY = max(maxWorldY, worldP["y"])
    
    return {
        "minTileX": math.floor(minWorldX / TILE_SIZE),
        "maxTileX": math.floor(maxWorldX / TILE_SIZE),
        "minTileY": math.floor(minWorldY / TILE_SIZE),
        "maxTileY": math.floor(maxWorldY / TILE_SIZE)
    }

def analyze_spatial_distribution(tree_list):
    """
    Phân tích không gian (K-NN):
    - Crowded: Dựa vào cây gần nhất.
    - Sparse: Dựa vào trung bình 3 cây gần nhất.
    """
    logger.info(">>> [Spatial] Starting SMART spatial analysis...")
    n_trees = len(tree_list)
    
    if n_trees < 2:
        if n_trees == 1: tree_list[0]['spacing_status'] = "SPARSE"
        return tree_list, []

    centers = np.array([[t['center_x'], t['center_y']] for t in tree_list])
    
    # Ma trận khoảng cách
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1).astype(float) 
    np.fill_diagonal(dist_sq, np.inf)
    
    dist_matrix_px = np.sqrt(dist_sq)
    sorted_dists_px = np.sort(dist_matrix_px, axis=1)

    warnings = []
    k_neighbors = min(n_trees - 1, 3) 
    
    for i, tree in enumerate(tree_list):
        gsd = tree['gsd_ref']
        nearest_dist_m = sorted_dists_px[i, 0] * gsd
        
        closest_k_dists_px = sorted_dists_px[i, :k_neighbors]
        avg_k_dist_m = np.mean(closest_k_dists_px) * gsd
        
        tree['nearest_neighbor_dist_m'] = round(nearest_dist_m, 2)
        tree['avg_3_neighbor_dist_m'] = round(avg_k_dist_m, 2)
        
        if nearest_dist_m < AGRI_SPECS["SPACING_CROWDED_THRESHOLD_M"]:
            tree['spacing_status'] = "CROWDED"
        elif avg_k_dist_m > AGRI_SPECS["SPACING_SPARSE_THRESHOLD_M"]:
            tree['spacing_status'] = "SPARSE"
            # English Warning
            warnings.append(f"Tree ID {tree['id']} (Gap Detected): Avg 3-neighbors distance is {avg_k_dist_m:.1f}m")
        else:
            tree['spacing_status'] = "OPTIMAL"
            
    return tree_list, warnings

# --- Load Model ---
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"
detection_model = None
try:
    if os.path.exists(MODEL_PATH):
        logger.info(f"INIT: Loading model from '{MODEL_PATH}' on '{device}'...")
        detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=MODEL_PATH, confidence_threshold=SAHI_CONFIDENCE_THRESHOLD, device=device)
        logger.info("INIT: ✅ Model loaded successfully!")
    else:
        logger.error(f"INIT: ❌ Model not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"INIT: ❌ Error loading model: {e}")

# --- App ---
app = FastAPI(title="Palm Tree Agri-Intelligence API", version="6.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class LatLng(BaseModel): latitude: float; longitude: float
class PredictRequest(BaseModel): 
    polygon: List[LatLng]
    area_m2: float | None = None
    area_ha: float | None = None

@app.post("/predict")
async def predict_from_polygon(data: PredictRequest = Body(...)):
    logger.info("="*50)
    logger.info("NEW REQUEST RECEIVED: /predict")
    
    if detection_model is None: raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        polygon_latlng = [p.dict() for p in data.polygon]
        if len(polygon_latlng) < 3: raise HTTPException(status_code=400, detail="Invalid polygon")

        # 0. Calculate GIS & Area
        avg_lat = sum(p["latitude"] for p in polygon_latlng) / len(polygon_latlng)
        gsd = calculate_gsd(avg_lat, TARGET_ZOOM)
        
        if data.area_m2 is None:
            area_m2 = calculate_polygon_area_m2(polygon_latlng)
            area_ha = area_m2 / 10000.0
        else:
            area_m2 = data.area_m2
            area_ha = data.area_ha if data.area_ha else area_m2 / 10000.0
            
        logger.info(f"1. GIS Info: Avg Lat={avg_lat:.5f}, GSD={gsd:.5f} m/px")

        # 1. Tile Download & Stitching
        tileBounds = getTileBoundingBox(polygon_latlng, TARGET_ZOOM)
        minTileX, maxTileX = tileBounds["minTileX"], tileBounds["maxTileX"]
        minTileY, maxTileY = tileBounds["minTileY"], tileBounds["maxTileY"]
        
        tiles_to_download = [(x, y) for x in range(minTileX, maxTileX + 1) for y in range(minTileY, maxTileY + 1)]
        num_tiles = len(tiles_to_download)
        
        if num_tiles > MAX_TILES: 
            raise HTTPException(status_code=400, detail=f"Area too large ({num_tiles} tiles). Max {MAX_TILES}.")
        
        stitch_width = (maxTileX - minTileX + 1) * TILE_SIZE
        stitch_height = (maxTileY - minTileY + 1) * TILE_SIZE
        stitched_image = Image.new("RGBA", (stitch_width, stitch_height))
        stitchOriginWorldX = minTileX * TILE_SIZE
        stitchOriginWorldY = minTileY * TILE_SIZE
        
        session = requests.Session()
        for x, y in tiles_to_download:
            url = GOOGLE_TILE_URL_TEMPLATE.format(x=x, y=y, z=TARGET_ZOOM)
            try:
                response = session.get(url, timeout=5)
                if response.status_code == 200:
                    tile_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
                    stitched_image.paste(tile_image, ((x - minTileX) * TILE_SIZE, (y - minTileY) * TILE_SIZE))
            except Exception: pass

        # 2. Run SAHI
        logger.info("4. AI Detection: Starting SAHI...")
        stitched_image_rgb_np = np.array(stitched_image.convert("RGB"))
        result = get_sliced_prediction(
            stitched_image_rgb_np, detection_model,
            slice_height=SAHI_SLICE_HEIGHT, slice_width=SAHI_SLICE_WIDTH,
            overlap_height_ratio=SAHI_OVERLAP_HEIGHT_RATIO, overlap_width_ratio=SAHI_OVERLAP_WIDTH_RATIO
        )
        all_predictions = result.object_prediction_list

        # 3. Masking
        polygon_px = []
        for p in polygon_latlng:
            worldP = latLngToWorldXY(p["latitude"], p["longitude"], TARGET_ZOOM)
            polygon_px.append((worldP["x"] - stitchOriginWorldX, worldP["y"] - stitchOriginWorldY))
        
        mask_pil = Image.new("L", (stitch_width, stitch_height), 0)
        ImageDraw.Draw(mask_pil).polygon(polygon_px, outline=1, fill=1)
        mask_np = np.array(mask_pil)

        # 4. Process Trees & GET LOCATION
        final_tree_list = []
        min_radius_m = float('inf')
        max_radius_m = float('-inf')
        tree_counter = 1

        for pred in all_predictions:
            box = pred.bbox
            center_x = int((box.minx + box.maxx) / 2)
            center_y = int((box.miny + box.maxy) / 2)

            if 0 <= center_y < mask_np.shape[0] and 0 <= center_x < mask_np.shape[1] and mask_np[center_y, center_x] > 0:
                wp = box.maxx - box.minx
                hp = box.maxy - box.miny
                radius_m = ((wp + hp) / 4.0) * gsd
                canopy_area_m2 = math.pi * (radius_m ** 2)
                
                if radius_m < min_radius_m: min_radius_m = radius_m
                if radius_m > max_radius_m: max_radius_m = radius_m

                # --- Tính tọa độ GPS ---
                abs_world_x = stitchOriginWorldX + center_x
                abs_world_y = stitchOriginWorldY + center_y
                gps_loc = worldXYToLatLng(abs_world_x, abs_world_y, TARGET_ZOOM)
                # -----------------------

                final_tree_list.append({
                    "id": tree_counter,
                    "location": {
                        "latitude": round(gps_loc["latitude"], 7),
                        "longitude": round(gps_loc["longitude"], 7)
                    },
                    "center_x": center_x,
                    "center_y": center_y,
                    "box_pixels_stitched": [int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)],
                    "confidence": round(float(pred.score.value), 4),
                    "canopy_radius_m": round(radius_m, 2),
                    "canopy_area_m2": round(canopy_area_m2, 2),
                    "gsd_ref": gsd
                })
                tree_counter += 1

        predicted_count = len(final_tree_list)
        if predicted_count == 0: 
            min_radius_m = 0.0
            max_radius_m = 0.0
            
        logger.info(f"5. Filtering: {predicted_count} trees detected.")

        # 5. Spatial Analysis
        final_tree_list, spatial_warnings = analyze_spatial_distribution(final_tree_list)

        # 6. Agri-Intelligence (TRANSLATED TO ENGLISH)
        current_density = (predicted_count / area_ha) if area_ha > 0 else 0
        
        recommendation_action = "MAINTAIN"
        recommendation_msg = "Planting density is within the optimal range."
        
        if current_density < AGRI_SPECS["OPTIMAL_DENSITY_MIN"]:
            missing = int((AGRI_SPECS["OPTIMAL_DENSITY_MIN"] * area_ha) - predicted_count)
            recommendation_action = "PLANT_MORE"
            recommendation_msg = f"Low density detected. You can plant approximately {missing} additional trees."
        elif current_density > AGRI_SPECS["OPTIMAL_DENSITY_MAX"]:
            excess = int(predicted_count - (AGRI_SPECS["OPTIMAL_DENSITY_MAX"] * area_ha))
            recommendation_action = "THINNING"
            recommendation_msg = f"High density detected. Consider thinning {excess} trees."

        yield_forecast = {
            "min_ton": round((predicted_count * AGRI_SPECS["YIELD_AVG_PER_TREE_KG"]) / 1000, 2),
            "max_ton": round((predicted_count * AGRI_SPECS["YIELD_HIGH_PER_TREE_KG"]) / 1000, 2),
        }

        # 7. Image Processing
        logger.info("7. Image: Generating Base64 overlay...")
        bbox_mask = mask_pil.getbbox()
        if not bbox_mask: raise HTTPException(status_code=400, detail="Empty area")
        cropped_rgba = stitched_image.crop(bbox_mask)
        
        buf_in = BytesIO()
        cropped_rgba.convert("RGB").save(buf_in, format="JPEG", quality=90)
        input_b64 = base64.b64encode(buf_in.getvalue()).decode('utf-8')

        overlay_bgra = cv2.cvtColor(np.array(cropped_rgba), cv2.COLOR_RGBA2BGRA)
        crop_origin_x, crop_origin_y = bbox_mask[0], bbox_mask[1]

        for tree in final_tree_list:
            box = tree['box_pixels_stitched']
            x1, y1 = box[0] - crop_origin_x, box[1] - crop_origin_y
            x2, y2 = box[2] - crop_origin_x, box[3] - crop_origin_y
            
            center_x_px = int((x1 + x2) / 2)
            center_y_px = int((y1 + y2) / 2)
            width_px = x2 - x1
            height_px = y2 - y1
            radius_px = int((width_px + height_px) / 4)

            color = (0, 255, 0)
            if tree.get('spacing_status') == 'CROWDED': color = (0, 0, 255)
            elif tree.get('spacing_status') == 'SPARSE': color = (0, 215, 255)
            
            cv2.circle(overlay_bgra, (center_x_px, center_y_px), radius_px, color, 2)
            
            text_pos_x = center_x_px - 10
            text_pos_y = center_y_px - radius_px - 5
            cv2.putText(overlay_bgra, str(tree['id']), (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        final_bgr = cv2.cvtColor(overlay_bgra, cv2.COLOR_BGRA2BGR)
        overlay_b64 = base64.b64encode(cv2.imencode(".jpg", final_bgr)[1]).decode("utf-8")

        logger.info("✅ SUCCESS: Response ready.")
        return JSONResponse(content={
            "summary": {
                "count": predicted_count,
                "area_ha": round(area_ha, 4),
                "density_per_ha": round(current_density, 2),
                "radius_stats": {
                    "min_m": round(min_radius_m, 2),
                    "max_m": round(max_radius_m, 2)
                }
            },
            "agri_intelligence": {
                "action": recommendation_action,
                "message": recommendation_msg,
                "yield_forecast_ton": yield_forecast,
                "spatial_warnings": spatial_warnings[:10]
            },
            "trees": final_tree_list,
            "input_image_base64": input_b64,
            "overlay_image_base64": overlay_b64
        })

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)