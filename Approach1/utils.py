import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.modules.block import C3
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.block import SPPF
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.modules.head import Detect
from torch.nn.modules.container import ModuleList
from ultralytics.nn.modules.block import DFL
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.loss import BboxLoss
import cv2


MODEL_PATH = "model/best.pt"

def load_model(device="cpu"):
    safe_classes = [
        Conv2d,
        ConvTranspose2d,
        Sequential,
        DetectionModel,
        Conv,
        BatchNorm2d,
        SiLU,
        C3,
        Bottleneck,
        SPPF,
        MaxPool2d,
        Upsample,
        Concat,
        Detect,
        ModuleList,
        DFL,
        IterableSimpleNamespace,
        v8DetectionLoss,
        BCEWithLogitsLoss,
        TaskAlignedAssigner,
        BboxLoss
    ]
    with torch.serialization.safe_globals(safe_classes):
        return YOLO(MODEL_PATH).to(device)

def detect_players(frame, model, conf=0.3):
    
    """Return list of detections as dicts with bbox, crop, center."""
    
    # Multi-scale detection for better accuracy
    all_detections = []
    
    # Original scale detection
    results = model(frame, conf=conf, verbose=False)[0]
    all_detections.extend(process_detections(results, frame, scale_factor=1.0))
    
    # Slightly scaled versions for better detection
    scales = [0.8, 1.2]  # Try smaller and larger scales
    for scale in scales:
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_frame = cv2.resize(frame, (new_w, new_h))
        
        results_scaled = model(scaled_frame, conf=conf*0.8, verbose=False)[0]  # Lower conf for scaled
        scaled_detections = process_detections(results_scaled, scaled_frame, scale_factor=1.0/scale)
        
        # Scale back bounding boxes to original image size
        for det in scaled_detections:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
            det["center"] = ((det["bbox"][0] + det["bbox"][2])//2, (det["bbox"][1] + det["bbox"][3])//2)
        
        all_detections.extend(scaled_detections)
    
    # Apply NMS to all detections combined
    all_detections = remove_overlapping_detections(all_detections, iou_threshold=0.4)
    
    # Create final player list with crops from original frame
    players = []
    for det in all_detections:
        x1, y1, x2, y2 = det["bbox"]
        # Ensure bbox is within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:  # Valid bbox
            crop = frame[y1:y2, x1:x2].copy()
            players.append({
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "center": ((x1+x2)//2, (y1+y2)//2),
                "confidence": det["confidence"]
            })
    
    return players

def process_detections(results, frame, scale_factor=1.0):
    """Process YOLO detection results."""
    detections = []
    
    # Debug: Print all detected classes occasionally
    if results.boxes is not None:
        all_classes = [int(b.cls) for b in results.boxes]
        if all_classes:
            unique_classes = set(all_classes)
            if len(unique_classes) > 1 and len(detections) == 0:
                print(f"Detected classes: {unique_classes}")
    
    if results.boxes is None:
        return detections
        
    for b in results.boxes:
        cls = int(b.cls)
        # Include multiple classes that might be players
        if cls in [0, 1, 2]:  # Expand classes for better recall              
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf_score = float(b.conf)
            
            # More flexible size filtering
            width, height = x2-x1, y2-y1
            aspect_ratio = height / width if width > 0 else 0
            
            # Adaptive size thresholds based on frame size
            min_width = max(15, frame.shape[1] // 100)   # At least 1% of frame width
            min_height = max(25, frame.shape[0] // 50)   # At least 2% of frame height
            max_width = frame.shape[1] // 3              # At most 1/3 of frame width
            max_height = frame.shape[0] // 2             # At most 1/2 of frame height
            
            # Filter by size and aspect ratio (players are usually taller than wide)
            if (min_width <= width <= max_width and 
                min_height <= height <= max_height and
                1.0 <= aspect_ratio <= 4.0):  # Reasonable aspect ratio for players
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1+x2)//2, (y1+y2)//2),
                    "confidence": conf_score,
                    "area": width * height,
                    "aspect_ratio": aspect_ratio
                })
    
    return detections

def remove_overlapping_detections(detections, iou_threshold=0.3):
    """Remove overlapping detections using simple NMS."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    for i, det in enumerate(detections):
        should_keep = True
        for kept_det in keep:
            if calculate_iou(det["bbox"], kept_det["bbox"]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(det)
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
