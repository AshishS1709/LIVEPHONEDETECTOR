from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import time
import asyncio
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

model = YOLO("best.pt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phone Detection API",
    description="Real-time phone detection API for driving violation monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PhoneDetectorAPI:
    def __init__(self):
        """Initialize the phone detection models and settings"""
        try:
            logger.info("🔄 Loading models...")
            
            # Check if model files exist
            model1_path = Path('best.pt')
            model2_path = Path('yolov8n.pt')
            
            if model1_path.exists():
                self.model1 = YOLO('best.pt')
                logger.info("✅ Custom model (best.pt) loaded")
            else:
                logger.warning("⚠️ Custom model (best.pt) not found, using YOLOv8n only")
                self.model1 = None
            
            # YOLOv8n will be downloaded automatically if not present
            self.model2 = YOLO('yolov8n.pt')
            logger.info("✅ YOLOv8n model loaded")
            
            self.confidence_threshold = 0.35
            
            # Statistics tracking
            self.detection_stats = {
                "total_frames": 0,
                "total_detections": 0,
                "total_violations": 0,
                "session_start": datetime.now().isoformat(),
                "last_detection": None
            }
            
            logger.info("✅ Phone Detection API initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise

    def is_in_driving_area(self, box: List[float], frame_shape: tuple) -> bool:
        """Check if detected phone is in the driving area (center region)"""
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Define driving area as center 60% of the frame
        return (frame_width * 0.2 < cx < frame_width * 0.8) and (frame_height * 0.15 < cy < frame_height * 0.85)

    def check_phone_size(self, box: List[float], frame_shape: tuple) -> bool:
        """Validate phone detection based on reasonable size"""
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        box_area = w * h
        frame_area = frame_width * frame_height
        
        # Phone should be between 0.1% and 25% of frame area
        return 0.001 < (box_area / frame_area) < 0.25

    def detect_phones(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect phones in a frame and return detection results"""
        detections = []
        phone_detected = False
        violation_detected = False
        
        # Prepare models to run
        models_to_run = []
        if self.model1 is not None:
            models_to_run.append((self.model1, "Custom Model"))
        models_to_run.append((self.model2, "YOLOv8n"))
        
        # Run detection on available models
        for model, model_name in models_to_run:
            try:
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            class_id = int(box.cls.item())
                            class_name = r.names.get(class_id, "unknown")
                            conf = float(box.conf.item())
                            
                            # Look for phone/cell phone detections
                            if 'phone' in class_name.lower() or 'cell' in class_name.lower():
                                box_coords = box.xyxy[0].cpu().numpy().tolist()
                                
                                # Validate detection
                                if not self.check_phone_size(box_coords, frame.shape):
                                    continue
                                
                                phone_detected = True
                                is_violation = self.is_in_driving_area(box_coords, frame.shape)
                                
                                if is_violation:
                                    violation_detected = True
                                
                                detection = {
                                    "bbox": box_coords,
                                    "confidence": conf,
                                    "class_name": class_name,
                                    "model": model_name,
                                    "is_violation": is_violation,
                                    "timestamp": datetime.now().isoformat()
                                }
                                detections.append(detection)
                                
            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                continue
        
        return {
            "phone_detected": phone_detected,
            "violation_detected": violation_detected,
            "detections": detections,
            "frame_shape": list(frame.shape),
            "processing_timestamp": datetime.now().isoformat()
        }

# Initialize detector
detector = PhoneDetectorAPI()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "service": "Phone Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /detect": "Upload image for phone detection",
            "POST /detect/base64": "Send base64 image for detection",
            "GET /stats": "Get detection statistics",
            "WebSocket /ws/detect": "Live detection stream",
            "GET /health": "Health check"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "custom_model": detector.model1 is not None,
            "yolov8n": True
        },
        "uptime": (datetime.now() - datetime.fromisoformat(detector.detection_stats["session_start"])).total_seconds()
    }

@app.get("/stats")
async def get_stats():
    """Get detection statistics"""
    return {
        "statistics": detector.detection_stats,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_phone_in_image(file: UploadFile = File(...)):
    """Upload an image and get phone detection results"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform detection
        detection_result = detector.detect_phones(frame)
        
        # Update statistics
        detector.detection_stats["total_frames"] += 1
        if detection_result["phone_detected"]:
            detector.detection_stats["total_detections"] += 1
            detector.detection_stats["last_detection"] = datetime.now().isoformat()
        if detection_result["violation_detected"]:
            detector.detection_stats["total_violations"] += 1
        
        return {
            "success": True,
            "detection_result": detection_result,
            "filename": file.filename,
            "file_size": len(contents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in detect_phone_in_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/base64")
async def detect_phone_base64(data: dict):
    """Detect phones in base64 encoded image"""
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64 image
        image_data = data["image"]
        if image_data.startswith('data:image'):
            # Remove data URL prefix if present
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Perform detection
        detection_result = detector.detect_phones(frame)
        
        # Update statistics
        detector.detection_stats["total_frames"] += 1
        if detection_result["phone_detected"]:
            detector.detection_stats["total_detections"] += 1
            detector.detection_stats["last_detection"] = datetime.now().isoformat()
        if detection_result["violation_detected"]:
            detector.detection_stats["total_violations"] += 1
        
        return {
            "success": True,
            "detection_result": detection_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in detect_phone_base64: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):
    """WebSocket endpoint for real-time phone detection"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Extract image data
            if "image" not in request_data:
                await manager.send_personal_message({
                    "error": "Missing image data"
                }, websocket)
                continue
            
            # Decode base64 frame
            image_data = request_data["image"]
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            try:
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await manager.send_personal_message({
                        "error": "Invalid image data"
                    }, websocket)
                    continue
                
                # Perform detection
                detection_result = detector.detect_phones(frame)
                
                # Update statistics
                detector.detection_stats["total_frames"] += 1
                if detection_result["phone_detected"]:
                    detector.detection_stats["total_detections"] += 1
                    detector.detection_stats["last_detection"] = datetime.now().isoformat()
                if detection_result["violation_detected"]:
                    detector.detection_stats["total_violations"] += 1
                
                # Send results back to client
                response = {
                    "success": True,
                    "detection_result": detection_result,
                    "frame_id": request_data.get("frame_id", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                await manager.send_personal_message(response, websocket)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await manager.send_personal_message({
                    "error": f"Processing error: {str(e)}"
                }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/reset-stats")
async def reset_statistics():
    """Reset detection statistics"""
    detector.detection_stats = {
        "total_frames": 0,
        "total_detections": 0,
        "total_violations": 0,
        "session_start": datetime.now().isoformat(),
        "last_detection": None
    }
    
    return {
        "success": True,
        "message": "Statistics reset successfully",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "GET /",
                "GET /health",
                "GET /stats",
                "POST /detect",
                "POST /detect/base64",
                "POST /reset-stats",
                "WebSocket /ws/detect"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")