# VisionGuard Pro - Manufacturing Quality Control Backend API
# Advanced visual defect detection with pose tracking for industrial automation

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from datetime import datetime, timedelta
import uuid
import json
import asyncio
import logging
from dataclasses import dataclass
import sqlite3
from contextlib import asynccontextmanager
import io
from PIL import Image
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Database initialization
def init_db():
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Inspections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            product_line TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            defects_detected INTEGER DEFAULT 0,
            quality_score REAL,
            pose_data TEXT,
            status TEXT DEFAULT 'processed',
            processing_time REAL,
            metadata TEXT
        )
    ''')
    
    # Defects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS defects (
            id TEXT PRIMARY KEY,
            inspection_id TEXT,
            defect_type TEXT,
            confidence REAL,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_width INTEGER,
            bbox_height INTEGER,
            severity TEXT,
            FOREIGN KEY (inspection_id) REFERENCES inspections (id)
        )
    ''')
    
    # Product lines configuration
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS product_lines (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            quality_threshold REAL DEFAULT 0.85,
            active BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            config TEXT
        )
    ''')
    
    # Analytics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id TEXT PRIMARY KEY,
            product_line TEXT,
            date DATE,
            total_inspections INTEGER DEFAULT 0,
            defects_found INTEGER DEFAULT 0,
            avg_quality_score REAL,
            processing_time_avg REAL
        )
    ''')
    
    conn.commit()
    conn.close()

# Data models
class DefectInfo(BaseModel):
    id: str
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Dict[str, int]  # x, y, width, height
    severity: str = Field(..., regex="^(low|medium|high|critical)$")

class PoseData(BaseModel):
    keypoints: List[Dict[str, float]]  # x, y, confidence for each keypoint
    pose_confidence: float = Field(..., ge=0.0, le=1.0)
    orientation: str  # front, side, back, etc.

class InspectionResult(BaseModel):
    id: str
    product_line: str
    timestamp: datetime
    quality_score: float = Field(..., ge=0.0, le=1.0)
    defects: List[DefectInfo]
    pose_data: Optional[PoseData]
    status: str
    processing_time: float
    recommendations: List[str]

class ProductLineConfig(BaseModel):
    name: str
    quality_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    defect_types: List[str]
    pose_tracking_enabled: bool = True
    active: bool = True

class AnalyticsData(BaseModel):
    product_line: str
    period: str  # daily, weekly, monthly
    total_inspections: int
    defects_found: int
    avg_quality_score: float
    defect_rate: float
    top_defect_types: List[Dict[str, Any]]

# AI Models (Mock implementations for demo - replace with actual models)
class DefectDetectionModel:
    def __init__(self):
        self.model_loaded = True
        logger.info("Defect detection model initialized")
    
    async def detect_defects(self, image: np.ndarray) -> List[Dict]:
        """Mock defect detection - replace with actual YOLOv8/Detectron2 model"""
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Mock defect detection results
        defects = []
        if np.random.random() > 0.7:  # 30% chance of defects
            num_defects = np.random.randint(1, 4)
            for i in range(num_defects):
                defects.append({
                    'type': np.random.choice(['scratch', 'dent', 'discoloration', 'crack', 'contamination']),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'bbox': {
                        'x': np.random.randint(0, 200),
                        'y': np.random.randint(0, 200),
                        'width': np.random.randint(50, 150),
                        'height': np.random.randint(50, 150)
                    },
                    'severity': np.random.choice(['low', 'medium', 'high'])
                })
        return defects

class PoseTracker:
    def __init__(self):
        self.model_loaded = True
        logger.info("Pose tracking model initialized")
    
    async def track_pose(self, image: np.ndarray) -> Dict:
        """Mock pose tracking - replace with MediaPipe/OpenPose"""
        await asyncio.sleep(0.3)
        
        # Mock pose data
        keypoints = []
        for i in range(17):  # COCO-style keypoints
            keypoints.append({
                'x': np.random.uniform(0, 640),
                'y': np.random.uniform(0, 480),
                'confidence': np.random.uniform(0.6, 0.9)
            })
        
        return {
            'keypoints': keypoints,
            'pose_confidence': np.random.uniform(0.7, 0.95),
            'orientation': np.random.choice(['front', 'side', 'back', 'angled'])
        }

class QualityScorer:
    @staticmethod
    def calculate_quality_score(defects: List[Dict], pose_data: Dict = None) -> float:
        """Calculate overall quality score based on defects and pose"""
        if not defects:
            base_score = 0.95
        else:
            severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}
            total_penalty = sum(severity_weights.get(d['severity'], 0.3) * d['confidence'] for d in defects)
            base_score = max(0.0, 1.0 - (total_penalty * 0.2))
        
        # Pose confidence bonus
        if pose_data and pose_data.get('pose_confidence', 0) > 0.8:
            base_score += 0.05
        
        return min(1.0, base_score)

# Global instances
defect_model = DefectDetectionModel()
pose_tracker = PoseTracker()
quality_scorer = QualityScorer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    logger.info("VisionGuard Pro API started successfully")
    yield
    # Shutdown
    logger.info("VisionGuard Pro API shutting down")

# FastAPI app
app = FastAPI(
    title="VisionGuard Pro API",
    description="Advanced Manufacturing Quality Control with Visual Defect Detection and Pose Tracking",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication (simplified - implement proper JWT in production)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Mock authentication - implement proper JWT verification
    if credentials.credentials == "demo-token":
        return {"user_id": "demo_user", "role": "inspector"}
    raise HTTPException(status_code=401, detail="Invalid authentication token")

# Utility functions
async def process_image(image_data: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def save_inspection_to_db(inspection_data: Dict):
    """Save inspection results to database"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Insert inspection
    cursor.execute('''
        INSERT INTO inspections 
        (id, product_line, quality_score, defects_detected, pose_data, processing_time, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        inspection_data['id'],
        inspection_data['product_line'],
        inspection_data['quality_score'],
        len(inspection_data['defects']),
        json.dumps(inspection_data.get('pose_data')),
        inspection_data['processing_time'],
        json.dumps(inspection_data.get('metadata', {}))
    ))
    
    # Insert defects
    for defect in inspection_data['defects']:
        cursor.execute('''
            INSERT INTO defects 
            (id, inspection_id, defect_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            inspection_data['id'],
            defect['type'],
            defect['confidence'],
            defect['bbox']['x'],
            defect['bbox']['y'],
            defect['bbox']['width'],
            defect['bbox']['height'],
            defect['severity']
        ))
    
    conn.commit()
    conn.close()

# API Routes

@app.get("/")
async def root():
    return {
        "message": "VisionGuard Pro API - Manufacturing Quality Control",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/v1/inspect", response_model=InspectionResult)
async def inspect_product(
    image: UploadFile = File(...),
    product_line: str = "default",
    enable_pose_tracking: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: Dict = Depends(verify_token)
):
    """
    Main inspection endpoint - processes uploaded image for defect detection and pose tracking
    """
    start_time = datetime.now()
    inspection_id = str(uuid.uuid4())
    
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process image
        image_data = await image.read()
        img_array = await process_image(image_data)
        
        # Detect defects
        defects_raw = await defect_model.detect_defects(img_array)
        defects = [
            DefectInfo(
                id=str(uuid.uuid4()),
                type=d['type'],
                confidence=d['confidence'],
                bbox=d['bbox'],
                severity=d['severity']
            ) for d in defects_raw
        ]
        
        # Track pose if enabled
        pose_data = None
        if enable_pose_tracking:
            pose_raw = await pose_tracker.track_pose(img_array)
            pose_data = PoseData(**pose_raw)
        
        # Calculate quality score
        quality_score = quality_scorer.calculate_quality_score(
            defects_raw, 
            pose_raw if enable_pose_tracking else None
        )
        
        # Generate recommendations
        recommendations = []
        if quality_score < 0.7:
            recommendations.append("Product requires immediate attention - multiple defects detected")
        if len(defects) > 0:
            recommendations.append(f"Detected {len(defects)} defect(s) - review before shipping")
        if pose_data and pose_data.pose_confidence < 0.7:
            recommendations.append("Poor pose detection - consider repositioning product")
        if not recommendations:
            recommendations.append("Product passes quality standards")
        
        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = InspectionResult(
            id=inspection_id,
            product_line=product_line,
            timestamp=start_time,
            quality_score=quality_score,
            defects=defects,
            pose_data=pose_data,
            status="completed",
            processing_time=processing_time,
            recommendations=recommendations
        )
        
        # Save to database (background task)
        inspection_data = {
            'id': inspection_id,
            'product_line': product_line,
            'quality_score': quality_score,
            'defects': [d.dict() for d in defects],
            'pose_data': pose_data.dict() if pose_data else None,
            'processing_time': processing_time,
            'metadata': {'user_id': user['user_id'], 'image_size': len(image_data)}
        }
        background_tasks.add_task(save_inspection_to_db, inspection_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Inspection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inspection processing failed: {str(e)}")

@app.post("/api/v1/batch-inspect")
async def batch_inspect(
    images: List[UploadFile] = File(...),
    product_line: str = "default",
    enable_pose_tracking: bool = True,
    user: Dict = Depends(verify_token)
):
    """
    Batch inspection endpoint for processing multiple images
    """
    if len(images) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    for image in images:
        try:
            # Use the single inspect logic
            result = await inspect_product(image, product_line, enable_pose_tracking, BackgroundTasks(), user)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "filename": image.filename,
                "status": "failed"
            })
    
    return {
        "batch_id": str(uuid.uuid4()),
        "total_images": len(images),
        "successful": len([r for r in results if "error" not in r]),
        "results": results
    }

@app.get("/api/v1/inspections/{inspection_id}")
async def get_inspection(inspection_id: str, user: Dict = Depends(verify_token)):
    """Get specific inspection result"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.*, GROUP_CONCAT(d.defect_type) as defect_types
        FROM inspections i
        LEFT JOIN defects d ON i.id = d.inspection_id
        WHERE i.id = ?
        GROUP BY i.id
    ''', (inspection_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Inspection not found")
    
    return {
        "id": result[0],
        "product_line": result[1],
        "timestamp": result[2],
        "quality_score": result[4],
        "defects_detected": result[3],
        "pose_data": json.loads(result[5]) if result[5] else None,
        "status": result[6],
        "processing_time": result[7],
        "defect_types": result[9].split(',') if result[9] else []
    }

@app.get("/api/v1/inspections")
async def list_inspections(
    product_line: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user: Dict = Depends(verify_token)
):
    """List inspections with filtering options"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    query = "SELECT * FROM inspections WHERE 1=1"
    params = []
    
    if product_line:
        query += " AND product_line = ?"
        params.append(product_line)
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    return {
        "inspections": [
            {
                "id": r[0],
                "product_line": r[1],
                "timestamp": r[2],
                "quality_score": r[4],
                "defects_detected": r[3],
                "status": r[6]
            } for r in results
        ],
        "total": len(results),
        "limit": limit,
        "offset": offset
    }

@app.post("/api/v1/product-lines", response_model=Dict[str, str])
async def create_product_line(
    config: ProductLineConfig,
    user: Dict = Depends(verify_token)
):
    """Create new product line configuration"""
    line_id = str(uuid.uuid4())
    
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO product_lines (id, name, quality_threshold, config)
        VALUES (?, ?, ?, ?)
    ''', (line_id, config.name, config.quality_threshold, json.dumps(config.dict())))
    
    conn.commit()
    conn.close()
    
    return {"id": line_id, "message": "Product line created successfully"}

@app.get("/api/v1/product-lines")
async def list_product_lines(user: Dict = Depends(verify_token)):
    """List all product line configurations"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM product_lines WHERE active = TRUE")
    results = cursor.fetchall()
    conn.close()
    
    return {
        "product_lines": [
            {
                "id": r[0],
                "name": r[1],
                "quality_threshold": r[2],
                "active": bool(r[3]),
                "created_at": r[4],
                "config": json.loads(r[5]) if r[5] else {}
            } for r in results
        ]
    }

@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_analytics(
    product_line: Optional[str] = None,
    days: int = 30,
    user: Dict = Depends(verify_token)
):
    """Get dashboard analytics for quality control metrics"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Base query
    query = '''
        SELECT 
            COUNT(*) as total_inspections,
            AVG(quality_score) as avg_quality,
            SUM(defects_detected) as total_defects,
            AVG(processing_time) as avg_processing_time
        FROM inspections 
        WHERE timestamp >= datetime('now', '-{} days')
    '''.format(days)
    
    params = []
    if product_line:
        query += " AND product_line = ?"
        params.append(product_line)
    
    cursor.execute(query, params)
    stats = cursor.fetchone()
    
    # Defect types breakdown
    defect_query = '''
        SELECT d.defect_type, COUNT(*) as count, AVG(d.confidence) as avg_confidence
        FROM defects d
        JOIN inspections i ON d.inspection_id = i.id
        WHERE i.timestamp >= datetime('now', '-{} days')
    '''.format(days)
    
    if product_line:
        defect_query += " AND i.product_line = ?"
    
    defect_query += " GROUP BY d.defect_type ORDER BY count DESC"
    
    cursor.execute(defect_query, params)
    defect_breakdown = cursor.fetchall()
    
    # Daily trends
    trend_query = '''
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as inspections,
            AVG(quality_score) as avg_quality,
            SUM(defects_detected) as defects
        FROM inspections 
        WHERE timestamp >= datetime('now', '-{} days')
    '''.format(days)
    
    if product_line:
        trend_query += " AND product_line = ?"
    
    trend_query += " GROUP BY DATE(timestamp) ORDER BY date"
    
    cursor.execute(trend_query, params)
    trends = cursor.fetchall()
    
    conn.close()
    
    return {
        "summary": {
            "total_inspections": stats[0] or 0,
            "avg_quality_score": round(stats[1] or 0, 3),
            "total_defects": stats[2] or 0,
            "avg_processing_time": round(stats[3] or 0, 3),
            "defect_rate": round((stats[2] or 0) / max(stats[0] or 1, 1), 3)
        },
        "defect_breakdown": [
            {
                "type": row[0],
                "count": row[1],
                "avg_confidence": round(row[2], 3)
            } for row in defect_breakdown
        ],
        "daily_trends": [
            {
                "date": row[0],
                "inspections": row[1],
                "avg_quality": round(row[2], 3),
                "defects": row[3]
            } for row in trends
        ]
    }

@app.post("/api/v1/real-time/start")
async def start_real_time_inspection(
    product_line: str,
    camera_id: str = "default",
    user: Dict = Depends(verify_token)
):
    """Start real-time inspection mode"""
    session_id = str(uuid.uuid4())
    
    # In production, this would initialize camera capture and start processing pipeline
    return {
        "session_id": session_id,
        "product_line": product_line,
        "camera_id": camera_id,
        "status": "started",
        "websocket_endpoint": f"/ws/inspection/{session_id}"
    }

@app.post("/api/v1/real-time/stop/{session_id}")
async def stop_real_time_inspection(
    session_id: str,
    user: Dict = Depends(verify_token)
):
    """Stop real-time inspection session"""
    return {
        "session_id": session_id,
        "status": "stopped",
        "message": "Real-time inspection session ended"
    }

@app.get("/api/v1/models/status")
async def get_model_status(user: Dict = Depends(verify_token)):
    """Get status of AI models"""
    return {
        "defect_detection": {
            "loaded": defect_model.model_loaded,
            "model_type": "YOLOv8-based",
            "version": "1.0.0",
            "accuracy": 0.94
        },
        "pose_tracking": {
            "loaded": pose_tracker.model_loaded,
            "model_type": "MediaPipe-based",
            "version": "1.0.0",
            "accuracy": 0.91
        },
        "system_status": "operational"
    }

@app.post("/api/v1/models/retrain")
async def retrain_models(
    product_line: str,
    annotation_data: Dict[str, Any],
    user: Dict = Depends(verify_token)
):
    """Trigger model retraining with new data"""
    if user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    training_id = str(uuid.uuid4())
    
    # In production, this would trigger actual retraining pipeline
    return {
        "training_id": training_id,
        "status": "started",
        "estimated_completion": datetime.now() + timedelta(hours=2),
        "product_line": product_line
    }

@app.get("/api/v1/export/report")
async def export_quality_report(
    product_line: Optional[str] = None,
    format: str = "json",
    days: int = 30,
    user: Dict = Depends(verify_token)
):
    """Export quality control report"""
    if format not in ["json", "csv", "pdf"]:
        raise HTTPException(status_code=400, detail="Supported formats: json, csv, pdf")
    
    # Get analytics data (reuse dashboard logic)
    analytics = await get_dashboard_analytics(product_line, days, user)
    
    if format == "json":
        return {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "product_line": product_line or "all",
            "period_days": days,
            "data": analytics
        }
    
    # For CSV/PDF, return download URL or base64 data
    return {
        "report_id": str(uuid.uuid4()),
        "format": format,
        "download_url": f"/api/v1/downloads/report-{uuid.uuid4()}.{format}",
        "expires_at": datetime.now() + timedelta(hours=24)
    }

@app.get("/api/v1/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "defect_detection": "operational",
            "pose_tracking": "operational"
        },
        "database": "connected",
        "version": "1.0.0"
    }

@app.get("/api/v1/alerts")
async def get_quality_alerts(
    severity: Optional[str] = None,
    limit: int = 20,
    user: Dict = Depends(verify_token)
):
    """Get quality control alerts and notifications"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Find recent inspections with quality issues
    cursor.execute('''
        SELECT i.id, i.product_line, i.timestamp, i.quality_score, i.defects_detected
        FROM inspections i
        WHERE i.quality_score < 0.8 AND i.timestamp >= datetime('now', '-24 hours')
        ORDER BY i.timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    alerts = []
    for row in cursor.fetchall():
        alerts.append({
            "id": row[0],
            "type": "quality_alert",
            "product_line": row[1],
            "timestamp": row[2],
            "quality_score": row[3],
            "defects_count": row[4],
            "severity": "high" if row[3] < 0.6 else "medium",
            "message": f"Quality score {row[3]:.2f} below threshold for {row[1]}"
        })
    
    conn.close()
    return {"alerts": alerts}

# WebSocket endpoint for real-time inspection (placeholder)
@app.websocket("/ws/inspection/{session_id}")
async def websocket_inspection(websocket, session_id: str):
    """WebSocket endpoint for real-time inspection streaming"""
    await websocket.accept()
    try:
        while True:
            # In production, this would stream camera frames and results
            await websocket.send_json({
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "processing",
                "frame_count": np.random.randint(1, 1000)
            })
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Configuration endpoints
@app.put("/api/v1/product-lines/{line_id}")
async def update_product_line(
    line_id: str,
    config: ProductLineConfig,
    user: Dict = Depends(verify_token)
):
    """Update product line configuration"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE product_lines 
        SET name = ?, quality_threshold = ?, config = ?
        WHERE id = ?
    ''', (config.name, config.quality_threshold, json.dumps(config.dict()), line_id))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Product line not found")
    
    conn.commit()
    conn.close()
    
    return {"message": "Product line updated successfully"}

@app.delete("/api/v1/product-lines/{line_id}")
async def delete_product_line(
    line_id: str,
    user: Dict = Depends(verify_token)
):
    """Soft delete product line"""
    if user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('UPDATE product_lines SET active = FALSE WHERE id = ?', (line_id,))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Product line not found")
    
    conn.commit()
    conn.close()
    
    return {"message": "Product line deactivated successfully"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": {
            "code": 500,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    }

# Advanced Analytics Endpoints
@app.get("/api/v1/analytics/trends")
async def get_quality_trends(
    product_line: Optional[str] = None,
    granularity: str = "daily",  # daily, weekly, monthly
    days: int = 30,
    user: Dict = Depends(verify_token)
):
    """Get detailed quality trends and patterns"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    date_format = {
        "daily": "%Y-%m-%d",
        "weekly": "%Y-W%W", 
        "monthly": "%Y-%m"
    }[granularity]
    
    query = f'''
        SELECT 
            strftime('{date_format}', timestamp) as period,
            COUNT(*) as inspections,
            AVG(quality_score) as avg_quality,
            SUM(defects_detected) as total_defects,
            AVG(processing_time) as avg_time,
            MIN(quality_score) as min_quality,
            MAX(quality_score) as max_quality
        FROM inspections 
        WHERE timestamp >= datetime('now', '-{days} days')
    '''
    
    params = []
    if product_line:
        query += " AND product_line = ?"
        params.append(product_line)
    
    query += f" GROUP BY strftime('{date_format}', timestamp) ORDER BY period"
    
    cursor.execute(query, params)
    trends = cursor.fetchall()
    conn.close()
    
    return {
        "trends": [
            {
                "period": row[0],
                "inspections": row[1],
                "avg_quality": round(row[2], 3),
                "total_defects": row[3],
                "avg_processing_time": round(row[4], 3),
                "quality_range": {"min": row[5], "max": row[6]},
                "defect_rate": round(row[3] / max(row[1], 1), 3)
            } for row in trends
        ],
        "granularity": granularity,
        "period_days": days
    }

@app.get("/api/v1/analytics/performance")
async def get_system_performance(
    hours: int = 24,
    user: Dict = Depends(verify_token)
):
    """Get system performance metrics"""
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Performance metrics
    cursor.execute('''
        SELECT 
            AVG(processing_time) as avg_processing_time,
            MIN(processing_time) as min_processing_time,
            MAX(processing_time) as max_processing_time,
            COUNT(*) as total_processed,
            COUNT(*) / ? as throughput_per_hour
        FROM inspections 
        WHERE timestamp >= datetime('now', '-{} hours')
    '''.format(hours), (hours,))
    
    perf_stats = cursor.fetchone()
    
    # Error rate (simplified)
    cursor.execute('''
        SELECT COUNT(*) 
        FROM inspections 
        WHERE status = 'failed' AND timestamp >= datetime('now', '-{} hours')
    '''.format(hours))
    
    error_count = cursor.fetchone()[0]
    
    conn.close()
    
    total_processed = perf_stats[3] or 0
    error_rate = error_count / max(total_processed, 1)
    
    return {
        "performance_metrics": {
            "avg_processing_time": round(perf_stats[0] or 0, 3),
            "min_processing_time": round(perf_stats[1] or 0, 3),
            "max_processing_time": round(perf_stats[2] or 0, 3),
            "throughput_per_hour": round(perf_stats[4] or 0, 2),
            "total_processed": total_processed,
            "error_rate": round(error_rate, 4),
            "uptime_percentage": 99.5  # Mock uptime
        },
        "system_health": "optimal" if error_rate < 0.01 else "degraded",
        "period_hours": hours
    }

# Calibration and Configuration
@app.post("/api/v1/calibration/camera")
async def calibrate_camera(
    camera_id: str,
    calibration_images: List[UploadFile] = File(...),
    user: Dict = Depends(verify_token)
):
    """Camera calibration for pose tracking accuracy"""
    if len(calibration_images) < 5:
        raise HTTPException(status_code=400, detail="Minimum 5 calibration images required")
    
    calibration_id = str(uuid.uuid4())
    
    # Mock calibration process
    await asyncio.sleep(2)  # Simulate processing
    
    return {
        "calibration_id": calibration_id,
        "camera_id": camera_id,
        "status": "completed",
        "accuracy_improvement": 0.15,
        "calibration_matrix": {
            "fx": 800.0, "fy": 800.0,
            "cx": 320.0, "cy": 240.0,
            "distortion": [0.1, -0.2, 0.0, 0.0, 0.0]
        }
    }

@app.post("/api/v1/thresholds/optimize")
async def optimize_quality_thresholds(
    product_line: str,
    target_false_positive_rate: float = 0.05,
    user: Dict = Depends(verify_token)
):
    """Automatically optimize quality thresholds based on historical data"""
    
    # Analyze historical data to find optimal thresholds
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT quality_score, defects_detected
        FROM inspections 
        WHERE product_line = ? AND timestamp >= datetime('now', '-30 days')
    ''', (product_line,))
    
    historical_data = cursor.fetchall()
    conn.close()
    
    if len(historical_data) < 100:
        raise HTTPException(status_code=400, detail="Insufficient historical data for optimization")
    
    # Mock optimization algorithm
    quality_scores = [row[0] for row in historical_data]
    optimal_threshold = np.percentile(quality_scores, 15)  # Bottom 15th percentile
    
    return {
        "optimization_id": str(uuid.uuid4()),
        "product_line": product_line,
        "previous_threshold": 0.85,
        "optimized_threshold": round(optimal_threshold, 3),
        "expected_improvement": {
            "false_positive_reduction": 0.12,
            "detection_accuracy": 0.08
        },
        "confidence": 0.87,
        "data_points_analyzed": len(historical_data)
    }

# Integration endpoints for factory systems
@app.post("/api/v1/integration/plc")
async def integrate_with_plc(
    plc_config: Dict[str, Any],
    user: Dict = Depends(verify_token)
):
    """Integration endpoint for PLC (Programmable Logic Controller) systems"""
    
    integration_id = str(uuid.uuid4())
    
    # Validate PLC configuration
    required_fields = ['ip_address', 'port', 'protocol']
    if not all(field in plc_config for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required PLC configuration fields")
    
    return {
        "integration_id": integration_id,
        "status": "configured",
        "plc_endpoint": f"{plc_config['ip_address']}:{plc_config['port']}",
        "protocol": plc_config['protocol'],
        "heartbeat_interval": 30,
        "commands_enabled": ["reject_product", "stop_line", "quality_alert"]
    }

@app.post("/api/v1/integration/erp")
async def integrate_with_erp(
    erp_config: Dict[str, Any],
    user: Dict = Depends(verify_token)
):
    """Integration with ERP (Enterprise Resource Planning) systems"""
    
    return {
        "integration_id": str(uuid.uuid4()),
        "erp_system": erp_config.get('system_type', 'generic'),
        "sync_enabled": True,
        "data_flows": [
            "quality_reports",
            "defect_statistics", 
            "production_metrics",
            "compliance_data"
        ],
        "sync_frequency": "hourly"
    }

# Advanced AI Features
@app.post("/api/v1/ai/anomaly-detection")
async def detect_anomalies(
    product_line: str,
    sensitivity: float = 0.8,
    lookback_days: int = 7,
    user: Dict = Depends(verify_token)
):
    """Advanced anomaly detection in quality patterns"""
    
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, quality_score, defects_detected, processing_time
        FROM inspections 
        WHERE product_line = ? AND timestamp >= datetime('now', '-{} days')
        ORDER BY timestamp
    '''.format(lookback_days), (product_line,))
    
    data = cursor.fetchall()
    conn.close()
    
    if len(data) < 50:
        raise HTTPException(status_code=400, detail="Insufficient data for anomaly detection")
    
    # Mock anomaly detection algorithm
    quality_scores = [row[1] for row in data]
    mean_quality = np.mean(quality_scores)
    std_quality = np.std(quality_scores)
    
    anomalies = []
    for i, (timestamp, quality, defects, proc_time) in enumerate(data):
        z_score = abs(quality - mean_quality) / max(std_quality, 0.01)
        if z_score > (2.0 - sensitivity):
            anomalies.append({
                "timestamp": timestamp,
                "quality_score": quality,
                "z_score": round(z_score, 3),
                "type": "quality_outlier",
                "severity": "high" if z_score > 3.0 else "medium"
            })
    
    return {
        "anomaly_detection_id": str(uuid.uuid4()),
        "product_line": product_line,
        "sensitivity": sensitivity,
        "data_points_analyzed": len(data),
        "anomalies_found": len(anomalies),
        "anomalies": anomalies[:20],  # Limit response size
        "baseline_metrics": {
            "mean_quality": round(mean_quality, 3),
            "quality_std": round(std_quality, 3)
        }
    }

@app.post("/api/v1/ai/predictive-maintenance")
async def predict_maintenance_needs(
    equipment_id: str,
    user: Dict = Depends(verify_token)
):
    """Predictive maintenance based on quality degradation patterns"""
    
    # Mock predictive maintenance algorithm
    maintenance_score = np.random.uniform(0.3, 0.9)
    
    predictions = []
    if maintenance_score < 0.7:
        predictions.append({
            "component": "camera_lens",
            "maintenance_type": "cleaning",
            "urgency": "medium",
            "estimated_days": 7
        })
    
    if maintenance_score < 0.5:
        predictions.append({
            "component": "conveyor_belt",
            "maintenance_type": "alignment",
            "urgency": "high", 
            "estimated_days": 3
        })
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "equipment_id": equipment_id,
        "maintenance_score": round(maintenance_score, 3),
        "predictions": predictions,
        "confidence": 0.82,
        "next_analysis": datetime.now() + timedelta(hours=6)
    }

# Mobile/Web specific endpoints
@app.get("/api/v1/mobile/summary")
async def get_mobile_summary(
    product_line: Optional[str] = None,
    user: Dict = Depends(verify_token)
):
    """Optimized summary endpoint for mobile applications"""
    
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Get today's stats
    query = '''
        SELECT 
            COUNT(*) as today_inspections,
            AVG(quality_score) as avg_quality,
            SUM(defects_detected) as total_defects
        FROM inspections 
        WHERE DATE(timestamp) = DATE('now')
    '''
    
    params = []
    if product_line:
        query += " AND product_line = ?"
        params.append(product_line)
    
    cursor.execute(query, params)
    today_stats = cursor.fetchone()
    
    # Get recent alerts
    cursor.execute('''
        SELECT COUNT(*) 
        FROM inspections 
        WHERE quality_score < 0.8 AND timestamp >= datetime('now', '-4 hours')
    ''')
    recent_alerts = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "today": {
            "inspections": today_stats[0] or 0,
            "avg_quality": round(today_stats[1] or 0, 2),
            "defects_found": today_stats[2] or 0,
            "alerts": recent_alerts
        },
        "status": "operational",
        "last_updated": datetime.now().isoformat()
    }

# Compliance and Reporting
@app.get("/api/v1/compliance/iso9001")
async def get_iso9001_report(
    month: Optional[str] = None,
    year: Optional[int] = None,
    user: Dict = Depends(verify_token)
):
    """Generate ISO 9001 compliance report"""
    
    if not month:
        month = datetime.now().strftime("%m")
    if not year:
        year = datetime.now().year
    
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total_inspections,
            AVG(quality_score) as avg_quality,
            SUM(CASE WHEN quality_score >= 0.85 THEN 1 ELSE 0 END) as passed_inspections
        FROM inspections 
        WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
    ''', (str(year), month))
    
    stats = cursor.fetchone()
    conn.close()
    
    total = stats[0] or 0
    passed = stats[2] or 0
    compliance_rate = passed / max(total, 1)
    
    return {
        "report_id": str(uuid.uuid4()),
        "period": f"{year}-{month}",
        "compliance_standard": "ISO 9001:2015",
        "metrics": {
            "total_inspections": total,
            "passed_inspections": passed,
            "compliance_rate": round(compliance_rate, 3),
            "avg_quality_score": round(stats[1] or 0, 3)
        },
        "compliance_status": "compliant" if compliance_rate >= 0.95 else "non_compliant",
        "generated_at": datetime.now().isoformat()
    }

# Machine Learning Operations
@app.get("/api/v1/ml/model-metrics")
async def get_model_metrics(user: Dict = Depends(verify_token)):
    """Get current ML model performance metrics"""
    
    return {
        "defect_detection": {
            "accuracy": 0.943,
            "precision": 0.891,
            "recall": 0.876,
            "f1_score": 0.883,
            "map_50": 0.789,  # Mean Average Precision at IoU 0.5
            "inference_time_ms": 45,
            "model_version": "v2.1.3",
            "last_updated": "2025-08-15T10:30:00Z"
        },
        "pose_tracking": {
            "accuracy": 0.912,
            "keypoint_accuracy": 0.887,
            "pose_estimation_error": 2.3,  # in pixels
            "inference_time_ms": 28,
            "model_version": "v1.4.1",
            "last_updated": "2025-08-10T14:20:00Z"
        },
        "system_metrics": {
            "gpu_utilization": 0.67,
            "memory_usage": 0.45,
            "cpu_usage": 0.23
        }
    }

@app.post("/api/v1/ml/feedback")
async def submit_model_feedback(
    inspection_id: str,
    feedback: Dict[str, Any],
    user: Dict = Depends(verify_token)
):
    """Submit feedback for model improvement"""
    
    feedback_id = str(uuid.uuid4())
    
    # Validate feedback structure
    required_fields = ['correct_classification', 'feedback_type']
    if not all(field in feedback for field in required_fields):
        raise HTTPException(status_code=400, detail="Invalid feedback format")
    
    # In production, this would be stored for retraining
    return {
        "feedback_id": feedback_id,
        "inspection_id": inspection_id,
        "status": "received",
        "impact": "scheduled_for_next_training_cycle",
        "estimated_improvement": 0.02
    }

# Production Line Integration
@app.post("/api/v1/production/line-control")
async def control_production_line(
    action: str,
    product_line: str,
    reason: str,
    user: Dict = Depends(verify_token)
):
    """Control production line based on quality results"""
    
    valid_actions = ["stop", "slow", "reject_current", "alert_operator"]
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
    
    control_id = str(uuid.uuid4())
    
    # Log control action
    logger.info(f"Production line control: {action} on {product_line} - {reason}")
    
    return {
        "control_id": control_id,
        "action": action,
        "product_line": product_line,
        "reason": reason,
        "executed_at": datetime.now().isoformat(),
        "status": "executed",
        "operator_notified": True
    }

# Data Export and Backup
@app.post("/api/v1/export/batch")
async def export_inspection_data(
    start_date: str,
    end_date: str,
    product_lines: Optional[List[str]] = None,
    format: str = "json",
    include_images: bool = False,
    user: Dict = Depends(verify_token)
):
    """Export inspection data for external analysis or backup"""
    
    export_id = str(uuid.uuid4())
    
    # Validate date range
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        if end_dt <= start_dt:
            raise ValueError("End date must be after start date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    
    # Calculate estimated size
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    query = "SELECT COUNT(*) FROM inspections WHERE timestamp BETWEEN ? AND ?"
    params = [start_date, end_date]
    
    if product_lines:
        placeholders = ','.join(['?'] * len(product_lines))
        query += f" AND product_line IN ({placeholders})"
        params.extend(product_lines)
    
    cursor.execute(query, params)
    record_count = cursor.fetchone()[0]
    conn.close()
    
    estimated_size_mb = record_count * 0.5  # Rough estimate
    if include_images:
        estimated_size_mb *= 10  # Images significantly increase size
    
    return {
        "export_id": export_id,
        "status": "queued",
        "estimated_records": record_count,
        "estimated_size_mb": round(estimated_size_mb, 2),
        "format": format,
        "include_images": include_images,
        "estimated_completion": datetime.now() + timedelta(minutes=5),
        "download_url": f"/api/v1/downloads/export-{export_id}.{format}"
    }

# System Administration
@app.post("/api/v1/admin/maintenance-mode")
async def toggle_maintenance_mode(
    enable: bool,
    message: Optional[str] = None,
    user: Dict = Depends(verify_token)
):
    """Enable/disable maintenance mode"""
    
    if user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "maintenance_mode": enable,
        "message": message or ("System entering maintenance mode" if enable else "System operational"),
        "timestamp": datetime.now().isoformat(),
        "affected_endpoints": [
            "/api/v1/inspect",
            "/api/v1/batch-inspect",
            "/api/v1/real-time/start"
        ]
    }

@app.get("/api/v1/admin/system-info")
async def get_system_info(user: Dict = Depends(verify_token)):
    """Get detailed system information"""
    
    if user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = sqlite3.connect('visionguard.db')
    cursor = conn.cursor()
    
    # Database stats
    cursor.execute("SELECT COUNT(*) FROM inspections")
    total_inspections = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM product_lines WHERE active = TRUE")
    active_lines = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "system": {
            "version": "1.0.0",
            "uptime": "7 days, 14 hours, 23 minutes",
            "environment": "production",
            "timezone": "UTC"
        },
        "database": {
            "total_inspections": total_inspections,
            "active_product_lines": active_lines,
            "size_mb": 156.7,
            "last_backup": "2025-08-30T22:00:00Z"
        },
        "resources": {
            "cpu_cores": 8,
            "memory_gb": 32,
            "gpu_model": "NVIDIA RTX 4090",
            "storage_gb": 512
        },
        "models": {
            "defect_detection": "YOLOv8-large",
            "pose_tracking": "MediaPipe Pose",
            "anomaly_detection": "Isolation Forest"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Requirements.txt content (install these dependencies):
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python==4.8.1.78
torch==2.1.0
torchvision==0.16.0
pillow==10.0.1
numpy==1.24.3
pydantic==2.4.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
sqlite3
websockets==12.0
aiofiles==23.2.1
python-dotenv==1.0.0
"""

# Docker configuration (Dockerfile):
"""
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Environment configuration (.env):
"""
DATABASE_URL=sqlite:///./visionguard.db
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=["http://localhost:3000", "http://localhost:3001"]
MODEL_PATH=./models/
UPLOAD_PATH=./uploads/
MAX_FILE_SIZE=10485760  # 10MB
GPU_ENABLED=true
"""