# 🔍 VisionGuard Pro

**Advanced Manufacturing Quality Control with AI-Powered Visual Defect Detection and Pose Tracking**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

VisionGuard Pro is an enterprise-grade backend API designed for automated visual quality control in manufacturing environments. It combines advanced computer vision, pose tracking, and anomaly detection to ensure product quality and manufacturing excellence.

## 🚀 Features

### **Core Capabilities**
- **🎯 Visual Defect Detection**: AI-powered detection of scratches, dents, cracks, discoloration, and contamination
- **🤖 Pose Tracking**: Product orientation and positioning analysis for comprehensive quality assessment
- **📊 Quality Scoring**: Intelligent scoring algorithm with configurable thresholds
- **⚡ Real-time Processing**: Live inspection capabilities with WebSocket streaming
- **📈 Advanced Analytics**: Comprehensive quality metrics and trend analysis

### **Industrial Integration**
- **🔌 PLC Integration**: Direct connection to Programmable Logic Controllers
- **🏢 ERP Connectivity**: Enterprise Resource Planning system synchronization
- **🎛️ Production Line Control**: Automated stop/reject/alert mechanisms
- **📋 Compliance Reporting**: ISO 9001 and industry standard compliance

### **AI & Machine Learning**
- **🧠 Anomaly Detection**: Pattern recognition for quality degradation
- **🔮 Predictive Maintenance**: Equipment health monitoring
- **🔄 Continuous Learning**: Model feedback and retraining capabilities
- **⚡ Batch Processing**: Handle multiple inspections simultaneously

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile App     │    │  Factory PLC    │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      VisionGuard Pro      │
                    │        Backend API        │
                    └─────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                        │
┌───────▼───────┐    ┌──────────▼──────────┐    ┌────────▼────────┐
│  Defect       │    │    Pose Tracking    │    │   Quality       │
│  Detection    │    │      Module         │    │   Scoring       │
│   (YOLOv8)    │    │   (MediaPipe)       │    │   Algorithm     │
└───────────────┘    └─────────────────────┘    └─────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      SQLite Database      │
                    │   (Inspections, Analytics)│
                    └───────────────────────────┘
```

## 🛠️ Installation

### **Prerequisites**
- Python 3.11+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 50GB+ storage space

### **Quick Start**

1. **Clone the repository:**
```bash
git clone https://github.com/yourorg/visionguard-pro.git
cd visionguard-pro
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database:**
```bash
python -c "from main import init_db; init_db()"
```

6. **Start the server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### **Docker Installation**

```bash
# Build the image
docker build -t visionguard-pro .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data visionguard-pro
```

### **Docker Compose (Recommended)**

```yaml
version: '3.8'
services:
  visionguard-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DATABASE_URL=sqlite:///./data/visionguard.db
      - GPU_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📖 API Documentation

### **Authentication**
All endpoints require authentication. Include the Bearer token in your requests:

```bash
curl -H "Authorization: Bearer your-token-here" \
     http://localhost:8000/api/v1/inspect
```

**Demo Token**: `demo-token` (for testing only)

### **Core Endpoints**

#### **1. Single Product Inspection**
```http
POST /api/v1/inspect
Content-Type: multipart/form-data

# Parameters:
# - image: Image file (required)
# - product_line: string (default: "default")
# - enable_pose_tracking: boolean (default: true)
```

**Example Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "product_line": "automotive_parts",
  "timestamp": "2025-08-31T10:30:00Z",
  "quality_score": 0.87,
  "defects": [
    {
      "id": "def-001",
      "type": "scratch",
      "confidence": 0.92,
      "bbox": {"x": 100, "y": 150, "width": 50, "height": 30},
      "severity": "medium"
    }
  ],
  "pose_data": {
    "keypoints": [...],
    "pose_confidence": 0.89,
    "orientation": "front"
  },
  "status": "completed",
  "processing_time": 1.23,
  "recommendations": ["Review defect before shipping"]
}
```

#### **2. Batch Inspection**
```http
POST /api/v1/batch-inspect
Content-Type: multipart/form-data

# Process up to 50 images simultaneously
```

#### **3. Real-time Inspection**
```http
POST /api/v1/real-time/start
{
  "product_line": "electronics",
  "camera_id": "cam_001"
}
```

#### **4. Analytics Dashboard**
```http
GET /api/v1/analytics/dashboard?product_line=automotive&days=30
```

### **Advanced Features**

#### **Anomaly Detection**
```http
POST /api/v1/ai/anomaly-detection
{
  "product_line": "electronics",
  "sensitivity": 0.8,
  "lookback_days": 7
}
```

#### **Production Line Control**
```http
POST /api/v1/production/line-control
{
  "action": "stop",
  "product_line": "automotive",
  "reason": "Quality threshold exceeded"
}
```

#### **Model Performance Metrics**
```http
GET /api/v1/ml/model-metrics
```

## 🔧 Configuration

### **Product Line Setup**
```http
POST /api/v1/product-lines
{
  "name": "Automotive Parts Line A",
  "quality_threshold": 0.85,
  "defect_types": ["scratch", "dent", "crack"],
  "pose_tracking_enabled": true,
  "active": true
}
```

### **Environment Variables**
```bash
# Database
DATABASE_URL=sqlite:///./visionguard.db

# Security
SECRET_KEY=your-super-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# AI Models
MODEL_PATH=./models/
GPU_ENABLED=true

# File Upload
UPLOAD_PATH=./uploads/
MAX_FILE_SIZE=10485760  # 10MB

# Performance
WORKER_PROCESSES=4
MAX_CONCURRENT_INSPECTIONS=10
```

## 🎯 Use Cases

### **1. Automotive Manufacturing**
- Engine component inspection
- Body panel defect detection
- Assembly line quality control

### **2. Electronics Production**
- PCB defect identification
- Component placement verification
- Solder joint quality assessment

### **3. Food & Packaging**
- Package integrity inspection
- Label placement verification
- Contamination detection

### **4. Textile & Apparel**
- Fabric defect detection
- Seam quality inspection
- Color consistency verification

## 📊 Monitoring & Analytics

### **Key Metrics Tracked**
- **Quality Score Distribution**: Real-time quality trends
- **Defect Rate by Type**: Breakdown of common defects
- **Processing Performance**: Throughput and latency metrics
- **Model Accuracy**: AI model performance monitoring

### **Dashboard Access**
```bash
# View interactive API documentation
http://localhost:8000/docs

# Health check
curl http://localhost:8000/api/v1/health

# System status
curl -H "Authorization: Bearer demo-token" \
     http://localhost:8000/api/v1/models/status
```

## 🔌 Industrial Integrations

### **PLC Integration**
```python
# Configure PLC connection
plc_config = {
    "ip_address": "192.168.1.100",
    "port": 502,
    "protocol": "modbus_tcp"
}

response = requests.post(
    "http://localhost:8000/api/v1/integration/plc",
    json=plc_config,
    headers={"Authorization": "Bearer your-token"}
)
```

### **ERP Integration**
```python
# Sync with ERP systems
erp_config = {
    "system_type": "SAP",
    "endpoint": "https://erp.company.com/api",
    "sync_frequency": "hourly"
}
```

## 📱 Mobile & Web Client Integration

### **Mobile Optimized Endpoints**
```http
# Lightweight summary for mobile apps
GET /api/v1/mobile/summary

# Quick inspection status
GET /api/v1/inspections/{id}
```

### **WebSocket Real-time Updates**
```javascript
// Connect to real-time inspection stream
const ws = new WebSocket('ws://localhost:8000/ws/inspection/session-id');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time inspection:', data);
};
```

## 🧪 Testing

### **Run Tests**
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=main --cov-report=html
```

### **Test Endpoints**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test inspection with sample image
curl -X POST \
  -H "Authorization: Bearer demo-token" \
  -F "image=@sample_product.jpg" \
  -F "product_line=test_line" \
  http://localhost:8000/api/v1/inspect
```

## 🚀 Deployment

### **Production Deployment**

#### **1. Environment Setup**
```bash
# Production environment variables
export DATABASE_URL="postgresql://user:pass@localhost/visionguard"
export SECRET_KEY="your-production-secret-key"
export CORS_ORIGINS='["https://your-domain.com"]'
export GPU_ENABLED=true
```

#### **2. Using Docker Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

#### **3. Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visionguard-pro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: visionguard-pro
  template:
    metadata:
      labels:
        app: visionguard-pro
    spec:
      containers:
      - name: api
        image: visionguard-pro:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
```

### **Scaling Considerations**
- **Load Balancing**: Use NGINX or cloud load balancer
- **Database**: Migrate to PostgreSQL for production
- **Caching**: Implement Redis for frequent queries
- **Storage**: Use S3 or similar for image storage
- **Monitoring**: Integrate with Prometheus/Grafana

## 🔧 Development

### **Project Structure**
```
visionguard-pro/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── .env.example          # Environment template
├── models/               # AI model files
├── uploads/              # Temporary image storage
├── tests/                # Test suite
├── docs/                 # Additional documentation
└── scripts/              # Utility scripts
```

### **Adding New Features**

#### **1. Custom Defect Types**
```python
# Add to DefectDetectionModel class
class CustomDefectDetector:
    def __init__(self):
        self.custom_models = {
            'automotive': 'models/automotive_defects.pt',
            'electronics': 'models/electronics_defects.pt'
        }
```

#### **2. New Analytics Endpoints**
```python
@app.get("/api/v1/analytics/custom")
async def custom_analytics(
    metric_type: str,
    user: Dict = Depends(verify_token)
):
    # Implementation here
    pass
```

### **Model Integration**

#### **Replace Mock Models with Real AI**

**Defect Detection (YOLOv8):**
```python
import ultralytics

class DefectDetectionModel:
    def __init__(self):
        self.model = ultralytics.YOLO('models/defect_detection.pt')
    
    async def detect_defects(self, image: np.ndarray):
        results = self.model(image)
        # Process YOLO results
        return processed_defects
```

**Pose Tracking (MediaPipe):**
```python
import mediapipe as mp

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
    
    async def track_pose(self, image: np.ndarray):
        results = self.pose.process(image)
        # Process MediaPipe results
        return pose_data
```

## 📋 API Reference

### **Authentication**
```http
Authorization: Bearer <your-jwt-token>
```

### **Core Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/inspect` | POST | Single product inspection |
| `/api/v1/batch-inspect` | POST | Batch product inspection |
| `/api/v1/inspections` | GET | List inspection history |
| `/api/v1/inspections/{id}` | GET | Get specific inspection |
| `/api/v1/product-lines` | GET/POST | Manage product lines |
| `/api/v1/analytics/dashboard` | GET | Quality control dashboard |
| `/api/v1/real-time/start` | POST | Start real-time inspection |

### **Advanced Features**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ai/anomaly-detection` | POST | Detect quality anomalies |
| `/api/v1/ai/predictive-maintenance` | POST | Predict maintenance needs |
| `/api/v1/ml/model-metrics` | GET | Model performance metrics |
| `/api/v1/production/line-control` | POST | Control production line |
| `/api/v1/compliance/iso9001` | GET | ISO 9001 compliance report |

### **Integration & Admin**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/integration/plc` | POST | Configure PLC integration |
| `/api/v1/integration/erp` | POST | Configure ERP integration |
| `/api/v1/admin/maintenance-mode` | POST | System maintenance mode |
| `/api/v1/export/batch` | POST | Export inspection data |

## 💡 Usage Examples

### **Python Client Example**
```python
import requests
import json

# Initialize client
API_BASE = "http://localhost:8000"
headers = {"Authorization": "Bearer demo-token"}

# Inspect a product
with open("product_image.jpg", "rb") as f:
    files = {"image": f}
    data = {"product_line": "automotive", "enable_pose_tracking": True}
    
    response = requests.post(
        f"{API_BASE}/api/v1/inspect",
        files=files,
        data=data,
        headers=headers
    )
    
    result = response.json()
    print(f"Quality Score: {result['quality_score']}")
    print(f"Defects Found: {len(result['defects'])}")
```

### **JavaScript/Node.js Example**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('image', fs.createReadStream('product_image.jpg'));
form.append('product_line', 'electronics');

const response = await axios.post(
  'http://localhost:8000/api/v1/inspect',
  form,
  {
    headers: {
      ...form.getHeaders(),
      'Authorization': 'Bearer demo-token'
    }
  }
);

console.log('Inspection Result:', response.data);
```

### **cURL Examples**
```bash
# Single inspection
curl -X POST \
  -H "Authorization: Bearer demo-token" \
  -F "image=@product.jpg" \
  -F "product_line=automotive" \
  http://localhost:8000/api/v1/inspect

# Get analytics
curl -H "Authorization: Bearer demo-token" \
     "http://localhost:8000/api/v1/analytics/dashboard?days=7"

# Health check
curl http://localhost:8000/api/v1/health
```

## 🏭 Industrial Deployment

### **Factory Network Integration**
```python
# Configure for factory network
FACTORY_CONFIG = {
    "network": "192.168.10.0/24",
    "plc_endpoints": ["192.168.10.100", "192.168.10.101"],
    "camera_streams": ["rtsp://192.168.10.200:554/stream1"],
    "quality_thresholds": {
        "line_a": 0.90,
        "line_b": 0.85,
        "line_c": 0.88
    }
}
```

### **Performance Optimization**
```python
# Production performance settings
PERFORMANCE_CONFIG = {
    "max_workers": 8,
    "gpu_batch_size": 16,
    "cache_size": 1000,
    "preprocessing_threads": 4,
    "model_optimization": "tensorrt"  # For NVIDIA GPUs
}
```

## 📈 Monitoring & Observability

### **Logging Configuration**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visionguard.log'),
        logging.StreamHandler()
    ]
)
```

### **Metrics Collection**
- **Prometheus Integration**: Custom metrics export
- **Health Checks**: Automated system monitoring
- **Performance Tracking**: Processing time and throughput
- **Error Monitoring**: Failed inspection tracking

### **Alerting**
```python
# Configure alerts for quality issues
ALERT_CONFIG = {
    "quality_threshold": 0.75,
    "defect_rate_threshold": 0.15,
    "processing_time_threshold": 5.0,
    "notification_channels": ["email", "slack", "webhook"]
}
```

## 🔒 Security

### **Authentication & Authorization**
- **JWT Tokens**: Secure API access
- **Role-based Access**: Inspector, Admin, Viewer roles
- **API Rate Limiting**: Prevent abuse
- **CORS Configuration**: Cross-origin security

### **Data Protection**
- **Image Encryption**: Secure image storage
- **Audit Logging**: Track all system access
- **Backup Encryption**: Secure data backups
- **Network Security**: VPN/firewall recommendations

## 🚨 Troubleshooting

### **Common Issues**

#### **Model Loading Errors**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify model files
ls -la models/
```

#### **Database Issues**
```bash
# Reset database
rm visionguard.db
python -c "from main import init_db; init_db()"
```

#### **Performance Issues**
```bash
# Monitor resource usage
htop
nvidia-smi

# Check logs
tail -f visionguard.log
```

### **Debug Mode**
```bash
# Start with debug logging
export LOG_LEVEL=DEBUG
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Documentation**
- **API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### **Contact**
- **Issues**: GitHub Issues
- **Email**: support@visionguard.com
- **Discord**: VisionGuard Community

### **Enterprise Support**
For enterprise deployments, custom integrations, and 24/7 support:
- **Enterprise Sales**: enterprise@visionguard.com
- **Technical Support**: tech-support@visionguard.com

## 🗺️ Roadmap

### **Q4 2025**
- [ ] Advanced 3D pose estimation
- [ ] Multi-camera synchronization
- [ ] Real-time model updates
- [ ] Cloud deployment templates

### **Q1 2026**
- [ ] Federated learning capabilities
- [ ] Advanced defect classification
- [ ] IoT sensor integration
- [ ] Mobile edge computing support

### **Future Features**
- [ ] AR/VR quality inspection interfaces
- [ ] Digital twin integration
- [ ] Blockchain quality certification
- [ ] Advanced robotics integration

---

**Built with ❤️ for manufacturing excellence**

*VisionGuard Pro - Ensuring quality, one inspection at a time.*