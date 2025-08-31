
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