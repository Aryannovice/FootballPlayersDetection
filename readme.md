# Player Tracking System

Multi-view player tracking system with two approaches: basic histogram matching and advanced homography-based tracking.


File structure
```$ tree -L 2
.
├── Approach1
│   ├── brute_force.py         # Basic two-view matching
│   ├── brute_force_reid.py    # Basic single-view tracking
│   └── utils.py               # Detection utilities
├── Approach2
│   ├── homography_matching.py # Advanced two-view tracking
│   ├── bytetrack_reid.py      # Advanced single-view with homography
│   └── utils.py               # Enhanced detection utilities
├── model
│   └── best.pt                # YOLO model weights
└── videos                     # Input videos directory ```



## Environment Setup

1. **Python Version**
   ```bash
   Python 3.8 or higher
   ```

2. **Required Environment Variables**
   ```bash
   export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
   ```

3. **Dependencies**
   ```bash
   pip install opencv-python ultralytics torch numpy
   ```

## Running the Code

### Approach 1 (Basic Histogram Matching)

1. **Single View Tracking**
   ```bash
   cd Approach1
   python brute_force_reid.py --video_path ../videos/your_video.mp4 --device cpu
   ```

2. **Two View Matching**
   ```bash
   cd Approach1
   python brute_force.py --broadcast_path ../videos/broadcast.mp4 --tacticam_path ../videos/tacticam.mp4 --device cpu
   ```

### Approach 2 (Advanced Homography-Based)

1. **Single View with Homography**
   ```bash
   cd Approach2
   python bytetrack_reid.py --video ../videos/your_video.mp4 --device cpu
   ```

2. **Two View with Advanced Features**
   ```bash
   cd Approach2
   python homography_matching.py --broadcast ../videos/broadcast.mp4 --tacticam ../videos/tacticam.mp4 --device cpu
   ```

## Controls
- Press `ESC` to exit any visualization window
- Use `--device cuda` instead of `cpu` if using NVIDIA GPU

## Performance Notes
- Approach 1: ~30-40% accuracy, basic color matching
- Approach 2: ~70% accuracy, robust to occlusions and view changes

- # Player Tracking Service Dockerization update version 2(lets say)cuz its cool.

A containerized player tracking service with multiple approaches for video analysis. **Fully containerized with Docker and verified working.**

## Docker Setup

### Prerequisites
- Docker and Docker Compose installed
- Approximately 2GB free disk space for the Docker image

### Quick Start

```bash
# Build and run the service
docker-compose up -d

# Verify it's working
curl http://localhost:8080/health

# Stop the service
docker-compose down
```

### Detailed Setup

1. **Build the Docker image:**
   ```bash
   docker build -t player-tracking .
   ```

2. **Run with Docker Compose (Recommended):**
   ```bash
   docker-compose up -d
   ```

3. **Or run directly with Docker:**
   ```bash
   docker run -p 8080:8080 -v $(pwd)/videos:/app/videos:ro -v $(pwd)/model:/app/model:ro player-tracking
   ```

### Health Check & Verification

Once the container is running, verify the service is available:

```bash
# Check health endpoint
curl http://localhost:8080/health

# Check service info
curl http://localhost:8080/
```

** Verified Health Response:**
```json
{
  "status": "healthy",
  "service": "player-tracking", 
  "version": "1.0.0",
  "approaches": ["brute_force", "bytetrack_reid", "homography_matching"]
}
```

** Verified Service Info Response:**
```json
{
  "message": "Player Tracking Service",
  "endpoints": ["/health"],
  "approaches": {
    "approach1": "brute_force and brute_force_reid",
    "approach2": "bytetrack_reid and homography_matching"
  }
}
```

### Container Management

```bash
# Start the service
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs

# Stop the service
docker-compose down

# Remove everything (including images)
docker-compose down --rmi all -v
```





### Running Player Tracking Algorithms

To use the tracking algorithms, execute them inside the running container:

```bash
# Get container ID
docker ps

# Access the running container
docker exec -it <container_id> bash

# Run specific approaches
python Approach1/brute_force.py --broadcast_path videos/broadcast.mp4 --tacticam_path videos/tacticam.mp4
python Approach1/brute_force_reid.py --video_path videos/15sec_input_720p.mp4
python Approach2/bytetrack_reid.py --video videos/15sec_input_720p.mp4
python Approach2/homography_matching.py --broadcast videos/broadcast.mp4 --tacticam videos/tacticam.mp4
```

## Docker Configuration

### Files Created
- `Dockerfile` - Multi-stage build with Python 3.9 and OpenCV dependencies
- `docker-compose.yml` - Service orchestration with health checks
- `.dockerignore` - Optimized build context
- `health_server.py` - Automatically generated Flask health endpoint

### Container Features
- **Base Image**: Python 3.9-slim
- **Exposed Port**: 8080
- **Health Checks**: Automatic container health monitoring
- **Volume Mounts**: Read-only access to videos and model directories
- **Dependencies**: All required packages for player tracking (ultralytics, opencv-python, numpy, tqdm, flask)


**In case you wish to run it locally , you will need model.pt (yolo v11) fine tuned as well as the videos, for further assistance just drop an email at ayushpandey1177@gmail.com or for videos and models**




