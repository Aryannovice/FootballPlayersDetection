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
