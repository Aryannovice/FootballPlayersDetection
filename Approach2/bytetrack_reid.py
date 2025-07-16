import cv2
import argparse
import numpy as np
from utils import load_model, detect_players
from collections import defaultdict

class SimpleHomography:
    def __init__(self):
        self.homography_matrix = None
        # Standard football field dimensions (in meters)
        self.field_length = 105
        self.field_width = 68
        # Reference field corners in real-world coordinates
        self.field_template = np.array([
            [0, 0],
            [self.field_length, 0],
            [self.field_length, self.field_width],
            [0, self.field_width]
        ], dtype=np.float32)
    
    def set_field_corners(self, frame):
        """Set approximate field corners based on frame dimensions."""
        h, w = frame.shape[:2]
        corners = np.array([
            [w*0.1, h*0.15],   # Top-left
            [w*0.9, h*0.15],   # Top-right
            [w*0.9, h*0.85],   # Bottom-right
            [w*0.1, h*0.85]    # Bottom-left
        ], dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(corners, self.field_template)
        return corners
    
    def image_to_field(self, point):
        """Transform image coordinates to field coordinates."""
        if self.homography_matrix is not None:
            point_array = np.array([[point]], dtype=np.float32)
            field_point = cv2.perspectiveTransform(point_array, self.homography_matrix)
            if field_point is not None:
                return field_point[0][0]
        return point

def is_valid_detection(detection, frame_shape):
    """Validate if detection is a real player."""
    bbox = detection["bbox"]
    x1, y1, x2, y2 = bbox
    
    # Check bounding box size
    width = x2 - x1
    height = y2 - y1
    
    if width < 15 or height < 25 or (width*height) < 500:
        return False
    
    # Aspect ratio check
    aspect_ratio = height / width if width > 0 else 0
    if aspect_ratio < 1.2 or aspect_ratio > 4.5:
        return False
    
    return True

def update_tracker(tracked_players, current_detections, frame_count, next_id):
    """Update tracker state: match, add new, and remove lost players."""
    
    # --- Step 1: Match existing players ---
    unmatched_detections = list(range(len(current_detections)))
    player_ids = list(tracked_players.keys())
    
    match_distances = []
    for i, det_idx in enumerate(unmatched_detections):
        for j, player_id in enumerate(player_ids):
            det_pos = current_detections[det_idx]['field_position']
            tracked_pos = tracked_players[player_id]['field_position']
            dist = np.linalg.norm(det_pos - tracked_pos)
            if dist < 25:  # Max distance for a match (in meters)
                match_distances.append((dist, det_idx, player_id))

    match_distances.sort()
    
    matches = {}
    used_detections = set()
    used_players = set()
    
    for dist, det_idx, player_id in match_distances:
        if det_idx not in used_detections and player_id not in used_players:
            matches[player_id] = current_detections[det_idx]
            used_detections.add(det_idx)
            used_players.add(player_id)
    
    # --- Step 2: Update matched players and mark as seen ---
    for player_id, detection in matches.items():
        tracked_players[player_id].update(detection)
        tracked_players[player_id]['last_seen'] = frame_count

    # --- Step 3: Add new players from unmatched detections ---
    for det_idx in range(len(current_detections)):
        if det_idx not in used_detections:
            new_player_data = current_detections[det_idx]
            new_player_data['last_seen'] = frame_count
            tracked_players[next_id] = new_player_data
            next_id += 1
            
    # --- Step 4: Remove players lost for too long ---
    lost_players = [
        pid for pid, data in tracked_players.items() 
        if frame_count - data.get('last_seen', frame_count) > 45  # Lost for 1.5 seconds
    ]
    for pid in lost_players:
        del tracked_players[pid]
        
    return tracked_players, next_id

def run(video_path, device="cpu"):
    """Track all players with persistent IDs using homography."""
    model = load_model(device)
    homography = SimpleHomography()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1 # Process as fast as possible
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    tracked_players = {}  # {player_id: data}
    next_id = 1
    frame_count = 0
    
    assignment_duration = 3.0
    reassignment_start = max(video_duration - 3.0, assignment_duration + 1.0)
    
    print("Player Tracking with Hide/Reveal Phases")
    print(f"Video duration: {video_duration:.1f}s, FPS: {fps}")
    print(f"Assignment phase (IDs visible): 0-{assignment_duration}s")
    print(f"Tracking phase (IDs hidden): {assignment_duration:.1f}s-{reassignment_start:.1f}s")
    print(f"Reassignment phase (IDs visible): {reassignment_start:.1f}s-{video_duration:.1f}s")
    print("Press ESC to exit")
    
    colors = [(np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)) for _ in range(50)]
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        original_h, original_w = frame.shape[:2]
        
        # --- Speed-up: Resize frame before detection ---
        target_w = 720  # A larger size for better detection quality
        if original_w > target_w:
            scale_factor = target_w / original_w
            target_h = int(original_h * scale_factor)
            processing_frame = cv2.resize(frame, (target_w, target_h))
        else:
            processing_frame = frame
            scale_factor = 1.0
        
        frame_count += 1
        current_time = frame_count / fps
        
        # Determine current phase
        is_assignment_phase = current_time < assignment_duration
        is_reassignment_phase = current_time >= reassignment_start
        is_tracking_phase = not is_assignment_phase and not is_reassignment_phase
        
        # Set up homography on first frame
        if homography.homography_matrix is None:
            field_corners = homography.set_field_corners(frame)
        
        # Detect and validate players on the resized frame
        detections = detect_players(processing_frame, model)
        
        # --- Scale detections back to original frame size ---
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = (
                int(x1 / scale_factor), int(y1 / scale_factor),
                int(x2 / scale_factor), int(y2 / scale_factor)
            )
            cx, cy = det['center']
            det['center'] = (int(cx / scale_factor), int(cy / scale_factor))
            # The crop is based on the resized frame, but it is not used in this script.

        # Validate detections using the original frame's dimensions for accuracy
        valid_detections = [det for det in detections if is_valid_detection(det, frame.shape)]
        
        # Convert to field coordinates
        current_players = []
        for det in valid_detections:
            center = det["center"]
            field_position = homography.image_to_field(center)
            if field_position is not None:
                det['field_position'] = field_position
                current_players.append(det)

        # Update tracking logic
        tracked_players, next_id = update_tracker(tracked_players, current_players, frame_count, next_id)
        
        # Draw all current valid detections (thin gray boxes)
        for player in current_players:
            x1, y1, x2, y2 = player['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
        
        # Draw tracked players with IDs only during assignment/reassignment phases
        if is_assignment_phase or is_reassignment_phase:
            for player_id, player_data in tracked_players.items():
                x1, y1, x2, y2 = player_data['bbox']
                field_pos = player_data['field_position']
                color = colors[player_id % len(colors)]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"ID {player_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display status on screen
        if is_assignment_phase:
            status = f"ASSIGNING IDs ({current_time:.1f}s)"
        elif is_tracking_phase:
            status = f"TRACKING (IDs HIDDEN, {current_time:.1f}s)"
        else:
            status = f"RE-ASSIGNING IDs ({current_time:.1f}s)"
            
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked Players: {len(tracked_players)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Player Tracking with Phases", frame)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nFinal Summary:")
    print(f"Total players tracked: {len(tracked_players)}")
    print("Final player positions:")
    for player_id, data in sorted(tracked_players.items()):
        pos = data['field_position']
        print(f"  Player {player_id}: Field position ({pos[0]:.1f}, {pos[1]:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player Tracking with Hide/Reveal Phases")
    parser.add_argument("--video", default="videos/15sec_input_720p.mp4", 
                       help="Path to video file")
    parser.add_argument("--device", default="cpu", 
                       help="Device for model inference (cpu/cuda)")
    
    args = parser.parse_args()
    run(args.video, args.device)
