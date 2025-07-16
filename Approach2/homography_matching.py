import cv2, numpy as np, argparse
from utils import load_model, detect_players

class UnifiedPlayerTracker:
    def __init__(self):
        self.global_player_bank = []  # Global database of all players
        self.next_global_id = 0
        self.missing_frames = {}  # Track missing frames per video per player
        self.max_missing = 15  # Frames to keep player in memory
        
    def extract_features(self, img):
        """Extract multiple features for better cross-video matching."""
        if img.size == 0:
            return {"hist": np.zeros(512), "color_mean": np.zeros(3), "dominant_colors": np.zeros(6)}
        
        img_resized = cv2.resize(img, (64, 64))
        
        # Color histogram
        hist = cv2.calcHist([img_resized], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).flatten()
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        
        # Mean color in different color spaces
        color_mean_bgr = np.mean(img_resized.reshape(-1, 3), axis=0)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        color_mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        
        # Dominant colors (simplified)
        pixels = img_resized.reshape(-1, 3)
        upper_half = pixels[pixels[:, 0] + pixels[:, 1] + pixels[:, 2] > np.median(pixels.sum(axis=1))]
        lower_half = pixels[pixels[:, 0] + pixels[:, 1] + pixels[:, 2] <= np.median(pixels.sum(axis=1))]
        
        dominant_upper = np.mean(upper_half, axis=0) if len(upper_half) > 0 else np.zeros(3)
        dominant_lower = np.mean(lower_half, axis=0) if len(lower_half) > 0 else np.zeros(3)
        
        return {
            "hist": hist,
            "color_mean": np.concatenate([color_mean_bgr, color_mean_hsv]),
            "dominant_colors": np.concatenate([dominant_upper, dominant_lower])
        }
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets."""
        # Histogram intersection
        hist_sim = np.sum(np.minimum(features1["hist"], features2["hist"]))
        
        # Color mean similarity (using negative distance)
        color_dist = np.linalg.norm(features1["color_mean"] - features2["color_mean"])
        color_sim = 1.0 / (1.0 + color_dist * 0.01)  # Normalize
        
        # Dominant color similarity
        dom_dist = np.linalg.norm(features1["dominant_colors"] - features2["dominant_colors"])
        dom_sim = 1.0 / (1.0 + dom_dist * 0.01)  # Normalize
        
        # Weighted combination
        total_sim = 0.5 * hist_sim + 0.3 * color_sim + 0.2 * dom_sim
        return total_sim
    
    def find_best_match(self, features, video_id, threshold=0.25):  # Lower threshold
        """Find best matching player across all videos."""
        if not self.global_player_bank:
            return None
        
        similarities = []
        for player_record in self.global_player_bank:
            player_id = player_record["id"]
            
            # Check all feature variants for this player
            best_sim = 0
            for video, stored_features in player_record["features"].items():
                similarity = self.calculate_similarity(features, stored_features)
                best_sim = max(best_sim, similarity)
            
            similarities.append((best_sim, player_id))
        
        if similarities:
            best_sim, best_id = max(similarities)
            print(f"Best match for {video_id}: ID{best_id} with similarity {best_sim:.3f}")
            return best_id if best_sim > threshold else None
        return None
    
    def add_global_player(self, features, video_id):
        """Add new player to global database."""
        player_id = self.next_global_id
        player_record = {
            "id": player_id,
            "features": {video_id: features},
            "last_seen": {video_id: 0}
        }
        self.global_player_bank.append(player_record)
        self.next_global_id += 1
        print(f"Created new player ID{player_id} in {video_id}")
        return player_id
    
    def update_global_player(self, player_id, features, video_id, alpha=0.3):
        """Update player's features for specific video."""
        for player_record in self.global_player_bank:
            if player_record["id"] == player_id:
                if video_id in player_record["features"]:
                    # Update existing features
                    old_features = player_record["features"][video_id]
                    new_features = {}
                    for key in features:
                        new_features[key] = (1-alpha) * old_features[key] + alpha * features[key]
                    player_record["features"][video_id] = new_features
                else:
                    # Add new video features for this player
                    player_record["features"][video_id] = features.copy()
                
                # Reset last seen counter
                player_record["last_seen"][video_id] = 0
                key = f"{player_id}_{video_id}"
                if key in self.missing_frames:
                    del self.missing_frames[key]
                print(f"Updated player ID{player_id} in {video_id}")
                break
    
    def cleanup_missing_players(self, video_id):
        """Clean up players missing for too long in specific video."""
        # Increment missing counters
        for player_record in self.global_player_bank:
            player_id = player_record["id"]
            key = f"{player_id}_{video_id}"
            if key not in self.missing_frames:
                self.missing_frames[key] = 0
            self.missing_frames[key] += 1
        
        # Remove video-specific data for players missing too long
        for player_record in self.global_player_bank:
            player_id = player_record["id"]
            key = f"{player_id}_{video_id}"
            if self.missing_frames.get(key, 0) > self.max_missing:
                if video_id in player_record["features"]:
                    del player_record["features"][video_id]
                if video_id in player_record["last_seen"]:
                    del player_record["last_seen"][video_id]
                if key in self.missing_frames:
                    del self.missing_frames[key]
        
        # Remove players with no remaining video data
        self.global_player_bank = [
            record for record in self.global_player_bank 
            if record["features"]
        ]
    
    def assign_unified_ids(self, players, video_id):
        """Assign consistent IDs across videos."""
        self.cleanup_missing_players(video_id)
        
        assigned_ids = set()
        
        for player in players:
            features = self.extract_features(player["crop"])
            
            # Find best match across all videos
            player_id = self.find_best_match(features, video_id)
            
            # Avoid double assignment in same frame
            if player_id is not None and player_id in assigned_ids:
                player_id = None
            
            if player_id is None:
                # Create new global player
                player_id = self.add_global_player(features, video_id)
            else:
                # Update existing player
                self.update_global_player(player_id, features, video_id)
                assigned_ids.add(player_id)
            
            player["id"] = player_id
        
        return players

def run(broadcast, tacticam, device="cpu"):
    model = load_model(device)
    capA, capB = cv2.VideoCapture(broadcast), cv2.VideoCapture(tacticam)
    
    # Check if videos opened successfully
    print("Broadcast opened:", capA.isOpened())
    print("Tacticam opened:", capB.isOpened())
    
    # Create unified tracker for both videos
    unified_tracker = UnifiedPlayerTracker()
    
    # Set window positions for side-by-side comparison
    cv2.namedWindow("Broadcast", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tacticam", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Broadcast", 50, 50)      # Left side
    cv2.moveWindow("Tacticam", 700, 50)      # Right side
    cv2.resizeWindow("Broadcast", 640, 360)
    cv2.resizeWindow("Tacticam", 640, 360)
    
    frame_count = 0
    
    while True:
        # Read from both videos
        retA, frameA = capA.read()
        retB, frameB = capB.read()
        
        # Create black frames if video ended
        if not retA:
            frameA = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frameA, "Broadcast Ended", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if not retB:
            frameB = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frameB, "Tacticam Ended", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Break if both videos ended
        if not retA and not retB:
            print("Both videos ended.")
            break
        
        frame_count += 1
        print(f"\n--- Frame {frame_count} ---")
        
        # Process broadcast video (if still playing)
        if retA:
            detA = detect_players(frameA, model, conf=0.3)
            detA = unified_tracker.assign_unified_ids(detA, "broadcast")
            broadcast_ids = [p["id"] for p in detA]
            print(f"Broadcast: {len(detA)} players, IDs: {broadcast_ids}")
            
            # Draw players with unified IDs
            for player in detA:
                x1, y1, x2, y2 = player["bbox"]
                player_id = player["id"]
                cv2.rectangle(frameA, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frameA, f"ID{player_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Process tacticam video (if still playing)
        if retB:
            detB = detect_players(frameB, model, conf=0.25)
            detB = unified_tracker.assign_unified_ids(detB, "tacticam")
            tacticam_ids = [p["id"] for p in detB]
            print(f"Tacticam: {len(detB)} players, IDs: {tacticam_ids}")
            
            # Draw players with unified IDs
            for player in detB:
                x1, y1, x2, y2 = player["bbox"]
                player_id = player["id"]
                
                # Use different colors but same ID system
                conf = player.get("confidence", 0.5)
                if conf > 0.4:
                    color = (0, 255, 0)  # High confidence - bright green
                else:
                    color = (0, 180, 180)  # Lower confidence - yellow-green
                cv2.rectangle(frameB, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frameB, f"ID{player_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show matching IDs in console
        if retA and retB:
            matching_ids = set(broadcast_ids) & set(tacticam_ids)
            if matching_ids:
                print(f"Matching IDs across videos: {matching_ids}")
        
        # Resize frames for display
        frameA = cv2.resize(frameA, (640, 360))
        frameB = cv2.resize(frameB, (640, 360))
        
        cv2.imshow("Broadcast", frameA)
        cv2.imshow("Tacticam", frameB)
        
        if cv2.waitKey(30) == 27:  # ESC key
            break
    
    capA.release()
    capB.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--broadcast",default="videos/broadcast.mp4")
    parser.add_argument("--tacticam",default="videos/tacticam.mp4")
    parser.add_argument("--device",default="cpu")
    run(**vars(parser.parse_args()))
