import os
import cv2
import time
import requests
import threading
from datetime import datetime
import numpy as np
from ultralytics import YOLO
# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# ESP32 Network Settings (Matching your Arduino static IP)
ESP32_IP = "192.168.137.100"     # Static IP configured on your ESP32
POLL_INTERVAL = 0.5             # Frequency (seconds) to poll traffic light status
UPDATE_INTERVAL = 1.5           # Frequency (seconds) to send vehicle counts to ESP32
# Camera Stream Configuration
# - Can be RTSP streams, MJPEG feeds, local video files, or USB webcams
LANE1_STREAM = "lane1.mp4"
LANE2_STREAM = "lane2.mp4"
# Base resolution at which the stop line coordinates were originally calibrated
BASE_RESOLUTION = (1920, 1080)
# Virtual Line Coordinates (Start and End points in (x, y))
LANE_LINES = {
    1: {"start": (566, 396), "end": (181, 487)},
    2: {"start": (1345, 465), "end": (1004, 548)}
}
# YOLOv8 Vehicle Class IDs (COCO Dataset)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
# Directory to save red-light violations
VIOLATIONS_DIR = "dataset/violations"
# Make sure the violation folder exists
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
# ==========================================
# ESP32 WIRELESS TRAFFIC CONTROLLER
# ==========================================
class ESP32Controller:
    """
    Manages wireless HTTP Wi-Fi integration with the ESP32 controller.
    - Polls GET /status to fetch current traffic light states for violation check.
    - Sends POST /update containing vehicle counts to drive the ESP32's AI mode.
    - Provides automatic local simulation fallback when offline.
    """
    def __init__(self, ip=ESP32_IP, poll_interval=POLL_INTERVAL, update_interval=UPDATE_INTERVAL):
        self.ip = ip
        self.poll_interval = poll_interval
        self.update_interval = update_interval
        self.running = True
        self.connected = False
        
        # Thread-safe states
        self.lock = threading.Lock()
        self.lane_states = {"lane1": "RED", "lane2": "GREEN"}
        self.vehicle_counts = {"lane1": 0, "lane2": 0}
        
        # Start background polling loop
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        
        # Start background update loop (Sends counts to ESP32)
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start offline simulation loop (In case ESP32 is not powered on)
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()
    def set_vehicle_count(self, lane_id, count):
        """Updates the current vehicle count for the given lane (thread-safe)."""
        with self.lock:
            self.vehicle_counts[f"lane{lane_id}"] = count
    def get_light_state(self, lane_id):
        """Returns the current traffic light state for the specified lane."""
        with self.lock:
            return self.lane_states.get(f"lane{lane_id}", "GREEN")
    def _poll_loop(self):
        """Fetches traffic light status from the ESP32 /status endpoint."""
        url = f"http://{self.ip}/status"
        while self.running:
            try:
                response = requests.get(url, timeout=0.8)
                if response.status_code == 200:
                    data = response.json()
                    with self.lock:
                        # Extract lane1_state and lane2_state matching ESP32 JSON schema
                        self.lane_states["lane1"] = data.get("lane1_state", "RED").upper()
                        self.lane_states["lane2"] = data.get("lane2_state", "GREEN").upper()
                    
                    if not self.connected:
                        print(f"[INFO] Connected wirelessly to ESP32 at {url}")
                        self.connected = True
            except Exception:
                if self.connected:
                    print(f"[WARNING] Lost Wi-Fi connection to ESP32 at {self.ip}. Falling back to simulation.")
                    self.connected = False
            
            time.sleep(self.poll_interval)
    def _update_loop(self):
        """Sends real-time vehicle counts to the ESP32 /update endpoint via HTTP POST."""
        url = f"http://{self.ip}/update"
        while self.running:
            if self.connected:
                with self.lock:
                    payload = {
                        "lane1": self.vehicle_counts["lane1"],
                        "lane2": self.vehicle_counts["lane2"],
                        "emergency": False,          # Set to True if emergency vehicle is detected
                        "ambulance_lane": 0          # Lane ID of ambulance if emergency is True
                    }
                try:
                    # Send payload in raw JSON body as expected by ArduinoJson deserializeJson
                    response = requests.post(url, json=payload, timeout=1.0)
                    if response.status_code != 200:
                        print(f"[API ERROR] Failed to send update. Status code: {response.status_code}")
                except Exception as e:
                    # Silently handle connection drops during updates
                    pass
            time.sleep(self.update_interval)
    def _sim_loop(self):
        """
        Runs a local simulated traffic light loop when the ESP32 is offline.
        Ensures the computer vision logic is testable without hardware.
        """
        phases = [
            ("GREEN", "RED"),
            ("YELLOW", "RED"),
            ("RED", "GREEN"),
            ("RED", "YELLOW")
        ]
        # Keep durations similar to ESP32 fallback (15s green, 3s yellow)
        durations = [15, 3, 15, 3] 
        idx = 0
        
        while self.running:
            if not self.connected:
                l1, l2 = phases[idx]
                with self.lock:
                    self.lane_states["lane1"] = l1
                    self.lane_states["lane2"] = l2
                time.sleep(durations[idx])
                idx = (idx + 1) % len(phases)
            else:
                time.sleep(1.0)
    def stop(self):
        self.running = False
# ==========================================
# GEOMETRIC LINE INTERSECTION MATH
# ==========================================
def ccw(A, B, C):
    """
    Determines if points A, B, and C are listed in counter-clockwise order.
    Used for 2D line segment intersection tests.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
def check_intersection(A, B, C, D):
    """
    Checks if line segment AB intersects line segment CD.
    - AB: Virtual violation line segment
    - CD: Bounding box bottom edge or vehicle trajectory vector
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# ==========================================
# CAMERA FEED LANE PROCESSOR (THREAD)
# ==========================================
class LaneProcessor(threading.Thread):
    """
    Processes a single camera stream, runs YOLOv8 tracking,
    renders overlays, and handles red light violation logic.
    """
    def __init__(self, lane_id, stream_source, esp_controller, yolo_model_path="yolov8n.pt"):
        super().__init__()
        self.lane_id = lane_id
        self.stream_source = stream_source
        self.esp = esp_controller
        self.running = True
        self.daemon = True
        
        # Load YOLOv8 model
        print(f"[YOLO] Loading YOLOv8 model for Lane {self.lane_id}...")
        self.model = YOLO(yolo_model_path)
        
        # Unscaled Stop Line coordinates
        self.line_coords = LANE_LINES[self.lane_id]
        
        # Tracking dictionaries
        # Stores {track_id: previous_bottom_center_point}
        self.track_history = {}
        # Stores set of track IDs that have already triggered a violation
        self.violated_ids = set()
        
        # Violation warning banner timer
        self.violation_flash_timer = 0
    def run(self):
        print(f"[STREAM] Opening stream for Lane {self.lane_id}: {self.stream_source}")
        
        # Handle index inputs if given as string integers
        src = self.stream_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
            
        cap = cv2.VideoCapture(src)
        
        # Fallback mechanism if stream fails
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera source {self.stream_source} for Lane {self.lane_id}.")
            print(f"[INFO] Lane {self.lane_id} processor will create a simulated black test stream.")
            self._run_simulated_stream()
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Video file loop logic (restart from beginning if it's a file)
                if isinstance(src, str) and os.path.exists(src):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                print(f"[WARNING] Stream interrupted on Lane {self.lane_id}. Reconnecting...")
                time.sleep(2.0)
                cap = cv2.VideoCapture(src)
                continue
            self._process_frame(frame)
            
            # Press 'q' inside OpenCV window to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cap.release()
        cv2.destroyAllWindows()
    def _run_simulated_stream(self):
        """
        Creates a synthetic video stream if a camera or video file is missing.
        Simulates vehicles crossing the virtual line to demonstrate functionality.
        """
        frame_w, frame_h = 1920, 1080
        frame_count = 0
        
        # Define virtual moving objects for simulation
        # {vehicle_id: {"x": x, "y": y, "speed": speed, "w": w, "h": h}}
        sim_vehicles = {}
        
        while self.running:
            # Create a dark gray road image
            frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 40
            
            # Draw lane lines and markings
            cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 255, 255), 4)
            cv2.putText(frame, f"Simulated Stream (Hardware Offline) - Lane {self.lane_id}", 
                        (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
            cv2.putText(frame, "Place video files 'lane1.mp4' / 'lane2.mp4' in project folder to run on real footage.", 
                        (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            # Periodically spawn a new vehicle in the simulation
            frame_count += 1
            if frame_count % 90 == 0:
                vid = frame_count // 90
                # Spawn coordinate depends on lane
                if self.lane_id == 1:
                    x_pos = 350
                    start_y = 100
                else:
                    x_pos = 1150
                    start_y = 100
                sim_vehicles[vid] = {"x": x_pos, "y": start_y, "speed": 12, "w": 180, "h": 260}
            # Update and draw simulated vehicles
            active_sims = {}
            for vid, v in list(sim_vehicles.items()):
                v["y"] += v["speed"]
                
                # Draw vehicle
                x1, y1 = v["x"] - v["w"] // 2, v["y"] - v["h"]
                x2, y2 = v["x"] + v["w"] // 2, v["y"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 3)
                cv2.putText(frame, f"car {vid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)
                
                # Mock a YOLO detection object to pass into the standard processor
                active_sims[vid] = [x1, y1, x2, y2]
                
                # Keep if on screen
                if v["y"] < frame_h + 100:
                    sim_vehicles[vid] = v
            # Process the simulated frame through the normal lane logic
            self._process_frame_logic(frame, mock_detections=active_sims)
            
            # Render window
            cv2.imshow(f"Smart Traffic Light - Lane {self.lane_id}", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                self.running = False
                break
    def _process_frame(self, frame):
        """Wrapper that performs YOLOv8 inference and calls logic."""
        # Run YOLOv8 tracking with persist=True
        # verbose=False reduces terminal spam
        results = self.model.track(source=frame, persist=True, verbose=False)
        
        detections = {}
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Ensure box has tracking ID and is a vehicle class
                if box.id is not None:
                    cls_id = int(box.cls[0].item())
                    if cls_id in VEHICLE_CLASS_IDS:
                        track_id = int(box.id[0].item())
                        xyxy = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                        detections[track_id] = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
        self._process_frame_logic(frame, mock_detections=detections)
        
        # Display the output window (resized for performance / display fitting)
        cv2.imshow(f"Smart Traffic Light - Lane {self.lane_id}", cv2.resize(frame, (960, 540)))
    def _process_frame_logic(self, frame, mock_detections):
        """
        Executes coordinates scaling, draws virtual lines and tracking boxes,
        calculates line crossing intersection, and triggers violation capture.
        """
        # 1. Update controller vehicle count for this lane
        vehicle_count = len(mock_detections)
        self.esp.set_vehicle_count(self.lane_id, vehicle_count)
        # 2. Get traffic light state wirelessly
        light_state = self.esp.get_light_state(self.lane_id)
        
        # Set line color based on wireless state
        if light_state == "RED":
            line_color = (0, 0, 255)       # Red (BGR)
        elif light_state == "YELLOW":
            line_color = (0, 255, 255)     # Yellow (BGR)
        else:
            line_color = (0, 255, 0)       # Green (BGR)
        # 3. Get scaled virtual line coordinates
        h, w = frame.shape[:2]
        line_start, line_end = self._scale_coordinates(
            self.line_coords["start"], 
            self.line_coords["end"], 
            (w, h)
        )
        # Draw the virtual crossing line
        cv2.line(frame, line_start, line_end, line_color, 4)
        cv2.putText(frame, f"LANE {self.lane_id} STOP LINE", 
                    (min(line_start[0], line_end[0]), min(line_start[1], line_end[1]) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        # Draw Traffic Light State Status on frame
        status_box_color = line_color
        cv2.rectangle(frame, (40, 40), (220, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (40, 40), (220, 100), status_box_color, 3)
        cv2.putText(frame, f"LIGHT: {light_state}", (55, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_box_color, 2)
        # Keep track of IDs seen in the current frame to prune history
        current_frame_ids = set()
        # 4. Process detections
        for track_id, bbox in mock_detections.items():
            current_frame_ids.add(track_id)
            x1, y1, x2, y2 = bbox
            
            # Bottom edge of the vehicle's bounding box
            bottom_edge_start = (x1, y2)
            bottom_edge_end = (x2, y2)
            
            # Bottom center point
            bottom_center = (int((x1 + x2) / 2), y2)
            # Draw vehicle bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(frame, bottom_center, 5, (0, 165, 255), -1)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # 5. Check for Red Light Violations
            # Only trigger crossing detection if the light state is wireless RED
            if light_state == "RED":
                # Check if bottom edge segment intersects the virtual line
                edge_intersects = check_intersection(line_start, line_end, bottom_edge_start, bottom_edge_end)
                
                # Check trajectory intersection (previous frame to current frame bottom center)
                trajectory_intersects = False
                if track_id in self.track_history:
                    prev_center = self.track_history[track_id]
                    trajectory_intersects = check_intersection(line_start, line_end, prev_center, bottom_center)
                # Trigger violation if crossed and not already captured
                if (edge_intersects or trajectory_intersects) and (track_id not in self.violated_ids):
                    self.violated_ids.add(track_id)
                    self.violation_flash_timer = 15  # Trigger visual alert on screen for 15 frames
                    
                    # Capture violation
                    self._trigger_violation(frame, bbox, track_id)
            # Update tracking history
            self.track_history[track_id] = bottom_center
        # Prune old tracking history to save memory
        self._prune_history(current_frame_ids)
        # Draw a visual flash/alert banner if a violation was recently detected
        if self.violation_flash_timer > 0:
            self.violation_flash_timer -= 1
            # Red warning banner at top of feed
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "!!! RED LIGHT VIOLATION DETECTED !!!", (w // 2 - 380, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4)
    def _scale_coordinates(self, start_pt, end_pt, current_res):
        """Scales virtual line coordinates if frame resolution differs from base."""
        base_w, base_h = BASE_RESOLUTION
        curr_w, curr_h = current_res
        
        if base_w == curr_w and base_h == curr_h:
            return start_pt, end_pt
            
        scale_x = curr_w / base_w
        scale_y = curr_h / base_h
        
        scaled_start = (int(start_pt[0] * scale_x), int(start_pt[1] * scale_y))
        scaled_end = (int(end_pt[0] * scale_x), int(end_pt[1] * scale_y))
        
        return scaled_start, scaled_end
    def _trigger_violation(self, frame, bbox, track_id):
        """Saves both the full annotated frame and a crop of the offending vehicle."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"lane{self.lane_id}_violation_{timestamp}"
        
        # Create a deep copy of the frame to draw violation markers on the saved image
        saved_frame = frame.copy()
        
        # Annotate saved proof image
        cv2.rectangle(saved_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # RED bounding box
        cv2.putText(saved_frame, f"VIOLATION ID: {track_id}", (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        # Path for full frame proof
        full_path = os.path.join(VIOLATIONS_DIR, f"{filename_base}.jpg")
        cv2.imwrite(full_path, saved_frame)
        print(f"[VIOLATION] Lane {self.lane_id} - Saved violation proof: {full_path}")
        
        # Crop the vehicle bounding box
        # Ensure crop coordinates stay within frame boundaries
        y1_c, y2_c = max(0, y1), min(h, y2)
        x1_c, x2_c = max(0, x1), min(w, x2)
        
        if (y2_c - y1_c) > 0 and (x2_c - x1_c) > 0:
            crop = frame[y1_c:y2_c, x1_c:x2_c]
            crop_path = os.path.join(VIOLATIONS_DIR, f"{filename_base}_crop.jpg")
            cv2.imwrite(crop_path, crop)
            print(f"[VIOLATION] Lane {self.lane_id} - Saved vehicle crop: {crop_path}")
    def _prune_history(self, current_ids):
        """Removes IDs from tracking history if they are no longer in the frame."""
        # If tracking history gets large, remove keys not in current_ids
        for old_id in list(self.track_history.keys()):
            if old_id not in current_ids:
                self.track_history.pop(old_id, None)
# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    print("====================================================")
    print("  Smart Traffic Light: Red Light Violation System  ")
    print("====================================================")
    print(f"Wireless ESP32 Target IP: {ESP32_IP}")
    print(f"Violation Folder: {os.path.abspath(VIOLATIONS_DIR)}\n")
    
    # 1. Initialize ESP32 Client Controller
    esp = ESP32Controller(ip=ESP32_IP)
    
    # 2. Spin up processors for both lanes
    processor1 = LaneProcessor(lane_id=1, stream_source=LANE1_STREAM, esp_controller=esp)
    processor2 = LaneProcessor(lane_id=2, stream_source=LANE2_STREAM, esp_controller=esp)
    
    processor1.start()
    processor2.start()
    
    try:
        # Keep main thread alive
        while processor1.is_alive() or processor2.is_alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Exiting threads... Please wait.")
    finally:
        esp.stop()
        processor1.running = False
        processor2.running = False
        processor1.join()
        processor2.join()
        print("[SHUTDOWN] System terminated successfully.")
if __name__ == "__main__":
    main()