# ============================================================
#  python.py  —  Python/YOLOv8 Traffic Controller
#  مشروع: AI-Driven Smart Traffic Management System
#
#  ── المهمة ─────────────────────────────────────────────────
#  1. قرا MJPEG stream من كاميرتين (ESP32-CAM) على port 81
#  2. عد العربيات وتتبعها بـ YOLOv8 مع عرض الشاشة المباشر
#  3. اكتشف الأسعاف بـ HSV Color Detection
#  4. اكتشف مخالفات الإشارة الحمراء عند تخطي الخطوط الظاهرية
#  5. ابعت النتيجة للـ ESP32 Main كل 2 ثانية
#
#  ── IPs ────────────────────────────────────────────────────
#  ESP32-CAM 1 : 192.168.137.101:81/stream
#  ESP32-CAM 2 : 192.168.137.102:81/stream
#  ESP32 Main  : 192.168.137.100/update
# ============================================================

import os
import cv2
import time
import socket
import requests
import threading
import logging
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import torch

# الكشف التلقائي عن كارت الشاشة (CUDA)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_START_TIME = time.time()  # وقت بداية النظام للـ time counter

# ============================================================
#  ⚙️ الإعدادات
# ============================================================
CAM1_URL  = "http://192.168.137.101:81/stream"  # ← port 81
CAM2_URL  = "http://192.168.137.102:81/stream"  # ← port 81
ESP32_URL = "http://192.168.137.100/update"     # ← port 80

MODEL_PATH = "yolov8n.pt"

SEND_INTERVAL          = 2.0   # ثانية بين كل POST
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY        = 2.0

# ── HSV: كشف الأسعاف (أحمر فقط) ────────────────────────────
RED_LOWER1  = np.array([0,   120, 70],  dtype=np.uint8)
RED_UPPER1  = np.array([10,  255, 255], dtype=np.uint8)
RED_LOWER2  = np.array([160, 120, 70],  dtype=np.uint8)
RED_UPPER2  = np.array([180, 255, 255], dtype=np.uint8)

RED_THRESHOLD = 0.40  # 40% من الـ bounding box لازم يكون أحمر

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# confidence منخفض عشان truck وmotorcycle ميتفوتوش
YOLO_CONF_THRESHOLD = 0.25  # الافتراضي 0.25 — قللّه لـ 0.15 لو لسه بيفوت
YOLO_IMGSZ          = 640   # دقة معالجة الصور لـ YOLO (640 أو 320 للسرعة القصوى)

# ── إعدادات المخالفات والخطوط الظاهرية ──────────────────────
BASE_RESOLUTION = (1920, 1080)

# مسافة الأمان للكشف (بالبكسل، بالنسبة لدقة 1080p).
# أي سيارة تقترب من الخط بمسافة أقل من هذه القيمة سيتم التقاطها فوراً كمخالفة
STOP_LINE_BUFFER = 120  # 120 بكسل على شاشة 1080p (حوالي 6% من العرض)

# إحداثيات الخطوط الظاهرية لكل حارة (البداية والنهاية)
LANE_LINES = {
    1: {"start": (566, 396), "end": (181, 487)},
    2: {"start": (1345, 465), "end": (1004, 548)}
}

# المجلد الذي سيتم حفظ المخالفات فيه
VIOLATIONS_DIR = "dataset/violations"

# التأكد من وجود مجلد المخالفات
os.makedirs(VIOLATIONS_DIR, exist_ok=True)


# ============================================================
#  Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ============================================================
#  LaneData — بيانات كل لين (thread-safe)
# ============================================================
class LaneData:
  def __init__(self, lane_id: int):
    self.lane_id         = lane_id
    self.vehicle_count   = 0
    self.ambulance_found = False
    self.cam_connected   = False
    self.green_remaining = 0   # ← التايمر من ESP32
    self.annotated_frame = None
    self._lock           = threading.Lock()

  def update(self, count: int, ambulance: bool, connected: bool, frame: np.ndarray = None):
    with self._lock:
      self.vehicle_count   = count
      self.ambulance_found = ambulance
      self.cam_connected   = connected
      if frame is not None:
        self.annotated_frame = frame

  def set_timer(self, value: int):
    with self._lock:
      self.green_remaining = value

  def read(self):
    with self._lock:
      return self.vehicle_count, self.ambulance_found, self.cam_connected

  def get_frame(self):
    with self._lock:
      return self.annotated_frame


# ============================================================
#  detect_ambulance() — HSV Color Detection
# ============================================================
def detect_ambulance(frame: np.ndarray, box) -> bool:
  x1, y1, x2, y2 = map(int, box.xyxy[0])
  h, w = frame.shape[:2]
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(w, x2), min(h, y2)

  if (x2 - x1) < 20 or (y2 - y1) < 20:
    return False

  cropped = frame[y1:y2, x1:x2]
  hsv     = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
  total   = cropped.shape[0] * cropped.shape[1]

  red_mask = cv2.bitwise_or(
    cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
    cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
  )
  red_ratio = cv2.countNonZero(red_mask) / total

  is_ambulance = red_ratio >= RED_THRESHOLD
  if is_ambulance:
    log.info(f"[AMBULANCE] Detected! red={red_ratio:.2f}")
  return is_ambulance


# ============================================================
#  أداة معايرة الخطوط الظاهرية بالضغط بالماوس (Visual Calibration Tool)
# ============================================================
calibration_clicks = {1: [], 2: []}

def make_mouse_callback(lane_id, frame_w, frame_h):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # The click x, y are already in native frame coordinates.
            # We scale them to the BASE_RESOLUTION (1920, 1080) for printing and saving,
            # so they remain resolution-independent.
            base_w, base_h = BASE_RESOLUTION
            scaled_x = int(x * base_w / frame_w)
            scaled_y = int(y * base_h / frame_h)
            
            calibration_clicks[lane_id].append((scaled_x, scaled_y))
            log.info(f"[CALIBRATION] Lane {lane_id} - Point {len(calibration_clicks[lane_id])} clicked (scaled to 1080p): ({scaled_x}, {scaled_y})")
            
            if len(calibration_clicks[lane_id]) == 2:
                pt1, pt2 = calibration_clicks[lane_id][0], calibration_clicks[lane_id][1]
                log.info("=" * 60)
                log.info(f"🎉 تم معايرة خط حارة {lane_id} بنجاح!")
                log.info(f"انسخ هذا السطر واستبدله في إعدادات LANE_LINES في كود python.py:")
                log.info(f"    {lane_id}: {{\"start\": {pt1}, \"end\": {pt2}}}")
                log.info("=" * 60)
                
                # Apply line immediately in memory
                LANE_LINES[lane_id] = {"start": pt1, "end": pt2}
                calibration_clicks[lane_id] = []
    return mouse_callback


# ============================================================
#  رياضيات اكتشاف عبور الخطوط الثنائية (Line Crossing Math)
# ============================================================
def ccw(A, B, C):
    """التحقق من الترتيب الدائري عكس عقارب الساعة للنقاط A, B, C"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersection(A, B, C, D):
    """التحقق مما إذا كان قطعة الخط AB تتقاطع مع قطعة الخط CD"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def point_to_segment_distance(P, A, B):
    """حساب المسافة العمودية بين نقطة P وقطعة مستقيمة AB"""
    ab_x = B[0] - A[0]
    ab_y = B[1] - A[1]
    ap_x = P[0] - A[0]
    ap_y = P[1] - A[1]
    
    ab2 = ab_x * ab_x + ab_y * ab_y
    if ab2 == 0:
        return np.hypot(ap_x, ap_y)
        
    t = (ap_x * ab_x + ap_y * ab_y) / ab2
    t = max(0.0, min(1.0, t))
    
    closest_x = A[0] + t * ab_x
    closest_y = A[1] + t * ab_y
    
    return np.hypot(P[0] - closest_x, P[1] - closest_y)

def check_box_line_intersection(bbox, line_start, line_end):
    """التحقق مما إذا كان مربع السيارة يتقاطع مع الخط الظاهري بأي شكل"""
    x1, y1, x2, y2 = bbox
    # الحواف الأربعة للمربع المحيط بالسيارة
    edges = [
        ((x1, y1), (x2, y1)), # الحافة العلوية
        ((x1, y2), (x2, y2)), # الحافة السفلية
        ((x1, y1), (x1, y2)), # الحافة اليسرى
        ((x2, y1), (x2, y2))  # الحافة اليمنى
    ]
    # التحقق من تقاطع خط الوقوف مع أي من الحواف الأربعة
    for edge_start, edge_end in edges:
        if check_intersection(line_start, line_end, edge_start, edge_end):
            return True
            
    # التحقق مما إذا كان خط الوقوف بالكامل يقع داخل المربع (حالة نادرة ولكن ممكنة)
    if (x1 <= line_start[0] <= x2 and y1 <= line_start[1] <= y2) or \
       (x1 <= line_end[0] <= x2 and y1 <= line_end[1] <= y2):
        return True
        
    return False

def scale_coordinates(start_pt, end_pt, current_res):
    """تعديل إحداثيات الخط الظاهري لتتوافق مع دقة إطار الفيديو الفعلي"""
    base_w, base_h = BASE_RESOLUTION
    curr_w, curr_h = current_res
    
    if base_w == curr_w and base_h == curr_h:
        return start_pt, end_pt
        
    scale_x = curr_w / base_w
    scale_y = curr_h / base_h
    
    scaled_start = (int(start_pt[0] * scale_x), int(start_pt[1] * scale_y))
    scaled_end = (int(end_pt[0] * scale_x), int(end_pt[1] * scale_y))
    
    return scaled_start, scaled_end


# ============================================================
#  حفظ وإلتقاط لقطة المخالفة (Capture Violation)
# ============================================================
def trigger_violation(frame: np.ndarray, bbox, track_id: int, lane_id: int):
    """يقوم بحفظ لقطة كاملة للمخالفة وقص صورة السيارة المخالفة في مجلد dataset/violations/"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"lane{lane_id}_violation_{timestamp}"
    
    # رسم مربع أحمر للمخالفة على الصورة المحفوظة
    saved_frame = frame.copy()
    cv2.rectangle(saved_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(saved_frame, f"VIOLATION ID: {track_id}", (x1, y1 - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    
    # حفظ الإطار الكامل كإثبات
    full_path = os.path.join(VIOLATIONS_DIR, f"{filename_base}.jpg")
    cv2.imwrite(full_path, saved_frame)
    log.info(f"[VIOLATION] Lane {lane_id} - Saved violation proof: {full_path}")
    
    # قص وحفظ السيارة المخالفة بشكل منفصل
    y1_c, y2_c = max(0, y1), min(h, y2)
    x1_c, x2_c = max(0, x1), min(w, x2)
    
    if (y2_c - y1_c) > 0 and (x2_c - x1_c) > 0:
        crop = frame[y1_c:y2_c, x1_c:x2_c]
        crop_path = os.path.join(VIOLATIONS_DIR, f"{filename_base}_crop.jpg")
        cv2.imwrite(crop_path, crop)
        log.info(f"[VIOLATION] Lane {lane_id} - Saved vehicle crop: {crop_path}")


# ============================================================
#  process_frame() — YOLOv8 Tracking + Violation + Ambulance + Overlay
# ============================================================
# قواميس التتبع المشتركة بين الفريمات
track_histories = {1: {}, 2: {}}
violated_ids = {1: set(), 2: set()}
violation_flash_timers = {1: 0, 2: 0}

def process_frame(model: YOLO, frame: np.ndarray, lane_id: int, sig_state: str, sig_remaining: int) -> tuple:
  # تشغيل التتبع لربط المركبات بمعرفات فريدة (persist=True)
  # نمرر الـ device والـ imgsz والـ half لتسريع الفريمات واستغلال كرت الشاشة بالـ FP16
  use_half = (DEVICE == "cuda")
  results = model.track(
      source=frame,
      conf=YOLO_CONF_THRESHOLD,
      persist=True,
      classes=list(VEHICLE_CLASSES),
      verbose=False,
      device=DEVICE,
      imgsz=YOLO_IMGSZ,
      half=use_half
  )[0]

  vehicle_count   = 0
  ambulance_found = False
  
  # الحصول على الصورة المرسومة من YOLO كقاعدة أساسية
  annotated_frame = results.plot()

  h, w = frame.shape[:2]

  # 1. إحضار ورسم الخطوط الظاهرية بناءً على حالة الإشارة
  line_config = LANE_LINES[lane_id]
  line_start, line_end = scale_coordinates(line_config["start"], line_config["end"], (w, h))
  
  color_map = {
    "GREEN":  (0, 200, 0),
    "YELLOW": (0, 200, 220),
    "RED":    (0, 0, 210),
  }
  sig_color = color_map.get(sig_state, (200, 200, 200))
  
  # رسم خط الوقوف الظاهري
  cv2.line(annotated_frame, line_start, line_end, sig_color, 4)
  cv2.putText(annotated_frame, f"LANE {lane_id} STOP LINE", 
              (min(line_start[0], line_end[0]), min(line_start[1], line_end[1]) - 15), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.65, sig_color, 2)

  current_frame_ids = set()

  if results.boxes is not None:
    for box in results.boxes:
      if int(box.cls[0]) not in VEHICLE_CLASSES:
        continue
      
      vehicle_count += 1
      
      # كشف الإسعاف عبر اللون الأحمر
      if detect_ambulance(frame, box):
        ambulance_found = True

      # كشف تتبع عبور خط الإشارة الحمراء
      if box.id is not None:
        track_id = int(box.id[0].item())
        current_frame_ids.add(track_id)
        
        xyxy = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
        bottom_edge_start = (xyxy[0], xyxy[3])
        bottom_edge_end = (xyxy[2], xyxy[3])
        bottom_center = (int((xyxy[0] + xyxy[2]) / 2), xyxy[3])
        
        # رسم دائرة صغيرة أسفل كل سيارة كدليل للتتبع
        cv2.circle(annotated_frame, bottom_center, 5, (0, 165, 255), -1)

        # التحقق من المخالفة فقط إذا كانت إشارة هذه الحارة حمراء
        if sig_state == "RED":
            # 1. التقاطع المباشر للحافة السفلية للسيارة مع الخط
            edge_intersects = check_intersection(line_start, line_end, bottom_edge_start, bottom_edge_end)
            
            # 2. تقاطع مسار حركة السيارة من الفريم السابق للفريم الحالي
            trajectory_intersects = False
            if track_id in track_histories[lane_id]:
                prev_center = track_histories[lane_id][track_id]
                trajectory_intersects = check_intersection(line_start, line_end, prev_center, bottom_center)
                
            # 3. تقاطع المربع المحيط بالسيارة بالكامل مع خط الوقوف (الحساسية ثلاثية الأبعاد)
            box_intersects = check_box_line_intersection(xyxy, line_start, line_end)
            
            # 4. مسافة الأمان للاقتراب (Buffer check)
            scaled_buffer = STOP_LINE_BUFFER * (w / BASE_RESOLUTION[0])
            dist_to_line = point_to_segment_distance(bottom_center, line_start, line_end)
            is_close = dist_to_line <= scaled_buffer
            
            # في حال حدوث تقاطع أو اقتراب من مسافة الأمان لأول مرة لهذه السيارة
            if (edge_intersects or trajectory_intersects or box_intersects or is_close) and (track_id not in violated_ids[lane_id]):
                violated_ids[lane_id].add(track_id)
                violation_flash_timers[lane_id] = 15  # تفعيل الفلاش التحذيري لـ 15 فريم
                
                # حفظ اللقطة والقص في المجلد
                trigger_violation(frame, xyxy, track_id, lane_id)

        # حفظ مركز الحركة لتتبعه في الفريم القادم
        track_histories[lane_id][track_id] = bottom_center

  # تنظيف الذاكرة للمركبات التي غادرت الكاميرا
  for old_id in list(track_histories[lane_id].keys()):
      if old_id not in current_frame_ids:
          track_histories[lane_id].pop(old_id, None)

  # فلاش تحذيري للمخالفة على الشاشة
  if violation_flash_timers[lane_id] > 0:
      violation_flash_timers[lane_id] -= 1
      overlay = annotated_frame.copy()
      cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 255), -1)
      cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
      cv2.putText(annotated_frame, "!!! RED LIGHT VIOLATION !!!", (w // 2 - 250, 75),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

  # ── التايمر والمعلومات على الشاشة ───────────────────────────
  elapsed_total = int(time.time() - _START_TIME)
  mins  = elapsed_total // 60
  secs  = elapsed_total % 60

  # خلفية شفافة لعلبة البيانات
  overlay = annotated_frame.copy()
  cv2.rectangle(overlay, (w-175, 8), (w-8, 90), (0, 0, 0), -1)
  cv2.addWeighted(overlay, 0.55, annotated_frame, 0.45, 0, annotated_frame)

  # السطر الأول — state + عداد تنازلي
  timer_text = f"{sig_state}  {sig_remaining}s" if sig_remaining > 0 else sig_state
  cv2.putText(annotated_frame, timer_text,
    (w-168, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sig_color, 2)

  # السطر التاني — عدد العربيات
  cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}",
    (w-168, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

  # السطر التالت — عداد تصاعدي من أول ما النظام اشتغل
  cv2.putText(annotated_frame, f"Time: {mins:02d}:{secs:02d}",
    (w-168, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
  # ────────────────────────────────────────────────────────────

  return vehicle_count, ambulance_found, annotated_frame


# ============================================================
#  FreshFrameReader — خيط مستقل لقراءة أحدث فريم وتجنب تأخير البث التراكمي
# ============================================================
class FreshFrameReader:
  def __init__(self, url: str, lane_id: int):
    self.url = url
    self.lane_id = lane_id
    self.cap = None
    self.frame = None
    self.ret = False
    self.running = True
    self.lock = threading.Lock()
    self.thread = threading.Thread(target=self._update_loop, daemon=True, name=f"Reader-Cam{lane_id}")
    self.thread.start()

  def _update_loop(self):
    while self.running:
      if self.cap is None or not self.cap.isOpened():
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
          log.warning(f"[CAM{self.lane_id}-Reader] Failed to open stream. Retrying in 2.0s...")
          time.sleep(2.0)
          continue
        log.info(f"[CAM{self.lane_id}-Reader] Connected to stream successfully!")

      ret, frame = self.cap.read()
      if ret and frame is not None:
        with self.lock:
          self.frame = frame
          self.ret = True
      else:
        with self.lock:
          self.ret = False
        if self.cap:
          self.cap.release()
        self.cap = None
        time.sleep(1.0)
      
      # نوم بسيط جداً لمنع استهلاك 100% من المعالج عند قراءة الفريمات
      time.sleep(0.001)

  def get_frame(self) -> tuple:
    with self.lock:
      if self.ret and self.frame is not None:
        return True, self.frame.copy()
      return False, None

  def release(self):
    self.running = False
    if self.cap:
      self.cap.release()
    log.info(f"[CAM{self.lane_id}-Reader] Released stream connection.")


# ============================================================
#  camera_thread() — Thread مستقل لكل كاميرا مع Reconnect ومحاكاة
# ============================================================
def camera_thread(
  lane:          LaneData,
  cam_url:       str,
  model:         YOLO,
  stop_event:    threading.Event,
  traffic_timer: "TrafficTimer"
):
  log.info(f"[CAM{lane.lane_id}] Thread started → {cam_url}")
  
  # التحقق مما إذا كان العنوان بثاً حياً أو ملف فيديو محلياً
  is_live = cam_url.startswith("http://") or cam_url.startswith("https://") or cam_url.startswith("rtsp://")
  
  reader = None
  cap_local = None
  local_fallback_file = f"lane{lane.lane_id}.mp4"
  using_fallback = False
  
  if is_live:
    reader = FreshFrameReader(cam_url, lane.lane_id)
  else:
    cap_local = cv2.VideoCapture(cam_url)
    
  while not stop_event.is_set():
    frame = None
    ret = False
    
    if is_live and not using_fallback:
      ret, frame = reader.get_frame()
      if not ret:
        # إذا انقطع البث المباشر، تحقق مما إذا كان هناك فيديو محلي كفولباك
        if os.path.exists(local_fallback_file):
          log.info(f"[CAM{lane.lane_id}] Live stream offline. Falling back to local video: {local_fallback_file}")
          if reader:
            reader.release()
            reader = None
          cap_local = cv2.VideoCapture(local_fallback_file)
          using_fallback = True
          is_live = False
        else:
          # إذا لم يكن هناك بديل محلي، حدّث الحالة لغير متصل وانتظر
          lane.update(count=0, ambulance=False, connected=False)
          time.sleep(1.0)
          continue
    else:
      # القراءة من ملف محلي (إما الملف الأصلي أو الفولباك)
      if cap_local is None or not cap_local.isOpened():
        target_file = local_fallback_file if using_fallback else cam_url
        cap_local = cv2.VideoCapture(target_file)
        if not cap_local.isOpened():
          log.error(f"[CAM{lane.lane_id}] Cannot open file: {target_file}")
          lane.update(count=0, ambulance=False, connected=False)
          time.sleep(5.0)
          continue
          
      ret, frame = cap_local.read()
      if not ret or frame is None:
        # إعادة تشغيل الفيديو المحلي للاستمرار بالاستعراض
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap_local.read()
        if not ret or frame is None:
          time.sleep(1.0)
          continue
      
      # تأخير صغير لمحاكاة معدل إطارات طبيعي (~30 FPS) عند قراءة الملف المحلي
      time.sleep(0.03)

    # تشغيل المعالجة والـ YOLO
    try:
      s1, r1, s2, r2 = traffic_timer.get_display()
      sig_state     = s1 if lane.lane_id == 1 else s2
      sig_remaining = r1 if lane.lane_id == 1 else r2
      count, ambulance, annotated_frame = process_frame(model, frame, lane.lane_id, sig_state, sig_remaining)
      
      # تحديث البيانات وإرسال الفريم الموشر لعرضه لاحقاً في خيط الواجهة الرئيسي (thread-safe)
      lane.update(count=count, ambulance=ambulance, connected=True, frame=annotated_frame)
      
    except Exception as e:
      log.error(f"[CAM{lane.lane_id}] YOLO error: {e}")
      lane.update(count=0, ambulance=False, connected=True)

  # تنظيف الموارد عند الإغلاق
  if reader:
    reader.release()
  if cap_local:
    cap_local.release()
  log.info(f"[CAM{lane.lane_id}] Thread stopped.")


# ============================================================
#  SoundState — حالة سنسور الصوت (من ESP32 /status)
# ============================================================
class SoundState:
  def __init__(self):
    self._active = False
    self._lock   = threading.Lock()

  def set(self, value: bool):
    with self._lock:
      self._active = value

  def get(self) -> bool:
    with self._lock:
      return self._active


# ============================================================
#  أداة اختبار الاتصال بالشبكة (Network Socket Check)
# ============================================================
def check_device_connection(ip, port, timeout=1.0):
    """التحقق السريع من وصول الشبكة للجهاز عبر الـ Sockets دون صلاحيات الأدمن"""
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False

def run_startup_diagnostics():
    """أداة التشخيص الأولى لفحص اتصال الكاميرات والمتحكم"""
    log.info("=" * 60)
    log.info("  ⚡ تشخيص شبكة الاتصال اللاسلكية (Startup Diagnostics) ⚡")
    log.info("=" * 60)
    
    esp_ip = "192.168.137.100"
    cam1_ip = "192.168.137.101"
    cam2_ip = "192.168.137.102"
    
    esp_ok = check_device_connection(esp_ip, 80, timeout=1.5)
    cam1_ok = check_device_connection(cam1_ip, 81, timeout=1.5)
    cam2_ok = check_device_connection(cam2_ip, 81, timeout=1.5)
    
    log.info(f"المتحكم الرئيسي ESP32 Main ({esp_ip}): {'متصل وعامل (ONLINE)' if esp_ok else 'غير متصل (OFFLINE)'}")
    log.info(f"كاميرا الحارة 1 ESP32-CAM ({cam1_ip}): {'متصلة وعاملة (ONLINE)' if cam1_ok else 'غير متصلة (OFFLINE)'}")
    log.info(f"كاميرا الحارة 2 ESP32-CAM ({cam2_ip}): {'متصلة وعاملة (ONLINE)' if cam2_ok else 'غير متصلة (OFFLINE)'}")
    
    if not esp_ok or not cam1_ok or not cam2_ok:
        log.warning("=" * 60)
        log.warning("             ⚠️ إرشادات حل مشكلات الاتصال ⚠️")
        log.warning("1. تأكد من أن جهاز الكمبيوتر متصل بنقطة الواي فاي SSID: 'hani'")
        log.warning("2. تأكد من تفعيل الهوتسبوت وتغذية الكاميرات والمتحكم بالطاقة الكافية.")
        log.warning("3. تأكد من عدم وجود تضارب في العناوين الثابتة (Static IPs) مع أجهزة أخرى.")
        log.warning("4. تأكد من تطابق الـ SSID والرقم السري في أكواد الـ Arduino المرفوعة للكاميرات.")
        log.warning("=" * 60)
    else:
        log.info("جميع الأجهزة متصلة بالشبكة وجاهزة للعمل بنجاح!\n")


# ============================================================
#  poll_esp32_sound() — Thread بيقرأ البيانات من خادم الـ ESP32
# ============================================================
def poll_esp32_sound(
  sound_state:  SoundState,
  esp32_base:   str,
  stop_event:   threading.Event,
  lane1:        "LaneData",
  lane2:        "LaneData",
  traffic_timer: "TrafficTimer"
):
  status_url = esp32_base.replace("/update", "/status")
  log.info(f"[SOUND] Polling → {status_url}")

  while not stop_event.is_set():
    try:
      resp = requests.get(status_url, timeout=1.5)
      if resp.status_code == 200:
        data = resp.json()
        sound_state.set(data.get("sound_detected", False))
        
        # مزامنة حالات الإشارات الحقيقية من خادم الـ ESP32 مع كود البايثون
        l1_state = data.get("lane1_state", "RED").upper()
        l2_state = data.get("lane2_state", "GREEN").upper()
        mode     = data.get("mode", "FALLBACK").upper()
        
        traffic_timer.sync_with_esp32(l1_state, l2_state, mode)

        # دعم استقبال المتبقي من العداد إذا تم تفعيله على الهاردوير
        if "lane1_remaining" in data:
            lane1.set_timer(int(data.get("lane1_remaining", 0)))
        if "lane2_remaining" in data:
            lane2.set_timer(int(data.get("lane2_remaining", 0)))
      else:
        # فحص تفصيلي للمخدم الذي استجاب بالـ 404
        server_header = resp.headers.get("Server", "Unknown Server")
        log.warning(f"[POLL ERROR] Status returned {resp.status_code} from {status_url}. Server Type: {server_header}")
        traffic_timer.set_offline()
    except Exception as e:
      # في حالة انقطاع الاتصال نتحول للمحاكاة المحلية تلقائياً
      traffic_timer.set_offline()
      
    time.sleep(1.0)


# ============================================================
#  send_to_esp32() — بيبعت البيانات للـ ESP32 كل SEND_INTERVAL
# ============================================================
def send_to_esp32(
  lane1:        LaneData,
  lane2:        LaneData,
  sound_active: bool
) -> bool:
  c1, amb1, _ = lane1.read()
  c2, amb2, _ = lane2.read()

  ambulance_lane = 0
  emergency      = False

  # Emergency = صوت صفارة + عربية حمراء في نفس الوقت
  if sound_active and (amb1 or amb2):
    emergency = True
    if amb1 and not amb2:
      ambulance_lane = 1
      log.info("[EMERGENCY] Ambulance in Lane 1 🚑")
    elif amb2 and not amb1:
      ambulance_lane = 2
      log.info("[EMERGENCY] Ambulance in Lane 2 🚑")
    else:
      ambulance_lane = 1 if c1 <= c2 else 2
      log.info(f"[EMERGENCY] Both lanes — priority Lane {ambulance_lane}")

  payload = {
    "lane1":          c1,
    "lane2":          c2,
    "emergency":      emergency,
    "ambulance_lane": ambulance_lane
  }

  log.info(f"[POST] Sending payload to ESP32: {payload}")

  try:
    resp = requests.post(
      ESP32_URL,
      json=payload,
      timeout=5.0
    )
    if resp.status_code == 200:
      return True
    
    # تفاصيل تشخيصية إذا كانت الإستجابة خاطئة (مثل 404)
    server_header = resp.headers.get("Server", "Unknown Server")
    log.error(f"[POST ERROR] ESP32 returned status {resp.status_code} at {ESP32_URL}. Server Type: {server_header}")
    log.error(f"[POST ERROR] Response Body: {resp.text[:300]}")
    return False

  except requests.exceptions.ConnectionError:
    # فحص ما إذا كان الـ ESP32 متصلاً بالشبكة أساساً
    if not check_device_connection("192.168.137.100", 80, timeout=0.8):
        log.error("[POST] Cannot reach ESP32 at 192.168.137.100 - Device is offline/unreachable")
    else:
        log.error("[POST] Connection failed - Web Server not running on ESP32 (Check Arduino Serial Monitor)")
    return False

  except requests.exceptions.Timeout:
    log.error("[POST] ESP32 request timed out")
    return False


# ============================================================
#  TrafficTimer — عداد الإشارة (أخضر/أصفر/أحمر) محسوب في Python
# ============================================================
YELLOW_DURATION = 3   # ثواني الأصفر

class TrafficTimer:
  """
  بتحسب وقت الأخضر من عدد العربيات وبتعد تنازلياً تلقائياً.
  الـ state بتتحدث كل ثانية في thread مستقل.
  """
  def __init__(self):
    self._lock        = threading.Lock()
    self.active_lane  = 1          # اللين الشغال دلوقتي
    self.state        = "GREEN"    # GREEN | YELLOW | RED
    self.remaining    = 15         # الثواني المتبقية للـ state الحالي
    self._green1      = 15         # الوقت المحسوب للحارة 1
    self._green2      = 15         # الوقت المحسوب للحارة 2
    self.esp32_connected = False
    self.esp32_mode   = "FALLBACK"

  def update_counts(self, c1: int, c2: int):
    """بتحدث أوقات الأخضر بناءً على عدد العربيات — بس مش بتقطع الـ cycle الحالي"""
    with self._lock:
      self._green1 = max(5, min(60, c1 * 1 + 5))
      self._green2 = max(5, min(60, c2 * 1 + 5))

  def sync_with_esp32(self, l1_state: str, l2_state: str, mode: str):
    """مزامنة مؤقت وحالات العرض المباشر مع الحالات الفيزيائية للـ ESP32 تلقائياً"""
    with self._lock:
      self.esp32_connected = True
      self.esp32_mode = mode
      
      new_active_lane = self.active_lane
      new_state = self.state
      
      if l1_state in ("GREEN", "YELLOW"):
        new_active_lane = 1
        new_state = l1_state
      elif l2_state in ("GREEN", "YELLOW"):
        new_active_lane = 2
        new_state = l2_state
      else:
        new_state = "RED"
        
      # في حالة حدوث انتقال للحالة أو الحارة بشكل فيزيائي، نقوم بإعادة تهيئة التوقيت المتبقي محلياً للتوافق
      if new_active_lane != self.active_lane or new_state != self.state:
        self.active_lane = new_active_lane
        self.state = new_state
        if self.state == "GREEN":
          self.remaining = self._green1 if self.active_lane == 1 else self._green2
        elif self.state == "YELLOW":
          self.remaining = YELLOW_DURATION
        else:
          self.remaining = 0

  def set_offline(self):
    """تفعيل وضع عدم الاتصال وتنشيط المحاكاة المحلية"""
    with self._lock:
      self.esp32_connected = False

  def tick(self):
    """بتتنادى كل ثانية من thread مستقل لإنقاص العداد تنازلياً"""
    with self._lock:
      # إذا كان الهاردوير متصلاً، الـ ESP32 هو من يقود الطور ولكننا نقوم بالعد التنازلي للشاشة
      if self.esp32_connected:
        if self.remaining > 0:
          self.remaining -= 1
        return
        
      # محاكاة محلية كاملة في حال عدم الاتصال
      if self.remaining > 0:
        self.remaining -= 1
        return

      # الـ remaining وصل صفر — انتقل للـ state الجاي محلياً
      if self.state == "GREEN":
        self.state     = "YELLOW"
        self.remaining = YELLOW_DURATION
      elif self.state == "YELLOW":
        self.active_lane = 2 if self.active_lane == 1 else 1
        self.state       = "GREEN"
        self.remaining   = self._green1 if self.active_lane == 1 else self._green2
      elif self.state == "RED":
        self.state     = "GREEN"
        self.remaining = self._green1 if self.active_lane == 1 else self._green2

  def set_emergency(self, amb_lane: int):
    """Emergency mode — افتح اللين المطلوب 20 ثانية"""
    with self._lock:
      self.active_lane = amb_lane
      self.state       = "GREEN"
      self.remaining   = 20

  def get_display(self) -> tuple:
    """بترجع (lane1_state, lane1_remaining, lane2_state, lane2_remaining)"""
    with self._lock:
      if self.active_lane == 1:
        if self.state == "GREEN":
          return ("GREEN",  self.remaining, "RED",    0)
        elif self.state == "YELLOW":
          return ("YELLOW", self.remaining, "RED",    0)
      else:
        if self.state == "GREEN":
          return ("RED",    0, "GREEN",  self.remaining)
        elif self.state == "YELLOW":
          return ("RED",    0, "YELLOW", self.remaining)
      return ("RED", 0, "RED", 0)


def timer_thread(traffic_timer: TrafficTimer, stop_event: threading.Event):
  """Thread بيعد كل ثانية"""
  while not stop_event.is_set():
    time.sleep(1.0)
    traffic_timer.tick()


# ============================================================
#  main()
# ============================================================
def main():
  log.info("=" * 55)
  log.info("  AI-Driven Smart Traffic Management System")
  log.info("=" * 55)
  
  # تشغيل فحص تشخيص الشبكة المبدئي
  run_startup_diagnostics()
  
  log.info(f"[YOLO] Loading model instances for Lane 1 and Lane 2...")
  model1 = YOLO(MODEL_PATH)
  model2 = YOLO(MODEL_PATH)
  log.info(f"[YOLO] PyTorch device detected: {DEVICE.upper()}")
  if DEVICE == "cuda":
    log.info(f"[YOLO] GPU Name: {torch.cuda.get_device_name(0)}")
  else:
    log.warning("[YOLO] RUNNING ON CPU! CUDA is not available. Please verify your PyTorch CUDA installation.")
  log.info("[YOLO] Models loaded!\n")

  lane1         = LaneData(lane_id=1)
  lane2         = LaneData(lane_id=2)
  sound_state   = SoundState()
  traffic_timer = TrafficTimer()
  stop_event    = threading.Event()

  t_cam1 = threading.Thread(
    target=camera_thread,
    args=(lane1, CAM1_URL, model1, stop_event, traffic_timer),
    daemon=True, name="CAM1-Thread"
  )
  t_cam2 = threading.Thread(
    target=camera_thread,
    args=(lane2, CAM2_URL, model2, stop_event, traffic_timer),
    daemon=True, name="CAM2-Thread"
  )
  t_sound = threading.Thread(
    target=poll_esp32_sound,
    args=(sound_state, ESP32_URL, stop_event, lane1, lane2, traffic_timer),
    daemon=True, name="Sound-Thread"
  )
  t_timer = threading.Thread(
    target=timer_thread,
    args=(traffic_timer, stop_event),
    daemon=True, name="Timer-Thread"
  )

  t_cam1.start()
  t_cam2.start()
  t_sound.start()
  t_timer.start()

  log.info("[SYS] All threads started")
  log.info(f"[SYS] Sending to ESP32 every {SEND_INTERVAL}s")
  log.info("[SYS] Press Ctrl+C or ESC in camera windows to stop\n")

  last_post_time = 0

  try:
    while not stop_event.is_set():
      # 1. عرض فريم حارة 1 في خيط الواجهة الرئيسي (Main Thread GUI)
      f1 = lane1.get_frame()
      if f1 is not None:
        window1 = "Lane 1 - AI Traffic Monitor"
        cv2.imshow(window1, f1)
        if not hasattr(lane1, 'callback_registered'):
          h, w = f1.shape[:2]
          cv2.setMouseCallback(window1, make_mouse_callback(1, w, h))
          lane1.callback_registered = True

      # 2. عرض فريم حارة 2 في خيط الواجهة الرئيسي (Main Thread GUI)
      f2 = lane2.get_frame()
      if f2 is not None:
        window2 = "Lane 2 - AI Traffic Monitor"
        cv2.imshow(window2, f2)
        if not hasattr(lane2, 'callback_registered'):
          h, w = f2.shape[:2]
          cv2.setMouseCallback(window2, make_mouse_callback(2, w, h))
          lane2.callback_registered = True

      # waitKey(1) مهم جداً لتحديث النافذة والتقاط الأزرار
      key = cv2.waitKey(1) & 0xFF
      if key == 27 or key == ord('q'): # ESC أو q للخروج
        break

      # 3. إرسال البيانات للـ ESP32 في Thread منفصل لتجنب أي تعليق في الـ GUI بسبب بطء الشبكة
      curr_time = time.time()
      if curr_time - last_post_time >= SEND_INTERVAL:
        last_post_time = curr_time
        
        c1, amb1, _ = lane1.read()
        c2, amb2, _ = lane2.read()

        # تحديث التايمر بعدد السيارات
        traffic_timer.update_counts(c1, c2)

        # قراءة حالة حساس الصوت
        sound_active = sound_state.get()

        # التحقق المزدوج للأسعاف
        if sound_active and (amb1 or amb2):
          amb_lane = 1 if amb1 else 2
          if amb1 and amb2:
            amb_lane = 1 if c1 <= c2 else 2
          traffic_timer.set_emergency(amb_lane)

        # إطلاق خيط الإرسال الخلفي لعدم تعليق الواجهة
        threading.Thread(
          target=send_to_esp32,
          args=(lane1, lane2, sound_active),
          daemon=True,
          name="HTTP-Post-Thread"
        ).start()

      # نوم قصير جداً لتجنب إشغال المعالج بنسبة 100% في الـ Loop الرئيسي
      time.sleep(0.005)

  except KeyboardInterrupt:
    pass

  log.info("\n[SYS] Stopping...")
  stop_event.set()
  t_cam1.join(timeout=2)
  t_cam2.join(timeout=2)
  t_sound.join(timeout=2)
  t_timer.join(timeout=2)
  cv2.destroyAllWindows()
  log.info("[SYS] Shutdown complete.")

# ============================================================
if __name__ == "__main__":
  main()