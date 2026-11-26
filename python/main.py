# ============================================================
#  main.py  —  Python/YOLOv8 Traffic Controller
#  مشروع: AI-Driven Smart Traffic Management System
#
#  ── المهام الأساسية ───────────────────────────────────────
#  1. قراءة MJPEG stream من كاميرتين (ESP32-CAM) على port 81 (مع Fallback لفيديو محلي)
#  2. عد المركبات وتتبعها بـ YOLOv8 مع عرض الشاشة المباشر
#  3. اكتشاف الإسعاف بـ HSV Color Detection
#  4. اكتشاف مخالفات الإشارة الحمراء عند تخطي الخطوط الظاهرية
#  5. إرسال النتيجة للـ ESP32 Main كل 2 ثانية + مزامنة الحالة
#
#  ── الفيتشرز المضافة ────────────────────────────────────────
#  6. تحسين الإضاءة المنخفضة تلقائياً (Low-Light CLAHE)
#  7. كشف السير عكس الاتجاه (Wrong-Way Driving Detection)
#  8. كشف التوقف المفاجئ والحوادث (Stalled / Accident Detection)
#  9. كشف المشاة في منطقة العبور (Pedestrian Crosswalk Alert)
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
#  ⚙️ الإعدادات الأساسية
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

PERSON_CLASS_ID = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
DETECT_CLASSES  = {PERSON_CLASS_ID} | VEHICLE_CLASSES

YOLO_CONF_THRESHOLD = 0.25
YOLO_IMGSZ          = 640

# ── إعدادات المخالفات والخطوط الظاهرية ──────────────────────
BASE_RESOLUTION = (1920, 1080)
STOP_LINE_BUFFER = 120

LANE_LINES = {
    1: {"start": (566, 396), "end": (181, 487)},
    2: {"start": (1345, 465), "end": (1004, 548)}
}

VIOLATIONS_DIR = "dataset/violations"
EVENTS_DIR     = "dataset/events"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)

# ============================================================
#   إعدادات الفيتشرز الجديدة (قابلة للتعديل حسب التجربة)
# ============================================================

# 1) Low-Light Enhancement (CLAHE)
ENABLE_LOW_LIGHT_ENHANCEMENT = True
LOW_LIGHT_BRIGHTNESS_THRESH  = 65.0     # لو متوسط الإضاءة أقل من القيمة دي يفعّل CLAHE
CLAHE_CLIP_LIMIT             = 2.5
CLAHE_TILE_GRID_SIZE         = (8, 8)

# 2) Wrong-Way Driving Detection
ENABLE_WRONG_WAY_DETECTION   = True
WRONG_WAY_MIN_DIST_1080P     = 90.0     # أقل مسافة حركة (على 1080p) قبل الحكم على الاتجاه
WRONG_WAY_COS_THRESHOLD      = -0.45    # أقل من -0.45 يعني ماشي عكس الاتجاه بوضوح
# متجه الاتجاه الصحيح لكل حارة (منسوب لـ 1080p) — هنعدله لسه حسب اتجاه الحركة في الكاميرا
LANE_FLOW_VECTORS = {
    1: (-1.0, 0.30),
    2: (-1.0, 0.25),
}

# 3) Stalled Vehicle & Accident Detection
ENABLE_STALLED_DETECTION     = True
STALL_SECONDS                = 8.0      # ثواني الثبات لاعتبار المركبة متوقفة/عطلانة
STALL_MOVE_THRESH_1080P      = 18.0     # مسافة حركة صغيرة جداً (بالبكسل على 1080p)

ENABLE_ACCIDENT_DETECTION    = True
ACCIDENT_IOU_THRESHOLD       = 0.22     # نسبة تداخل مربعي مركبتين للشك في حادثة
ACCIDENT_CONFIRM_SECONDS     = 2.0      # مدة استمرار التداخل لتأكيد التنبيه

# 4) Pedestrian Crosswalk Alert
ENABLE_CROSSWALK_ALERT       = True
ALERT_ONLY_ON_GREEN_LANES    = True     # تنبيه عند وجود مشاة في منطقة العبور لما الاشاره تكون اخضر
# مضلعات مناطق عبور المشاة لكل حارة (مؤقتة على 1080p — هنعدلها حسب مكان العبور)
CROSSWALK_ZONES = {
    1: [(160, 495), (580, 410), (620, 470), (200, 565)],
    2: [(980, 555), (1370, 475), (1410, 540), (1020, 625)],
}


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
    self.green_remaining = 0
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
#  Feature 1: Low-Light Enhancement (CLAHE)
# ============================================================
def enhance_low_light(frame: np.ndarray) -> tuple:
  """
  يفحص متوسط سطوع الفريم؛ إذا كان منخفضاً يطبق CLAHE على قناة L في LAB.
  يرجع: (processed_frame, was_enhanced, mean_brightness)
  """
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  mean_brightness = float(np.mean(gray))

  if not ENABLE_LOW_LIGHT_ENHANCEMENT or mean_brightness >= LOW_LIGHT_BRIGHTNESS_THRESH:
    return frame, False, mean_brightness

  lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
  l_channel, a_channel, b_channel = cv2.split(lab)

  clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
  enhanced_l = clahe.apply(l_channel)

  merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
  enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
  return enhanced_bgr, True, mean_brightness


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
#  أداة معايرة الخطوط الظاهرية بالضغط بالماوس
# ============================================================
calibration_clicks = {1: [], 2: []}

def make_mouse_callback(lane_id, frame_w, frame_h):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            base_w, base_h = BASE_RESOLUTION
            scaled_x = int(x * base_w / frame_w)
            scaled_y = int(y * base_h / frame_h)
            
            calibration_clicks[lane_id].append((scaled_x, scaled_y))
            log.info(f"[CALIBRATION] Lane {lane_id} - Point {len(calibration_clicks[lane_id])} clicked (scaled to 1080p): ({scaled_x}, {scaled_y})")
            
            if len(calibration_clicks[lane_id]) == 2:
                pt1, pt2 = calibration_clicks[lane_id][0], calibration_clicks[lane_id][1]
                log.info("=" * 60)
                log.info(f"🎉 تم معايرة خط حارة {lane_id} بنجاح!")
                log.info(f"انسخ هذا السطر واستبدله في إعدادات LANE_LINES في كود main.py:")
                log.info(f"    {lane_id}: {{\"start\": {pt1}, \"end\": {pt2}}}")
                log.info("=" * 60)
                
                LANE_LINES[lane_id] = {"start": pt1, "end": pt2}
                calibration_clicks[lane_id] = []
    return mouse_callback


# ============================================================
#  رياضيات اكتشاف عبور الخطوط والأشكال الهندسية
# ============================================================
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersection(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def point_to_segment_distance(P, A, B):
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
    x1, y1, x2, y2 = bbox
    edges = [
        ((x1, y1), (x2, y1)),
        ((x1, y2), (x2, y2)),
        ((x1, y1), (x1, y2)),
        ((x2, y1), (x2, y2))
    ]
    for edge_start, edge_end in edges:
        if check_intersection(line_start, line_end, edge_start, edge_end):
            return True
            
    if (x1 <= line_start[0] <= x2 and y1 <= line_start[1] <= y2) or \
       (x1 <= line_end[0] <= x2 and y1 <= line_end[1] <= y2):
        return True
        
    return False

def scale_coordinates(start_pt, end_pt, current_res):
    base_w, base_h = BASE_RESOLUTION
    curr_w, curr_h = current_res
    
    if base_w == curr_w and base_h == curr_h:
        return start_pt, end_pt
        
    scale_x = curr_w / base_w
    scale_y = curr_h / base_h
    
    scaled_start = (int(start_pt[0] * scale_x), int(start_pt[1] * scale_y))
    scaled_end = (int(end_pt[0] * scale_x), int(end_pt[1] * scale_y))
    
    return scaled_start, scaled_end

def scale_polygon(points, current_res):
    base_w, base_h = BASE_RESOLUTION
    curr_w, curr_h = current_res
    scale_x = curr_w / base_w
    scale_y = curr_h / base_h
    return np.array([(int(px * scale_x), int(py * scale_y)) for (px, py) in points], dtype=np.int32)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter_area / float(areaA + areaB - inter_area)


# ============================================================
#  حفظ لقطات المخالفات والأحداث
# ============================================================
def trigger_violation(frame: np.ndarray, bbox, track_id: int, lane_id: int):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename_base = f"lane{lane_id}_violation_{timestamp}"
    
    saved_frame = frame.copy()
    cv2.rectangle(saved_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(saved_frame, f"VIOLATION ID: {track_id}", (x1, max(25, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    
    full_path = os.path.join(VIOLATIONS_DIR, f"{filename_base}.jpg")
    cv2.imwrite(full_path, saved_frame)
    log.info(f"[VIOLATION] Lane {lane_id} - Saved violation proof: {full_path}")
    
    y1_c, y2_c = max(0, y1), min(h, y2)
    x1_c, x2_c = max(0, x1), min(w, x2)
    
    if (y2_c - y1_c) > 0 and (x2_c - x1_c) > 0:
        crop = frame[y1_c:y2_c, x1_c:x2_c]
        crop_path = os.path.join(VIOLATIONS_DIR, f"{filename_base}_crop.jpg")
        cv2.imwrite(crop_path, crop)
        log.info(f"[VIOLATION] Lane {lane_id} - Saved vehicle crop: {crop_path}")

def save_event_snapshot(frame: np.ndarray, bbox, label: str, lane_id: int):
    x1, y1, x2, y2 = bbox
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"lane{lane_id}_{label.lower().replace(' ', '_')}_{timestamp}.jpg"
    snap = frame.copy()
    cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 140, 255), 3)
    cv2.putText(snap, f"{label} (Lane {lane_id})", (x1, max(25, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    out_path = os.path.join(EVENTS_DIR, fname)
    cv2.imwrite(out_path, snap)
    log.info(f"[EVENT] Lane {lane_id} - Saved {label}: {out_path}")


# ============================================================
#  Feature 2: Wrong-Way Driving Detection
# ============================================================
origin_positions = {1: {}, 2: {}}
wrong_way_ids      = {1: set(), 2: set()}

def check_wrong_way(lane_id: int, track_id: int, current_pos: tuple, frame_w: int, frame: np.ndarray, bbox) -> bool:
  if not ENABLE_WRONG_WAY_DETECTION:
    return False

  if track_id not in origin_positions[lane_id]:
    origin_positions[lane_id][track_id] = current_pos
    return False

  ox, oy = origin_positions[lane_id][track_id]
  cx, cy = current_pos
  dx = float(cx - ox)
  dy = float(cy - oy)
  dist = float(np.hypot(dx, dy))

  min_dist = WRONG_WAY_MIN_DIST_1080P * (frame_w / BASE_RESOLUTION[0])
  if dist < min_dist:
    return False

  flow_x, flow_y = LANE_FLOW_VECTORS.get(lane_id, (-1.0, 0.0))
  flow_norm = float(np.hypot(flow_x, flow_y))
  if flow_norm == 0:
    return False

  # Cosine similarity بين متجه حركة المركبة ومتجه الحارة الصحيح
  cos_sim = (dx * flow_x + dy * flow_y) / (dist * flow_norm)

  if cos_sim < WRONG_WAY_COS_THRESHOLD:
    if track_id not in wrong_way_ids[lane_id]:
      wrong_way_ids[lane_id].add(track_id)
      log.warning(f"[WRONG WAY] Lane {lane_id} - Track ID {track_id} moving opposite to flow (cos={cos_sim:.2f})")
      save_event_snapshot(frame, bbox, f"WRONG_WAY_ID_{track_id}", lane_id)
    return True

  return track_id in wrong_way_ids[lane_id]


# ============================================================
#  Feature 3: Stalled Vehicle & Accident Detection
# ============================================================
# لكل track_id: {"anchor": (x, y), "last_move_t": float, "stalled": bool, "saved": bool}
stall_tracker = {1: {}, 2: {}}
# لتتبع أزواج المركبات المتداخلة: {(id1, id2): first_overlap_t}
overlap_pairs   = {1: {}, 2: {}}
accident_saved  = {1: set(), 2: set()}

def update_stalled_and_accidents(lane_id: int, vehicles_info: list, now: float, frame_w: int, frame: np.ndarray):
  """
  vehicles_info: قائمة من tuples بالشكل (track_id, bbox_int, bottom_center)
  ترجع: (stalled_ids_set, accident_boxes_list)
  """
  stalled_ids = set()
  accident_boxes = []

  move_thresh = STALL_MOVE_THRESH_1080P * (frame_w / BASE_RESOLUTION[0])
  active_ids = set()

  for tid, bbox, center in vehicles_info:
    active_ids.add(tid)
    if tid not in stall_tracker[lane_id]:
      stall_tracker[lane_id][tid] = {
          "anchor": center,
          "last_move_t": now,
          "stalled": False,
          "saved": False,
      }
      continue

    entry = stall_tracker[lane_id][tid]
    ax, ay = entry["anchor"]
    moved_dist = float(np.hypot(center[0] - ax, center[1] - ay))

    if moved_dist > move_thresh:
      entry["anchor"] = center
      entry["last_move_t"] = now
      entry["stalled"] = False
    else:
      if ENABLE_STALLED_DETECTION and (now - entry["last_move_t"] >= STALL_SECONDS):
        entry["stalled"] = True
        stalled_ids.add(tid)
        if not entry["saved"]:
          entry["saved"] = True
          log.warning(f"[STALLED] Lane {lane_id} - Vehicle ID {tid} stationary for >= {STALL_SECONDS}s")
          save_event_snapshot(frame, bbox, f"STALLED_ID_{tid}", lane_id)

  # تنظيف المركبات المختفية
  for old_id in list(stall_tracker[lane_id].keys()):
    if old_id not in active_ids:
      stall_tracker[lane_id].pop(old_id, None)

  # كشف التداخل المحتمل للحوادث بين المركبات
  if ENABLE_ACCIDENT_DETECTION and len(vehicles_info) >= 2:
    current_pairs = set()
    n = len(vehicles_info)
    for i in range(n):
      for j in range(i + 1, n):
        id1, box1, _ = vehicles_info[i]
        id2, box2, _ = vehicles_info[j]
        pair_key = (min(id1, id2), max(id1, id2))
        iou = compute_iou(box1, box2)

        if iou >= ACCIDENT_IOU_THRESHOLD:
          current_pairs.add(pair_key)
          if pair_key not in overlap_pairs[lane_id]:
            overlap_pairs[lane_id][pair_key] = now
          elif now - overlap_pairs[lane_id][pair_key] >= ACCIDENT_CONFIRM_SECONDS:
            union_box = (
                min(box1[0], box2[0]),
                min(box1[1], box2[1]),
                max(box1[2], box2[2]),
                max(box1[3], box2[3]),
            )
            accident_boxes.append((union_box, pair_key))
            if pair_key not in accident_saved[lane_id]:
              accident_saved[lane_id].add(pair_key)
              log.warning(f"[ACCIDENT] Lane {lane_id} - High overlap between IDs {pair_key} (IoU={iou:.2f})")
              save_event_snapshot(frame, union_box, f"ACCIDENT_{pair_key[0]}_{pair_key[1]}", lane_id)

    for p_key in list(overlap_pairs[lane_id].keys()):
      if p_key not in current_pairs:
        overlap_pairs[lane_id].pop(p_key, None)

  return stalled_ids, accident_boxes


# ============================================================
#  Feature 4: Pedestrian Crosswalk Alert
# ============================================================
def check_pedestrians_on_crosswalk(lane_id: int, person_boxes: list, sig_state: str, frame_res: tuple, annotated_frame: np.ndarray) -> int:
  """
  person_boxes: قائمة من (x1, y1, x2, y2) للأشخاص المكتشفين
  ترجع عدد المشاة الموجودين داخل منطقة العبور في هذه الحارة
  """
  if not ENABLE_CROSSWALK_ALERT or lane_id not in CROSSWALK_ZONES:
    return 0

  poly = scale_polygon(CROSSWALK_ZONES[lane_id], frame_res)
  # رسم حدود منطقة العبور بشفافية خفيفة
  zone_overlay = annotated_frame.copy()
  cv2.polylines(zone_overlay, [poly], isClosed=True, color=(255, 200, 0), thickness=2)
  cv2.fillPoly(zone_overlay, [poly], color=(255, 200, 0))
  cv2.addWeighted(zone_overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)

  pedestrians_in_zone = 0
  for (px1, py1, px2, py2) in person_boxes:
    foot_pt = (float((px1 + px2) / 2.0), float(py2))
    inside = cv2.pointPolygonTest(poly, foot_pt, measureDist=False) >= 0
    if inside:
      pedestrians_in_zone += 1
      cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 215, 255), 2)
      cv2.putText(annotated_frame, "PEDESTRIAN", (px1, max(20, py1 - 8)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

  if pedestrians_in_zone > 0:
    if not ALERT_ONLY_ON_GREEN_LANES or sig_state == "GREEN":
      log.warning(f"[CROSSWALK] Lane {lane_id} - {pedestrians_in_zone} pedestrian(s) on crosswalk while signal is {sig_state}!")
      return pedestrians_in_zone

  return 0


# ============================================================
#  process_frame() — YOLOv8 Tracking + All Features + Overlay
# ============================================================
track_histories = {1: {}, 2: {}}
violated_ids = {1: set(), 2: set()}
violation_flash_timers = {1: 0, 2: 0}

def process_frame(model: YOLO, frame: np.ndarray, lane_id: int, sig_state: str, sig_remaining: int) -> tuple:
  now = time.time()
  h, w = frame.shape[:2]

  # 1) تحسين الإضاءة المنخفضة تلقائياً قبل الكشف
  enhanced_frame, low_light_applied, brightness_val = enhance_low_light(frame)

  use_half = (DEVICE == "cuda")
  results = model.track(
      source=enhanced_frame,
      conf=YOLO_CONF_THRESHOLD,
      persist=True,
      classes=list(DETECT_CLASSES),
      verbose=False,
      device=DEVICE,
      imgsz=YOLO_IMGSZ,
      half=use_half
  )[0]

  vehicle_count   = 0
  ambulance_found = False
  annotated_frame = results.plot()

  # رسم خط الوقوف الظاهري
  line_config = LANE_LINES[lane_id]
  line_start, line_end = scale_coordinates(line_config["start"], line_config["end"], (w, h))
  
  color_map = {
    "GREEN":  (0, 200, 0),
    "YELLOW": (0, 200, 220),
    "RED":    (0, 0, 210),
  }
  sig_color = color_map.get(sig_state, (200, 200, 200))
  
  cv2.line(annotated_frame, line_start, line_end, sig_color, 4)
  cv2.putText(annotated_frame, f"LANE {lane_id} STOP LINE",
              (min(line_start[0], line_end[0]), min(line_start[1], line_end[1]) - 15),
              cv2.FONT_HERSHEY_SIMPLEX, 0.65, sig_color, 2)

  current_frame_ids = set()
  vehicles_info = []
  person_boxes = []
  wrong_way_count = 0

  if results.boxes is not None:
    for box in results.boxes:
      cls_id = int(box.cls[0])
      xyxy = box.xyxy[0].cpu().numpy().astype(int)
      x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

      if cls_id == PERSON_CLASS_ID:
        person_boxes.append((x1, y1, x2, y2))
        continue

      if cls_id not in VEHICLE_CLASSES:
        continue
      
      vehicle_count += 1
      
      # كشف الإسعاف عبر اللون الأحمر
      if detect_ambulance(enhanced_frame, box):
        ambulance_found = True

      if box.id is not None:
        track_id = int(box.id[0].item())
        current_frame_ids.add(track_id)
        
        bottom_edge_start = (x1, y2)
        bottom_edge_end = (x2, y2)
        bottom_center = (int((x1 + x2) / 2), y2)
        vehicles_info.append((track_id, (x1, y1, x2, y2), bottom_center))
        
        cv2.circle(annotated_frame, bottom_center, 5, (0, 165, 255), -1)

        # فحص السير عكس الاتجاه
        if check_wrong_way(lane_id, track_id, bottom_center, w, enhanced_frame, (x1, y1, x2, y2)):
          wrong_way_count += 1
          cv2.putText(annotated_frame, f"WRONG WAY #{track_id}", (x1, max(20, y1 - 35)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # التحقق من مخالفة الإشارة الحمراء
        if sig_state == "RED":
            edge_intersects = check_intersection(line_start, line_end, bottom_edge_start, bottom_edge_end)
            
            trajectory_intersects = False
            if track_id in track_histories[lane_id]:
                prev_center = track_histories[lane_id][track_id]
                trajectory_intersects = check_intersection(line_start, line_end, prev_center, bottom_center)
                
            box_intersects = check_box_line_intersection((x1, y1, x2, y2), line_start, line_end)
            
            scaled_buffer = STOP_LINE_BUFFER * (w / BASE_RESOLUTION[0])
            dist_to_line = point_to_segment_distance(bottom_center, line_start, line_end)
            is_close = dist_to_line <= scaled_buffer
            
            if (edge_intersects or trajectory_intersects or box_intersects or is_close) and (track_id not in violated_ids[lane_id]):
                violated_ids[lane_id].add(track_id)
                violation_flash_timers[lane_id] = 15
                trigger_violation(enhanced_frame, (x1, y1, x2, y2), track_id, lane_id)

        track_histories[lane_id][track_id] = bottom_center

  # تحديث التوقف المفاجئ والحوادث
  stalled_ids, accident_boxes = update_stalled_and_accidents(lane_id, vehicles_info, now, w, enhanced_frame)

  for tid, bbox, _ in vehicles_info:
    if tid in stalled_ids:
      bx1, by1, bx2, by2 = bbox
      cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 140, 255), 3)
      cv2.putText(annotated_frame, f"STALLED #{tid}", (bx1, max(20, by1 - 10)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

  for (ux1, uy1, ux2, uy2), _ in accident_boxes:
    cv2.rectangle(annotated_frame, (ux1, uy1), (ux2, uy2), (0, 0, 255), 3)

  # تحديث تنبيه المشاة على ممر العبور
  pedestrians_alert_count = check_pedestrians_on_crosswalk(lane_id, person_boxes, sig_state, (w, h), annotated_frame)

  # تنظيف الذاكرة للمركبات التي غادرت الكاميرا
  for old_id in list(track_histories[lane_id].keys()):
      if old_id not in current_frame_ids:
          track_histories[lane_id].pop(old_id, None)
          origin_positions[lane_id].pop(old_id, None)

  # فلاش تحذيري للمخالفة على الشاشة
  if violation_flash_timers[lane_id] > 0:
      violation_flash_timers[lane_id] -= 1
      overlay = annotated_frame.copy()
      cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 255), -1)
      cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
      cv2.putText(annotated_frame, "!!! RED LIGHT VIOLATION !!!", (max(20, w // 2 - 250), 75),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

  # شريط تنبيهات الفيتشرز الجديدة أعلى يسار الشاشة
  alerts = []
  if len(accident_boxes) > 0:
    alerts.append(f"ACCIDENT ({len(accident_boxes)})")
  if len(stalled_ids) > 0:
    alerts.append(f"STALLED ({len(stalled_ids)})")
  if wrong_way_count > 0:
    alerts.append(f"WRONG WAY ({wrong_way_count})")
  if pedestrians_alert_count > 0:
    alerts.append(f"PEDESTRIAN ({pedestrians_alert_count})")

  if alerts:
    alert_text = " | ".join(alerts)
    cv2.rectangle(annotated_frame, (10, 10), (min(w - 10, 10 + len(alert_text) * 12 + 20), 46), (0, 0, 180), -1)
    cv2.putText(annotated_frame, alert_text, (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

  # ── التايمر والمعلومات على الشاشة ───────────────────────────
  elapsed_total = int(time.time() - _START_TIME)
  mins  = elapsed_total // 60
  secs  = elapsed_total % 60

  box_w = 195
  overlay = annotated_frame.copy()
  cv2.rectangle(overlay, (w - box_w - 8, 8), (w - 8, 108), (0, 0, 0), -1)
  cv2.addWeighted(overlay, 0.55, annotated_frame, 0.45, 0, annotated_frame)

  timer_text = f"{sig_state}  {sig_remaining}s" if sig_remaining > 0 else sig_state
  cv2.putText(annotated_frame, timer_text,
    (w - box_w, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sig_color, 2)

  cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}",
    (w - box_w, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

  ll_tag = f"CLAHE({brightness_val:.0f})" if low_light_applied else f"LUX:{brightness_val:.0f}"
  cv2.putText(annotated_frame, f"Time: {mins:02d}:{secs:02d} | {ll_tag}",
    (w - box_w, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1)

  cv2.putText(annotated_frame, f"Cam{lane_id} | Q:Quit",
    (w - box_w, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

  return vehicle_count, ambulance_found, annotated_frame


# ============================================================
#  FreshFrameReader
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
#  camera_thread()
# ============================================================
def camera_thread(
  lane:          LaneData,
  cam_url:       str,
  model:         YOLO,
  stop_event:    threading.Event,
  traffic_timer: "TrafficTimer"
):
  log.info(f"[CAM{lane.lane_id}] Thread started → {cam_url}")
  
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
        if os.path.exists(local_fallback_file):
          log.info(f"[CAM{lane.lane_id}] Live stream offline. Falling back to local video: {local_fallback_file}")
          if reader:
            reader.release()
            reader = None
          cap_local = cv2.VideoCapture(local_fallback_file)
          using_fallback = True
          is_live = False
        else:
          lane.update(count=0, ambulance=False, connected=False)
          time.sleep(1.0)
          continue
    else:
      target_file = local_fallback_file if using_fallback else cam_url
      if cap_local is None or not cap_local.isOpened():
        cap_local = cv2.VideoCapture(target_file)
        if not cap_local.isOpened():
          log.error(f"[CAM{lane.lane_id}] Cannot open file: {target_file}")
          lane.update(count=0, ambulance=False, connected=False)
          time.sleep(5.0)
          continue
          
      ret, frame = cap_local.read()
      if not ret or frame is None:
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap_local.read()
        if not ret or frame is None:
          time.sleep(1.0)
          continue
      
      time.sleep(0.03)

    try:
      s1, r1, s2, r2 = traffic_timer.get_display()
      sig_state     = s1 if lane.lane_id == 1 else s2
      sig_remaining = r1 if lane.lane_id == 1 else r2
      count, ambulance, annotated_frame = process_frame(model, frame, lane.lane_id, sig_state, sig_remaining)
      lane.update(count=count, ambulance=ambulance, connected=True, frame=annotated_frame)
      
    except Exception as e:
      log.error(f"[CAM{lane.lane_id}] YOLO error: {e}")
      lane.update(count=0, ambulance=False, connected=True)

  if reader:
    reader.release()
  if cap_local:
    cap_local.release()
  log.info(f"[CAM{lane.lane_id}] Thread stopped.")


# ============================================================
#  SoundState & Network Diagnostics
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

def check_device_connection(ip, port, timeout=1.0):
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False

def run_startup_diagnostics():
    log.info("=" * 60)
    log.info("  ⚡ تشخيص شبكة الاتصال اللاسلكية (Startup Diagnostics) ⚡")
    log.info("=" * 60)
    
    esp_ip  = "192.168.137.100"
    cam1_ip = "192.168.137.101"
    cam2_ip = "192.168.137.102"
    
    esp_ok  = check_device_connection(esp_ip, 80, timeout=1.5)
    cam1_ok = check_device_connection(cam1_ip, 81, timeout=1.5)
    cam2_ok = check_device_connection(cam2_ip, 81, timeout=1.5)
    
    log.info(f"المتحكم الرئيسي ESP32 Main ({esp_ip}): {'متصل وعامل (ONLINE)' if esp_ok else 'غير متصل (OFFLINE)'}")
    log.info(f"كاميرا الحارة 1 ESP32-CAM ({cam1_ip}): {'متصلة وعاملة (ONLINE)' if cam1_ok else 'غير متصلة (OFFLINE)'}")
    log.info(f"كاميرا الحارة 2 ESP32-CAM ({cam2_ip}): {'متصلة وعاملة (ONLINE)' if cam2_ok else 'غير متصلة (OFFLINE)'}")
    
    if not esp_ok or not cam1_ok or not cam2_ok:
        log.warning("=" * 60)
        log.warning("             ⚠️ إرشادات حل مشكلات الاتصال ⚠️")
        log.warning("1. تأكد من الاتصال بنقطة الواي فاي الصحيحة وتطابق الـ SSID/Password.")
        log.warning("2. تأكد من تغذية الكاميرات والمتحكم بالطاقة الكافية.")
        log.warning("3. في حال عدم وجود بث مباشر، سيتم الاعتماد على الفيديوهات المحلية (lane1.mp4 / lane2.mp4) أو المحاكاة.")
        log.warning("=" * 60)
    else:
        log.info("جميع الأجهزة متصلة بالشبكة وجاهزة للعمل بنجاح!\n")


# ============================================================
#  poll_esp32_sound() & send_to_esp32()
# ============================================================
def poll_esp32_sound(
  sound_state:   SoundState,
  esp32_base:    str,
  stop_event:    threading.Event,
  lane1:         "LaneData",
  lane2:         "LaneData",
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
        
        l1_state = data.get("lane1_state", "RED").upper()
        l2_state = data.get("lane2_state", "GREEN").upper()
        mode     = data.get("mode", "FALLBACK").upper()
        
        traffic_timer.sync_with_esp32(l1_state, l2_state, mode)

        if "lane1_remaining" in data:
            lane1.set_timer(int(data.get("lane1_remaining", 0)))
        if "lane2_remaining" in data:
            lane2.set_timer(int(data.get("lane2_remaining", 0)))
      else:
        traffic_timer.set_offline()
    except Exception:
      traffic_timer.set_offline()
      
    time.sleep(1.0)


def send_to_esp32(
  lane1:         LaneData,
  lane2:         LaneData,
  sound_active:  bool,
  traffic_timer: "TrafficTimer"
) -> bool:
  c1, amb1, _ = lane1.read()
  c2, amb2, _ = lane2.read()

  traffic_timer.update_counts(c1, c2)

  ambulance_lane = 0
  emergency      = False

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

    traffic_timer.set_emergency(ambulance_lane)

  payload = {
    "lane1":          c1,
    "lane2":          c2,
    "emergency":      emergency,
    "ambulance_lane": ambulance_lane
  }

  log.info(f"[POST] Sending payload to ESP32: {payload}")

  try:
    resp = requests.post(ESP32_URL, json=payload, timeout=5.0)
    return resp.status_code == 200
  except Exception:
    return False


# ============================================================
#  TrafficTimer & timer_thread
# ============================================================
YELLOW_DURATION = 3

class TrafficTimer:
  def __init__(self):
    self._lock           = threading.Lock()
    self.active_lane     = 1
    self.state           = "GREEN"
    self.remaining       = 15
    self._green1         = 15
    self._green2         = 15
    self.esp32_connected = False
    self.esp32_mode      = "FALLBACK"

  def update_counts(self, c1: int, c2: int):
    with self._lock:
      self._green1 = max(5, min(60, c1 * 1 + 5))
      self._green2 = max(5, min(60, c2 * 1 + 5))

  def sync_with_esp32(self, l1_state: str, l2_state: str, mode: str):
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
    with self._lock:
      self.esp32_connected = False

  def tick(self):
    with self._lock:
      if self.esp32_connected:
        if self.remaining > 0:
          self.remaining -= 1
        return
        
      if self.remaining > 0:
        self.remaining -= 1
        return

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
    with self._lock:
      if amb_lane in (1, 2):
        self.active_lane = amb_lane
        self.state       = "GREEN"
        self.remaining   = 20

  def get_display(self) -> tuple:
    with self._lock:
      if self.active_lane == 1:
        if self.state in ("GREEN", "YELLOW"):
          return (self.state, self.remaining, "RED", 0)
      else:
        if self.state in ("GREEN", "YELLOW"):
          return ("RED", 0, self.state, self.remaining)
      return ("RED", 0, "RED", 0)


def timer_thread(traffic_timer: TrafficTimer, stop_event: threading.Event):
  while not stop_event.is_set():
    time.sleep(1.0)
    traffic_timer.tick()


# ============================================================
#  Main Loop & UI Display
# ============================================================
def main():
  run_startup_diagnostics()

  log.info(f"[MODEL] Loading {MODEL_PATH} on {DEVICE}...")
  model = YOLO(MODEL_PATH)

  lane1 = LaneData(1)
  lane2 = LaneData(2)
  sound_state = SoundState()
  traffic_timer = TrafficTimer()
  stop_event = threading.Event()

  t_timer = threading.Thread(target=timer_thread, args=(traffic_timer, stop_event), daemon=True)
  t_poll  = threading.Thread(target=poll_esp32_sound, args=(sound_state, ESP32_URL, stop_event, lane1, lane2, traffic_timer), daemon=True)
  t_cam1  = threading.Thread(target=camera_thread, args=(lane1, CAM1_URL, model, stop_event, traffic_timer), daemon=True)
  t_cam2  = threading.Thread(target=camera_thread, args=(lane2, CAM2_URL, model, stop_event, traffic_timer), daemon=True)

  t_timer.start()
  t_poll.start()
  t_cam1.start()
  t_cam2.start()

  win1 = "Lane 1 - Smart Traffic"
  win2 = "Lane 2 - Smart Traffic"
  cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
  cv2.namedWindow(win2, cv2.WINDOW_NORMAL)

  callbacks_bound = {1: False, 2: False}
  last_send_time = 0.0

  try:
    while not stop_event.is_set():
      now = time.time()
      if now - last_send_time >= SEND_INTERVAL:
        last_send_time = now
        send_to_esp32(lane1, lane2, sound_state.get(), traffic_timer)

      for lane_obj, win_name, lid in [(lane1, win1, 1), (lane2, win2, 2)]:
        frame = lane_obj.get_frame()
        if frame is not None:
          fh, fw = frame.shape[:2]
          if not callbacks_bound[lid]:
            cv2.setMouseCallback(win_name, make_mouse_callback(lid, fw, fh))
            callbacks_bound[lid] = True
          cv2.imshow(win_name, frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord("q") or key == ord("Q"):
        log.info("[SYS] Quit key pressed. Shutting down...")
        break

  except KeyboardInterrupt:
    log.info("[SYS] Interrupted by user.")
  finally:
    stop_event.set()
    t_cam1.join(timeout=2.0)
    t_cam2.join(timeout=2.0)
    t_poll.join(timeout=2.0)
    t_timer.join(timeout=2.0)
    cv2.destroyAllWindows()
    log.info("[SYS] Clean shutdown complete.")


if __name__ == "__main__":
  main()
