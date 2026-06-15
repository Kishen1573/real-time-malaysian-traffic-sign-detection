import os
import cv2
import pytz
from datetime import datetime, timedelta
from ultralytics import YOLO
import torch
from torchvision import transforms, models
from PIL import Image

from quality import compute_visibility_metrics
from log_utils import (
    open_csv_for_today, append_csv_row,
    open_xlsx_for_today, append_xlsx_row
)

# -----------------------------
# Load YOLO and CNN models
# -----------------------------
yolo_model = YOLO("models/best.pt")

cnn_model = models.resnet18(weights=None)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 9)
cnn_model.load_state_dict(torch.load("models/best_speed_class_cnn.pt", map_location="cpu"))
cnn_model.eval()

cnn_classes = [
    'Speed limit 110', 'Speed limit 20', 'Speed limit 30', 'Speed limit 40',
    'Speed limit 50', 'Speed limit 60', 'Speed limit 70', 'Speed limit 80', 'Speed limit 90'
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------- Model-aware names / exclusions ----------
EXCLUDED_LABELS = {'Road cones','Train Gate','Weight limit','Zebra crossing','pedestrian crossing opt1'}
def _names_dict(model):
    names = model.names
    return names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
_MODEL_NAMES = _names_dict(yolo_model)
EXCLUDED_IDS = {i for i, n in _MODEL_NAMES.items() if n in EXCLUDED_LABELS}
# ---------------------------------------------------

critical_alerts = [
    'Stop','No entry','Children','Road work',
    'No U-turns','No left turn','No right turn',
    'Speed limit 20','Speed limit 30','Speed limit 40',
    'Speed limit 50','Speed limit 60','Speed limit 70',
    'Speed limit 80','Speed limit 90','Speed limit 110'
]

# Alert debounce
last_non_speed_alert = None
non_speed_alert_start = None
non_speed_alert_min_duration = timedelta(seconds=2)
latest_alert = None

MYT = pytz.timezone("Asia/Kuala_Lumpur")
DETECTION_BASE = "static/detections"

# ---- Thresholds ----
GLARE_TH = 0.15
CONTRAST_LOW = 14.0
BLUR_LOW = 100.0
TRUSTED_CONF = 0.70

# ---- Drawing colors ----
YELLOW = (0, 255, 255)
BOX = (0, 200, 255)
BLACK = (0, 0, 0)

def _normalize_label(label: str) -> str:
    if label == "Bumps ahead": return "Bumps"
    if label in ["Expressway signs 1", "Expressway signs 2"]: return "Expressway sign"
    if label in ["Roadway diverges", "Pass either side"]: return "Pass either side"
    if label in ["Traffic merging from the left","Traffic merging to the left","Traffic from Left Merges Ahead"]:
        return "Traffic from Left Merges Ahead"
    if label in ["Traffic merging from the right","Traffic from Right Merges Ahead"]:
        return "Traffic from Right Merges Ahead"
    if label == "pedestrian crossing opt1": return "Pedestrian Crossing"
    return label

def _derive_problem_hint(qm: dict) -> str:
    if qm.get("glare_ratio") is not None and qm["glare_ratio"] > GLARE_TH: return "glare"
    if qm["contrast_score"] < CONTRAST_LOW: return "faded"
    if qm["blur_score"] < BLUR_LOW: return "motion"
    return "general"

def _suggest_for_hint(hint: str) -> str:
    if hint == "glare":
        return "Re-angle sign or add anti-glare visor; replace with proper retroreflective sheeting if needed."
    if hint == "faded":
        return "Clean or repaint/replace the sign face with compliant retroreflective material."
    if hint == "motion":
        return "Improve local lighting or camera stability; verify capture mounting and night settings."
    return "Site check for occlusion or weather; trim vegetation or relocate for clear line-of-sight."

def _draw_text_bg(img, text, x, y_baseline, place_above=True,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, thick=2,
                  pad=4, bg=YELLOW, color=BLACK):
    H, W = img.shape[:2]
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    if place_above:
        top = y_baseline - th - pad; bottom = y_baseline + bl + pad
        if top < 0: top = y_baseline; bottom = y_baseline + th + bl + 2*pad
    else:
        top = y_baseline - th - pad; bottom = y_baseline + bl + pad
        if bottom > H: bottom = y_baseline; top = y_baseline - (th + bl + 2*pad)
    x = max(0, min(W - (tw + 2*pad), x))
    top = max(0, top); bottom = min(H - 1, bottom)
    cv2.rectangle(img, (x, top), (x + tw + 2*pad, bottom), bg, -1)
    cv2.putText(img, text, (x + pad, bottom - pad - bl // 2), font, scale, color, thick, cv2.LINE_AA)

def detect_stream(video_path):
    global last_non_speed_alert, non_speed_alert_start, latest_alert

    today_str = datetime.now(MYT).strftime("%Y-%m-%d")
    output_dir = os.path.join(DETECTION_BASE, today_str)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = open_csv_for_today(DETECTION_BASE)
    xlsx_path = open_xlsx_for_today(DETECTION_BASE)

    cap = cv2.VideoCapture(video_path)
    last_saved_second = None
    saved_count_this_second = 0
    last_saved_image_rel = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        now_myt = datetime.now(MYT)
        now_myt_iso = now_myt.isoformat()
        current_second_key = now_myt.strftime("%Y-%m-%d %H:%M:%S")

        if current_second_key != last_saved_second:
            last_saved_second = current_second_key
            saved_count_this_second = 0
            last_saved_image_rel = None

        annotated = frame.copy()
        results = yolo_model.predict(source=frame, conf=0.2, iou=0.6)[0]

        detection_entries = []
        detected_alerts = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            raw_name = _MODEL_NAMES.get(cls_id, "")
            if cls_id in EXCLUDED_IDS or raw_name in EXCLUDED_LABELS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0].item())
            raw_label = _MODEL_NAMES.get(cls_id, str(cls_id))
            label = _normalize_label(raw_label)

            # Speed limit sub-classification
            if label == "Speed limit":
                crop_sl = frame[y1:y2, x1:x2]
                if crop_sl.size != 0:
                    pil = Image.fromarray(cv2.cvtColor(crop_sl, cv2.COLOR_BGR2RGB))
                    input_tensor = transform(pil).unsqueeze(0)
                    with torch.no_grad():
                        out = cnn_model(input_tensor)
                        pred = torch.argmax(out, 1).item()
                        label = cnn_classes[pred]

            # Quality metrics (still computed & saved, just not drawn)
            crop = frame[y1:y2, x1:x2]
            qm = compute_visibility_metrics(crop)

            # Draw rectangle + ONLY the label (visibility badge removed to speed up)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX, 2)
            _draw_text_bg(annotated, f"{label} ({conf:.2f})", x1, y1 - 5, place_above=True)

            problem_hint = ""; jkr_action = ""
            if qm["low_visibility"]:
                problem_hint = _derive_problem_hint(qm)
                jkr_action = _suggest_for_hint(problem_hint)

            is_trusted = conf >= TRUSTED_CONF

            detection_entries.append({
                "label": label, "conf": conf, "qm": qm,
                "problem_hint": problem_hint, "jkr_action": jkr_action,
                "is_trusted": is_trusted
            })

            # Alerts
            if label in critical_alerts and conf >= 0.80:
                detected_alerts.append((label, conf))

        # Alert selection (unchanged)
        current_alert = None
        for lab, _ in detected_alerts:
            if lab.startswith("Speed limit"):
                current_alert = lab
                last_non_speed_alert = None
                non_speed_alert_start = None
                break
        if not current_alert and detected_alerts:
            non_speed = [(lab, conf) for (lab, conf) in detected_alerts if not lab.startswith("Speed limit")]
            if non_speed:
                top_non_speed_label, _ = max(non_speed, key=lambda p: p[1])
                if top_non_speed_label != last_non_speed_alert:
                    last_non_speed_alert = top_non_speed_label
                    non_speed_alert_start = now_myt
                elif non_speed_alert_start and (now_myt - non_speed_alert_start) >= non_speed_alert_min_duration:
                    current_alert = last_non_speed_alert
        latest_alert = current_alert

        # Save 1 annotated frame per second
        image_relpath_for_second = last_saved_image_rel
        if detection_entries and saved_count_this_second < 1:
            fname = f"detection_{now_myt.strftime('%H-%M-%S-%f')}.jpg"
            image_abs = os.path.join(output_dir, fname)
            cv2.imwrite(image_abs, annotated)
            image_relpath_for_second = f"{today_str}/{fname}"
            last_saved_image_rel = image_relpath_for_second
            saved_count_this_second += 1

        # Append rows (MYT only; no UTC; no bbox coords)
        if detection_entries:
            for de in detection_entries:
                row = {
                    "timestamp_myt": now_myt_iso,
                    "date_myt": today_str,
                    "frame_idx": frame_idx,
                    "label": de["label"],
                    "confidence": round(de["conf"], 3),
                    "image_relpath": image_relpath_for_second or "",
                    "blur_score": de["qm"]["blur_score"],
                    "contrast_score": de["qm"]["contrast_score"],
                    "glare_ratio": de["qm"]["glare_ratio"],
                    "visibility_score": de["qm"]["visibility_score"],
                    "low_visibility": de["qm"]["low_visibility"],
                    "problem_hint": de["problem_hint"],
                    "jkr_action": de["jkr_action"],
                    "is_trusted": de["is_trusted"],
                }
                append_csv_row(csv_path, row)
                append_xlsx_row(xlsx_path, row)

        # Stream frame
        _, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

def get_latest_alert():
    return latest_alert
