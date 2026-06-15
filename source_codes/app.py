from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, Response
import os, csv
from datetime import datetime, date
import pytz
from werkzeug.utils import secure_filename
from detector import detect_stream, get_latest_alert

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECTION_FOLDER = 'static/detections'
ALLOWED_EXTENSIONS = {'mp4','avi','mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

latest_video_filename = None
MYT = pytz.timezone("Asia/Kuala_Lumpur")

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    today = datetime.now(MYT).strftime("%Y-%m-%d")
    return render_template('index.html', today=today, uploaded_video=latest_video_filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    global latest_video_filename
    if 'video' not in request.files:
        return redirect(url_for('index'))
    f = request.files['video']
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        latest_video_filename = filename
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    if latest_video_filename:
        return Response(
            detect_stream(os.path.join(UPLOAD_FOLDER, latest_video_filename)),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    return "No video uploaded", 404

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/latest-alert')
def latest_alert():
    a = get_latest_alert()
    if a:
        return jsonify({'alert': a, 'timestamp': datetime.now(MYT).isoformat()})
    return jsonify({'alert': None, 'timestamp': None})

# ---- helper: always show MYT time string
def _to_myt_iso(ts_myt: str, ts_utc: str) -> str:
    # We removed UTC in new logs; keep fallback for older rows.
    if ts_myt:
        return ts_myt
    if not ts_utc:
        return ""
    try:
        txt = ts_utc.replace('Z', '+00:00')
        dt_utc = datetime.fromisoformat(txt)
        dt_myt = dt_utc.astimezone(MYT)
        return dt_myt.isoformat()
    except Exception:
        return ts_utc

def _cleanup_daily_pdfs(base_dir: str):
    """Remove any daily_summary.pdf files (we don’t use PDFs anymore)."""
    if not os.path.isdir(base_dir):
        return
    for d in os.listdir(base_dir):
        p = os.path.join(base_dir, d, "daily_summary.pdf")
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass

# -----------------------------
# JKR Low-Visibility Dashboard
# -----------------------------
@app.route('/lowvis')
def lowvis():
    # remove leftover PDFs so they don’t linger in the file explorer
    _cleanup_daily_pdfs(DETECTION_FOLDER)

    date_str = request.args.get('date') or datetime.now(MYT).strftime("%Y-%m-%d")

    only_lowvis = str(request.args.get('only_lowvis', '1')).lower() in ('1','true','yes','on')
    label_filter = (request.args.get('label') or '').strip()
    hint_filter  = (request.args.get('hint')  or '').strip()
    trusted_only = str(request.args.get('trusted_only', '0')).lower() in ('1','true','yes','on')  # show BOTH by default

    csv_path = os.path.join(DETECTION_FOLDER, date_str, "detections.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                # numeric
                try: r['visibility_score'] = float(r.get('visibility_score') or 0.0)
                except: r['visibility_score'] = 0.0
                try: r['confidence'] = float(r.get('confidence') or 0.0)
                except: r['confidence'] = 0.0

                # booleans
                r['low_visibility'] = str(r.get('low_visibility','')).strip().lower() in ('true','1','yes')
                if 'is_trusted' in r and r['is_trusted'] != '':
                    r['is_trusted'] = str(r['is_trusted']).strip().lower() in ('true','1','yes')
                else:
                    r['is_trusted'] = r['confidence'] >= 0.70

                # preview & time
                rel = (r.get('image_relpath') or '').strip()
                r['image_url'] = url_for('static', filename=f"detections/{rel}") if rel else None
                r['time_display'] = _to_myt_iso(r.get('timestamp_myt',''), r.get('timestamp_utc',''))

                rows.append(r)

    # filters
    if only_lowvis:
        rows = [r for r in rows if r['low_visibility']]
    if trusted_only:
        rows = [r for r in rows if r['is_trusted']]
    if label_filter:
        rows = [r for r in rows if r.get('label','').lower() == label_filter.lower()]
    if hint_filter:
        rows = [r for r in rows if r.get('problem_hint','').lower() == hint_filter.lower()]

    rows.sort(key=lambda r: r['visibility_score'])
    label_opts = sorted({r.get('label','') for r in rows if r.get('label')})
    hint_opts  = sorted({r.get('problem_hint','') for r in rows if r.get('problem_hint')})
    top20 = rows[:20]

    # Only Excel download remains
    xlsx_url = url_for('static', filename=f"detections/{date_str}/detections.xlsx")

    available_dates = []
    if os.path.isdir(DETECTION_FOLDER):
        for d in sorted(os.listdir(DETECTION_FOLDER)):
            if os.path.isfile(os.path.join(DETECTION_FOLDER, d, "detections.csv")):
                available_dates.append(d)

    return render_template(
        'lowvis.html',
        date=date_str,
        rows=rows,
        top20=top20,
        only_lowvis=only_lowvis,
        label_filter=label_filter,
        hint_filter=hint_filter,
        trusted_only=trusted_only,
        label_opts=label_opts,
        hint_opts=hint_opts,
        xlsx_url=xlsx_url,
        available_dates=available_dates
    )

# -------------------------------------------------
# NEW: JSON feed for “Today’s Detections” thumbnails
# -------------------------------------------------
@app.route('/detection-feed')
def detection_feed():
    """Return list of today's annotated images for the gallery."""
    day = request.args.get('date') or datetime.now(MYT).strftime("%Y-%m-%d")
    day_dir = os.path.join(DETECTION_FOLDER, day)
    out = []

    csv_path = os.path.join(day_dir, "detections.csv")
    seen = set()

    if os.path.exists(csv_path):
        # Prefer the CSV so we only show images that actually correspond to saved detections.
        with open(csv_path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rel = (r.get('image_relpath') or '').strip()
                if not rel or rel in seen:
                    continue
                img_abs = os.path.join("static", "detections", rel)
                if os.path.exists(img_abs):
                    out.append({"filename": f"detections/{rel}"})
                    seen.add(rel)
    else:
        # Fallback: just list images in the folder
        if os.path.isdir(day_dir):
            for fname in sorted(os.listdir(day_dir)):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    out.append({"filename": f"detections/{day}/{fname}"})

    # Show newest first (feel free to reverse if you prefer oldest first)
    return jsonify(out[::-1])

# ---------------------------------------
# NEW: dated log page using popup.html
# ---------------------------------------
@app.route('/logs')
def logs():
    log_date = (request.args.get('log_date') or "").strip()
    if not log_date:
        return redirect(url_for('index'))

    # prevent future dates
    try:
        chosen = date.fromisoformat(log_date)
    except Exception:
        chosen = date.today()
    future = chosen > date.fromisoformat(datetime.now(MYT).strftime("%Y-%m-%d"))

    detections = []
    day_dir = os.path.join(DETECTION_FOLDER, log_date)
    csv_path = os.path.join(day_dir, "detections.csv")
    if os.path.exists(csv_path) and not future:
        with open(csv_path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rel = (r.get('image_relpath') or '').strip()
                if not rel:
                    continue
                detections.append({
                    "filename": f"detections/{rel}",
                    "label": r.get("label",""),
                    "timestamp": _to_myt_iso(r.get("timestamp_myt",""), r.get("timestamp_utc",""))
                })

    # Deduplicate by filename (keep first)
    seen = set()
    deduped = []
    for d in detections:
        if d["filename"] in seen:
            continue
        seen.add(d["filename"])
        deduped.append(d)

    return render_template(
        'popup.html',
        detections=deduped,
        log_date=log_date,
        future=future
    )

if __name__ == '__main__':
    app.run(debug=True)
