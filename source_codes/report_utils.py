import os, csv
from collections import Counter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def _to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")

def _read_csv_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def ensure_daily_summary_pdf(detection_folder, date, topn=12):
    date_dir = os.path.join(detection_folder, date)
    csv_path = os.path.join(date_dir, "detections.csv")
    out_path = os.path.join(date_dir, "daily_summary.pdf")
    if not os.path.exists(csv_path):
        return None

    regenerate = (not os.path.exists(out_path)) or (os.path.getmtime(out_path) < os.path.getmtime(csv_path))
    if not regenerate:
        return f"{date}/daily_summary.pdf"

    rows = _read_csv_rows(csv_path)

    # parse helpers (fallbacks for old CSVs)
    def _float(x, d=0.0):
        try: return float(x)
        except: return d

    for r in rows:
        r["visibility_score"] = _float(r.get("visibility_score"), 0.0)
        r["confidence"] = _float(r.get("confidence"), 0.0)
        r["low_visibility"] = _to_bool(r.get("low_visibility", "false"))
        if "is_trusted" in r and r["is_trusted"] != "":
            r["is_trusted"] = _to_bool(r["is_trusted"])
        else:
            r["is_trusted"] = r["confidence"] >= 0.70

    # Counts by label: TRUSTED rows only
    trusted_rows = [r for r in rows if r["is_trusted"]]
    label_counts = Counter([r.get("label","") for r in trusted_rows if r.get("label")])

    # Worst low-vis thumbnails (independent of trusted flag)
    worst = [r for r in rows if r["low_visibility"]]
    worst.sort(key=lambda r: r["visibility_score"])
    worst = worst[:topn]

    c = canvas.Canvas(out_path, pagesize=A4)
    W, H = A4
    margin = 1.5*cm
    y = H - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"Daily Summary — {date}")
    y -= 0.8*cm

    total = len(rows)
    total_low = sum(1 for r in rows if r["low_visibility"])
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Total detections: {total}    Low-visibility: {total_low}    (Counts by label: trusted only)")
    y -= 0.6*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Counts by label (trusted only):")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    col_w = (W - 2*margin) / 2
    i = 0
    for lab, cnt in label_counts.most_common():
        text = f"{lab}: {cnt}"
        x = margin + (i % 2) * col_w
        c.drawString(x, y, text)
        if i % 2 == 1:
            y -= 0.45*cm
        i += 1
        if y < 6*cm:
            c.showPage()
            y = H - margin
            c.setFont("Helvetica", 10)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"Worst {len(worst)} low-visibility detections:")
    y -= 0.6*cm

    cell_w = (W - 2*margin) / 3
    cell_h = 5.0*cm
    pad = 0.15*cm

    for idx, r in enumerate(worst):
        if y < (margin + cell_h + 1.5*cm):
            c.showPage()
            y = H - margin
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, f"Worst (cont’d):")
            y -= 0.6*cm

        col = idx % 3
        if col == 0 and idx > 0:
            y -= (cell_h + 1.1*cm)

        x0 = margin + col * cell_w
        rel = (r.get("image_relpath") or "").strip()
        img_abs = os.path.join(detection_folder, rel) if rel else None
        if img_abs and os.path.exists(img_abs):
            img_w = cell_w - 2*pad
            img_h = cell_h - 2*pad
            c.drawImage(img_abs, x0 + pad, y - cell_h + pad, width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')

        lbl = r.get("label","")
        hint = r.get("problem_hint","")
        act = r.get("jkr_action","")
        vs = _float(r.get("visibility_score"), 0.0)
        t = (r.get("timestamp_utc") or "")
        c.setFont("Helvetica", 9)
        caption = f"{lbl} | vis:{vs:.1f} | {hint} | {t}"
        c.drawString(x0, y - cell_h - 0.3*cm, caption)
        if act:
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(x0, y - cell_h - 0.6*cm, act)

    c.save()
    return f"{date}/daily_summary.pdf"
