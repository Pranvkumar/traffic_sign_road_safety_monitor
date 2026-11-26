"""Streamlit dashboard (optional dependency): `streamlit run app/dashboard.py`.

Shows sign boxes, class labels, road overlay, timings and model status in plain language.
It never shows fabricated predictions: if a checkpoint is missing it says so ("model not
loaded"), which is different from "loaded, no sign detected".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_monitor import DISCLAIMER  # noqa: E402

UNKNOWN_THRESHOLD_TEXT = "60%"


def load_pipeline_safely(config_path: str = "configs/yolo.yaml"):
    """Return (pipeline or None, message). Testable without Streamlit."""
    from traffic_monitor.pipeline.inference import Pipeline

    try:
        return Pipeline.from_config(config_path), "All configured models loaded."
    except (FileNotFoundError, ImportError) as exc:
        return None, f"Model not loaded: {exc}"


def pretty(name: str) -> str:
    """'speed_limit_50' -> 'Speed limit 50'."""
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def describe_result(result: dict | None, pipeline_loaded: bool) -> str:
    """Human-readable headline that distinguishes 'not loaded' from 'nothing detected'."""
    if not pipeline_loaded:
        return "Model not loaded — no predictions shown."
    if result is None:
        return "No image processed yet."
    if result["n_signs"] == 0:
        return "Models loaded: no traffic sign detected above the threshold."
    n = result["n_signs"]
    return f"Found {n} possible traffic sign{'s' if n != 1 else ''}."


def detections_table(result: dict) -> list[dict]:
    """One plain-language row per detected sign (numbers match the boxes drawn on the image)."""
    rows = []
    for i, d in enumerate(result.get("detections", []), 1):
        x1, y1, x2, y2 = (int(round(v)) for v in d["box_xyxy"])
        if d.get("class_name") == "unknown":
            sign = "Unknown sign"
            note = (f"Not sure what it is (best guess: {pretty(d.get('top_guess', '?'))}, "
                    f"only {d['class_confidence']:.0%} sure — below the confidence limit)")
        else:
            sign = pretty(d.get("class_name", "sign"))
            note = "Confident" if d.get("class_confidence", 0) >= 0.9 else "Fairly confident"
        rows.append({"#": i, "Sign": sign,
                     "How sure it is a sign": f"{d['detection_score']:.0%}",
                     "How sure of the sign type": f"{d.get('class_confidence', 0):.0%}",
                     "Size": f"{x2 - x1}×{y2 - y1} px",
                     "Position (left, top)": f"({x1}, {y1})",
                     "Comment": note})
    return rows


def human_summary(result: dict) -> list[str]:
    """Short bullet points explaining the whole result in plain language."""
    lines = []
    dets = result.get("detections", [])
    named = [pretty(d["class_name"]) for d in dets if d.get("class_name") not in (None, "unknown")]
    unknown = sum(1 for d in dets if d.get("class_name") == "unknown")
    if dets:
        part = f"recognised as: {', '.join(named)}" if named else "none could be named confidently"
        lines.append(f"**Signs:** {len(dets)} possible sign(s) found — {part}"
                     + (f"; {unknown} marked *unknown*." if unknown and named else "."))
    else:
        lines.append("**Signs:** no traffic sign found in this image.")
    road = result.get("road")
    if road is not None:
        lines.append(f"**Road:** about {road['road_coverage']:.0%} of the image is road surface (shaded pink).")
    else:
        lines.append("**Road:** road segmentation was not run.")
    t = result.get("timing_ms", {})
    if t:
        lines.append(f"**Speed:** {t.get('total_ms', 0) / 1000:.1f} s in total "
                     f"(finding signs {t.get('detection_ms', 0) / 1000:.1f} s, "
                     f"naming them {t.get('classification_ms', 0):.0f} ms, "
                     f"road {t.get('segmentation_ms', 0):.0f} ms).")
    hz = result.get("hazards")
    if hz is None:
        lines.append("**Hazards:** not checked — no hazard model is loaded.")
    elif hz:
        lines.append(f"**Hazards:** {len(hz)} pothole(s) found (orange boxes; best score {max(h['score'] for h in hz):.2f}).")
    else:
        lines.append("**Hazards:** no pothole found.")
    lines.append("**Keep in mind:** sign names come from European traffic-sign data (ETSC, 171 sign types). "
                 "Indian signs mostly follow the same Vienna-convention shapes, but unusual designs can still be "
                 "named wrongly or marked *unknown*, and low-confidence boxes may be false alarms.")
    return lines


def main() -> None:  # pragma: no cover - UI code
    import cv2
    import numpy as np
    import pandas as pd
    import streamlit as st

    from traffic_monitor.pipeline.inference import draw, to_json

    st.set_page_config(page_title="Traffic Sign Monitor (research demo)", layout="wide")
    st.title("Traffic Sign Recognition & Road Monitor")
    st.warning(DISCLAIMER)
    pipe, msg = st.cache_resource(load_pipeline_safely)()     # load models once, not on every rerun
    st.sidebar.header("Model status")
    st.sidebar.write(msg)
    if pipe:
        for k, v in pipe.status().items():
            st.sidebar.write(f"**{pretty(k)}**: {v}")
    upload = st.file_uploader("Upload a road image (jpg/png/webp/bmp). Files are processed in memory and not stored.",
                              type=["jpg", "jpeg", "png", "webp", "bmp"])
    if upload and pipe:
        img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not decode this image. Try saving it as JPG or PNG.")
            st.stop()
        with st.spinner("Analysing image..."):
            res = pipe.predict(img)
        vis = draw(img, res)                                     # boxes are numbered to match the table
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width=min(1100, vis.shape[1]),
                 caption="Red boxes = possible signs (#numbers match the table). Pink = road surface.")
        st.subheader(describe_result(res, True))
        for line in human_summary(res):
            st.markdown(f"- {line}")
        rows = detections_table(res)
        if rows:
            st.markdown("#### Signs found")
            st.table(pd.DataFrame(rows).set_index("#"))
        with st.expander("Raw JSON output (for developers)"):
            st.json(to_json(res))
    elif upload:
        st.error(describe_result(None, False))


if __name__ == "__main__":  # pragma: no cover
    main()
