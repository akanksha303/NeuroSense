"""
NeuroSense - Multimodal EEG & MRI Analysis Workspace
Run:
    streamlit run Frontend/streamlit_app.py

Environment / Secrets:
    NEUROSENSE_API: Base URL for the FastAPI backend (default: http://localhost:8000)
"""

from __future__ import annotations

import html
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

EEG_LEN = 178
DEFAULT_TIMEOUT_HEALTH = 5.0
DEFAULT_TIMEOUT_PREDICT = 45.0

st.set_page_config(
    page_title="NeuroSense | Multimodal EEG & MRI Workspace",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CONFIGURATION & STYLING
# =====================================================================
def resolve_backend_url() -> str:
    """Resolve backend URL from st.secrets, environment variable, or localhost."""
    try:
        if "NEUROSENSE_API" in st.secrets and st.secrets["NEUROSENSE_API"]:
            return str(st.secrets["NEUROSENSE_API"]).strip().rstrip("/")
    except Exception:
        pass
    return os.getenv("NEUROSENSE_API", "http://localhost:8000").strip().rstrip("/")


DEFAULT_API_URL = resolve_backend_url()

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #F3F6F8;
    --surface: #FFFFFF;
    --surface-subtle: #F7F9FB;
    --ink: #102A3C;
    --muted: #526778;
    --faint: #7E90A0;
    --line: #D8E1E8;
    --teal: #0B7A75;
    --teal-dark: #095F5B;
    --teal-soft: #E3F2F1;
    --ok: #256F4F;
    --ok-soft: #E4F3EB;
    --warn: #A66300;
    --warn-soft: #FDF1DC;
    --alert: #B03535;
    --alert-soft: #FBE7E7;
    --radius: 10px;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}

[data-testid="stHeader"] {
    background: rgba(243, 246, 248, 0.88);
    backdrop-filter: blur(6px);
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

.stApp h2 { font-size: 1.42rem !important; }
.stApp h3 { font-size: 1.15rem !important; }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
}

[data-testid="stFileUploader"] section {
    background: var(--surface-subtle);
    border: 1.5px dashed #9FB4C3;
    border-radius: 8px;
}

[data-testid="stFileUploader"] button {
    background: var(--surface) !important;
    border: 1px solid var(--teal) !important;
    color: var(--teal-dark) !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: var(--teal-soft) !important;
}

.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: var(--teal);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.98rem;
    min-height: 48px;
}

.stButton > button[kind="primary"]:hover:not(:disabled),
button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    background: var(--teal-dark);
    color: #FFFFFF;
}

.stButton > button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    background: #E2E8EE;
    color: var(--faint);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 18px;
    font-weight: 500;
    color: var(--muted);
}

.stTabs [aria-selected="true"] {
    color: var(--teal-dark) !important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: var(--teal) !important;
    height: 3px;
}

.stTabs [data-baseweb="tab-border"] { display: none; }

.ns-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 14px;
}

.ns-brand { display: flex; align-items: center; gap: 12px; }

.ns-mark {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--teal);
    display: flex;
    align-items: center;
    justify-content: center;
}

.ns-mark svg { width: 22px; height: 22px; }
.ns-title { font-size: 1.35rem; font-weight: 700; color: var(--ink); line-height: 1.15; }
.ns-subtitle { font-size: 0.85rem; color: var(--muted); }

.ns-pill {
    background: var(--teal-soft);
    color: var(--teal-dark);
    border: 1px solid #BFE0DD;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
}

.ns-disclaimer {
    background: var(--warn-soft);
    border: 1px solid #F0D9A8;
    border-left: 4px solid var(--warn);
    border-radius: 8px;
    padding: 12px 16px;
    color: #5C3B00;
    font-size: 0.88rem;
    line-height: 1.55;
    margin-bottom: 18px;
}

.ns-conn {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 10px;
}

.ns-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.ns-dot.ok { background: var(--ok); }
.ns-dot.bad { background: var(--alert); }
.ns-dot.unknown { background: var(--faint); }

.ns-split {
    display: flex;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--line);
    margin-top: 6px;
}

.ns-split-eeg { background: var(--teal); }
.ns-split-mri { background: #4F7FA8; }

.ns-split-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
}

.ns-panel-title { font-size: 1.04rem; font-weight: 600; color: var(--ink); margin-bottom: 2px; }
.ns-panel-hint { font-size: 0.84rem; color: var(--muted); margin-bottom: 10px; }

.ns-status {
    background: var(--surface-subtle);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
}

.ns-status-k { font-size: 0.78rem; color: var(--muted); }
.ns-status-v { font-size: 0.98rem; font-weight: 600; margin-top: 2px; }
.ns-tone-ok { color: var(--ok); }
.ns-tone-bad { color: var(--alert); }
.ns-tone-idle { color: var(--muted); }

.ns-report {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 20px 22px;
}

.ns-score {
    font-size: 3.1rem;
    font-weight: 700;
    line-height: 1.05;
    margin: 4px 0 10px 0;
    font-variant-numeric: tabular-nums;
}

.ns-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 600;
}

.ns-scale {
    position: relative;
    display: flex;
    height: 11px;
    border-radius: 999px;
    margin: 24px 0 6px 0;
}

.ns-seg-low { width: 40%; background: #BFE3D0; border-radius: 999px 0 0 999px; }
.ns-seg-mod { width: 30%; background: #F3D9A0; }
.ns-seg-high { width: 30%; background: #EDB4B4; border-radius: 0 999px 999px 0; }

.ns-marker {
    position: absolute;
    top: -6px;
    width: 4px;
    height: 23px;
    border-radius: 2px;
    background: var(--ink);
    transform: translateX(-2px);
}

.ns-ticks {
    position: relative;
    height: 16px;
    font-size: 0.75rem;
    color: var(--faint);
    font-variant-numeric: tabular-nums;
}

.ns-ticks span { position: absolute; transform: translateX(-50%); }

.ns-note {
    border-top: 1px solid var(--line);
    margin-top: 16px;
    padding-top: 12px;
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--ink);
}

.ns-note small {
    display: block;
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.8rem;
}

.ns-bar-row { margin: 10px 0; }
.ns-bar-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    margin-bottom: 4px;
}

.ns-bar-head span:last-child {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
}

.ns-bar-track {
    height: 8px;
    border-radius: 999px;
    background: #E6ECF1;
    overflow: hidden;
}

.ns-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: #8FB9C7;
}

.ns-bar-fill.top { background: var(--teal); }

.ns-metric-big {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--ink);
    margin: 2px 0 8px 0;
    font-variant-numeric: tabular-nums;
}

.ns-svg-wrap svg { width: 100%; height: auto; display: block; }
.ns-svg-wrap .grid { stroke: #E3EAF0; stroke-width: 1; }
.ns-svg-wrap .axis { fill: var(--muted); font-size: 12px; font-family: 'IBM Plex Sans', sans-serif; }
.ns-svg-wrap .trace { stroke: var(--teal); stroke-width: 2; fill: none; stroke-linejoin: round; }
.ns-svg-wrap .trace-smooth { stroke: #1D4ED8; stroke-width: 2.2; stroke-dasharray: 6 4; fill: none; }
.ns-svg-wrap .hit { fill: rgba(11, 122, 117, 0); cursor: crosshair; }
.ns-svg-wrap .hit:hover { fill: var(--teal); stroke: #FFFFFF; stroke-width: 2; }

.ns-legend {
    display: flex;
    gap: 18px;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 6px;
}

.ns-legend i {
    display: inline-block;
    width: 18px;
    height: 3px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 6px;
}

.ns-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 18px 20px;
    height: 100%;
}

.ns-card h4 { margin: 0 0 8px 0; font-size: 1rem; }
.ns-card p, .ns-card li { color: var(--muted); font-size: 0.9rem; line-height: 1.65; }
.ns-card ul { margin: 0; padding-left: 18px; }

.ns-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--teal);
    color: #FFFFFF;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 8px;
}

.ns-footer {
    text-align: center;
    color: var(--faint);
    font-size: 0.8rem;
    padding-top: 24px;
}
</style>
"""

PULSE_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M2 12h4l2.5-6 4 12 2.5-6H22"/></svg>'
)

BAND_GUIDANCE = {
    "Low": (
        "Score falls in the lower display band. A low score does not rule out "
        "clinical findings and is not a medical diagnosis."
    ),
    "Moderate": (
        "Score falls in the middle display band. Treat this output strictly as "
        "an exploratory signal and consult a qualified clinician."
    ),
    "High": (
        "Score falls in the higher display band. This is an unvalidated model "
        "output, not a diagnosis. Seek professional medical evaluation."
    ),
}


# =====================================================================
# DOMAIN LOGIC & BACKEND CLIENT
# =====================================================================
@dataclass(frozen=True)
class HealthStatus:
    reachable: bool
    detail: str
    payload: dict[str, Any] | None = None


def render_html(markup: str) -> None:
    cleaned = "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def parse_eeg_bytes(raw: bytes) -> list[float] | None:
    """Parse comma-, semicolon-, or whitespace-delimited finite floats."""
    try:
        text = raw.decode("utf-8", errors="ignore")
        tokens = text.replace(",", " ").replace(";", " ").split()
        values = [float(tok) for tok in tokens]
    except ValueError:
        return None
    if not values or any(not math.isfinite(v) for v in values):
        return None
    return values


def generate_sample_eeg_csv(seed: int = 42, anomalous: bool = False) -> bytes:
    """Generate a deterministic 178-sample EEG CSV for quick evaluation."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, EEG_LEN)
    signal = (
        0.65 * np.sin(2 * np.pi * 8.0 * t)
        + 0.30 * np.cos(2 * np.pi * 16.0 * t)
        + rng.normal(0.0, 0.12, size=EEG_LEN)
    )
    if anomalous:
        burst = np.exp(-0.5 * ((t - 0.55) / 0.06) ** 2) * np.sin(2 * np.pi * 28.0 * t)
        signal = signal + 2.2 * burst
    df = pd.DataFrame({"sample": np.arange(EEG_LEN), "amplitude": np.round(signal, 6)})
    buf = io.StringIO()
    df["amplitude"].to_csv(buf, index=False, header=False)
    return buf.getvalue().encode("utf-8")


def smooth_values(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    series = pd.Series(values, dtype="float64")
    return series.rolling(window=window, center=True, min_periods=1).mean().tolist()


def classify_band(probability: float) -> tuple[str, str, str]:
    """Return (band_label, text_color, badge_bg) for display purposes."""
    p = min(max(float(probability), 0.0), 1.0)
    if p >= 0.70:
        return "High", "#B03535", "#FBE7E7"
    if p >= 0.40:
        return "Moderate", "#A66300", "#FDF1DC"
    return "Low", "#256F4F", "#E4F3EB"


@st.cache_data(ttl=15, show_spinner=False)
def check_backend_health(base_url: str) -> HealthStatus:
    """Query GET /health with short caching to keep navigation responsive."""
    if not base_url:
        return HealthStatus(False, "No backend URL configured")
    try:
        resp = requests.get(f"{base_url}/health", timeout=DEFAULT_TIMEOUT_HEALTH)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = {"status": "ok (non-JSON response)"}
        return HealthStatus(True, f"HTTP {resp.status_code} · Healthy", data)
    except requests.RequestException as exc:
        return HealthStatus(False, str(exc))


def request_prediction(
    base_url: str,
    eeg_payload: tuple[str, bytes] | None,
    mri_payload: tuple[str, bytes, str] | None,
    alpha: float,
) -> tuple[bool, dict[str, Any] | str]:
    """Send selected modalities to POST /predict and return (ok, result_or_error)."""
    files: dict[str, tuple[Any, ...]] = {}
    if eeg_payload is not None:
        name, raw_bytes = eeg_payload
        files["eeg"] = (name, raw_bytes, "text/csv")
    if mri_payload is not None:
        name, raw_bytes, mime = mri_payload
        files["mri"] = (name, raw_bytes, mime)

    try:
        resp = requests.post(
            f"{base_url}/predict",
            files=files,
            data={"alpha": f"{alpha:.4f}"},
            timeout=DEFAULT_TIMEOUT_PREDICT,
        )
        resp.raise_for_status()
        parsed = resp.json()
        if not isinstance(parsed, dict) or "final_risk" not in parsed:
            return False, "Backend response missing required 'final_risk' field."
        return True, parsed
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"
    except ValueError:
        return False, "Backend returned invalid JSON from POST /predict."


# =====================================================================
# VISUALIZATION & REPORT COMPONENTS
# =====================================================================
def build_eeg_svg(
    values: list[float], start: int, end: int, smoothed: list[float] | None = None
) -> str:
    seg = values[start : end + 1]
    n = len(seg)
    width, height = 900, 310
    pad_l, pad_r, pad_t, pad_b = 64, 18, 14, 42

    lo, hi = min(seg), max(seg)
    if math.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    def to_x(idx: int) -> float:
        return pad_l + (width - pad_l - pad_r) * (idx / max(n - 1, 1))

    def to_y(val: float) -> float:
        return pad_t + (height - pad_t - pad_b) * (1.0 - (val - lo) / (hi - lo))

    svg_parts = [
        "<defs>"
        '<linearGradient id="ns-eeg-grad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#0B7A75" stop-opacity="0.15"/>'
        '<stop offset="1" stop-color="#0B7A75" stop-opacity="0"/>'
        "</linearGradient></defs>"
    ]

    for k in range(5):
        frac = k / 4.0
        y_pos = pad_t + (height - pad_t - pad_b) * frac
        tick_val = hi - (hi - lo) * frac
        svg_parts.append(
            f'<line x1="{pad_l}" y1="{y_pos:.1f}" x2="{width - pad_r}" y2="{y_pos:.1f}" class="grid"/>'
        )
        svg_parts.append(
            f'<text x="{pad_l - 8}" y="{y_pos + 4:.1f}" text-anchor="end" class="axis">'
            f"{tick_val:.3g}</text>"
        )

    for k in range(5):
        frac = k / 4.0
        x_pos = pad_l + (width - pad_l - pad_r) * frac
        sample_idx = start + round((n - 1) * frac)
        svg_parts.append(
            f'<text x="{x_pos:.1f}" y="{height - 20}" text-anchor="middle" class="axis">'
            f"{sample_idx}</text>"
        )

    svg_parts.append(
        f'<text x="{(pad_l + width - pad_r) / 2:.0f}" y="{height - 3}" '
        'text-anchor="middle" class="axis">Sample index</text>'
    )
    svg_parts.append(
        f'<text x="14" y="{(pad_t + height - pad_b) / 2:.0f}" text-anchor="middle" '
        f'class="axis" transform="rotate(-90 14 {(pad_t + height - pad_b) / 2:.0f})">Amplitude</text>'
    )

    line_coords = " ".join(f"{to_x(i):.1f},{to_y(v):.1f}" for i, v in enumerate(seg))
    svg_parts.append(
        f'<polygon points="{to_x(0):.1f},{height - pad_b} {line_coords} '
        f'{to_x(n - 1):.1f},{height - pad_b}" fill="url(#ns-eeg-grad)"/>'
    )
    svg_parts.append(f'<polyline points="{line_coords}" class="trace"/>')

    if smoothed is not None:
        sm_seg = smoothed[start : end + 1]
        sm_coords = " ".join(f"{to_x(i):.1f},{to_y(v):.1f}" for i, v in enumerate(sm_seg))
        svg_parts.append(f'<polyline points="{sm_coords}" class="trace-smooth"/>')

    for i, v in enumerate(seg):
        svg_parts.append(
            f'<circle cx="{to_x(i):.1f}" cy="{to_y(v):.1f}" r="6.5" class="hit">'
            f"<title>Sample {start + i}: {v:.5g}</title></circle>"
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="EEG waveform plot across selected sample window">'
        + "".join(svg_parts)
        + "</svg>"
    )


def render_prob_bar(label: str, probability: float, highlight: bool = False) -> str:
    pct = min(max(float(probability) * 100.0, 0.0), 100.0)
    fill_cls = "ns-bar-fill top" if highlight else "ns-bar-fill"
    return (
        '<div class="ns-bar-row">'
        f'<div class="ns-bar-head"><span>{html.escape(str(label))}</span>'
        f"<span>{pct:.1f}%</span></div>"
        f'<div class="ns-bar-track"><div class="{fill_cls}" style="width:{pct:.1f}%"></div></div>'
        "</div>"
    )


def render_score_card(final_risk: Any) -> None:
    try:
        prob = min(max(float(final_risk), 0.0), 1.0)
    except (TypeError, ValueError):
        st.warning("Screening score could not be parsed as a numeric value.")
        return

    pct = prob * 100.0
    band, color, bg = classify_band(prob)

    render_html(
        f"""
        <div class="ns-report">
            <div class="ns-panel-hint" style="margin-bottom:0">Combined screening score</div>
            <div class="ns-score" style="color:{color}">{pct:.1f}%</div>
            <span class="ns-badge" style="background:{bg}; color:{color}">{band} display band</span>
            <div class="ns-scale" role="img" aria-label="Score {pct:.1f}% on a 0 to 100 scale">
                <div class="ns-seg-low"></div>
                <div class="ns-seg-mod"></div>
                <div class="ns-seg-high"></div>
                <div class="ns-marker" style="left:{pct:.1f}%"></div>
            </div>
            <div class="ns-ticks">
                <span style="left:0%">0</span>
                <span style="left:40%">40</span>
                <span style="left:70%">70</span>
                <span style="left:100%">100</span>
            </div>
            <div class="ns-note">
                {html.escape(BAND_GUIDANCE[band])}
                <small>
                    Acute neurological symptoms (such as sudden severe headache, focal weakness,
                    confusion, or seizure) require emergency medical care.
                </small>
            </div>
        </div>
        """
    )
    st.caption(
        "Display bands (below 40%, 40–70%, 70% and above) are UI groupings only, "
        "not validated clinical thresholds."
    )


def render_eeg_card(eeg_data: dict[str, Any]) -> None:
    with st.container(border=True):
        render_html(
            '<div class="ns-panel-title">EEG model output</div>'
            '<div class="ns-panel-hint">1D signal anomaly estimation</div>'
        )
        raw_prob = eeg_data.get("probability")
        try:
            prob = min(max(float(raw_prob), 0.0), 1.0)
        except (TypeError, ValueError):
            st.info("EEG probability unavailable in response payload.")
            return

        render_html(
            f'<div class="ns-panel-hint" style="margin-bottom:0">Estimated probability</div>'
            f'<div class="ns-metric-big">{prob * 100:.1f}%</div>'
            + render_prob_bar("EEG anomaly probability", prob, highlight=True)
        )


def render_mri_card(mri_data: dict[str, Any]) -> None:
    with st.container(border=True):
        render_html(
            '<div class="ns-panel-title">MRI model output</div>'
            '<div class="ns-panel-hint">2D scan slice classification</div>'
        )
        label = str(mri_data.get("label", "Unknown"))
        probs = mri_data.get("probs", {})
        if isinstance(probs, dict):
            entries = list(probs.items())
        elif isinstance(probs, list):
            entries = [(f"Class {idx + 1}", val) for idx, val in enumerate(probs)]
        else:
            entries = []

        scored: list[tuple[str, float]] = []
        for cls_name, val in entries:
            try:
                scored.append((str(cls_name), min(max(float(val), 0.0), 1.0)))
            except (TypeError, ValueError):
                continue
        scored.sort(key=lambda item: item[1], reverse=True)

        bars_html = "".join(
            render_prob_bar(name, p, highlight=(idx == 0))
            for idx, (name, p) in enumerate(scored)
        )
        render_html(
            f'<div class="ns-panel-hint" style="margin-bottom:0">Predicted class</div>'
            f'<div class="ns-metric-big">{html.escape(label)}</div>'
            f'<div class="ns-panel-hint" style="margin-top:8px">Class distribution</div>'
            f"{bars_html}"
        )


def render_results(result: dict[str, Any]) -> None:
    st.markdown("## Evaluation report")
    st.caption("Output returned by POST /predict on the connected backend.")

    col_summary, col_models = st.columns([0.92, 1.08], gap="large")

    with col_summary:
        render_score_card(result.get("final_risk"))
        mode = result.get("mode")
        if mode == "fused" and "alpha" in result:
            try:
                a = float(result["alpha"])
                st.caption(f"Fusion mode active · EEG weight: {a:.0%} | MRI weight: {1.0 - a:.0%}")
            except (TypeError, ValueError):
                pass
        elif mode:
            st.caption(f"Execution mode: {mode}")

    with col_models:
        if isinstance(result.get("eeg"), dict):
            render_eeg_card(result["eeg"])
        if isinstance(result.get("mri"), dict):
            render_mri_card(result["mri"])

    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download report (JSON)",
        data=serialized.encode("utf-8"),
        file_name="neurosense_prediction_report.json",
        mime="application/json",
    )

    with st.expander("Inspect raw JSON response"):
        st.code(serialized, language="json")


# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
st.markdown(THEME_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Backend configuration")
    backend_url = st.text_input(
        "FastAPI endpoint base URL",
        value=DEFAULT_API_URL,
        help="Set via NEUROSENSE_API environment variable or Streamlit secrets.",
    ).strip().rstrip("/")

    col_ping, col_clear = st.columns(2)
    with col_ping:
        ping_clicked = st.button("Check health", use_container_width=True)
    with col_clear:
        if st.button("Reset session", use_container_width=True):
            check_backend_health.clear()
            st.session_state.pop("prediction_result", None)
            st.rerun()

    if ping_clicked:
        check_backend_health.clear()

    health = check_backend_health(backend_url)
    dot_cls = "ok" if health.reachable else "bad"
    status_txt = "Backend reachable" if health.reachable else "Backend unreachable"
    render_html(
        f'<div class="ns-conn"><span class="ns-dot {dot_cls}"></span>'
        f"<span>{html.escape(status_txt)}</span></div>"
    )
    if health.reachable:
        st.caption(health.detail)
        if health.payload is not None:
            with st.expander("Health payload"):
                st.json(health.payload)
    else:
        st.caption("Ensure the FastAPI service is running and reachable.")

    st.divider()
    st.markdown("### Multimodal fusion")
    alpha = st.slider(
        "EEG fusion weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Applies when both EEG and MRI inputs are provided (MRI weight = 1 - α).",
    )
    render_html(
        f"""
        <div class="ns-split">
            <div class="ns-split-eeg" style="width:{alpha * 100:.0f}%"></div>
            <div class="ns-split-mri" style="width:{(1.0 - alpha) * 100:.0f}%"></div>
        </div>
        <div class="ns-split-labels">
            <span>EEG: {alpha:.0%}</span>
            <span>MRI: {1.0 - alpha:.0%}</span>
        </div>
        """
    )

    st.divider()
    st.markdown("### Quick sample data")
    st.caption("Download a synthetic 178-sample CSV to test the EEG pipeline.")
    sample_mode = st.radio(
        "Sample profile",
        options=["Baseline waveform", "Elevated burst waveform"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_anomalous = sample_mode == "Elevated burst waveform"
    st.download_button(
        label="Download sample EEG (.csv)",
        data=generate_sample_eeg_csv(seed=101 if is_anomalous else 42, anomalous=is_anomalous),
        file_name="sample_eeg_burst_178.csv" if is_anomalous else "sample_eeg_baseline_178.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =====================================================================
# HEADER & DISCLAIMER
# =====================================================================
render_html(
    f"""
    <div class="ns-topbar">
        <div class="ns-brand">
            <div class="ns-mark">{PULSE_SVG}</div>
            <div>
                <div class="ns-title">NeuroSense</div>
                <div class="ns-subtitle">Multimodal EEG &amp; MRI Research Workspace</div>
            </div>
        </div>
        <div class="ns-pill">Capstone Research Build</div>
    </div>
    <div class="ns-disclaimer">
        <strong>Research &amp; Educational Notice:</strong> NeuroSense is a non-clinical
        prototype developed for research and engineering evaluation. Model outputs are not
        medical diagnoses and must not be used for clinical decision-making.
    </div>
    """
)


# =====================================================================
# MAIN WORKSPACE TABS
# =====================================================================
tab_analysis, tab_architecture, tab_limitations = st.tabs(
    ["Analysis workspace", "System & methodology", "Limitations & ethics"]
)

with tab_analysis:
    st.markdown("## Modality inputs")
    st.caption("Provide an EEG file, an MRI slice image, or both for fused inference.")

    col_eeg, col_mri = st.columns(2, gap="large")

    eeg_bytes: bytes | None = None
    eeg_values: list[float] | None = None
    eeg_valid = True
    eeg_status = ("Awaiting file", "idle")

    mri_bytes: bytes | None = None
    mri_status = ("Awaiting file", "idle")

    with col_eeg:
        with st.container(border=True):
            render_html(
                f'<div class="ns-panel-title">EEG recording</div>'
                f'<div class="ns-panel-hint">CSV or TXT containing exactly {EEG_LEN} numeric samples</div>'
            )
            eeg_file = st.file_uploader(
                "Upload EEG recording",
                type=["csv", "txt"],
                key="eeg_uploader",
                label_visibility="collapsed",
            )
            if eeg_file is not None:
                eeg_bytes = eeg_file.getvalue()
                parsed_eeg = parse_eeg_bytes(eeg_bytes)
                if parsed_eeg is None or len(parsed_eeg) != EEG_LEN:
                    found = 0 if parsed_eeg is None else len(parsed_eeg)
                    st.error(f"Invalid EEG input: expected {EEG_LEN} finite numbers, found {found}.")
                    eeg_valid = False
                    eeg_status = ("Validation failed", "bad")
                else:
                    eeg_values = parsed_eeg
                    eeg_status = ("Validated (178 samples)", "ok")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Samples", len(eeg_values))
                    m2.metric("Min", f"{min(eeg_values):.3g}")
                    m3.metric("Max", f"{max(eeg_values):.3g}")

            s_label, s_tone = eeg_status
            render_html(
                f'<div class="ns-status"><div class="ns-status-k">EEG status</div>'
                f'<div class="ns-status-v ns-tone-{s_tone}">{html.escape(s_label)}</div></div>'
            )

    with col_mri:
        with st.container(border=True):
            render_html(
                '<div class="ns-panel-title">MRI scan image</div>'
                '<div class="ns-panel-hint">PNG, JPG, or JPEG axial/sagittal/coronal slice</div>'
            )
            mri_file = st.file_uploader(
                "Upload MRI scan image",
                type=["png", "jpg", "jpeg"],
                key="mri_uploader",
                label_visibility="collapsed",
            )
            if mri_file is not None:
                mri_bytes = mri_file.getvalue()
                mri_status = ("Loaded image", "ok")
                st.image(
                    io.BytesIO(mri_bytes),
                    caption=f"{mri_file.name} ({len(mri_bytes) / 1024:.1f} KB)",
                    use_container_width=True,
                )

            s_label, s_tone = mri_status
            render_html(
                f'<div class="ns-status"><div class="ns-status-k">MRI status</div>'
                f'<div class="ns-status-v ns-tone-{s_tone}">{html.escape(s_label)}</div></div>'
            )

    if eeg_values is not None:
        st.markdown("### Waveform inspection")
        with st.container(border=True):
            render_html(
                '<div class="ns-panel-title">Interactive EEG signal viewer</div>'
                '<div class="ns-panel-hint">Inspect sample ranges and optional moving-average smoothing</div>'
            )
            c1, c2, c3 = st.columns([1.3, 1.3, 1.2], gap="medium")
            with c1:
                idx_start = st.slider("Start sample", 0, EEG_LEN - 10, 0)
            with c2:
                idx_end = st.slider("End sample", idx_start + 9, EEG_LEN - 1, EEG_LEN - 1)
            with c3:
                smooth_win = st.select_slider("Smoothing window", options=[1, 3, 5, 9, 15], value=1)

            smoothed = smooth_values(eeg_values, smooth_win) if smooth_win > 1 else None
            legend = '<div class="ns-legend"><span><i style="background:#0B7A75"></i>Raw signal</span>'
            if smoothed is not None:
                legend += '<span><i style="background:#1D4ED8"></i>Moving average</span>'
            legend += "</div>"
            render_html(legend)

            svg_markup = build_eeg_svg(eeg_values, idx_start, idx_end, smoothed)
            render_html(f'<div class="ns-svg-wrap">{svg_markup}</div>')

            with st.expander("View sample table"):
                df_samples = pd.DataFrame(
                    {
                        "sample_index": np.arange(EEG_LEN),
                        "amplitude": eeg_values,
                    }
                )
                st.dataframe(df_samples, use_container_width=True, hide_index=True)

    has_input = (eeg_values is not None) or (mri_bytes is not None)
    ready_to_run = eeg_valid and has_input

    st.write("")
    run_btn = st.button(
        "Run multimodal inference",
        type="primary",
        disabled=not ready_to_run,
        use_container_width=True,
    )

    if not has_input:
        st.caption("Upload at least one valid modality file to run inference.")

    if run_btn and ready_to_run:
        eeg_tuple = (eeg_file.name, eeg_bytes) if (eeg_values is not None and eeg_file and eeg_bytes) else None
        mri_tuple = (
            (mri_file.name, mri_bytes, mri_file.type or "image/png")
            if (mri_bytes is not None and mri_file)
            else None
        )

        with st.spinner("Sending inputs to FastAPI backend..."):
            ok, payload = request_prediction(backend_url, eeg_tuple, mri_tuple, alpha)

        if ok and isinstance(payload, dict):
            st.session_state["prediction_result"] = payload
        else:
            st.error(str(payload))

    if "prediction_result" in st.session_state:
        st.divider()
        render_results(st.session_state["prediction_result"])


with tab_architecture:
    st.markdown("## System & methodology")
    st.caption("Overview of client-side validation, backend inference contract, and fusion weighting.")

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        render_html(
            """
            <div class="ns-card">
                <div class="ns-step">1</div>
                <h4>Client-side pre-validation</h4>
                <p>
                    Verifies finite floating-point samples and exact 178-point length
                    for EEG files prior to network upload, reducing malformed requests.
                </p>
            </div>
            """
        )
    with c2:
        render_html(
            """
            <div class="ns-card">
                <div class="ns-step">2</div>
                <h4>FastAPI inference contract</h4>
                <p>
                    Submits multipart inputs to <code>POST /predict</code> with fusion
                    parameter <code>alpha</code> and monitors service readiness via
                    <code>GET /health</code>.
                </p>
            </div>
            """
        )
    with c3:
        render_html(
            """
            <div class="ns-card">
                <div class="ns-step">3</div>
                <h4>Transparent reporting</h4>
                <p>
                    Presents combined scores alongside individual EEG and MRI outputs,
                    downloadable JSON reports, and full raw payloads.
                </p>
            </div>
            """
        )


with tab_limitations:
    st.markdown("## Limitations & ethics")
    st.caption("Key technical boundaries and responsible-use guardrails for this prototype.")

    left, right = st.columns(2, gap="large")
    with left:
        render_html(
            """
            <div class="ns-card">
                <h4>Technical boundaries</h4>
                <ul>
                    <li>Display bands (0–40%, 40–70%, 70–100%) are arbitrary visual ranges, not calibrated medical cutoffs.</li>
                    <li>Single-modality and fused outputs depend on backend model weights, calibration, and input domain shift.</li>
                    <li>EEG windowing expects fixed-length 178-sample segments formatted consistently with training data.</li>
                </ul>
            </div>
            """
        )
    with right:
        render_html(
            """
            <div class="ns-card">
                <h4>Ethical &amp; clinical guardrails</h4>
                <ul>
                    <li>Not a medical device and not approved for clinical diagnosis, screening, or treatment planning.</li>
                    <li>Avoid uploading identifiable patient metadata or PHI unless governed by approved research protocols.</li>
                    <li>Always consult licensed clinicians for medical evaluation and emergency care.</li>
                </ul>
            </div>
            """
        )

    render_html(
        '<div class="ns-footer">NeuroSense · Multimodal EEG &amp; MRI Research Prototype</div>'
    )