"""
NeuroSense - Multimodal EEG & MRI Analysis Workspace
Run:
    streamlit run Frontend/streamlit_app.py

Environment / Secrets:
    NEUROSENSE_API: Base URL for the FastAPI backend
                    (default: https://neurosense-roir.onrender.com)
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
DEFAULT_TIMEOUT_HEALTH = 6.0
DEFAULT_TIMEOUT_PREDICT = 60.0
FALLBACK_RENDER_URL = "https://neurosense-roir.onrender.com"

st.set_page_config(
    page_title="NeuroSense | Multimodal EEG & MRI Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CONFIGURATION & STYLING
# =====================================================================
def resolve_backend_url() -> str:
    """Silently resolve backend URL from st.secrets, env var, or Render default."""
    try:
        if "NEUROSENSE_API" in st.secrets and st.secrets["NEUROSENSE_API"]:
            return str(st.secrets["NEUROSENSE_API"]).strip().rstrip("/")
    except Exception:
        pass
    return os.getenv("NEUROSENSE_API", FALLBACK_RENDER_URL).strip().rstrip("/")


BACKEND_API_URL = resolve_backend_url()

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #F4F7FA;
    --surface: #FFFFFF;
    --surface-subtle: #F8FAFC;
    --ink: #0F172A;
    --muted: #334155;
    --faint: #64748B;
    --line: #CBD5E1;
    --teal: #0F766E;
    --teal-dark: #115E59;
    --teal-soft: #CCFBF1;
    --blue: #2563EB;
    --blue-dark: #1D4ED8;
    --blue-soft: #DBEAFE;
    --ok: #059669;
    --ok-soft: #ECFDF5;
    --warn: #D97706;
    --warn-soft: #FFFBEB;
    --alert: #DC2626;
    --alert-soft: #FEF2F2;
    --radius: 12px;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}

[data-testid="stHeader"] {
    background: rgba(244, 247, 250, 0.85);
    backdrop-filter: blur(8px);
}

/* Force crisp, high-contrast typography across Streamlit widgets */
.stApp p, .stApp label, .stApp span, .stMarkdown {
    color: var(--ink);
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
}

.stApp h2 { font-size: 1.38rem !important; margin-bottom: 0.2rem !important; }
.stApp h3 { font-size: 1.12rem !important; }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

/* Color-coded modality cards in the 2-column upload grid */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) [data-testid="stVerticalBlockBorderWrapper"] {
    border-top: 4px solid var(--teal) !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) [data-testid="stVerticalBlockBorderWrapper"] {
    border-top: 4px solid var(--blue) !important;
}

[data-testid="stFileUploader"] section {
    background: var(--surface-subtle);
    border: 1.5px dashed #94A3B8;
    border-radius: 10px;
    transition: border-color 0.2s ease, background 0.2s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--teal);
    background: #F0FDFA;
}

[data-testid="stFileUploader"] button {
    background: var(--surface) !important;
    border: 1.5px solid var(--teal) !important;
    color: var(--teal-dark) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: var(--teal-soft) !important;
}

/* Secondary buttons (sidebar / download) */
.stButton > button:not([kind="primary"]),
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease;
}

.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:hover {
    border-color: var(--teal) !important;
    color: var(--teal-dark) !important;
    background: #F0FDFA !important;
}

/* Primary Run Inference CTA button */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, var(--teal) 0%, var(--blue) 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    min-height: 50px;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stButton > button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] p {
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover:not(:disabled),
button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
}

.stButton > button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    background: #CBD5E1 !important;
    color: #64748B !important;
    box-shadow: none !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 12px;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 18px;
    font-weight: 600;
    color: var(--muted) !important;
    border-radius: 8px 8px 0 0;
}

.stTabs [data-baseweb="tab"] p {
    color: var(--muted) !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    background: #EFF6FF;
}

.stTabs [aria-selected="true"] p {
    color: var(--teal-dark) !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, var(--teal), var(--blue)) !important;
    height: 3px;
}

.stTabs [data-baseweb="tab-border"] { display: none; }

/* Hero Banner */
.ns-hero {
    background: linear-gradient(135deg, #0F766E 0%, #115E59 48%, #1E40AF 100%);
    border-radius: 14px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 14px;
    box-shadow: 0 6px 20px rgba(15, 118, 110, 0.18);
}

.ns-brand { display: flex; align-items: center; gap: 14px; }

.ns-mark {
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
}

.ns-mark svg { width: 24px; height: 24px; }
.ns-title { font-size: 1.45rem; font-weight: 700; color: #FFFFFF !important; line-height: 1.15; }
.ns-subtitle { font-size: 0.88rem; color: #E0F2FE !important; margin-top: 2px; }

.ns-pill {
    background: rgba(255, 255, 255, 0.18);
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 999px;
    padding: 6px 15px;
    font-size: 0.82rem;
    font-weight: 600;
    backdrop-filter: blur(4px);
}

/* Research Disclaimer */
.ns-disclaimer {
    background: var(--warn-soft);
    border: 1px solid #FDE68A;
    border-left: 4px solid var(--warn);
    border-radius: 10px;
    padding: 12px 16px;
    color: #78350F !important;
    font-size: 0.88rem;
    line-height: 1.55;
    margin-bottom: 18px;
}

.ns-disclaimer strong { color: #78350F !important; }

/* Sidebar Engine Status Card */
.ns-engine-card {
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.ns-engine-card.ok {
    background: var(--ok-soft);
    border: 1px solid #6EE7B7;
}

.ns-engine-card.bad {
    background: var(--warn-soft);
    border: 1px solid #FCD34D;
}

.ns-engine-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 0.9rem;
}

.ns-engine-card.ok .ns-engine-head { color: #065F46 !important; }
.ns-engine-card.bad .ns-engine-head { color: #92400E !important; }

.ns-engine-sub {
    font-size: 0.8rem;
    margin-top: 3px;
}

.ns-engine-card.ok .ns-engine-sub { color: #047857 !important; }
.ns-engine-card.bad .ns-engine-sub { color: #B45309 !important; }

.ns-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.ns-dot.ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.2); }
.ns-dot.bad { background: var(--warn); box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.2); }

/* Fusion Split Bar */
.ns-split {
    display: flex;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: #E2E8F0;
    margin-top: 8px;
}

.ns-split-eeg { background: var(--teal); transition: width 0.2s ease; }
.ns-split-mri { background: var(--blue); transition: width 0.2s ease; }

.ns-split-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
}

/* Modality Upload Cards */
.ns-panel-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 3px 9px;
    border-radius: 6px;
    margin-bottom: 6px;
}

.ns-badge-eeg { background: var(--teal-soft); color: var(--teal-dark) !important; }
.ns-badge-mri { background: var(--blue-soft); color: var(--blue-dark) !important; }

.ns-panel-title { font-size: 1.08rem; font-weight: 700; color: var(--ink) !important; margin-bottom: 2px; }
.ns-panel-hint { font-size: 0.85rem; color: var(--muted) !important; margin-bottom: 12px; }

.ns-status {
    background: var(--surface-subtle);
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 12px;
}

.ns-status-k { font-size: 0.78rem; color: var(--faint) !important; font-weight: 500; }
.ns-status-v { font-size: 0.96rem; font-weight: 700; margin-top: 2px; }
.ns-tone-ok { color: var(--ok) !important; }
.ns-tone-bad { color: var(--alert) !important; }
.ns-tone-idle { color: var(--muted) !important; }

/* Evaluation Report */
.ns-report {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
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
    font-weight: 700;
}

.ns-scale {
    position: relative;
    display: flex;
    height: 12px;
    border-radius: 999px;
    margin: 24px 0 6px 0;
}

.ns-seg-low { width: 40%; background: #6EE7B7; border-radius: 999px 0 0 999px; }
.ns-seg-mod { width: 30%; background: #FCD34D; }
.ns-seg-high { width: 30%; background: #FCA5A5; border-radius: 0 999px 999px 0; }

.ns-marker {
    position: absolute;
    top: -6px;
    width: 4px;
    height: 24px;
    border-radius: 2px;
    background: var(--ink);
    transform: translateX(-2px);
}

.ns-ticks {
    position: relative;
    height: 16px;
    font-size: 0.76rem;
    color: var(--faint) !important;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.ns-ticks span { position: absolute; transform: translateX(-50%); }

.ns-note {
    border-top: 1px solid #E2E8F0;
    margin-top: 16px;
    padding-top: 12px;
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--ink) !important;
}

.ns-note small {
    display: block;
    margin-top: 8px;
    color: var(--muted) !important;
    font-size: 0.8rem;
}

.ns-bar-row { margin: 11px 0; }
.ns-bar-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 4px;
}

.ns-bar-head span:last-child {
    color: var(--muted) !important;
    font-variant-numeric: tabular-nums;
}

.ns-bar-track {
    height: 9px;
    border-radius: 999px;
    background: #E2E8F0;
    overflow: hidden;
}

.ns-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: #93C5FD;
}

.ns-bar-fill.top { background: linear-gradient(90deg, var(--teal), var(--blue)); }

.ns-metric-big {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 2px 0 8px 0;
    font-variant-numeric: tabular-nums;
}

/* SVG Waveform Plot */
.ns-svg-wrap svg { width: 100%; height: auto; display: block; }
.ns-svg-wrap .grid { stroke: #E2E8F0; stroke-width: 1; }
.ns-svg-wrap .axis { fill: #475569; font-size: 12px; font-family: 'Plus Jakarta Sans', sans-serif; }
.ns-svg-wrap .trace { stroke: var(--teal); stroke-width: 2.2; fill: none; stroke-linejoin: round; }
.ns-svg-wrap .trace-smooth { stroke: var(--blue); stroke-width: 2.2; stroke-dasharray: 6 4; fill: none; }
.ns-svg-wrap .hit { fill: rgba(15, 118, 110, 0); cursor: crosshair; }
.ns-svg-wrap .hit:hover { fill: var(--teal); stroke: #FFFFFF; stroke-width: 2; }

.ns-legend {
    display: flex;
    gap: 18px;
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--muted) !important;
    margin-bottom: 8px;
}

.ns-legend i {
    display: inline-block;
    width: 18px;
    height: 3px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 6px;
}

/* Info Cards */
.ns-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 20px 22px;
    height: 100%;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}

.ns-card h4 { margin: 0 0 8px 0; font-size: 1.02rem; color: var(--ink) !important; }
.ns-card p, .ns-card li { color: var(--muted) !important; font-size: 0.9rem; line-height: 1.65; }
.ns-card ul { margin: 0; padding-left: 18px; }

.ns-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--teal), var(--blue));
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 10px;
}

.ns-footer {
    text-align: center;
    color: var(--faint) !important;
    font-size: 0.82rem;
    padding-top: 28px;
}
</style>
"""

PULSE_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" '
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
        return "High", "#DC2626", "#FEF2F2"
    if p >= 0.40:
        return "Moderate", "#D97706", "#FFFBEB"
    return "Low", "#059669", "#ECFDF5"


@st.cache_data(ttl=20, show_spinner=False)
def check_backend_health(base_url: str) -> HealthStatus:
    """Query GET /health silently with caching for fast navigation."""
    if not base_url:
        return HealthStatus(False, "No backend configured")
    try:
        resp = requests.get(f"{base_url}/health", timeout=DEFAULT_TIMEOUT_HEALTH)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = {"status": "ok"}
        return HealthStatus(True, "Models loaded & ready", data)
    except requests.RequestException:
        return HealthStatus(False, "Cold start takes ~30s on Render")


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
    except requests.Timeout:
        return False, "The inference server is still waking up from sleep. Please click Run Multimodal Inference again in a few seconds."
    except requests.RequestException as exc:
        return False, f"Unable to reach the inference service ({exc})."
    except ValueError:
        return False, "Backend returned an unreadable JSON payload."


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
        '<stop offset="0" stop-color="#0F766E" stop-opacity="0.20"/>'
        '<stop offset="1" stop-color="#0F766E" stop-opacity="0.0"/>'
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
            '<span class="ns-panel-badge ns-badge-eeg">1D-CNN Branch</span>'
            '<div class="ns-panel-title">EEG model output</div>'
            '<div class="ns-panel-hint">Temporal signal anomaly estimation</div>'
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
            '<span class="ns-panel-badge ns-badge-mri">MobileNetV2 Branch</span>'
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
    st.divider()
    st.markdown("## Evaluation report")
    st.caption("Multimodal inference results generated by the NeuroSense neural engine.")

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
    col_dl, _ = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="Download report (JSON)",
            data=serialized.encode("utf-8"),
            file_name="neurosense_prediction_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Inspect raw JSON response"):
        st.code(serialized, language="json")


# =====================================================================
# SIDEBAR CONTROLS (Clean User-Facing Layout)
# =====================================================================
st.markdown(THEME_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### NeuroSense Engine")
    health = check_backend_health(BACKEND_API_URL)
    card_cls = "ok" if health.reachable else "bad"
    dot_cls = "ok" if health.reachable else "bad"
    title_txt = "Inference API Online" if health.reachable else "Waking Up Server..."

    render_html(
        f"""
        <div class="ns-engine-card {card_cls}">
            <div class="ns-engine-head">
                <span class="ns-dot {dot_cls}"></span>
                <span>{html.escape(title_txt)}</span>
            </div>
            <div class="ns-engine-sub">{html.escape(health.detail)}</div>
        </div>
        """
    )

    c_ref, c_rst = st.columns(2)
    with c_ref:
        if st.button("Refresh status", use_container_width=True):
            check_backend_health.clear()
            st.rerun()
    with c_rst:
        if st.button("Reset session", use_container_width=True):
            check_backend_health.clear()
            st.session_state.pop("prediction_result", None)
            st.rerun()

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
            <span style="color:#0F766E">EEG: {alpha:.0%}</span>
            <span style="color:#2563EB">MRI: {1.0 - alpha:.0%}</span>
        </div>
        """
    )

    st.divider()
    st.markdown("### Quick sample data")
    st.caption("Download a synthetic 178-sample CSV to test the EEG pipeline.")
    sample_mode = st.radio(
        "Sample profile",
        options=["Baseline waveform", "Elevated burst waveform"],
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
# HERO HEADER & DISCLAIMER
# =====================================================================
render_html(
    f"""
    <div class="ns-hero">
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
    eeg_filename: str = "eeg_recording.csv"
    eeg_values: list[float] | None = None
    eeg_valid = True
    eeg_status = ("Awaiting file", "idle")

    mri_bytes: bytes | None = None
    mri_filename: str = "mri_scan.png"
    mri_mime: str = "image/png"
    mri_status = ("Awaiting file", "idle")

    with col_eeg:
        with st.container(border=True):
            render_html(
                '<span class="ns-panel-badge ns-badge-eeg">Modality 01 · Signal</span>'
                '<div class="ns-panel-title">EEG recording</div>'
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
                eeg_filename = eeg_file.name
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
                '<span class="ns-panel-badge ns-badge-mri">Modality 02 · Imaging</span>'
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
                mri_filename = mri_file.name
                mri_mime = mri_file.type or "image/png"
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
                smooth_win = st.slider("Moving average window", 1, 15, 1, step=2)

            smoothed = smooth_values(eeg_values, smooth_win) if smooth_win > 1 else None
            legend_extra = (
                f'<span><i style="background:#2563EB"></i>Smoothed ({smooth_win}-pt)</span>'
                if smoothed is not None
                else ""
            )
            render_html(
                f'<div class="ns-legend">'
                f'<span><i style="background:#0F766E"></i>Raw EEG amplitude</span>'
                f'{legend_extra}</div>'
                f'<div class="ns-svg-wrap">{build_eeg_svg(eeg_values, idx_start, idx_end, smoothed)}</div>'
            )

    st.markdown("")
    has_any_valid_input = (eeg_values is not None and eeg_valid) or (mri_bytes is not None)
    can_run = has_any_valid_input and eeg_valid

    run_clicked = st.button(
        "Run Multimodal Inference",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )

    if not has_any_valid_input:
        st.caption("Upload at least one valid modality file to run inference.")

    if run_clicked:
        eeg_payload = (eeg_filename, eeg_bytes) if (eeg_bytes is not None and eeg_valid) else None
        mri_payload = (mri_filename, mri_bytes, mri_mime) if mri_bytes is not None else None

        with st.spinner("Running multimodal inference (waking server if idle)..."):
            ok, outcome = request_prediction(
                base_url=BACKEND_API_URL,
                eeg_payload=eeg_payload,
                mri_payload=mri_payload,
                alpha=alpha,
            )

        if ok and isinstance(outcome, dict):
            st.session_state["prediction_result"] = outcome
            check_backend_health.clear()
        else:
            st.error(str(outcome))

    if "prediction_result" in st.session_state:
        render_results(st.session_state["prediction_result"])


with tab_architecture:
    st.markdown("## System & methodology")
    st.caption("Dual-branch multimodal pipeline combining 1D temporal signal processing with 2D spatial imaging.")

    s1, s2, s3 = st.columns(3, gap="large")
    with s1:
        render_html(
            """
            <div class="ns-card">
                <div class="ns-step">01</div>
                <h4>1D-CNN EEG Branch</h4>
                <p>Processes standardized 178-sample EEG segments using multi-scale 1D convolutions,
                batch normalization, and pooling layers to capture high-frequency transient bursts
                and rhythmic anomalies.</p>
            </div>
            """
        )
    with s2:
        render_html(
            """
            <div class="ns-card">
                <div class="ns-step">02</div>
                <h4>MobileNetV2 MRI Branch</h4>
                <p>Extracts spatial representations from 2D MRI slices via depthwise separable
                convolutions, outputting multi-class probability distributions across structural
                brain categories.</p>
            </div>
            """
        )
    with s3:
        render_html(
            """
            <div class="ns-card">
                <div class="ns-step">03</div>
                <h4>Decision-Level Late Fusion</h4>
                <p>Combines normalized modality risk scores using configurable weight α
                (<code>α · P_EEG + (1 - α) · P_MRI</code>), while gracefully falling back to
                single-modality inference when only one input is supplied.</p>
            </div>
            """
        )


with tab_limitations:
    st.markdown("## Limitations & ethical considerations")
    st.caption("Important boundaries for interpreting experimental outputs from this build.")

    l1, l2 = st.columns(2, gap="large")
    with l1:
        render_html(
            """
            <div class="ns-card">
                <h4>Technical &amp; Dataset Constraints</h4>
                <ul>
                    <li><strong>Single-slice evaluation:</strong> The MRI branch inspects individual 2D slices rather than full 3D volumetric NIfTI studies.</li>
                    <li><strong>Fixed EEG window:</strong> Signals are constrained to 178-sample windows and do not capture multi-channel montage correlations.</li>
                    <li><strong>Domain shift:</strong> Scanner hardware, electrode impedance, and preprocessing pipelines can alter prediction calibration.</li>
                </ul>
            </div>
            """
        )
    with l2:
        render_html(
            """
            <div class="ns-card">
                <h4>Intended Use &amp; Safety</h4>
                <ul>
                    <li><strong>Non-clinical research only:</strong> This software has not undergone regulatory clearance (FDA/CE) or prospective clinical trials.</li>
                    <li><strong>Human-in-the-loop:</strong> Outputs should only support academic benchmarking and algorithm exploration, never patient triage.</li>
                    <li><strong>Data privacy:</strong> Avoid uploading identifiable patient health information (PHI) or un-anonymized DICOM metadata.</li>
                </ul>
            </div>
            """
        )

render_html('<div class="ns-footer">NeuroSense · Multimodal EEG &amp; MRI Research Workspace</div>')