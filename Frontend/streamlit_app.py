"""
NeuroSense - EEG and MRI analysis workspace (research prototype)
Run: streamlit run Frontend/streamlit_app.py

Single file, no extra config needed.

UI: clinical light theme (white surfaces, deep teal accent, restrained colour),
health-themed loading screen that checks the backend, report-style results.
Built only with Streamlit components, CSS and inline SVG.
Backend contract (POST /predict, GET /health) is unchanged.

Optional: add .streamlit/config.toml so Streamlit's own widgets match the theme:

    [theme]
    base = "light"
    primaryColor = "#0B7A75"
    backgroundColor = "#F3F6F8"
    secondaryBackgroundColor = "#FFFFFF"
    textColor = "#102A3C"
"""

import html
import io
import json
import math
import os
import time

import pandas as pd
import requests
import streamlit as st

EEG_LEN = 178


def _default_api() -> str:
    """Backend URL: Streamlit secret, then env var, then localhost."""
    try:
        if "NEUROSENSE_API" in st.secrets:
            return str(st.secrets["NEUROSENSE_API"])
    except Exception:  # no secrets file configured
        pass
    return os.getenv("NEUROSENSE_API", "http://localhost:8000")


DEFAULT_API = _default_api()

st.set_page_config(
    page_title="NeuroSense | EEG and MRI analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# THEME
# All colours are CSS variables in :root. Change them there to re-theme.
# =====================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #F3F6F8;
    --surface: #FFFFFF;
    --ink: #102A3C;
    --muted: #5B6F80;
    --faint: #8A9AA8;
    --line: #D8E1E8;
    --teal: #0B7A75;
    --teal-dark: #095F5B;
    --teal-soft: #E3F2F1;
    --ok: #2E7D5B;
    --ok-soft: #E4F3EB;
    --warn: #B26A00;
    --warn-soft: #FDF1DC;
    --alert: #B83B3B;
    --alert-soft: #FBE7E7;
    --radius: 10px;
}

/* ---------- page ---------- */
.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.block-container { padding-top: 1.6rem; max-width: 1160px; }
[data-testid="stHeader"] { background: rgba(243, 246, 248, .85); backdrop-filter: blur(6px); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    letter-spacing: -0.005em;
}
.stApp h2 { font-size: 1.45rem !important; }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] p { color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stImageCaption"] { color: var(--muted) !important; }
[data-testid="stAlert"] { border-radius: 8px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-baseweb="input"], [data-baseweb="base-input"] { background: #F7F9FB !important; border-radius: 8px !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within { border-color: var(--teal) !important; }
[data-testid="stSidebar"] input { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
hr { border-color: var(--line) !important; }

/* ---------- widgets ---------- */
[data-baseweb="slider"] [role="slider"] { background: var(--teal) !important; border-color: var(--teal) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="linear-gradient"],
[data-testid="stSelectSlider"] [data-baseweb="slider"] div[style*="linear-gradient"] {
    background: var(--teal) !important;
}
[data-testid="stFileUploader"] section {
    background: #F7F9FB; border: 1.5px dashed #9FB4C3; border-radius: 8px;
}
[data-testid="stFileUploader"] section * { color: var(--muted) !important; }
[data-testid="stFileUploader"] button {
    background: var(--surface) !important; border: 1px solid var(--teal) !important;
    color: var(--teal-dark) !important; border-radius: 8px !important;
}
[data-testid="stFileUploader"] button:hover { background: var(--teal-soft) !important; }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); }

.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-radius: 8px;
}
.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--teal); color: var(--teal-dark);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: var(--teal); color: #FFFFFF; border: none; border-radius: 8px;
    font-weight: 600; font-size: 16px; min-height: 52px;
}
.stButton > button[kind="primary"]:hover:not(:disabled),
button[data-testid="stBaseButton-primary"]:hover:not(:disabled) { background: var(--teal-dark); color: #FFFFFF; }
.stButton > button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled { background: #E3E9EE; color: var(--faint); }
.stButton > button:focus-visible { outline: 3px solid rgba(11, 122, 117, .45); outline-offset: 2px; }

/* bordered containers are the main white panels */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface); border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
}

/* tabs = main navigation */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; font-weight: 500; color: var(--muted); }
.stTabs [data-baseweb="tab"] * { color: inherit !important; }
.stTabs [aria-selected="true"] { color: var(--teal-dark) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--teal) !important; height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- top bar ---------- */
.topbar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    padding: 4px 0 18px 0; border-bottom: 1px solid var(--line); margin-bottom: 14px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 40px; height: 40px; border-radius: 10px; background: var(--teal);
    display: flex; align-items: center; justify-content: center;
}
.brand-mark svg { width: 24px; height: 24px; }
.brand-name { font-size: 22px; font-weight: 700; color: var(--ink); line-height: 1.1; }
.brand-sub { font-size: 13.5px; color: var(--muted); }
.pill {
    background: var(--warn-soft); color: var(--warn); border: 1px solid #F0D9A8;
    border-radius: 999px; padding: 5px 14px; font-size: 13px; font-weight: 600;
}
.intro { color: var(--muted); font-size: 15px; line-height: 1.65; max-width: 760px; margin: 0 0 14px 0; }

/* ---------- disclaimer ---------- */
.disclaimer {
    background: var(--warn-soft); border: 1px solid #F0D9A8; border-left: 4px solid var(--warn);
    border-radius: 8px; padding: 12px 16px; color: #5C3B00; font-size: 14px; line-height: 1.6; margin: 0 0 18px 0;
}
.disclaimer b { color: #3F2800; }

/* ---------- loading screen ---------- */
.boot {
    position: fixed; inset: 0; z-index: 99999; background: var(--bg);
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px;
    font-family: 'IBM Plex Sans', system-ui, sans-serif; padding: 24px; text-align: center;
}
.boot svg { width: min(460px, 86vw); height: auto; }
.boot-ecg { stroke-dasharray: 1; stroke-dashoffset: 1; animation: ns-ecg 2.4s linear infinite; }
@keyframes ns-ecg { 0% { stroke-dashoffset: 1; } 70% { stroke-dashoffset: 0; } 100% { stroke-dashoffset: 0; opacity: .15; } }
.boot-title { font-size: 26px; font-weight: 700; color: var(--ink); }
.boot-msg { font-size: 15px; color: var(--muted); min-height: 22px; }
.boot-bar { width: min(320px, 70vw); height: 4px; background: var(--line); border-radius: 999px; overflow: hidden; }
.boot-bar span { display: block; height: 100%; width: 40%; background: var(--teal); border-radius: 999px; animation: ns-slide 1.4s ease-in-out infinite; }
@keyframes ns-slide { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }
.boot-note { font-size: 12.5px; color: var(--faint); }

/* ---------- sidebar pieces ---------- */
.conn { display: flex; align-items: center; gap: 8px; font-size: 14px; margin: 2px 0 10px 0; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--faint); }
.dot.ok { background: var(--ok); } .dot.bad { background: var(--alert); }
.split { display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: var(--line); margin-top: 4px; }
.split-eeg { background: var(--teal); }
.split-mri { background: #5B8DB8; }
.split-legend { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); margin-top: 6px; }

/* ---------- panels ---------- */
.panel-title { font-size: 17px; font-weight: 600; color: var(--ink); margin: 0 0 2px 0; }
.panel-hint { color: var(--muted); font-size: 13.5px; margin: 0 0 10px 0; }

.status { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; }
.status-label { color: var(--muted); font-size: 13px; }
.status-value { font-size: 17px; font-weight: 600; margin-top: 2px; }
.tone-ok { color: var(--ok); } .tone-bad { color: var(--alert); }
.tone-idle { color: var(--faint); } .tone-info { color: var(--ink); }

.stat-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 12px 0; }
.stat { flex: 1 1 110px; background: #F7F9FB; border: 1px solid var(--line); border-radius: 8px; padding: 9px 14px; }
.stat-label { color: var(--muted); font-size: 12.5px; }
.stat-value { color: var(--ink); font-size: 17px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px 0; }
.chip { background: var(--teal-soft); border: 1px solid #BFE0DD; color: var(--teal-dark); border-radius: 999px; padding: 3px 12px; font-size: 12.5px; }

/* ---------- signal viewer ---------- */
.viewer svg { width: 100%; height: auto; display: block; }
.viewer .grid { stroke: #E3EAF0; stroke-width: 1; }
.viewer .axis { fill: var(--muted); font-size: 12px; font-family: 'IBM Plex Sans', sans-serif; }
.viewer .trace { stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; fill: none; stroke: var(--teal); }
.viewer .trace-smooth { stroke: #1D4ED8; stroke-width: 2.4; stroke-dasharray: 7 5; fill: none; stroke-linecap: round; }
.viewer .hit { fill: rgba(11, 122, 117, 0); cursor: crosshair; }
.viewer .hit:hover { fill: var(--teal); stroke: #FFFFFF; stroke-width: 2; }
.legend { display: flex; gap: 18px; color: var(--muted); font-size: 13px; margin: 2px 0 6px 0; }
.legend i { display: inline-block; width: 22px; height: 3px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }

/* ---------- results ---------- */
.report { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px 24px; }
.report-label { color: var(--muted); font-size: 13.5px; }
.report-score { font-size: 54px; font-weight: 700; line-height: 1.05; margin: 2px 0 8px 0; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600; font-size: 14px; }
.scale { position: relative; display: flex; height: 12px; border-radius: 999px; overflow: visible; margin: 26px 0 6px 0; }
.scale .seg { height: 100%; }
.scale .seg.low { width: 40%; background: #BFE3D0; border-radius: 999px 0 0 999px; }
.scale .seg.mod { width: 30%; background: #F3D9A0; }
.scale .seg.high { width: 30%; background: #EDB4B4; border-radius: 0 999px 999px 0; }
.scale .marker {
    position: absolute; top: -7px; width: 4px; height: 26px; border-radius: 2px;
    background: var(--ink); transform: translateX(-2px);
}
.scale-ticks { position: relative; height: 16px; font-size: 12px; color: var(--faint); }
.scale-ticks span { position: absolute; transform: translateX(-50%); }
.band-note { border-top: 1px solid var(--line); margin-top: 18px; padding-top: 14px; color: var(--ink); font-size: 14.5px; line-height: 1.65; }
.band-note small { display: block; color: var(--muted); margin-top: 8px; font-size: 13px; }

.bar-row { margin: 11px 0; }
.bar-head { display: flex; justify-content: space-between; font-size: 14px; color: var(--ink); margin-bottom: 5px; }
.bar-head span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
.bar-track { height: 8px; border-radius: 999px; background: #E6ECF1; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; background: #8FB9C7; }
.bar-fill.top { background: var(--teal); }

.big-value { font-size: 30px; font-weight: 700; color: var(--ink); margin: 2px 0 4px 0; }
.big-label { color: var(--muted); font-size: 13.5px; }

.rawjson {
    background: #F7F9FB; border: 1px solid var(--line); border-radius: 8px; color: var(--ink);
    padding: 12px 14px; font-size: 12.5px; line-height: 1.5; overflow-x: auto; max-height: 320px;
}

/* ---------- info tabs ---------- */
.info { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px; height: 100%; }
.info h4 { margin: 0 0 8px 0; font-size: 16.5px; }
.info p, .info li { color: var(--muted); font-size: 14.5px; line-height: 1.7; }
.info ul { margin: 0; padding-left: 18px; }
.step-no {
    display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 50%; background: var(--teal); color: #FFFFFF; font-weight: 600; font-size: 14px; margin-bottom: 8px;
}
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 6px 0 2px 0; }
.flow span { background: #F7F9FB; border: 1px solid var(--line); color: var(--ink); border-radius: 8px; padding: 7px 12px; font-size: 14px; }
.flow em { color: var(--teal); font-style: normal; }

.footer { color: var(--faint); text-align: center; font-size: 12.5px; padding: 26px 0 6px 0; }

@media (max-width: 820px) { .report-score { font-size: 44px; } }
@media (prefers-reduced-motion: reduce) { .boot-ecg, .boot-bar span { animation: none; stroke-dashoffset: 0; } }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# =====================================================================
# HELPERS
# =====================================================================
def md(markup: str):
    """Render an HTML snippet. Indentation and blank lines are removed so
    Streamlit's markdown parser never mistakes the HTML for a code block."""
    cleaned = "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def json_pre(obj):
    """Show JSON in a readable block."""
    text = html.escape(json.dumps(obj, indent=2, ensure_ascii=False))
    st.markdown(f'<pre class="rawjson">{text}</pre>', unsafe_allow_html=True)


def parse_eeg(raw: bytes):
    """Return a list of floats, or None if the file is not purely numeric."""
    try:
        text = raw.decode("utf-8", errors="ignore")
        values = [
            float(value)
            for value in text.replace(",", " ").replace(";", " ").split()
        ]
    except ValueError:
        return None
    if any(not math.isfinite(v) for v in values):  # rejects nan / inf
        return None
    return values


def risk_level(probability: float):
    """Return (label, text colour, badge background)."""
    if probability >= 0.70:
        return "High", "#B83B3B", "#FBE7E7"
    if probability >= 0.40:
        return "Moderate", "#B26A00", "#FDF1DC"
    return "Low", "#2E7D5B", "#E4F3EB"


BAND_NOTES = {
    "Low": (
        "The model output is in the lower range. A low score does not rule "
        "anything out, and it is not a diagnosis."
    ),
    "Moderate": (
        "The model output is in a middle range. Treat it as a research "
        "signal only, and discuss any symptoms or concerns with a clinician."
    ),
    "High": (
        "The model output is in the higher range. This is an experimental "
        "score, not a diagnosis. If you have symptoms or concerns, please "
        "talk to a qualified clinician."
    ),
}

HEART_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M2 12h4l2.5-6 4 12 2.5-6H22"/></svg>'
)


def panel_head(title: str, hint: str):
    md(
        f"""
        <div class="panel-title">{html.escape(title)}</div>
        <div class="panel-hint">{html.escape(hint)}</div>
        """
    )


def status_tile(label: str, value: str, tone: str):
    md(
        f"""
        <div class="status">
            <div class="status-label">{html.escape(label)}</div>
            <div class="status-value tone-{tone}">{html.escape(value)}</div>
        </div>
        """
    )


def stat_row(items):
    cells = "".join(
        f'<div class="stat"><div class="stat-label">{html.escape(k)}</div>'
        f'<div class="stat-value">{html.escape(v)}</div></div>'
        for k, v in items
    )
    md(f'<div class="stat-row">{cells}</div>')


def prob_bar(label: str, probability: float, highlight: bool = False):
    percent = min(max(float(probability), 0.0), 1.0) * 100
    return (
        '<div class="bar-row">'
        f'<div class="bar-head"><span>{html.escape(str(label))}</span>'
        f"<span>{percent:.1f}%</span></div>"
        '<div class="bar-track">'
        f'<div class="bar-fill{" top" if highlight else ""}" '
        f'style="width:{percent:.1f}%"></div></div></div>'
    )


# =====================================================================
# LOADING SCREEN (runs once per session, checks the backend for real)
# =====================================================================
def _ecg_path(width=460, height=90, beats=3) -> str:
    """One ECG-style trace made of repeated P-QRS-T beats."""
    mid = height / 2
    beat_w = width / beats
    pts = [f"0 {mid}"]
    for b in range(beats):
        x0 = b * beat_w
        shape = [
            (0.10, 0), (0.16, -6), (0.22, 0),            # P wave
            (0.34, 0), (0.38, 8), (0.44, -38), (0.50, 30), (0.55, 0),  # QRS
            (0.68, 0), (0.76, -12), (0.84, 0), (1.00, 0),  # T wave
        ]
        for frac, dy in shape:
            pts.append(f"{x0 + frac * beat_w:.1f} {mid + dy:.1f}")
    return "M" + " L".join(pts)


def boot_html(message: str, note: str = "") -> str:
    return (
        '<div class="boot" role="status" aria-live="polite">'
        '<svg viewBox="0 0 460 90" aria-hidden="true">'
        '<path d="M0 45 H460" stroke="#D8E1E8" stroke-width="1.5" fill="none"/>'
        f'<path class="boot-ecg" d="{_ecg_path()}" pathLength="1" fill="none" '
        'stroke="#0B7A75" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
        '<div class="boot-title">NeuroSense</div>'
        f'<div class="boot-msg">{html.escape(message)}</div>'
        '<div class="boot-bar"><span></span></div>'
        f'<div class="boot-note">{html.escape(note)}</div>'
        "</div>"
    )


def run_boot_screen():
    """Show the loading screen while we wake up and ping the backend."""
    slot = st.empty()
    slot.markdown(
        boot_html("Starting the analysis workspace", "Research prototype - not for clinical use"),
        unsafe_allow_html=True,
    )
    started = time.time()
    backend_ok = False
    for _ in range(4):  # free hosts can take a few seconds to wake up
        try:
            reply = requests.get(f"{DEFAULT_API.rstrip('/')}/health", timeout=4)
            reply.raise_for_status()
            backend_ok = True
            break
        except requests.RequestException:
            time.sleep(1.5)
    if time.time() - started < 2.0:  # avoid a flash on fast connections
        time.sleep(2.0 - (time.time() - started))

    slot.markdown(
        boot_html(
            "Analysis server connected" if backend_ok else "Analysis server not reachable yet",
            "You can change the server address in the sidebar.",
        ),
        unsafe_allow_html=True,
    )
    time.sleep(0.7)
    slot.empty()
    st.session_state["booted"] = True
    st.session_state["backend_ok"] = backend_ok


if not st.session_state.get("booted"):
    run_boot_screen()


# ---------- interactive EEG viewer (inline SVG built from the uploaded values) ----------
def signal_svg(values, start, end, smoothed=None) -> str:
    seg = values[start : end + 1]
    n = len(seg)
    W, H = 900, 320
    pl, pr, pt, pb = 66, 18, 16, 42

    lo, hi = min(seg), max(seg)
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    margin = (hi - lo) * 0.08
    lo, hi = lo - margin, hi + margin

    def px(i):
        return pl + (W - pl - pr) * (i / (n - 1))

    def py(v):
        return pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo))

    parts = [
        "<defs>"
        '<linearGradient id="ns-sig-fill" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#0B7A75" stop-opacity=".14"/>'
        '<stop offset="1" stop-color="#0B7A75" stop-opacity="0"/></linearGradient></defs>'
    ]

    for k in range(5):
        frac = k / 4
        y = pt + (H - pt - pb) * frac
        parts.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{W - pr}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{pl - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">'
            f"{hi - (hi - lo) * frac:.3g}</text>"
        )
    for k in range(5):
        frac = k / 4
        x = pl + (W - pl - pr) * frac
        parts.append(
            f'<text x="{x:.1f}" y="{H - 20}" text-anchor="middle" class="axis">'
            f"{start + round((n - 1) * frac)}</text>"
        )
    parts.append(
        f'<text x="{(pl + W - pr) / 2:.0f}" y="{H - 3}" text-anchor="middle" class="axis">Sample index</text>'
    )
    parts.append(
        f'<text x="14" y="{(pt + H - pb) / 2:.0f}" text-anchor="middle" class="axis" '
        f'transform="rotate(-90 14 {(pt + H - pb) / 2:.0f})">Amplitude</text>'
    )

    line_pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(seg))
    parts.append(
        f'<polygon points="{px(0):.1f},{H - pb} {line_pts} {px(n - 1):.1f},{H - pb}" fill="url(#ns-sig-fill)"/>'
    )
    parts.append(f'<polyline points="{line_pts}" class="trace"/>')

    if smoothed is not None:
        sm = smoothed[start : end + 1]
        sm_pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(sm))
        parts.append(f'<polyline points="{sm_pts}" class="trace-smooth"/>')

    for i, v in enumerate(seg):
        parts.append(
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="7" class="hit">'
            f"<title>Sample {start + i}: {v:.5g}</title></circle>"
        )

    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        'aria-label="EEG signal plot. Hover over the line to read individual sample values.">'
        + "".join(parts)
        + "</svg>"
    )


# ---------- results ----------
def show_risk(probability: float):
    level, color, background = risk_level(probability)
    percent = min(max(float(probability) * 100, 0), 100)

    md(
        f"""
        <div class="report">
            <div class="report-label">Combined screening score</div>
            <div class="report-score" style="color:{color}">{percent:.1f}%</div>
            <span class="badge" style="background:{background}; color:{color}">{level} range</span>
            <div class="scale" role="img" aria-label="Score {percent:.1f} percent on a scale from low to high">
                <div class="seg low"></div><div class="seg mod"></div><div class="seg high"></div>
                <div class="marker" style="left:{percent:.1f}%"></div>
            </div>
            <div class="scale-ticks">
                <span style="left:0%">0</span><span style="left:40%">40</span>
                <span style="left:70%">70</span><span style="left:100%">100</span>
            </div>
            <div class="band-note">
                {BAND_NOTES[level]}
                <small>
                    Sudden severe headache, weakness on one side, confusion or a
                    seizure needs emergency care. Do not wait for a screening result.
                </small>
            </div>
        </div>
        """
    )
    st.caption(
        "Ranges are display bands (below 40%, 40-70%, 70% and above), not clinical "
        "thresholds. This score reflects the configured model output and is not a "
        "validated estimate of clinical risk."
    )


def show_eeg_result(eeg_result: dict):
    probability = float(eeg_result["probability"])
    with st.container(border=True):
        panel_head("EEG model", "Signal analysis")
        md(
            f"""
            <div class="big-label">EEG anomaly probability</div>
            <div class="big-value">{probability * 100:.1f}%</div>
            {prob_bar("Model output", probability, highlight=True)}
            """
        )


def show_mri_result(mri_result: dict):
    probabilities = mri_result.get("probs", {})
    if isinstance(probabilities, dict):
        pairs = list(probabilities.items())
    else:
        pairs = [(f"Class {i + 1}", p) for i, p in enumerate(probabilities)]

    cleaned = []
    for name, value in pairs:
        try:
            cleaned.append((str(name), float(value)))
        except (TypeError, ValueError):
            continue
    cleaned.sort(key=lambda item: item[1], reverse=True)

    with st.container(border=True):
        panel_head("MRI model", "Image classification")
        bars = "".join(
            prob_bar(name, value, highlight=(i == 0))
            for i, (name, value) in enumerate(cleaned)
        )
        md(
            f"""
            <div class="big-label">Model label</div>
            <div class="big-value">{html.escape(str(mri_result["label"]))}</div>
            <div class="big-label" style="margin-top:10px">Class probabilities</div>
            {bars}
            """
        )


def show_results(result: dict):
    st.markdown("## Analysis results")
    st.caption("Model output from the connected FastAPI backend.")

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        show_risk(float(result["final_risk"]))
        if result.get("mode") == "fused":
            fused_alpha = float(result["alpha"])
            md(
                f"""
                <div class="chip-row">
                    <span class="chip">Combined analysis: EEG {fused_alpha:.0%} / MRI {1 - fused_alpha:.0%}</span>
                </div>
                """
            )

    with right:
        if "eeg" in result:
            show_eeg_result(result["eeg"])
        if "mri" in result:
            show_mri_result(result["mri"])

    with st.expander("View raw API response"):
        json_pre(result)

    st.caption(
        "Clinical interpretation requires qualified healthcare professionals "
        "and appropriate validated diagnostic procedures."
    )


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Analysis server")
    backend_ok = st.session_state.get("backend_ok")
    if backend_ok is True:
        md('<div class="conn"><span class="dot ok"></span>Connected</div>')
    elif backend_ok is False:
        md('<div class="conn"><span class="dot bad"></span>Not reachable</div>')

    api_url = st.text_input("FastAPI backend URL", DEFAULT_API).rstrip("/")

    if st.button("Check connection", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            st.session_state["backend_ok"] = True
            st.success("Backend is responding")
            json_pre(response.json())
        except requests.RequestException as exc:
            st.session_state["backend_ok"] = False
            st.error(f"Backend unavailable: {exc}")
        except ValueError:
            st.error("Backend returned invalid JSON.")

    st.divider()
    st.markdown("### Combined analysis")
    alpha = st.slider(
        "EEG fusion weight (α)",
        0.0,
        1.0,
        0.5,
        0.05,
        help="Used only when both EEG and MRI files are supplied.",
    )
    md(
        f"""
        <div class="split">
            <div class="split-eeg" style="width:{alpha * 100:.0f}%"></div>
            <div class="split-mri" style="width:{(1 - alpha) * 100:.0f}%"></div>
        </div>
        <div class="split-legend"><span>EEG {alpha:.0%}</span><span>MRI {1 - alpha:.0%}</span></div>
        """
    )

    st.divider()
    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()


# =====================================================================
# HEADER + DISCLAIMER (always visible)
# =====================================================================
md(
    f"""
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">{HEART_ICON}</div>
            <div>
                <div class="brand-name">NeuroSense</div>
                <div class="brand-sub">EEG and MRI analysis workspace</div>
            </div>
        </div>
        <div class="pill">Research prototype</div>
    </div>
    <p class="intro">
        Upload an EEG recording, an MRI image, or both. NeuroSense runs experimental
        machine-learning models and shows each output with its limits.
    </p>
    """
)

md(
    """
    <div class="disclaimer">
        <b>Medical disclaimer.</b> NeuroSense is an experimental research
        prototype. Its outputs are not clinical diagnoses and must not be
        used to diagnose, rule out or treat any medical condition. Always
        consult a qualified healthcare professional.
    </div>
    """
)

# =====================================================================
# NAVIGATION
# =====================================================================
tab_analyze, tab_how, tab_safety = st.tabs(
    ["Analysis", "How it works", "Safety and limits"]
)

# ---------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------
with tab_analyze:
    st.markdown("## Upload patient data")
    st.caption("Upload one modality, or provide both for a combined analysis.")

    col_eeg, col_mri = st.columns(2, gap="large")

    eeg_bytes = None
    mri_bytes = None
    eeg_file = None
    mri_file = None
    eeg_values = None
    eeg_ok = True
    eeg_state = ("Waiting for file", "idle")
    mri_state = ("Waiting for file", "idle")

    with col_eeg:
        with st.container(border=True):
            panel_head(
                "EEG recording",
                f"CSV or TXT file with exactly {EEG_LEN} numeric values.",
            )
            eeg_file = st.file_uploader(
                "Choose EEG file",
                type=["csv", "txt"],
                key="eeg",
                label_visibility="collapsed",
            )

            if eeg_file is not None:
                eeg_bytes = eeg_file.getvalue()
                eeg_values = parse_eeg(eeg_bytes)

                if eeg_values is None or len(eeg_values) != EEG_LEN:
                    count = 0 if eeg_values is None else len(eeg_values)
                    st.error(f"Expected {EEG_LEN} numeric values; found {count}.")
                    eeg_ok = False
                    eeg_values = None
                    eeg_state = ("Needs attention", "bad")
                else:
                    st.success(f"{len(eeg_values)} numeric values loaded")
                    eeg_state = ("Ready", "ok")

    with col_mri:
        with st.container(border=True):
            panel_head(
                "MRI scan",
                "JPG or PNG brain image for model inference.",
            )
            mri_file = st.file_uploader(
                "Choose MRI image",
                type=["jpg", "jpeg", "png"],
                key="mri",
                label_visibility="collapsed",
            )

            if mri_file is not None:
                mri_bytes = mri_file.getvalue()
                st.image(
                    mri_bytes,
                    caption="Uploaded image",
                    use_container_width=True,
                )
                details = [mri_file.name, f"{len(mri_bytes) / 1024:.0f} KB"]
                try:
                    from PIL import Image

                    width, height = Image.open(io.BytesIO(mri_bytes)).size
                    details.insert(1, f"{width}×{height} px")
                except Exception:  # preview details are optional
                    pass
                chips = "".join(
                    f'<span class="chip">{html.escape(d)}</span>' for d in details
                )
                md(f'<div class="chip-row">{chips}</div>')
                mri_state = ("Ready", "ok")

    # ---------- interactive EEG viewer ----------
    if eeg_values is not None:
        st.write("")
        with st.container(border=True):
            panel_head(
                "Signal viewer",
                "Zoom into a range of samples and hover over the line to read exact values.",
            )

            ctl1, ctl2, ctl3 = st.columns([2.2, 1, 1.2], gap="medium")
            with ctl1:
                sample_range = st.slider(
                    "Sample range",
                    0,
                    EEG_LEN - 1,
                    (0, EEG_LEN - 1),
                    key="eeg_range",
                )
            with ctl2:
                show_smooth = st.checkbox("Show smoothed trace", key="eeg_smooth")
            with ctl3:
                smooth_window = st.select_slider(
                    "Smoothing window",
                    options=[3, 5, 9, 15, 21],
                    value=5,
                    key="eeg_window",
                    disabled=not show_smooth,
                )

            start, end = sample_range
            if end - start < 4:  # keep at least 5 samples visible
                end = min(start + 4, EEG_LEN - 1)
                start = max(end - 4, 0)

            segment = eeg_values[start : end + 1]
            smoothed = None
            if show_smooth:
                smoothed = (
                    pd.Series(eeg_values)
                    .rolling(smooth_window, center=True, min_periods=1)
                    .mean()
                    .tolist()
                )

            stat_row(
                [
                    ("Samples shown", str(len(segment))),
                    ("Minimum", f"{min(segment):.4g}"),
                    ("Maximum", f"{max(segment):.4g}"),
                    ("Mean", f"{sum(segment) / len(segment):.4g}"),
                ]
            )

            legend = '<span><i style="background:#0B7A75"></i>Raw signal</span>'
            if show_smooth:
                legend += (
                    f'<span><i style="background:#1D4ED8"></i>Moving average ({smooth_window})</span>'
                )
            md(
                f"""
                <div class="viewer">
                    <div class="legend">{legend}</div>
                    {signal_svg(eeg_values, start, end, smoothed)}
                </div>
                """
            )

    # ---------- selected mode + status ----------
    if eeg_bytes is not None and mri_bytes is not None:
        mode = "EEG + MRI combined"
    elif eeg_bytes is not None:
        mode = "EEG only"
    elif mri_bytes is not None:
        mode = "MRI only"
    else:
        mode = None

    st.write("")
    t1, t2, t3 = st.columns(3, gap="medium")
    with t1:
        status_tile("EEG file", *eeg_state)
    with t2:
        status_tile("MRI image", *mri_state)
    with t3:
        status_tile(
            "Analysis mode",
            mode or "Upload a file to begin",
            "info" if mode else "idle",
        )

    # ---------- analysis ----------
    st.write("")
    run = st.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
        disabled=(mode is None or not eeg_ok),
    )

    if run:
        st.session_state.pop("result", None)  # never show a stale result
        files = {}

        if eeg_bytes is not None:
            files["eeg"] = (
                eeg_file.name,
                eeg_bytes,
                "text/csv",
            )

        if mri_bytes is not None:
            files["mri"] = (
                mri_file.name,
                mri_bytes,
                mri_file.type,
            )

        try:
            with st.spinner("Analysing your uploaded data..."):
                response = requests.post(
                    f"{api_url}/predict",
                    files=files,
                    data={"alpha": alpha},
                    timeout=60,
                )

            if response.status_code != 200:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Analysis failed: {detail}")
            else:
                try:
                    st.session_state["result"] = response.json()
                except ValueError:
                    st.error("Backend returned invalid JSON.")

        except requests.RequestException as exc:
            st.error(
                f"Could not reach the backend at {api_url}. "
                f"Check the address in the sidebar. Details: {exc}"
            )

    # ---------- results ----------
    result = st.session_state.get("result")
    if result:
        st.divider()
        if "final_risk" not in result:
            st.error("The backend response did not include a combined score.")
            with st.expander("View raw API response"):
                json_pre(result)
        else:
            show_results(result)

# ---------------------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------------------
with tab_how:
    st.markdown("## How NeuroSense works")
    st.caption("From upload to results in three steps.")

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        md(
            f"""
            <div class="info">
                <div class="step-no">1</div>
                <h4>Upload</h4>
                <p>Add an EEG file with exactly {EEG_LEN} numeric values, an MRI image
                (JPG or PNG), or both.</p>
            </div>
            """
        )
    with s2:
        md(
            """
            <div class="info">
                <div class="step-no">2</div>
                <h4>Analyse</h4>
                <p>Your files are sent to the FastAPI backend set in the sidebar.
                The models run there and return their outputs.</p>
            </div>
            """
        )
    with s3:
        md(
            """
            <div class="info">
                <div class="step-no">3</div>
                <h4>Review</h4>
                <p>See each model's output. When both inputs are supplied, a combined
                score is weighted by the fusion setting in the sidebar.</p>
            </div>
            """
        )

    st.write("")
    i1, i2 = st.columns(2, gap="medium")
    with i1:
        md(
            f"""
            <div class="info">
                <h4>Accepted inputs</h4>
                <ul>
                    <li><b>EEG:</b> CSV or TXT, {EEG_LEN} numbers separated by commas,
                    semicolons or whitespace.</li>
                    <li><b>MRI:</b> a JPG, JPEG or PNG image.</li>
                    <li><b>Modes:</b> EEG only, MRI only, or both together.</li>
                </ul>
            </div>
            """
        )
    with i2:
        md(
            """
            <div class="info">
                <h4>What you get back</h4>
                <ul>
                    <li><b>EEG model:</b> an anomaly probability.</li>
                    <li><b>MRI model:</b> a predicted label with a probability for each class.</li>
                    <li><b>Combined score:</b> one screening score, shown in a Low,
                    Moderate or High display range.</li>
                </ul>
            </div>
            """
        )

    st.write("")
    md(
        """
        <div class="info">
            <h4>Data flow</h4>
            <div class="flow">
                <span>Your browser</span><em>→</em>
                <span>Streamlit dashboard</span><em>→</em>
                <span>POST /predict</span><em>→</em>
                <span>FastAPI backend and models</span><em>→</em>
                <span>JSON results</span>
            </div>
        </div>
        """
    )

# ---------------------------------------------------------------------
# SAFETY
# ---------------------------------------------------------------------
with tab_safety:
    st.markdown("## Safety and limits")
    st.caption("Please read this before interpreting any result.")

    a1, a2 = st.columns(2, gap="medium")
    with a1:
        md(
            """
            <div class="info">
                <h4>What this is</h4>
                <ul>
                    <li>An experimental research prototype for exploring model outputs.</li>
                    <li>A way to visualise EEG signals and image-classification results.</li>
                </ul>
            </div>
            """
        )
    with a2:
        md(
            """
            <div class="info">
                <h4>What this is not</h4>
                <ul>
                    <li>Not a medical device and not a diagnostic tool.</li>
                    <li>Not validated for clinical use. Scores can be wrong.</li>
                    <li>Not a replacement for a qualified healthcare professional.</li>
                </ul>
            </div>
            """
        )

    st.write("")
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        md(
            """
            <div class="info">
                <h4>Your data</h4>
                <ul>
                    <li>Uploaded files are sent only to the backend URL set in the sidebar.</li>
                    <li>Avoid uploading identifiable patient data to a backend you do not control.</li>
                </ul>
            </div>
            """
        )
    with b2:
        md(
            """
            <div class="info">
                <h4>When to seek care</h4>
                <ul>
                    <li>Sudden severe headache, weakness on one side, confusion, trouble
                    speaking or a seizure needs emergency care right away.</li>
                    <li>For any ongoing symptoms or concerns, talk to a clinician.</li>
                </ul>
            </div>
            """
        )

# =====================================================================
# FOOTER
# =====================================================================
md(
    """
    <div class="footer">
        NeuroSense - AI-assisted neurological research prototype - Not for clinical use<br>
        Built with Streamlit, FastAPI and machine learning
    </div>
    """
)