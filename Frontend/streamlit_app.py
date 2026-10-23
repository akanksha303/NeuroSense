"""
NeuroSense — AI Neurological Screening Dashboard (research prototype)
Run: streamlit run Frontend/streamlit_app.py

Single file, no extra config needed.

UI: deep navy / purple / cyan theme, built only with Streamlit components,
CSS and inline SVG (no Plotly). Backend contract (POST /predict, GET /health)
is unchanged.
"""

import html
import io
import json
import math
import os

import pandas as pd
import requests
import streamlit as st

EEG_LEN = 178
DEFAULT_API = os.getenv("NEUROSENSE_API", "http://localhost:8000")

st.set_page_config(
    page_title="NeuroSense | AI Brain Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# THEME
# All colours are CSS variables in :root. Change them there to re-theme.
# =====================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg0: #060A1C;
    --bg1: #0B1030;
    --panel: rgba(18, 25, 66, .66);
    --line: rgba(150, 160, 255, .18);
    --text: #EAF0FF;
    --muted: #A3AEDB;
    --faint: #7782B5;
    --violet: #8B5CF6;
    --indigo: #4F46E5;
    --cyan: #22D3EE;
    --pink: #F472B6;
    --ok: #34D399;
    --warn: #FBBF24;
    --alert: #FB7185;
    --grad: linear-gradient(120deg, #7C3AED 0%, #4F46E5 48%, #06B6D4 100%);
    --radius: 20px;
}

/* ---------- page ---------- */
.stApp {
    background:
        radial-gradient(900px 520px at 6% -6%, rgba(124, 58, 237, .34), transparent 60%),
        radial-gradient(820px 520px at 100% 0%, rgba(6, 182, 212, .20), transparent 55%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    background-attachment: fixed;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }

[data-testid="stHeader"] {
    background: rgba(6, 10, 28, .55);
    backdrop-filter: blur(8px);
}
[data-testid="stHeader"] * { color: #C9D1F5; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Sora', sans-serif !important;
    color: #FFFFFF !important;
    letter-spacing: -0.01em;
}
[data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"],
[data-testid="stImageCaption"] { color: var(--muted) !important; }
[data-testid="stSpinner"] * { color: var(--muted) !important; }
[data-testid="stAlert"] { border-radius: 12px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0F2C 0%, #131A4A 60%, #1C1F66 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--text); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
[data-baseweb="input"], [data-baseweb="base-input"] {
    background: #0C1238 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] input {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

/* ---------- widgets ---------- */
[data-testid="stSlider"] *,
[data-testid="stSelectSlider"] * { color: var(--text); }
[data-baseweb="slider"] [role="slider"] {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 4px rgba(34, 211, 238, .25) !important;
}
/* recolour Streamlit's default red accents (no config.toml needed) */
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="linear-gradient"],
[data-testid="stSelectSlider"] [data-baseweb="slider"] div[style*="linear-gradient"] {
    background: linear-gradient(90deg, #8B5CF6, #22D3EE) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {
    border-color: var(--cyan) !important;
}
[data-testid="stSidebar"] hr, [data-testid="stMain"] hr { border-color: var(--line) !important; }
[data-testid="stCheckbox"] p,
[data-testid="stWidgetLabel"] p { color: var(--text) !important; }

[data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, .04);
    border: 1.5px dashed rgba(139, 92, 246, .65);
    border-radius: 14px;
}
[data-testid="stFileUploader"] section * { color: var(--muted) !important; }
[data-testid="stFileUploader"] button {
    background: rgba(139, 92, 246, .18) !important;
    border: 1px solid rgba(139, 92, 246, .55) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] button:hover { border-color: var(--cyan) !important; }

[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text) !important; }

/* secondary buttons */
.stButton > button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, .06);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 12px;
    transition: border-color .2s, transform .2s;
}
.stButton > button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--cyan);
    color: #FFFFFF;
    transform: translateY(-1px);
}

/* primary action */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: var(--grad);
    color: #FFFFFF;
    border: none;
    border-radius: 14px;
    font-weight: 700;
    font-size: 17px;
    min-height: 58px;
    box-shadow: 0 10px 30px rgba(99, 102, 241, .45), 0 0 0 1px rgba(255, 255, 255, .08) inset;
    transition: transform .2s, box-shadow .2s, filter .2s;
}
.stButton > button[kind="primary"]:hover:not(:disabled),
button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    transform: translateY(-2px);
    filter: brightness(1.08);
    box-shadow: 0 14px 38px rgba(34, 211, 238, .38);
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    background: rgba(255, 255, 255, .08);
    color: var(--faint);
    box-shadow: none;
}

/* bordered containers act as cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 14px 40px rgba(3, 6, 24, .45);
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(34, 211, 238, .42) !important;
}

/* tabs = main navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255, 255, 255, .04);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 11px;
    padding: 0 20px;
    font-weight: 600;
    color: var(--muted);
}
.stTabs [data-baseweb="tab"] * { color: inherit !important; }
.stTabs [aria-selected="true"] { background: var(--grad); color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 46px 50px 78px 50px;
    margin-bottom: 16px;
    background:
        radial-gradient(620px 340px at 86% 22%, rgba(34, 211, 238, .30), transparent 62%),
        radial-gradient(540px 360px at 8% 0%, rgba(139, 92, 246, .46), transparent 62%),
        linear-gradient(135deg, #0B1240 0%, #1B1672 55%, #0D587F 100%);
    border: 1px solid rgba(160, 170, 255, .25);
    box-shadow: 0 24px 60px rgba(5, 8, 30, .55), inset 0 1px 0 rgba(255, 255, 255, .08);
}
.hero-net {
    position: absolute; inset: 0; width: 100%; height: 100%;
    opacity: .34; pointer-events: none;
}
.hero-inner {
    position: relative;
    display: flex; align-items: center; justify-content: space-between; gap: 28px;
}
.hero-copy { max-width: 600px; }
.hero-chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(34, 211, 238, .12);
    border: 1px solid rgba(34, 211, 238, .4);
    color: #A5F3FC;
    padding: 6px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 600;
}
.pulse {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 0 0 rgba(34, 211, 238, .7);
    animation: ns-pulse 2s infinite;
}
@keyframes ns-pulse {
    70% { box-shadow: 0 0 0 9px rgba(34, 211, 238, 0); }
    100% { box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 44px; font-weight: 700; line-height: 1.12;
    color: #FFFFFF; margin: 16px 0 12px 0;
}
.hero-sub { color: #CFD8FF; font-size: 16px; line-height: 1.7; margin: 0 0 20px 0; }
.hero-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.hero-tags span {
    background: rgba(255, 255, 255, .09);
    border: 1px solid rgba(255, 255, 255, .16);
    color: #E3E9FF;
    padding: 6px 12px; border-radius: 999px;
    font-size: 13px; font-weight: 500;
}
.hero-brain {
    width: 300px; height: auto; flex-shrink: 0;
    filter: drop-shadow(0 0 26px rgba(139, 92, 246, .6));
}
.hero-brain .node { animation: ns-node 2.8s ease-in-out infinite; }
.hero-brain .node:nth-of-type(2n) { animation-delay: .9s; }
.hero-brain .node:nth-of-type(3n) { animation-delay: 1.8s; }
@keyframes ns-node { 50% { opacity: .35; } }
.hero-wave {
    position: absolute; left: 0; right: 0; bottom: 14px;
    width: 100%; height: 56px; opacity: .92; pointer-events: none;
}
.hero-wave path { stroke-dasharray: 1; stroke-dashoffset: 1; animation: ns-draw 3.2s ease-out .3s forwards; }
@keyframes ns-draw { to { stroke-dashoffset: 0; } }

/* ---------- disclaimer ---------- */
.disclaimer {
    display: flex; gap: 14px; align-items: flex-start;
    background: rgba(251, 191, 36, .08);
    border: 1px solid rgba(251, 191, 36, .35);
    border-left: 5px solid var(--warn);
    border-radius: 14px;
    padding: 14px 18px;
    color: #FDE7A8; font-size: 14px; line-height: 1.6;
    margin: 0 0 20px 0;
}
.disclaimer b { color: #FFFFFF; }
.disclaimer-icon { font-size: 22px; line-height: 1.3; }

/* ---------- sidebar brand and split bar ---------- */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.brand-mark {
    width: 42px; height: 42px; border-radius: 12px;
    background: var(--grad); display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 0 18px rgba(99, 102, 241, .6);
}
.brand-name { font-family: 'Sora', sans-serif; font-size: 20px; font-weight: 700; color: #FFFFFF; line-height: 1.1; }
.brand-sub { font-size: 12px; color: var(--muted); }
.split { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,.08); margin-top: 4px; }
.split-eeg { background: var(--violet); }
.split-mri { background: var(--cyan); }
.split-legend { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-top: 6px; }

/* ---------- cards ---------- */
.panel-head { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.panel-icon {
    width: 40px; height: 40px; border-radius: 12px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
    background: rgba(139, 92, 246, .22); border: 1px solid rgba(139, 92, 246, .5);
}
.panel-icon.cyan { background: rgba(34, 211, 238, .16); border-color: rgba(34, 211, 238, .5); }
.panel-title { font-family: 'Sora', sans-serif; font-size: 19px; font-weight: 600; color: #FFFFFF; line-height: 1.2; }
.panel-hint { color: var(--muted); font-size: 13px; margin: 0 0 10px 0; }

.status {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
    transition: transform .2s, border-color .2s;
}
.status:hover { transform: translateY(-2px); border-color: rgba(34, 211, 238, .42); }
.status-label { color: var(--muted); font-size: 13px; font-weight: 600; }
.status-value { font-family: 'Sora', sans-serif; font-size: 20px; font-weight: 600; margin-top: 4px; }
.tone-ok { color: var(--ok); }
.tone-bad { color: var(--alert); }
.tone-idle { color: var(--faint); }
.tone-info { color: #FFFFFF; }

.stat-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 12px 0; }
.stat {
    flex: 1 1 110px;
    background: rgba(255, 255, 255, .05);
    border: 1px solid var(--line);
    border-radius: 12px; padding: 10px 14px;
}
.stat-label { color: var(--muted); font-size: 12px; }
.stat-value { font-family: 'Sora', sans-serif; color: #FFFFFF; font-size: 18px; font-weight: 600; margin-top: 2px; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px 0; }
.chip {
    background: rgba(34, 211, 238, .10); border: 1px solid rgba(34, 211, 238, .32);
    color: #B6F3FF; border-radius: 999px; padding: 4px 12px; font-size: 12px;
}

/* ---------- signal viewer ---------- */
.viewer svg { width: 100%; height: auto; display: block; }
.viewer .grid { stroke: rgba(150, 160, 255, .16); stroke-width: 1; }
.viewer .axis { fill: var(--muted); font-size: 12px; font-family: 'Plus Jakarta Sans', sans-serif; }
.viewer .trace { stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; filter: drop-shadow(0 0 4px rgba(34,211,238,.55)); }
.viewer .trace-smooth { stroke: #C4B5FD; stroke-width: 2.6; stroke-dasharray: 7 5; fill: none; stroke-linecap: round; }
.viewer .hit { fill: rgba(34, 211, 238, 0); cursor: crosshair; }
.viewer .hit:hover { fill: rgba(34, 211, 238, .95); stroke: #FFFFFF; stroke-width: 2; }
.legend { display: flex; gap: 18px; color: var(--muted); font-size: 13px; margin: 2px 0 6px 0; }
.legend i { display: inline-block; width: 22px; height: 3px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }

/* ---------- results ---------- */
.score-panel {
    background: linear-gradient(160deg, rgba(31, 36, 105, .85), rgba(14, 20, 56, .85));
    border: 1px solid rgba(139, 92, 246, .38);
    border-radius: 24px;
    padding: 26px 22px 22px 22px;
    text-align: center;
    box-shadow: 0 18px 46px rgba(30, 20, 100, .45);
}
.score-panel svg { width: 100%; max-width: 360px; height: auto; }
.gauge-num { font-family: 'Sora', sans-serif; font-size: 34px; font-weight: 700; }
.gauge-cap { fill: var(--muted); font-size: 12px; font-family: 'Plus Jakarta Sans', sans-serif; }
.badge { display: inline-block; padding: 7px 18px; border-radius: 999px; font-weight: 700; font-size: 15px; margin-top: 4px; }
.band-scale { display: flex; justify-content: space-between; color: var(--faint); font-size: 12px; margin: 14px 10px 0 10px; }
.band-note {
    background: rgba(255, 255, 255, .05); border: 1px solid var(--line);
    border-radius: 14px; padding: 14px 16px; margin-top: 16px;
    color: var(--text); font-size: 14px; line-height: 1.6; text-align: left;
}
.band-note small { display: block; color: var(--muted); margin-top: 6px; }

.bar-row { margin: 12px 0; }
.bar-head { display: flex; justify-content: space-between; font-size: 14px; color: var(--text); margin-bottom: 6px; }
.bar-head span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
.bar-track { height: 10px; border-radius: 999px; background: rgba(255, 255, 255, .08); overflow: hidden; }
.bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366F1, #22D3EE);
    transform-origin: left; animation: ns-grow .8s ease-out;
    transition: filter .2s;
}
.bar-fill.top { background: linear-gradient(90deg, #A78BFA, #F472B6); box-shadow: 0 0 14px rgba(167, 139, 250, .55); }
.bar-row:hover .bar-fill { filter: brightness(1.2); }
@keyframes ns-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

.big-value { font-family: 'Sora', sans-serif; font-size: 32px; font-weight: 700; color: #FFFFFF; margin: 4px 0 2px 0; }
.big-label { color: var(--muted); font-size: 13px; }

.rawjson {
    background: #070B24; border: 1px solid var(--line); border-radius: 12px;
    color: #C9D3FF; padding: 14px 16px; font-size: 12.5px; line-height: 1.5;
    overflow-x: auto; max-height: 340px;
}

/* ---------- info tabs ---------- */
.glass {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 22px;
    height: 100%;
    transition: transform .2s, border-color .2s;
}
.glass:hover { transform: translateY(-2px); border-color: rgba(34, 211, 238, .42); }
.glass h4 { margin: 0 0 8px 0; font-size: 18px; }
.glass p, .glass li { color: var(--muted); font-size: 14.5px; line-height: 1.7; }
.glass ul { margin: 0; padding-left: 18px; }
.step-no {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--grad); color: #FFFFFF; font-weight: 700; margin-bottom: 10px;
}
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 6px 0 4px 0; }
.flow span {
    background: rgba(255, 255, 255, .06); border: 1px solid var(--line);
    color: var(--text); border-radius: 12px; padding: 8px 14px; font-size: 14px;
}
.flow em { color: var(--cyan); font-style: normal; }

.footer { color: var(--faint); text-align: center; font-size: 12.5px; padding: 28px 0 6px 0; }

/* ---------- responsive and accessibility ---------- */
@media (max-width: 820px) {
    .hero { padding: 28px 22px 70px 22px; }
    .hero-inner { flex-direction: column; align-items: flex-start; }
    .hero-title { font-size: 31px; }
    .hero-brain { width: 190px; align-self: center; }
}
@media (prefers-reduced-motion: reduce) {
    .pulse, .hero-brain .node, .bar-fill { animation: none; }
    .hero-wave path { animation: none; stroke-dashoffset: 0; }
}
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
    """Show JSON in a dark, readable block (no dependency on the Streamlit theme)."""
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
        return "High", "#FB7185", "rgba(251,113,133,.16)"
    if probability >= 0.40:
        return "Moderate", "#FBBF24", "rgba(251,191,36,.16)"
    return "Low", "#34D399", "rgba(52,211,153,.16)"


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


def panel_head(icon: str, title: str, hint: str, cyan: bool = False):
    md(
        f"""
        <div class="panel-head">
            <div class="panel-icon{' cyan' if cyan else ''}">{icon}</div>
            <div class="panel-title">{html.escape(title)}</div>
        </div>
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


# ---------- decorative illustration (not data) ----------
def network_svg() -> str:
    """Subtle neural-network pattern for the hero background."""
    nodes = [
        (70 + i * 66 + 16 * math.sin(i * 1.9), 50 + (i * 97) % 210 + 14 * math.cos(i * 2.4))
        for i in range(15)
    ]
    parts = []
    for i, (x1, y1) in enumerate(nodes):
        for j in range(i + 1, min(i + 5, len(nodes))):
            x2, y2 = nodes[j]
            if math.hypot(x2 - x1, y2 - y1) < 190:
                parts.append(
                    f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                    'stroke="#8FB8FF" stroke-width="1"/>'
                )
    for x, y in nodes:
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.2" fill="#7DD3FC"/>')
    return (
        '<svg class="hero-net" viewBox="0 0 1000 300" preserveAspectRatio="xMidYMid slice" '
        'aria-hidden="true">' + "".join(parts) + "</svg>"
    )


def wave_svg() -> str:
    """Decorative EEG-style waveform along the bottom of the hero."""
    width, height, steps = 1000, 56, 240
    mid = height / 2
    points = []
    for i in range(steps + 1):
        x = width * i / steps
        t = i / steps * math.pi * 14
        y = mid + 8 * math.sin(t) + 5 * math.sin(t * 2.7 + 1)
        if i % 60 == 30:
            y = mid - 22
        elif i % 60 == 31:
            y = mid + 16
        points.append(f"{x:.1f} {y:.1f}")
    path = "M" + " L".join(points)
    return (
        f'<svg class="hero-wave" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        'aria-hidden="true"><defs>'
        '<linearGradient id="ns-wave-g" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#A78BFA"/><stop offset="1" stop-color="#22D3EE"/>'
        "</linearGradient></defs>"
        f'<path d="{path}" pathLength="1" fill="none" stroke="url(#ns-wave-g)" '
        'stroke-width="2.2" vector-effect="non-scaling-stroke"/></svg>'
    )


BRAIN_SVG = """
<svg class="hero-brain" viewBox="0 0 320 270" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs>
<linearGradient id="ns-brain-g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#C4B5FD"/><stop offset="1" stop-color="#22D3EE"/>
</linearGradient>
<radialGradient id="ns-brain-halo" cx="50%" cy="50%" r="50%">
<stop offset="0" stop-color="#7C3AED" stop-opacity=".55"/>
<stop offset="1" stop-color="#7C3AED" stop-opacity="0"/>
</radialGradient>
<filter id="ns-brain-glow" x="-30%" y="-30%" width="160%" height="160%">
<feGaussianBlur stdDeviation="3.5" result="b"/>
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
</defs>
<circle cx="160" cy="130" r="135" fill="url(#ns-brain-halo)"/>
<g filter="url(#ns-brain-glow)" fill="rgba(124,58,237,.16)" stroke="url(#ns-brain-g)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round">
<path d="M158 38 C122 28 72 42 52 88 C36 124 46 164 78 188 C98 206 130 214 158 206 Z"/>
<path d="M162 38 C198 28 248 42 268 88 C284 124 274 164 242 188 C222 206 190 214 162 206 Z"/>
<path d="M146 206 C146 226 152 238 160 244 C168 238 174 226 174 206" fill="none"/>
</g>
<g fill="none" stroke="url(#ns-brain-g)" stroke-width="2.2" stroke-linecap="round" opacity=".92" filter="url(#ns-brain-glow)">
<path d="M150 62 C128 56 102 64 90 84"/>
<path d="M146 96 C120 90 92 102 84 124"/>
<path d="M150 132 C126 128 100 140 98 164"/>
<path d="M120 70 C112 92 118 108 134 118"/>
<path d="M70 112 C80 104 92 108 96 118"/>
<path d="M118 150 C112 166 120 180 138 188"/>
<path d="M170 62 C192 56 218 64 230 84"/>
<path d="M174 96 C200 90 228 102 236 124"/>
<path d="M170 132 C194 128 220 140 222 164"/>
<path d="M200 70 C208 92 202 108 186 118"/>
<path d="M250 112 C240 104 228 108 224 118"/>
<path d="M202 150 C208 166 200 180 182 188"/>
</g>
<g fill="#22D3EE" filter="url(#ns-brain-glow)">
<circle class="node" cx="90" cy="84" r="3.6"/><circle class="node" cx="84" cy="124" r="3.6"/>
<circle class="node" cx="98" cy="164" r="3.6"/><circle class="node" cx="134" cy="118" r="3.6"/>
<circle class="node" cx="138" cy="188" r="3.6"/><circle class="node" cx="230" cy="84" r="3.6"/>
<circle class="node" cx="236" cy="124" r="3.6"/><circle class="node" cx="222" cy="164" r="3.6"/>
<circle class="node" cx="186" cy="118" r="3.6"/><circle class="node" cx="182" cy="188" r="3.6"/>
<circle class="node" cx="150" cy="62" r="3.6"/><circle class="node" cx="170" cy="62" r="3.6"/>
</g>
</svg>
"""


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
        f'<linearGradient id="ns-sig-stroke" gradientUnits="userSpaceOnUse" x1="{pl}" y1="0" x2="{W - pr}" y2="0">'
        '<stop offset="0" stop-color="#A78BFA"/><stop offset="1" stop-color="#22D3EE"/></linearGradient>'
        '<linearGradient id="ns-sig-fill" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#22D3EE" stop-opacity=".28"/>'
        '<stop offset="1" stop-color="#22D3EE" stop-opacity="0"/></linearGradient></defs>'
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
    parts.append(f'<polyline points="{line_pts}" class="trace" fill="none" stroke="url(#ns-sig-stroke)"/>')

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
def gauge_svg(percent: float, color: str) -> str:
    radius = 90
    length = math.pi * radius
    filled = length * percent / 100
    return (
        f'<svg viewBox="0 0 220 135" role="img" aria-label="Screening score {percent:.1f} percent">'
        '<defs><linearGradient id="ns-gauge-g" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#22D3EE"/><stop offset=".55" stop-color="#8B5CF6"/>'
        '<stop offset="1" stop-color="#F472B6"/></linearGradient>'
        '<filter id="ns-gauge-glow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="3" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        '<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="rgba(160,170,255,.16)" '
        'stroke-width="16" stroke-linecap="round"/>'
        '<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="url(#ns-gauge-g)" '
        f'stroke-width="16" stroke-linecap="round" stroke-dasharray="{filled:.2f} {length:.2f}" '
        'filter="url(#ns-gauge-glow)"/>'
        f'<text x="110" y="98" text-anchor="middle" class="gauge-num" fill="{color}">{percent:.1f}%</text>'
        '<text x="110" y="124" text-anchor="middle" class="gauge-cap">Combined screening score</text>'
        "</svg>"
    )


def show_risk(probability: float):
    level, color, background = risk_level(probability)
    percent = min(max(float(probability) * 100, 0), 100)

    md(
        f"""
        <div class="score-panel">
            {gauge_svg(percent, color)}
            <div><span class="badge" style="background:{background}; color:{color}">
                ● {level} range
            </span></div>
            <div class="band-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
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
        "Ranges are display bands (below 40%, 40–70%, 70% and above), not clinical "
        "thresholds. This score reflects the configured model output and is not a "
        "validated estimate of clinical risk."
    )


def show_eeg_result(eeg_result: dict):
    probability = float(eeg_result["probability"])
    with st.container(border=True):
        panel_head("〰️", "EEG model", "Signal analysis")
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
        panel_head("🧠", "MRI model", "Image classification", cyan=True)
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
                    <span class="chip">Combined analysis · EEG {fused_alpha:.0%} / MRI {1 - fused_alpha:.0%}</span>
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
    md(
        """
        <div class="brand">
            <div class="brand-mark">🧠</div>
            <div>
                <div class="brand-name">NeuroSense</div>
                <div class="brand-sub">AI brain analytics</div>
            </div>
        </div>
        """
    )
    st.divider()

    st.markdown("### Backend")
    api_url = st.text_input("FastAPI backend URL", DEFAULT_API).rstrip("/")

    if st.button("Check backend connection", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            st.success("Backend is responding")
            json_pre(response.json())
        except requests.RequestException as exc:
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
# HERO + DISCLAIMER (always visible)
# =====================================================================
md(
    f"""
    <div class="hero">
        {network_svg()}
        <div class="hero-inner">
            <div class="hero-copy">
                <div class="hero-chip"><span class="pulse"></span>Research prototype</div>
                <div class="hero-title">Explore EEG and MRI screening signals in one place</div>
                <div class="hero-sub">
                    NeuroSense runs experimental machine-learning models on EEG
                    recordings and MRI images, then presents each output clearly.
                    It is a research tool, not a medical device.
                </div>
                <div class="hero-tags">
                    <span>〰️ EEG · {EEG_LEN} values</span>
                    <span>🧠 MRI · JPG or PNG</span>
                    <span>⚡ FastAPI inference</span>
                </div>
            </div>
            {BRAIN_SVG}
        </div>
        {wave_svg()}
    </div>
    """
)

md(
    """
    <div class="disclaimer">
        <div class="disclaimer-icon">⚕️</div>
        <div>
            <b>Medical disclaimer.</b> NeuroSense is an experimental research
            prototype. Its outputs are not clinical diagnoses and must not be
            used to diagnose, rule out or treat any medical condition. Always
            consult a qualified healthcare professional.
        </div>
    </div>
    """
)

# =====================================================================
# NAVIGATION
# =====================================================================
tab_analyze, tab_how, tab_safety = st.tabs(
    ["🔬 Analyze", "🧭 How it works", "⚕️ Safety and limits"]
)

# ---------------------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------------------
with tab_analyze:
    st.markdown("## Upload your data")
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
                "〰️",
                "EEG recording",
                f"Upload a CSV or TXT file containing exactly {EEG_LEN} numeric values.",
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
                    st.success(f"✓ {len(eeg_values)} numeric values loaded")
                    eeg_state = ("Ready", "ok")

    with col_mri:
        with st.container(border=True):
            panel_head(
                "🧠",
                "MRI scan",
                "Upload a supported JPG or PNG brain image for model inference.",
                cyan=True,
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
                "📈",
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

            legend = '<span><i style="background:linear-gradient(90deg,#A78BFA,#22D3EE)"></i>Raw signal</span>'
            if show_smooth:
                legend += (
                    f'<span><i style="background:#C4B5FD"></i>Moving average ({smooth_window})</span>'
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
        "✨  Run AI analysis",
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
            with st.spinner("🧠 Processing your uploaded data..."):
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
                f"Details: {exc}"
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
            <div class="glass">
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
            <div class="glass">
                <div class="step-no">2</div>
                <h4>Analyze</h4>
                <p>Your files are sent to the FastAPI backend set in the sidebar.
                The models run there and return their outputs.</p>
            </div>
            """
        )
    with s3:
        md(
            """
            <div class="glass">
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
            <div class="glass">
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
            <div class="glass">
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
        <div class="glass">
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
            <div class="glass">
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
            <div class="glass">
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
            <div class="glass">
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
            <div class="glass">
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
        NeuroSense · AI-assisted neurological research prototype · Not for clinical use<br>
        Built with Streamlit, FastAPI and machine learning
    </div>
    """
)