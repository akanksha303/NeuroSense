"""
NeuroSense — AI Neurological Screening Dashboard
Run: streamlit run Frontend/streamlit_app.py
"""

import html
import math
import os

import pandas as pd
import requests
import streamlit as st

EEG_LEN = 178
DEFAULT_API = os.getenv("NEUROSENSE_API", "http://localhost:8000")

st.set_page_config(
    page_title="NeuroSense | Brain Health Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- theme ----------
# All colours live in CSS variables on :root. Change a value here and the
# whole app re-themes, so you never hunt through the file for a hex code.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=Figtree:wght@400;500;600;700&display=swap');

    :root {
        --sun: #FFC93C;          /* primary yellow */
        --sun-deep: #E8A500;     /* hover / strong accent */
        --sun-soft: #FFF4CC;     /* tinted backgrounds */
        --sun-wash: #FFFBEB;     /* page background */
        --ink: #2B2208;          /* main text */
        --ink-soft: #6F6342;     /* secondary text */
        --line: #F1E3B0;         /* borders */
        --ok: #2E9B6A;           /* low risk */
        --warn: #D9810B;         /* moderate risk */
        --alert: #D2453D;        /* high risk */
        --radius: 14px;
    }

    .stApp {
        background: var(--sun-wash);
        font-family: 'Figtree', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3, h4 {
        font-family: 'Bricolage Grotesque', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.01em;
    }

    /* sidebar: light and warm, dark text (no forced white-on-white bugs) */
    [data-testid="stSidebar"] {
        background: var(--sun-soft);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--ink); }

    /* hero: the one bold element on the page */
    .hero {
        background: var(--sun);
        border-radius: 20px;
        padding: 30px 34px;
        margin: 0 0 18px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
    }
    .hero-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 40px;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.1;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #4A3C0E;
        font-size: 16px;
        line-height: 1.6;
        max-width: 560px;
        margin: 0;
    }
    .hero-icon {
        font-size: 64px;
        line-height: 1;
        background: rgba(255,255,255,.55);
        border-radius: 50%;
        width: 104px;
        height: 104px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .notice {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 5px solid var(--sun-deep);
        border-radius: 10px;
        padding: 12px 16px;
        color: var(--ink-soft);
        font-size: 14px;
        margin: 0 0 22px 0;
    }
    .notice b { color: var(--ink); }

    /* status tiles */
    .status {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 16px 18px;
        min-height: 96px;
    }
    .status-label { color: var(--ink-soft); font-size: 14px; font-weight: 600; }
    .status-value {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
    }

    /* card headings inside bordered containers */
    .card-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 21px;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
    }
    .card-hint { color: var(--ink-soft); font-size: 14px; margin: 2px 0 10px 0; }

    /* risk result */
    .risk-panel {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 26px;
        text-align: center;
    }
    .risk-caption { color: var(--ink-soft); font-size: 15px; font-weight: 600; }
    .risk-number {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 56px;
        font-weight: 700;
        line-height: 1.1;
        margin: 6px 0 10px 0;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
    }
    .gauge {
        position: relative;
        height: 12px;
        border-radius: 999px;
        margin: 24px 6px 6px 6px;
        background: linear-gradient(
            90deg,
            var(--ok) 0%, var(--ok) 38%,
            var(--sun) 42%, var(--warn) 66%,
            var(--alert) 72%, var(--alert) 100%
        );
    }
    .gauge-marker {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 24px;
        background: var(--ink);
        border: 2px solid #FFFFFF;
        border-radius: 4px;
        transform: translateX(-50%);
    }
    .gauge-scale {
        display: flex;
        justify-content: space-between;
        color: var(--ink-soft);
        font-size: 12px;
        margin: 0 6px;
    }

    .guidance {
        background: var(--sun-soft);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 16px 18px;
        color: var(--ink);
        font-size: 15px;
        line-height: 1.6;
        margin-top: 14px;
    }
    .guidance b { display: block; margin-bottom: 4px; }
    .guidance small { color: var(--ink-soft); }

    /* buttons and widgets */
    .stButton > button[kind="primary"] {
        background: var(--sun);
        color: var(--ink);
        border: none;
        border-radius: 12px;
        font-weight: 700;
        min-height: 50px;
        font-size: 16px;
    }
    .stButton > button[kind="primary"]:hover:not(:disabled) {
        background: var(--sun-deep);
        color: var(--ink);
    }
    .stButton > button[kind="primary"]:disabled {
        background: #EDE6CB;
        color: #A89E7C;
    }
    .stButton > button { border-radius: 10px; }

    div[data-testid="stFileUploader"] section {
        background: var(--sun-wash);
        border: 1.5px dashed var(--sun-deep);
        border-radius: 12px;
    }
    div[data-testid="stProgressBar"] > div > div { background: var(--sun-deep); }

    .footer {
        color: var(--ink-soft);
        text-align: center;
        font-size: 13px;
        padding: 26px 0 6px 0;
    }

    @media (max-width: 700px) {
        .hero { padding: 22px 20px; }
        .hero-title { font-size: 30px; }
        .hero-icon { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
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
        return "High", "#D2453D", "#FCE4E2"
    if probability >= 0.40:
        return "Moderate", "#B86E00", "#FFECC2"
    return "Low", "#2E9B6A", "#DDF5E8"


GUIDANCE = {
    "High": (
        "Consider speaking with a neurologist soon",
        "The model flagged patterns worth a professional look. Bring this "
        "result along as supporting context, together with any symptoms "
        "you have noticed.",
    ),
    "Moderate": (
        "Consider mentioning this to your doctor",
        "The result is in an uncertain range. If you have symptoms such as "
        "recurring headaches, memory changes or dizziness, a check-up is a "
        "good next step.",
    ),
    "Low": (
        "No strong signal in this screening",
        "A low score does not rule out a condition. If you have symptoms, "
        "or are worried, speak with a healthcare professional anyway.",
    ),
}


def status_tile(label: str, value: str, color: str):
    st.markdown(
        f"""
        <div class="status">
            <div class="status-label">{html.escape(label)}</div>
            <div class="status-value" style="color:{color}">
                {html.escape(value)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_header(icon: str, title: str, hint: str):
    st.markdown(
        f"""
        <div class="card-title">{icon} {html.escape(title)}</div>
        <div class="card-hint">{html.escape(hint)}</div>
        """,
        unsafe_allow_html=True,
    )


def show_risk(probability: float):
    level, color, background = risk_level(probability)
    percent = min(max(float(probability) * 100, 0), 100)
    marker = min(max(percent, 2), 98)  # keep the marker inside the bar

    st.markdown(
        f"""
        <div class="risk-panel">
            <div class="risk-caption">Combined screening score</div>
            <div class="risk-number" style="color:{color}">{percent:.1f}%</div>
            <span class="risk-badge"
                  style="background:{background}; color:{color}">
                {level} risk category
            </span>
            <div class="gauge">
                <div class="gauge-marker" style="left:{marker}%"></div>
            </div>
            <div class="gauge-scale">
                <span>Low</span><span>Moderate</span><span>High</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    heading, body = GUIDANCE[level]
    st.markdown(
        f"""
        <div class="guidance">
            <b>{heading}</b>
            {body}
            <br><small>
                Sudden severe headache, weakness on one side, confusion or a
                seizure needs emergency care. Do not wait for a screening.
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This score reflects the configured model output and is not a "
        "validated estimate of clinical risk."
    )


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 🩺 NeuroSense")
    st.caption("Brain health screening")
    st.divider()

    st.markdown("### Settings")
    api_url = st.text_input("Backend URL", DEFAULT_API).strip().rstrip("/")

    alpha = st.slider(
        "EEG weight in combined score",
        0.0,
        1.0,
        0.5,
        0.05,
        help="Used only when both an EEG file and an MRI image are uploaded.",
    )
    st.caption(f"EEG {alpha:.0%}  ·  MRI {1 - alpha:.0%}")

    st.divider()

    if st.button("Check backend connection", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            st.success("Backend is responding")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")
        except ValueError:
            st.error("Backend returned invalid JSON.")

    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()


# ---------- hero ----------
st.markdown(
    """
    <div class="hero">
        <div>
            <div class="hero-title">Brain health screening,<br>made clearer</div>
            <p class="hero-subtitle">
                Upload an EEG recording, an MRI image, or both. NeuroSense
                shows what the models found in plain language.
            </p>
        </div>
        <div class="hero-icon">🩺</div>
    </div>
    <div class="notice">
        <b>Research prototype.</b> Results are experimental and must not be
        used to diagnose, rule out or treat a medical condition.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- uploads ----------
st.markdown("## Upload your data")
st.caption("Provide one file, or both for a combined analysis.")

col_eeg, col_mri = st.columns(2, gap="large")

eeg_bytes = None
mri_bytes = None
eeg_file = None
mri_file = None
eeg_ok = True
eeg_state = ("Waiting for file", "#8A7F5C")
mri_state = ("Waiting for file", "#8A7F5C")

with col_eeg:
    with st.container(border=True):
        card_header(
            "〰️",
            "EEG recording",
            f"A CSV or TXT file with exactly {EEG_LEN} numeric values.",
        )
        eeg_file = st.file_uploader(
            "Choose EEG file",
            type=["csv", "txt"],
            key="eeg",
            label_visibility="collapsed",
        )

        if eeg_file is not None:
            eeg_bytes = eeg_file.getvalue()
            values = parse_eeg(eeg_bytes)

            if values is None:
                st.error(
                    "This file contains text or invalid numbers. "
                    "Use only numeric values separated by commas or spaces."
                )
                eeg_ok = False
                eeg_state = ("Invalid file", "#D2453D")
            elif len(values) != EEG_LEN:
                st.error(
                    f"Expected {EEG_LEN} values but found {len(values)}."
                )
                eeg_ok = False
                eeg_state = ("Wrong length", "#D2453D")
            else:
                st.success(f"{len(values)} values loaded")
                st.caption("Signal preview")
                st.line_chart(
                    pd.DataFrame({"Signal amplitude": values}),
                    height=200,
                    color="#E8A500",
                )
                eeg_state = ("Ready", "#2E9B6A")

with col_mri:
    with st.container(border=True):
        card_header(
            "🧠",
            "MRI scan",
            "A JPG or PNG image of a brain scan.",
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
            mri_state = ("Ready", "#2E9B6A")

# ---------- selected mode ----------
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
        "#2B2208" if mode else "#8A7F5C",
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
        files["eeg"] = (eeg_file.name, eeg_bytes, "text/csv")

    if mri_bytes is not None:
        files["mri"] = (mri_file.name, mri_bytes, mri_file.type)

    try:
        with st.spinner("Analysing your data..."):
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
            f"Check that it is running. Details: {exc}"
        )


# ---------- results ----------
result = st.session_state.get("result")

if result:
    st.divider()
    st.markdown("## Your results")
    st.caption("Model output from the connected backend.")

    if "final_risk" not in result:
        st.error("The backend response did not include a risk score.")
    else:
        left, right = st.columns([0.9, 1.1], gap="large")

        with left:
            show_risk(float(result["final_risk"]))

            if result.get("mode") == "fused":
                used_alpha = float(result.get("alpha", alpha))
                st.info(f"Combined using EEG weight {used_alpha:.2f}")

        with right:
            if "eeg" in result:
                probability = float(result["eeg"].get("probability", 0))
                with st.container(border=True):
                    card_header("〰️", "EEG model", "Signal analysis")
                    st.metric(
                        "Anomaly probability",
                        f"{probability * 100:.1f}%",
                    )
                    st.progress(min(max(probability, 0.0), 1.0))

            if "mri" in result:
                mri_result = result["mri"]
                with st.container(border=True):
                    card_header("🧠", "MRI model", "Image classification")
                    st.metric("Predicted label", str(mri_result.get("label", "—")))

                    probabilities = mri_result.get("probs", {})

                    if isinstance(probabilities, dict) and probabilities:
                        chart_df = pd.DataFrame(
                            {
                                "Class": list(probabilities.keys()),
                                "Probability": [
                                    float(v) for v in probabilities.values()
                                ],
                            }
                        ).set_index("Class")
                    elif isinstance(probabilities, list) and probabilities:
                        chart_df = pd.DataFrame({"Probability": probabilities})
                    else:
                        chart_df = None

                    if chart_df is not None:
                        st.bar_chart(
                            chart_df,
                            horizontal=True,
                            height=220,
                            color="#E8A500",
                        )

    with st.expander("View raw API response"):
        st.json(result)

    st.caption(
        "Clinical interpretation requires qualified healthcare professionals "
        "and appropriate validated diagnostic procedures."
    )

# ---------- footer ----------
st.markdown(
    """
    <div class="footer">
        NeuroSense · AI-assisted neurological research prototype<br>
        Built with Streamlit, FastAPI and machine learning
    </div>
    """,
    unsafe_allow_html=True,
)