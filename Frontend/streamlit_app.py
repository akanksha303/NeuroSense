import streamlit as st

st.set_page_config(
    page_title="NeuroSense — Multimodal Neuro & Tumor Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling (Teal & Coral Palette) ---
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f4fbfa 0%, #ffffff 45%, #fff9f5 100%);
        }
        .hero-card {
            background: #ffffff;
            border: 1px solid #d9efec;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 10px 28px rgba(24, 90, 82, 0.06);
        }
        .metric-pill {
            display: inline-block;
            background: #e6f6f3;
            color: #16655c;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 4px 12px;
            border-radius: 999px;
            margin-bottom: 10px;
        }
        .coral-badge {
            display: inline-block;
            background: #fff0e8;
            color: #c85a28;
            font-weight: 600;
            font-size: 0.82rem;
            padding: 4px 10px;
            border-radius: 999px;
        }
        .feature-box {
            background: #ffffff;
            border: 1px solid #e7efee;
            border-radius: 14px;
            padding: 18px;
            height: 100%;
        }
        div[data-testid="stMetric"] {
            background-color: #f9fdfc;
            border: 1px solid #e2f1ee;
            padding: 12px 16px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "reviewed_cases" not in st.session_state:
    st.session_state.reviewed_cases = 12
if "active_clinician" not in st.session_state:
    st.session_state.active_clinician = None
if "checklist_state" not in st.session_state:
    st.session_state.checklist_state = {
        "EEG Signal Triage": [True, True, False],
        "MRI Scan Review": [True, False, False],
        "Late-Fusion Summary": [False, False, False],
    }

# --- Top Navigation & Session Bar ---
nav_col1, nav_col2, nav_col3 = st.columns([4, 3, 3])
with nav_col1:
    st.markdown("### 🧠 **NeuroSense**")
    st.caption("Multimodal EEG 1D-CNN & MRI MobileNetV2 neurological screening")
with nav_col2:
    cohort_view = st.segmented_control(
        "Cohort Window",
        options=["Current Batch", "Prior Run", "7d Benchmark"],
        default="Current Batch",
        label_visibility="collapsed",
    )
with nav_col3:
    with st.popover("Reviewer Sign In / Workspace", use_container_width=True):
        if st.session_state.active_clinician:
            st.success(f"Active session: **{st.session_state.active_clinician}**")
            if st.button("Sign Out", key="neuro_sign_out"):
                st.session_state.active_clinician = None
                st.rerun()
        else:
            reviewer_id = st.text_input("Reviewer ID or Email", placeholder="reviewer@neurosense.ai")
            track = st.selectbox("Inference Backend", ["FastAPI + Late Fusion", "TFLite Edge Mode", "EEG / MRI Ablation"])
            if st.button("Open Workspace", type="primary", use_container_width=True):
                st.session_state.active_clinician = reviewer_id.strip() or f"Reviewer ({track})"
                st.rerun()

st.divider()

# --- Hero Section: Value Proposition + Live Inference Preview ---
hero_left, hero_right = st.columns([1.15, 1], gap="large")

with hero_left:
    st.markdown('<span class="metric-pill">MULTIMODAL EEG + MRI DECISION FUSION</span>', unsafe_allow_html=True)
    st.markdown("# Clear neurological risk signals across wave and scan.")
    st.write(
        "NeuroSense pairs a 1D-CNN for EEG time-series dynamics with a MobileNetV2 2D-CNN "
        "for brain MRI classification—combining modality probabilities via decision-level late "
        "fusion for transparent screening support and triage."
    )
    cta1, cta2 = st.columns([1, 1])
    with cta1:
        if st.button("Load Sample Case", type="primary", use_container_width=True):
            st.toast("Sample EEG + MRI case loaded for interactive inspection.", icon="🧠")
    with cta2:
        st.link_button("Explore Architecture", "#neuro-pillars", use_container_width=True)

    st.markdown("")
    t1, t2, t3 = st.columns(3)
    t1.metric("Fusion Confidence", "94.2%", "+2.8% vs single-mod")
    t2.metric("Runtime Target", "TFLite", "Edge & FastAPI ready")
    t3.metric("Modalities", "2 Streams", "1D-CNN EEG + 2D MRI")

with hero_right:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<span class="coral-badge">LIVE FUSION PREVIEW</span>', unsafe_allow_html=True)
    st.subheader(f"Case Telemetry — {cohort_view or 'Current Batch'}")

    presets = {
        "Current Batch": {
            "eeg": "0.89",
            "eeg_d": "Alpha/Theta shift",
            "mri": "0.93",
            "mri_d": "MobileNetV2 score",
            "fused": "Elevated",
            "fused_d": "Late fusion alert",
            "series": [0.42, 0.48, 0.55, 0.63, 0.74, 0.85, 0.91],
        },
        "Prior Run": {
            "eeg": "0.31",
            "eeg_d": "Baseline rhythm",
            "mri": "0.24",
            "mri_d": "Low anomaly prob",
            "fused": "Low Risk",
            "fused_d": "Routine follow-up",
            "series": [0.29, 0.27, 0.30, 0.25, 0.26, 0.24, 0.27],
        },
        "7d Benchmark": {
            "eeg": "0.76",
            "eeg_d": "Rolling mean",
            "mri": "0.81",
            "mri_d": "Rolling mean",
            "fused": "Moderate",
            "fused_d": "7-day fusion avg",
            "series": [0.61, 0.65, 0.70, 0.72, 0.75, 0.78, 0.79],
        },
    }
    snap = presets[cohort_view or "Current Batch"]

    m1, m2, m3 = st.columns(3)
    m1.metric("EEG 1D-CNN", snap["eeg"], snap["eeg_d"])
    m2.metric("MRI 2D-CNN", snap["mri"], snap["mri_d"])
    m3.metric("Fusion Status", snap["fused"], snap["fused_d"])

    st.caption("Late-Fusion Risk Trajectory (Normalized Window Scores)")
    st.line_chart(snap["series"], height=120)

    # Interactive Case Review Counter
    target_cases = 20
    progress_val = min(st.session_state.reviewed_cases / target_cases, 1.0)
    c_col1, c_col2 = st.columns([2.2, 1])
    with c_col1:
        st.write(f"**Cases Verified Today:** {st.session_state.reviewed_cases} / {target_cases}")
        st.progress(progress_val)
    with c_col2:
        st.write("")
        if st.button("+1 Verified", use_container_width=True):
            st.session_state.reviewed_cases += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- Core Architecture Pillars ---
st.markdown('<div id="neuro-pillars"></div>', unsafe_allow_html=True)
st.subheader("Four Pillars of the NeuroSense Pipeline")
st.write("Structured for reproducible inference, clear modality weighting, and practical deployment.")

p1, p2, p3, p4 = st.columns(4, gap="medium")
pillars = [
    (p1, "🌊 EEG 1D-CNN Encoder", "Extracts temporal and frequency-band patterns from multi-channel EEG signals."),
    (p2, "🧲 MRI MobileNetV2 Net", "Classifies structural brain MRI slices with lightweight convolutional features."),
    (p3, "🔀 Decision-Level Fusion", "Combines independent branch probabilities so missing or noisy inputs stay interpretable."),
    (p4, "⚡ FastAPI & TFLite Stack", "Serves predictions cleanly via REST endpoints and quantized edge models."),
]
for col, title, desc in pillars:
    with col:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        st.write(desc)
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- Interactive Stage Explorer ---
st.subheader("Explore Screening & Triage Stages")
st.write("Switch across workflow stages to inspect stage guidance, verification items, and branch agreement.")

stage = st.radio(
    "Select workflow stage",
    ["EEG Signal Triage", "MRI Scan Review", "Late-Fusion Summary"],
    horizontal=True,
)

stage_details = {
    "EEG Signal Triage": {
        "guidance": "Inspect bandpass filtering, artifact rejection, and 1D-CNN class probabilities before fusion.",
        "tag": "Branch A · 1D-CNN",
        "items": [
            "Confirm sampling rate and baseline artifact removal",
            "Check band-power anomalies against reference window",
            "Log 1D-CNN softmax confidence score",
        ],
    },
    "MRI Scan Review": {
        "guidance": "Verify slice normalization, orientation alignment, and MobileNetV2 lesion/tumor class probabilities.",
        "tag": "Branch B · MobileNetV2",
        "items": [
            "Validate image resolution and intensity normalization",
            "Review top predicted tumor/structural class probabilities",
            "Flag low-contrast or motion-affected slices",
        ],
    },
    "Late-Fusion Summary": {
        "guidance": "Compare EEG and MRI branch outputs side by side to detect agreement or modality conflict.",
        "tag": "Combined Decision",
        "items": [
            "Check fusion weight balance between EEG and MRI branches",
            "Highlight cases where branch confidences diverge",
            "Export structured summary for clinical/research review",
        ],
    },
}

current_stage = stage_details[stage]
exp_left, exp_right = st.columns([1.2, 1], gap="large")

with exp_left:
    st.info(f"**{stage} ({current_stage['tag']}):** {current_stage['guidance']}")
    st.markdown("**Stage Verification Checklist**")
    checks = st.session_state.checklist_state[stage]
    for idx, label in enumerate(current_stage["items"]):
        checks[idx] = st.checkbox(label, value=checks[idx], key=f"neuro_{stage}_{idx}")
    completed = sum(checks)
    st.caption(f"Completed {completed} of {len(checks)} checks for {stage}")

with exp_right:
    st.markdown("**Branch Confidence Across Recent Cases (%)**")
    branch_scores = {
        "EEG Signal Triage": [88, 91, 79, 85, 92, 87, 89],
        "MRI Scan Review": [92, 89, 94, 86, 90, 93, 93],
        "Late-Fusion Summary": [91, 92, 89, 88, 94, 92, 94],
    }
    st.bar_chart(branch_scores[stage], height=180)

st.caption(
    "Research & Decision-Support Notice: NeuroSense is built for neurological research and screening "
    "support and does not replace formal clinical diagnosis by a licensed physician."
)