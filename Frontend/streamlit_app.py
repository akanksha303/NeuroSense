import streamlit as st

st.set_page_config(
    page_title="VitalWave — Daily Health & Wellness",
    page_icon="💚",
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
if "water_ml" not in st.session_state:
    st.session_state.water_ml = 1250
if "signed_in_user" not in st.session_state:
    st.session_state.signed_in_user = None
if "habit_checks" not in st.session_state:
    st.session_state.habit_checks = {
        "Morning Routine": [True, True, False],
        "Active Focus": [True, False, False],
        "Night Recovery": [False, False, False],
    }

# --- Top Navigation & Auth Bar ---
nav_col1, nav_col2, nav_col3 = st.columns([4, 3, 3])
with nav_col1:
    st.markdown("### 💚 **VitalWave**")
    st.caption("Calm daily health telemetry, sleep recovery, and habit guidance")
with nav_col2:
    timeframe = st.segmented_control(
        "Telemetry Window",
        options=["Today", "Yesterday", "7d Avg"],
        default="Today",
        label_visibility="collapsed",
    )
with nav_col3:
    with st.popover("Sign In / Get Started", use_container_width=True):
        if st.session_state.signed_in_user:
            st.success(f"Signed in as **{st.session_state.signed_in_user}**")
            if st.button("Sign Out", key="sign_out_btn"):
                st.session_state.signed_in_user = None
                st.rerun()
        else:
            email = st.text_input("Email address", placeholder="you@example.com")
            goal = st.selectbox("Primary focus", ["Stress & Recovery", "Steady Energy", "Sleep Quality"])
            if st.button("Continue to VitalWave", type="primary", use_container_width=True):
                st.session_state.signed_in_user = email.strip() or f"Member ({goal})"
                st.rerun()

st.divider()

# --- Hero Section: Value Proposition + Live Wellness Preview ---
hero_left, hero_right = st.columns([1.15, 1], gap="large")

with hero_left:
    st.markdown('<span class="metric-pill">CLINICIAN-REVIEWED DAILY WELLNESS</span>', unsafe_allow_html=True)
    st.markdown("# Understand your body's rhythm without the noise.")
    st.write(
        "VitalWave brings together resting heart rate, sleep quality, daily movement, "
        "and hydration into one calm morning-to-night view—so you know when to push "
        "and when to recover."
    )
    cta1, cta2 = st.columns([1, 1])
    with cta1:
        if st.button("Start Free Trial", type="primary", use_container_width=True):
            st.toast("Trial activated! Explore your daily wellness routines below.", icon="💚")
    with cta2:
        st.link_button("Explore Pillars", "#core-pillars", use_container_width=True)

    st.markdown("")
    t1, t2, t3 = st.columns(3)
    t1.metric("Member Satisfaction", "96%", "+3% vs last quarter")
    t2.metric("Privacy Standard", "HIPAA", "End-to-end encrypted")
    t3.metric("Guided Protocols", "45+", "Reviewed by clinicians")

with hero_right:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<span class="coral-badge">LIVE WELLNESS PREVIEW</span>', unsafe_allow_html=True)
    st.subheader(f"Snapshot — {timeframe or 'Today'}")

    # Adjust sample figures smoothly across timeframe tabs
    presets = {
        "Today": {"hr": "64 bpm", "hr_d": "-2 bpm", "sleep": "86 / 100", "sleep_d": "+4 pts", "steps": "7,420", "steps_d": "74% of 10k", "series": [68, 66, 65, 67, 64, 63, 64]},
        "Yesterday": {"hr": "66 bpm", "hr_d": "+1 bpm", "sleep": "82 / 100", "sleep_d": "-2 pts", "steps": "9,180", "steps_d": "92% of 10k", "series": [70, 69, 67, 66, 65, 66, 66]},
        "7d Avg": {"hr": "65 bpm", "hr_d": "Steady", "sleep": "84 / 100", "sleep_d": "+3 pts", "steps": "8,150", "steps_d": "81% of 10k", "series": [67, 66, 65, 66, 64, 65, 65]},
    }
    snap = presets[timeframe or "Today"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Resting HR", snap["hr"], snap["hr_d"])
    m2.metric("Sleep Score", snap["sleep"], snap["sleep_d"])
    m3.metric("Steps", snap["steps"], snap["steps_d"])

    st.caption("Heart Rate Trend (Last 7 Checkpoints)")
    st.line_chart(snap["series"], height=120)

    # Interactive Hydration Tracker
    goal_ml = 2500
    progress_val = min(st.session_state.water_ml / goal_ml, 1.0)
    w_col1, w_col2 = st.columns([2.2, 1])
    with w_col1:
        st.write(f"**Hydration:** {st.session_state.water_ml} ml / {goal_ml} ml")
        st.progress(progress_val)
    with w_col2:
        st.write("")
        if st.button("+250 ml", use_container_width=True):
            st.session_state.water_ml += 250
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- Core Health Pillars ---
st.markdown('<div id="core-pillars"></div>', unsafe_allow_html=True)
st.subheader("Four Pillars of Everyday Health")
st.write("Designed around small, repeatable signals rather than overwhelming charts.")

p1, p2, p3, p4 = st.columns(4, gap="medium")
pillars = [
    (p1, "🫀 Cardiovascular Calm", "Track resting heart rate trends and recovery shifts after high-strain days."),
    (p2, "🌙 Deep Sleep Architecture", "See how consistent wind-down windows support restorative overnight recovery."),
    (p3, "🚶 Natural Movement", "Balance desk hours with short mobility breaks and steady daily step targets."),
    (p4, "💧 Hydration & Energy", "Pair fluid intake with afternoon focus cues to prevent energy dips."),
]
for col, title, desc in pillars:
    with col:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        st.write(desc)
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- Interactive Daily Wellness Explorer ---
st.subheader("Explore Your Daily Wellness")
st.write("Switch between time-of-day modes to preview tailored guidance and check off habits.")

mode = st.radio(
    "Select wellness mode",
    ["Morning Routine", "Active Focus", "Night Recovery"],
    horizontal=True,
)

mode_details = {
    "Morning Routine": {
        "focus": "Hydrate first, get 10 minutes of natural light, and review readiness before opening notifications.",
        "readiness": "88% Ready",
        "habits": ["Drink 500 ml water on waking", "10 min outdoor daylight walk", "Log morning energy check-in"],
    },
    "Active Focus": {
        "focus": "Protect deep-work blocks with 5-minute posture and eye-rest resets every 50 minutes.",
        "readiness": "Steady Pace",
        "habits": ["Take a 5 min stretch break", "Midday hydration check", "Step away from screens at lunch"],
    },
    "Night Recovery": {
        "focus": "Lower evening stimulation, cool your room, and wind down 45 minutes before target sleep.",
        "readiness": "Wind-Down Mode",
        "habits": ["Dim bright overhead lights", "Set phone to Do Not Disturb", "Finish caffeine & heavy snacks early"],
    },
}

current_mode = mode_details[mode]
exp_left, exp_right = st.columns([1.2, 1], gap="large")

with exp_left:
    st.info(f"**{mode} ({current_mode['readiness']}):** {current_mode['focus']}")
    st.markdown("**Daily Habit Check-In**")
    checks = st.session_state.habit_checks[mode]
    for idx, habit_label in enumerate(current_mode["habits"]):
        checks[idx] = st.checkbox(habit_label, value=checks[idx], key=f"{mode}_{idx}")
    completed = sum(checks)
    st.caption(f"Completed {completed} of {len(checks)} habits in {mode}")

with exp_right:
    st.markdown("**Weekly Adherence Preview**")
    adherence_data = {
        "Morning Routine": [80, 100, 67, 100, 100, 67, 67],
        "Active Focus": [67, 67, 100, 67, 100, 100, 33],
        "Night Recovery": [100, 67, 67, 100, 67, 67, 0],
    }
    st.bar_chart(adherence_data[mode], height=180)