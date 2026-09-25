"""
Retention Radar
A Streamlit UI for the bagging-ensemble employee attrition model
(employee_attrition_bagging.pkl).

Run with:
    streamlit run app.py
"""

import base64
import io
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config & theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Retention Radar",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "employee_attrition_bagging.pkl")

NAVY = "#0B1F3A"
TEAL = "#1FB6A6"
AMBER = "#F5A623"
CREAM = "#F7F5F0"
SLATE = "#2C4970"

NUMERIC_COLS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary",
]
DEPT_COLS = [
    "dept_IT",
    "dept_RandD",
    "dept_accounting",
    "dept_hr",
    "dept_management",
    "dept_marketing",
    "dept_product_mng",
    "dept_sales",
    "dept_support",
    "dept_technical",
]
FEATURE_ORDER = NUMERIC_COLS + DEPT_COLS
DEPT_LABELS = [c.replace("dept_", "") for c in DEPT_COLS]
SALARY_MAP = {"low": 0, "medium": 1, "high": 2}


def load_svg(name: str) -> str:
    path = os.path.join(ASSETS_DIR, name)
    with open(path, "r") as f:
        return f.read()


def svg_to_data_uri(svg_text: str) -> str:
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


# --------------------------------------------------------------------------
# Global CSS — deliberately distinct from the generic Streamlit look
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {CREAM};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: {CREAM} !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem;
    }}
    h1, h2, h3 {{
        font-family: Georgia, 'Times New Roman', serif;
        color: {NAVY};
    }}
    .rr-hero {{
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(11,31,58,0.18);
    }}
    .rr-card {{
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #E7E2D6;
        box-shadow: 0 2px 10px rgba(11,31,58,0.05);
        margin-bottom: 1rem;
    }}
    .rr-metric-label {{
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.72rem;
        color: {SLATE};
        font-weight: 600;
    }}
    .rr-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }}
    .rr-icon-tile {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.5rem 0;
    }}
    .rr-icon-tile img {{
        width: 38px;
        height: 38px;
    }}
    div.stButton > button {{
        background-color: {NAVY};
        color: {CREAM};
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        background-color: {SLATE};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_feature_row(satisfaction, evaluation, projects, hours, tenure,
                       accident, promotion, dept, salary_level):
    row = {c: 0 for c in FEATURE_ORDER}
    row["satisfaction_level"] = satisfaction
    row["last_evaluation"] = evaluation
    row["number_project"] = projects
    row["average_montly_hours"] = hours
    row["time_spend_company"] = tenure
    row["Work_accident"] = int(accident)
    row["promotion_last_5years"] = int(promotion)
    row["salary"] = SALARY_MAP[salary_level]
    dept_col = f"dept_{dept}"
    if dept_col in row:
        row[dept_col] = 1
    return pd.DataFrame([row])[FEATURE_ORDER]


def preprocess_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding used at training time to a raw HR dataframe."""
    work = df.copy()
    if "salary" in work.columns and work["salary"].dtype == object:
        work["salary"] = work["salary"].map(SALARY_MAP)
    x = work[[c for c in [
        "satisfaction_level", "last_evaluation", "number_project",
        "average_montly_hours", "time_spend_company", "Work_accident",
        "promotion_last_5years", "dept", "salary",
    ] if c in work.columns]]
    x = pd.get_dummies(x)
    x = x.reindex(columns=FEATURE_ORDER, fill_value=0)
    return x


def feature_importance_frame(model):
    try:
        if hasattr(model, "estimators_") and model.estimators_:
            imps = np.array([
                est.feature_importances_ for est in model.estimators_
                if hasattr(est, "feature_importances_")
            ])
            if len(imps):
                avg = imps.mean(axis=0)
                return pd.DataFrame({"feature": FEATURE_ORDER, "importance": avg}) \
                    .sort_values("importance", ascending=True)
    except Exception:
        pass
    return None


def risk_badge(prob):
    if prob >= 0.66:
        return "High risk", "#C0392B", "#FBE7E4"
    if prob >= 0.33:
        return "Watch", AMBER, "#FCF1DE"
    return "Stable", TEAL, "#E5F7F4"


def risk_meter_svg(prob: float) -> str:
    """Custom semi-circular gauge rendered as inline SVG (no external image libs)."""
    angle = 180 * prob
    import math
    r = 90
    cx, cy = 110, 110
    rad = math.radians(180 - angle)
    x = cx + r * math.cos(rad)
    y = cy - r * math.sin(rad)
    large_arc = 1 if angle > 180 else 0
    color = "#C0392B" if prob >= 0.66 else (AMBER if prob >= 0.33 else TEAL)
    pct = int(round(prob * 100))
    return f"""
    <svg viewBox="0 0 220 140" xmlns="http://www.w3.org/2000/svg">
      <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#E7E2D6" stroke-width="16" stroke-linecap="round"/>
      <path d="M 20 110 A 90 90 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>
      <text x="110" y="105" text-anchor="middle" font-family="Georgia, serif" font-size="34" fill="{NAVY}" font-weight="bold">{pct}%</text>
      <text x="110" y="128" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{SLATE}" letter-spacing="0.05em">ATTRITION PROBABILITY</text>
    </svg>
    """


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🧭 Retention Radar")
    st.caption("Bagging-ensemble HR attrition model")
    page = st.radio(
        "Navigate",
        ["Single employee", "Batch upload", "Model insights", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    model = load_model()
    if model is not None:
        st.success("Model loaded", icon="✅")
        n_est = getattr(model, "n_estimators", None)
        if n_est:
            st.caption(f"{n_est} estimators in ensemble")
    else:
        st.error("Model file not found", icon="⚠️")
        st.caption(f"Place **employee_attrition_bagging.pkl** next to app.py")

# --------------------------------------------------------------------------
# Hero banner
# --------------------------------------------------------------------------
hero_svg = load_svg("hero.svg")
st.markdown(f'<div class="rr-hero">{hero_svg}</div>', unsafe_allow_html=True)

model = load_model()

# --------------------------------------------------------------------------
# PAGE: Single employee
# --------------------------------------------------------------------------
if page == "Single employee":
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("#### Employee profile")

        icons = [
            ("icon_pulse.svg", "Satisfaction & evaluation"),
            ("icon_workload.svg", "Workload"),
            ("icon_tenure.svg", "Tenure & history"),
            ("icon_dept.svg", "Department & pay"),
        ]

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                satisfaction = st.slider("Satisfaction level", 0.0, 1.0, 0.55, 0.01)
                projects = st.slider("Number of projects", 1, 10, 4)
                accident = st.checkbox("Had a work accident")
            with c2:
                evaluation = st.slider("Last evaluation score", 0.0, 1.0, 0.65, 0.01)
                hours = st.slider("Average monthly hours", 80, 320, 200)
                promotion = st.checkbox("Promoted in last 5 years")

            tenure = st.slider("Years at company", 1, 10, 3)

            c3, c4 = st.columns(2)
            with c3:
                dept = st.selectbox("Department", DEPT_LABELS, index=DEPT_LABELS.index("sales"))
            with c4:
                salary_level = st.selectbox("Salary band", ["low", "medium", "high"], index=0)

            predict_clicked = st.button("Assess retention risk", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        icon_cols = st.columns(4)
        for col, (icon_file, label) in zip(icon_cols, icons):
            with col:
                data_uri = svg_to_data_uri(load_svg(icon_file))
                st.markdown(
                    f'<div class="rr-icon-tile"><img src="{data_uri}"/>'
                    f'<span style="font-size:0.8rem;color:{SLATE};">{label}</span></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown("#### Risk read-out")
        if model is None:
            st.info("Load `employee_attrition_bagging.pkl` in the app folder to see a live prediction.")
        elif predict_clicked:
            row = build_feature_row(
                satisfaction, evaluation, projects, hours, tenure,
                accident, promotion, dept, salary_level,
            )
            proba = model.predict_proba(row)[0][1]
            label, color, bg = risk_badge(proba)

            st.markdown(f'<div class="rr-card">{risk_meter_svg(proba)}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="rr-badge" style="background:{bg};color:{color};">{label}</span>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="rr-card">', unsafe_allow_html=True)
                st.markdown('<div class="rr-metric-label">Notable signals</div>', unsafe_allow_html=True)
                notes = []
                if satisfaction < 0.4:
                    notes.append("Low satisfaction score")
                if hours > 250 or hours < 130:
                    notes.append("Monthly hours are far from the typical band")
                if tenure >= 4 and promotion == 0:
                    notes.append("Tenured without a recent promotion")
                if projects >= 6:
                    notes.append("Carrying a high project load")
                if not notes:
                    notes.append("No strong risk signals detected")
                for n in notes:
                    st.markdown(f"- {n}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("Set the profile on the left, then click **Assess retention risk**.")

# --------------------------------------------------------------------------
# PAGE: Batch upload
# --------------------------------------------------------------------------
elif page == "Batch upload":
    st.markdown("#### Batch scoring")
    st.caption(
        "Upload a CSV with the original HR columns: satisfaction_level, last_evaluation, "
        "number_project, average_montly_hours, time_spend_company, Work_accident, "
        "promotion_last_5years, dept, salary."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.markdown('<div class="rr-card">', unsafe_allow_html=True)
        st.markdown("**Preview**")
        st.dataframe(raw.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if model is None:
            st.info("Load `employee_attrition_bagging.pkl` in the app folder to score this file.")
        else:
            if st.button("Score all rows", use_container_width=True):
                x = preprocess_batch(raw)
                proba = model.predict_proba(x)[:, 1]
                pred = (proba >= 0.5).astype(int)
                result = raw.copy()
                result["attrition_probability"] = proba.round(3)
                result["predicted_left"] = pred

                st.markdown('<div class="rr-card">', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows scored", len(result))
                m2.metric("Flagged high risk (≥50%)", int(pred.sum()))
                m3.metric("Average risk", f"{proba.mean()*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

                st.dataframe(
                    result.sort_values("attrition_probability", ascending=False),
                    use_container_width=True,
                )

                csv_buf = io.StringIO()
                result.to_csv(csv_buf, index=False)
                st.download_button(
                    "Download scored CSV",
                    data=csv_buf.getvalue(),
                    file_name="retention_radar_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# --------------------------------------------------------------------------
# PAGE: Model insights
# --------------------------------------------------------------------------
elif page == "Model insights":
    st.markdown("#### Ensemble insights")
    if model is None:
        st.info("Load `employee_attrition_bagging.pkl` in the app folder to see ensemble insights.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="rr-card">', unsafe_allow_html=True)
            st.markdown('<div class="rr-metric-label">Ensemble size</div>', unsafe_allow_html=True)
            st.markdown(f"### {getattr(model, 'n_estimators', 'n/a')} trees")
            oob = getattr(model, "oob_score_", None)
            if oob is not None:
                st.caption(f"Out-of-bag accuracy: {oob*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            base = getattr(model, "estimator", getattr(model, "base_estimator_", None))
            st.markdown('<div class="rr-card">', unsafe_allow_html=True)
            st.markdown('<div class="rr-metric-label">Base learner</div>', unsafe_allow_html=True)
            st.markdown(f"### {type(base).__name__ if base is not None else 'Decision Tree'}")
            depth = getattr(base, "max_depth", None) if base is not None else None
            if depth:
                st.caption(f"Max depth: {depth}")
            st.markdown('</div>', unsafe_allow_html=True)

        fi = feature_importance_frame(model)
        if fi is not None:
            st.markdown('<div class="rr-card">', unsafe_allow_html=True)
            st.markdown("**Average feature importance across trees**")
            st.bar_chart(fi.set_index("feature"), horizontal=True, color=TEAL)
            st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# PAGE: About
# --------------------------------------------------------------------------
else:
    st.markdown("#### About Retention Radar")
    st.markdown(
        """
        <div class="rr-card">
        Retention Radar wraps a <b>BaggingClassifier</b> of 20 depth-4 decision trees,
        trained on the classic HR attrition dataset, in a single-employee and batch
        scoring interface.
        <br><br>
        <b>Features used by the model:</b> satisfaction level, last evaluation score,
        number of projects, average monthly hours, time spent at the company, work
        accident history, promotion in the last 5 years, department, and salary band.
        <br><br>
        Drop your trained <code>employee_attrition_bagging.pkl</code> file next to
        <code>app.py</code> and run:
        <br><br>
        <code>streamlit run app.py</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
