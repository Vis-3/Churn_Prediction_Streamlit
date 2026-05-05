import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR= os.path.join(ROOT, "results")

# ── Color palette ─────────────────────────────────────────────────────────────
C = dict(
    primary   = "#0f172a",
    secondary = "#3b82f6",
    churn     = "#f87171",
    retain    = "#34d399",
    warning   = "#fbbf24",
    purple    = "#a78bfa",
    bg        = "#0a0a0f",
    card      = "#111118",
    card2     = "#1a1a24",
    text      = "#f1f5f9",
    muted     = "#94a3b8",
    border    = "#1e2030",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, section.main > div {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
  }}
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"] {{ background: transparent !important; }}

  /* Sidebar (if opened) */
  [data-testid="stSidebar"] {{ background: {C['card']} !important; }}

  /* Streamlit generic text */
  p, li, span, label, div {{ color: {C['text']}; }}

  /* ── Hero banner ── */
  .hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #6d28d9 100%);
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: #f1f5f9;
  }}
  .hero h1 {{ font-size: 2.2rem; font-weight: 800; margin: 0 0 .5rem;
               letter-spacing: -0.5px; color: #ffffff; }}
  .hero p  {{ font-size: 1.05rem; margin: 0; opacity: .88;
               max-width: 680px; line-height: 1.6; color: #e2e8f0; }}

  /* ── KPI cards ── */
  .kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .kpi-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    flex: 1; min-width: 140px;
    box-shadow: 0 2px 12px rgba(0,0,0,.4);
  }}
  .kpi-label {{ font-size: .72rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: .08em; color: {C['muted']}; margin-bottom: .4rem; }}
  .kpi-value {{ font-size: 1.9rem; font-weight: 800; color: #ffffff; line-height: 1; }}
  .kpi-delta {{ font-size: .78rem; margin-top: .3rem; color: {C['muted']}; }}

  /* ── Section headers ── */
  .section-header {{
    font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid {C['secondary']};
    padding-left: .75rem;
    margin: 2rem 0 1rem;
  }}

  /* ── Insight callout boxes ── */
  .insight {{
    border-radius: 10px; padding: 1rem 1.25rem;
    margin: 1rem 0; font-size: .92rem; line-height: 1.6;
  }}
  .insight-blue  {{ background: #1e2f4a; border-left: 4px solid {C['secondary']}; color: #bfdbfe; }}
  .insight-red   {{ background: #2a1515; border-left: 4px solid {C['churn']};     color: #fca5a5; }}
  .insight-green {{ background: #0f2a1e; border-left: 4px solid {C['retain']};    color: #6ee7b7; }}
  .insight-amber {{ background: #2a1f0a; border-left: 4px solid {C['warning']};   color: #fcd34d; }}

  /* ── Tabs ── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {C['card']} !important;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
  }}
  [data-testid="stTabs"] [data-baseweb="tab"] {{
    font-size: .88rem; font-weight: 600; padding: .55rem 1.1rem;
    color: {C['muted']} !important;
    border-radius: 8px !important;
    background: transparent !important;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
    background: {C['card2']} !important;
    color: #ffffff !important;
  }}
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    display: none !important;
  }}
  [data-testid="stTabs"] [data-baseweb="tab-border"] {{
    background: {C['border']} !important;
  }}

  /* ── Dataframes ── */
  [data-testid="stDataFrame"] > div {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
  }}
  .dvn-scroller {{ background: {C['card']} !important; }}

  /* ── Sliders ── */
  [data-testid="stSlider"] label {{ color: {C['muted']} !important; }}

  /* ── Step badge ── */
  .step-badge {{
    display: inline-block;
    background: {C['secondary']};
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px; text-align: center;
    font-weight: 700; font-size: .85rem;
    margin-right: .5rem;
  }}
</style>
""", unsafe_allow_html=True)

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(os.path.join(DATA_DIR, "telco_churn.csv"))

@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(DATA_DIR, "churn_features.csv"))

@st.cache_data
def load_risk():
    p = os.path.join(DATA_DIR, "churn_risk_results.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_roi():
    p = os.path.join(DATA_DIR, "roi_analysis.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_resource
def load_model_artifacts():
    try:
        with open(os.path.join(MODELS_DIR, "advanced_ensemble.pkl"), "rb") as f:
            model = pickle.load(f)
        X_test  = np.load(os.path.join(DATA_DIR, "X_test_adv.npy"))
        y_test  = np.load(os.path.join(DATA_DIR, "y_test_adv.npy"))
        X_train = np.load(os.path.join(DATA_DIR, "X_train_adv.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train_adv.npy"))
        with open(os.path.join(MODELS_DIR, "threshold.pkl"), "rb") as f:
            threshold = float(pickle.load(f))
        return model, X_train, X_test, y_train, y_test, threshold
    except Exception:
        return None, None, None, None, None, 0.31

# ── Plotly theme helper ───────────────────────────────────────────────────────
LAYOUT = dict(
    font_family="Inter, system-ui, sans-serif",
    font_color=C["text"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d0d14",
    margin=dict(l=20, r=20, t=44, b=20),
    title_font_size=14,
    title_font_color="#e2e8f0",
)

def styled(fig, height=360):
    fig.update_layout(**LAYOUT, height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#1e2030", zeroline=False,
                     tickfont_color=C["muted"])
    fig.update_yaxes(showgrid=True, gridcolor="#1e2030", zeroline=False,
                     tickfont_color=C["muted"])
    return fig

def kpi(label, value, delta=""):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta">{delta}</div>
    </div>"""

def insight(text, kind="blue"):
    return f'<div class="insight insight-{kind}">{text}</div>'

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
raw      = load_raw()
features = load_features()
risk_df  = load_risk()
roi_df   = load_roi()
model, X_train, X_test, y_train, y_test, THRESHOLD = load_model_artifacts()

raw["Churn_bin"] = (raw["Churn"] == "Yes").astype(int)
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Churn Intelligence Platform</h1>
  <p>An end-to-end data science case study: from raw customer data to causal intervention strategy.
  We don't just predict who will leave; we identify <strong>why</strong> they leave and
  <strong>who will respond</strong> to retention efforts, quantified in ROI.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "The Problem",
    "Data Explorer",
    "Feature Engineering",
    "Modeling",
    "Causal Inference",
    "Business Impact",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THE PROBLEM
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <p style="font-size:1.05rem;color:#475569;line-height:1.8;max-width:780px">
    Telecom churn costs the industry <strong>$62B annually</strong>. For a company with 7,000 customers
    charging ~$65/month, every percentage-point drop in churn rate is worth <strong>~$55,000/year</strong>.
    The challenge: most churn models tell you <em>who</em> will leave but not <em>what to do about it</em>.
    Blindly contacting at-risk customers can make things worse.
    </p>
    """, unsafe_allow_html=True)

    churn_rate = raw["Churn_bin"].mean()
    n_churn    = raw["Churn_bin"].sum()
    avg_mc     = raw["MonthlyCharges"].mean()
    annual_rev = raw["MonthlyCharges"].sum() * 12

    st.markdown(
        '<div class="kpi-row">' +
        kpi("Total Customers", f"{len(raw):,}") +
        kpi("Churned (26.5%)", f"{n_churn:,}", "High-risk segment") +
        kpi("Avg Monthly Charge", f"${avg_mc:.2f}") +
        kpi("Annual Revenue at Risk", f"${n_churn * avg_mc * 12 / 1e6:.1f}M") +
        kpi("Class Imbalance", "73.5 / 26.5", "Requires SMOTE") +
        '</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        section("Churn Distribution")
        counts = raw["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        fig = px.pie(
            counts, names="Churn", values="Count",
            color="Churn",
            color_discrete_map={"Yes": C["churn"], "No": C["retain"]},
            hole=0.55,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker_line_width=2, marker_line_color="white")
        fig.update_layout(**LAYOUT, height=320, showlegend=False,
                          annotations=[dict(text="<b>26.5%</b><br>churn",
                                            x=0.5, y=0.5, font_size=16,
                                            showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Churn Rate by Contract Type")
        ct = raw.groupby("Contract")["Churn_bin"].mean().reset_index()
        ct.columns = ["Contract", "Churn Rate"]
        ct["Churn Rate %"] = ct["Churn Rate"] * 100
        fig = px.bar(
            ct, x="Contract", y="Churn Rate %",
            color="Churn Rate %",
            color_continuous_scale=[[0, C["retain"]], [0.5, C["warning"]], [1, C["churn"]]],
            text=ct["Churn Rate %"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**LAYOUT, height=320, coloraxis_showscale=False,
                          yaxis_ticksuffix="%", yaxis_range=[0, 55])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(insight(
        "<strong>Key finding:</strong> Month-to-month customers churn at <strong>42.7%</strong> vs "
        "just <strong>6.8%</strong> for annual/two-year contracts. Contract type is the single strongest "
        "observable driver, but is it causal? (We test this in Tab 5.)",
        "blue"
    ), unsafe_allow_html=True)

    section("The Data Science Journey")
    steps = [
        ("1", "Raw Data (7,043 customers, 21 features)", "Telco dataset with demographics, services, billing"),
        ("2", "Exploratory Analysis", "Understand distributions, class imbalance, correlations"),
        ("3", "Feature Engineering", "35 features including 5 causal interaction terms"),
        ("4", "Predictive Modeling", "11 baselines → Optuna tuning → Calibrated Voting Ensemble (ROC-AUC 0.847)"),
        ("5", "Causal Inference", "PSM + Uplift Models + CausalForestDML isolate true causal effects"),
        ("6", "Business Strategy", "4-segment uplift strategy with quantified ROI per intervention"),
    ]
    for badge, title, desc in steps:
        st.markdown(
            f'<div style="padding:.6rem 0;border-bottom:1px solid {C["border"]}">'
            f'<span class="step-badge">{badge}</span>'
            f'<strong style="color:#e2e8f0">{title}</strong>'
            f'<span style="color:{C["muted"]};font-size:.88rem;margin-left:.5rem">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        '<p style="font-size:1rem;color:#475569;line-height:1.7;max-width:760px">'
        'Before building any model, we need to understand <em>what the data tells us</em>. '
        'Each chart below surfaces a pattern that will later justify a modeling choice or a causal test.'
        '</p>', unsafe_allow_html=True
    )

    # ── Numerical distributions ───────────────────────────────────────────────
    section("Numerical Feature Distributions by Churn")
    col1, col2, col3 = st.columns(3)

    num_cols = [("tenure", "Tenure (months)"), ("MonthlyCharges", "Monthly Charges ($)"), ("TotalCharges", "Total Charges ($)")]
    for col, (feat, label) in zip([col1, col2, col3], num_cols):
        with col:
            fig = px.histogram(
                raw, x=feat, color="Churn",
                barmode="overlay", nbins=40,
                color_discrete_map={"Yes": C["churn"], "No": C["retain"]},
                opacity=0.75,
                labels={feat: label, "Churn": "Churned"},
            )
            fig.update_layout(**LAYOUT, height=280, title=label,
                              legend=dict(orientation="h", y=1.15, x=0))
            col.plotly_chart(fig, use_container_width=True)

    st.markdown(insight(
        "Churners are heavily concentrated in <strong>early tenure (0–12 months)</strong> "
        "and <strong>higher monthly charges</strong>. Customers who survive past 2 years rarely leave. "
        "This motivates both the <code>early_customer</code> flag and <code>price_tenure_risk</code> feature.",
        "blue"
    ), unsafe_allow_html=True)

    # ── Categorical churn rates ───────────────────────────────────────────────
    section("Churn Rate by Key Categorical Features")
    cat_feats = [
        ("InternetService", "Internet Service"),
        ("PaymentMethod", "Payment Method"),
        ("SeniorCitizen", "Senior Citizen"),
        ("Partner", "Has Partner"),
        ("Dependents", "Has Dependents"),
        ("PaperlessBilling", "Paperless Billing"),
    ]
    col1, col2 = st.columns(2)
    for i, (feat, label) in enumerate(cat_feats):
        col = col1 if i % 2 == 0 else col2
        with col:
            grp = (
                raw.groupby(feat)["Churn_bin"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Churn Rate", "count": "n"})
            )
            grp["Churn Rate %"] = grp["Churn Rate"] * 100
            grp[feat] = grp[feat].astype(str)
            fig = px.bar(
                grp, x=feat, y="Churn Rate %",
                color="Churn Rate %",
                color_continuous_scale=[[0, C["retain"]], [0.5, C["warning"]], [1, C["churn"]]],
                text=grp["Churn Rate %"].apply(lambda x: f"{x:.1f}%"),
                labels={feat: label},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**LAYOUT, height=260, title=label,
                              coloraxis_showscale=False, yaxis_ticksuffix="%")
            col.plotly_chart(fig, use_container_width=True)

    # ── Tenure survival ───────────────────────────────────────────────────────
    section("Churn Hazard Across Tenure Bands")
    raw["TenureBand"] = pd.cut(
        raw["tenure"],
        bins=[0, 6, 12, 24, 36, 48, 60, 72],
        labels=["0–6m", "7–12m", "13–24m", "25–36m", "37–48m", "49–60m", "61–72m"],
    )
    tenure_churn = raw.groupby("TenureBand", observed=True)["Churn_bin"].mean().reset_index()
    tenure_churn.columns = ["Tenure Band", "Churn Rate"]
    tenure_churn["Churn Rate %"] = tenure_churn["Churn Rate"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tenure_churn["Tenure Band"], y=tenure_churn["Churn Rate %"],
        marker_color=[C["churn"] if v > 30 else C["warning"] if v > 15 else C["retain"]
                      for v in tenure_churn["Churn Rate %"]],
        text=tenure_churn["Churn Rate %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT, height=320, yaxis_ticksuffix="%",
                      title="Churn Rate Drops Sharply After 12 Months")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(insight(
        "<strong>51% of churners leave within the first 12 months.</strong> The drop from the 0–6m band "
        "to the 61–72m band is extreme: early-tenure customers are a fundamentally different risk cohort. "
        "This directly motivated the <code>early_customer</code> (tenure ≤ 6) binary feature.",
        "amber"
    ), unsafe_allow_html=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    section("Correlation Among Numeric Features")
    num_df = raw[["tenure", "MonthlyCharges", "TotalCharges", "Churn_bin", "SeniorCitizen"]].copy()
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        color_continuous_scale=[[0, C["churn"]], [0.5, "white"], [1, C["secondary"]]],
        zmin=-1, zmax=1,
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(**LAYOUT, height=360, title="Pearson Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        '<p style="font-size:1rem;color:#475569;line-height:1.7;max-width:760px">'
        'Raw features describe customers; <em>engineered features</em> capture the mechanisms behind churn. '
        'We expanded from 21 raw fields to 35 features, each grounded in a business hypothesis '
        'and validated by SHAP importance scores.'
        '</p>', unsafe_allow_html=True
    )

    section("Engineered Feature Catalogue")
    feat_rows = [
        ("price_stress",              "MonthlyCharges × IsMonthToMonth",    "High charges on a flexible contract, the riskiest combination"),
        ("price_tenure_risk",         "MonthlyCharges ÷ (tenure + 1)",      "Price burden relative to relationship depth; spikes for new, expensive customers"),
        ("fiber_contract_risk",       "IsFiber × IsMonthToMonth",           "Fiber optic customers on short-term contracts, the highest empirical churn group"),
        ("early_customer",            "tenure ≤ 6 months  (binary)",        "First 6 months are the critical retention window"),
        ("fiber_new_customer",        "IsFiber × early_customer",           "Brand-new fiber customers face both onboarding friction and service issues"),
        ("ContractMonths",            "Contract → {1, 12, 24}",             "Numeric encoding of contractual lock-in"),
        ("Contract_Tenure_Interaction","ContractMonths × tenure",           "Rewards customers who have stayed longer than their contract requires"),
        ("HasMultipleServices",       "Count of add-on services (0–6)",     "Engagement proxy: customers with more services are stickier"),
        ("AvgMonthlySpend",           "TotalCharges ÷ tenure",              "Normalised spend; corrects for tenure length when comparing bills"),
    ]

    rows_html = "".join(
        f"""<tr>
          <td style="font-family:monospace;font-size:.85rem;color:{C['secondary']};
                     white-space:nowrap;padding:.65rem 1rem;border-bottom:1px solid {C['border']}">{feat}</td>
          <td style="font-family:monospace;font-size:.82rem;color:{C['warning']};
                     white-space:nowrap;padding:.65rem 1rem;border-bottom:1px solid {C['border']}">{formula}</td>
          <td style="font-size:.88rem;color:{C['muted']};padding:.65rem 1rem;
                     border-bottom:1px solid {C['border']};line-height:1.5">{rationale}</td>
        </tr>"""
        for feat, formula, rationale in feat_rows
    )
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:{C['card']};
                  border:1px solid {C['border']};border-radius:10px;overflow:hidden">
      <thead>
        <tr style="background:{C['card2']}">
          <th style="text-align:left;padding:.7rem 1rem;font-size:.72rem;font-weight:700;
                     text-transform:uppercase;letter-spacing:.07em;color:{C['muted']};
                     border-bottom:1px solid {C['border']}">Feature</th>
          <th style="text-align:left;padding:.7rem 1rem;font-size:.72rem;font-weight:700;
                     text-transform:uppercase;letter-spacing:.07em;color:{C['muted']};
                     border-bottom:1px solid {C['border']}">Formula / Definition</th>
          <th style="text-align:left;padding:.7rem 1rem;font-size:.72rem;font-weight:700;
                     text-transform:uppercase;letter-spacing:.07em;color:{C['muted']};
                     border-bottom:1px solid {C['border']}">Business Rationale</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # ── Interaction feature distributions ────────────────────────────────────
    section("Interaction Features vs Churn Rate")
    if "price_stress" in features.columns and "Churn" in features.columns:
        feat_churn = features.copy()
        feat_churn["Churn_bin"] = feat_churn["Churn"].astype(int) if feat_churn["Churn"].dtype != object else (feat_churn["Churn"] == "Yes").astype(int)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                feat_churn, x="Churn_bin", y="price_stress",
                color="Churn_bin",
                color_discrete_map={0: C["retain"], 1: C["churn"]},
                labels={"Churn_bin": "Churned", "price_stress": "price_stress"},
                category_orders={"Churn_bin": [0, 1]},
            )
            fig.update_layout(**LAYOUT, height=300, title="price_stress by Churn",
                              showlegend=False, xaxis_tickvals=[0, 1],
                              xaxis_ticktext=["No Churn", "Churned"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                feat_churn, x="Churn_bin", y="price_tenure_risk",
                color="Churn_bin",
                color_discrete_map={0: C["retain"], 1: C["churn"]},
                labels={"Churn_bin": "Churned", "price_tenure_risk": "price_tenure_risk"},
                category_orders={"Churn_bin": [0, 1]},
            )
            fig.update_layout(**LAYOUT, height=300, title="price_tenure_risk by Churn",
                              showlegend=False, xaxis_tickvals=[0, 1],
                              xaxis_ticktext=["No Churn", "Churned"])
            st.plotly_chart(fig, use_container_width=True)

    # ── Num services ─────────────────────────────────────────────────────────
    section("Service Depth and Churn")
    if "NumServices" in features.columns:
        feat_c = features.copy()
        if "Churn_bin" not in feat_c.columns:
            feat_c["Churn_bin"] = feat_c["Churn"].astype(int) if feat_c["Churn"].dtype != object else (feat_c["Churn"] == "Yes").astype(int)
        svc = feat_c.groupby("NumServices")["Churn_bin"].mean().reset_index()
        svc.columns = ["Num Services", "Churn Rate"]
        svc["Churn Rate %"] = svc["Churn Rate"] * 100
        fig = px.bar(
            svc, x="Num Services", y="Churn Rate %",
            color="Churn Rate %",
            color_continuous_scale=[[0, C["retain"]], [1, C["churn"]]],
            text=svc["Churn Rate %"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**LAYOUT, height=320, coloraxis_showscale=False,
                          yaxis_ticksuffix="%", title="Customers with more services churn less")
        st.plotly_chart(fig, use_container_width=True)

    # ── SHAP plot ─────────────────────────────────────────────────────────────
    shap_path = os.path.join(RESULTS_DIR, "shap_summary_plot.png")
    if os.path.exists(shap_path):
        section("SHAP Feature Importance (XGBoost)")
        st.markdown(insight(
            "SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each prediction, "
            "enabling both global importance ranking and per-customer explanations. "
            "Red = feature value pushes towards churn; blue = towards retention.",
            "blue"
        ), unsafe_allow_html=True)
        st.image(shap_path, use_container_width=True)

    st.markdown(insight(
        "<strong>All 5 interaction features rank in the top 15 by SHAP importance</strong>, "
        "validating the feature engineering hypotheses. The ensemble ROC-AUC improved from "
        "<strong>0.834 to 0.847</strong> after adding them.",
        "green"
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODELING
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(
        '<p style="font-size:1rem;color:#475569;line-height:1.7;max-width:760px">'
        'We benchmarked 9 model families, tuned the top performers with Optuna, '
        'and combined them into a calibrated voting ensemble. '
        'Calibration ensures the predicted probabilities are reliable, critical for the '
        'uplift segmentation downstream.'
        '</p>', unsafe_allow_html=True
    )

    # ── Baseline comparison ───────────────────────────────────────────────────
    section("Model Comparison (ROC-AUC on Held-out Test Set)")
    baselines = pd.DataFrame([
        ("Gradient Boosting",    0.847, True),
        ("Logistic Regression",  0.846, True),
        ("LDA",                  0.843, True),
        ("Naive Bayes",          0.832, False),
        ("Random Forest",        0.830, False),
        ("XGBoost",              0.820, True),
        ("SVM",                  0.800, False),
        ("KNN",                  0.782, False),
        ("Decision Tree",        0.656, False),
    ], columns=["Model", "ROC-AUC", "In Ensemble"])

    fig = px.bar(
        baselines.sort_values("ROC-AUC"),
        x="ROC-AUC", y="Model",
        orientation="h",
        color="In Ensemble",
        color_discrete_map={True: C["secondary"], False: C["muted"]},
        text=baselines.sort_values("ROC-AUC")["ROC-AUC"].apply(lambda x: f"{x:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **LAYOUT, height=380,
        xaxis_range=[0.6, 0.88],
        legend=dict(orientation="h", y=-0.15),
        title="Blue bars = included in final voting ensemble",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(insight(
        "The top-3 models (Gradient Boosting, Logistic Regression, LDA) were tuned with Optuna "
        "(40/30/20 trials respectively) and combined via <strong>soft-voting ensemble</strong>. "
        "Probabilities were then calibrated with <strong>isotonic regression</strong> to improve "
        "reliability for downstream uplift modeling.",
        "blue"
    ), unsafe_allow_html=True)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        section("ROC Curve (Test Set)")
        if model is not None and X_test is not None:
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=C["secondary"], width=2.5),
                name=f"Ensemble (AUC = {roc_auc:.3f})",
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color=C["muted"], dash="dash", width=1),
                name="Random Baseline",
            ))
            fig.update_layout(**LAYOUT, height=360,
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              legend=dict(x=0.45, y=0.08))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model artefacts not loaded.")

    with col2:
        section("Confusion Matrix (Threshold = 0.30)")
        if model is not None and X_test is not None:
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= THRESHOLD).astype(int)
            cm    = confusion_matrix(y_test, preds)
            labels = ["No Churn", "Churn"]

            fig = px.imshow(
                cm, x=labels, y=labels,
                color_continuous_scale=[[0, "#f8fafc"], [1, C["secondary"]]],
                text_auto=True, aspect="auto",
                labels=dict(x="Predicted", y="Actual"),
            )
            fig.update_layout(**LAYOUT, height=360,
                              title="Confusion Matrix",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Performance metrics ───────────────────────────────────────────────────
    if model is not None and X_test is not None:
        from sklearn.metrics import classification_report
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)
        rep   = classification_report(y_test, preds, output_dict=True)

        section("Key Metrics at Threshold = 0.30")
        st.markdown(
            '<div class="kpi-row">' +
            kpi("ROC-AUC",   f"{auc(roc_curve(y_test, probs)[0], roc_curve(y_test, probs)[1]):.3f}") +
            kpi("Churn Recall",    f"{rep['1']['recall']:.1%}", "Catching actual churners") +
            kpi("Churn Precision", f"{rep['1']['precision']:.1%}", "Predicted churners correct") +
            kpi("F1 (Churn)",      f"{rep['1']['f1-score']:.1%}") +
            kpi("Accuracy",        f"{rep['accuracy']:.1%}") +
            '</div>', unsafe_allow_html=True
        )

    st.markdown(insight(
        "<strong>Why threshold 0.30 instead of 0.50?</strong> The cost of a missed churner "
        "(losing the customer entirely) far exceeds the cost of a false alarm (a wasted retention offer). "
        "We tuned the threshold to maximise F1 on the churn class, accepting more false positives "
        "in exchange for catching more true churners.",
        "amber"
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CAUSAL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(
        '<p style="font-size:1rem;color:#475569;line-height:1.7;max-width:760px">'
        'A predictive model tells you <em>who is likely to churn</em>. '
        'A causal model tells you <em>what is causing the churn</em>. '
        'These are fundamentally different questions. Conflating them leads to wasted budgets, '
        'or worse, interventions that accelerate the very behaviour you want to prevent.'
        '</p>', unsafe_allow_html=True
    )

    # ── Correlation vs causation ──────────────────────────────────────────────
    section("Why Correlation Isn't Enough")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};border-radius:12px;padding:1.5rem">
        <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;
                    color:{C['muted']};font-weight:700;margin-bottom:.8rem">
          CORRELATIONAL MODEL OUTPUT
        </div>
        <p style="color:{C['text']};line-height:1.7">
          "Customer X has <strong style="color:#f1f5f9">78% churn probability</strong>.<br>
          They have fiber optic internet and a month-to-month contract.<br>
          ➜ Contact them with a retention offer."
        </p>
        <div style="background:#2a1515;border-left:3px solid {C['churn']};border-radius:8px;
                    padding:.8rem;margin-top:1rem;font-size:.88rem;color:#fca5a5">
          ❌ <strong>Problem:</strong> We don't know if contacting them helps, hurts, or does nothing.
          Maybe they're happy and the model picked up on proxy signals.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};border-radius:12px;padding:1.5rem">
        <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;
                    color:{C['muted']};font-weight:700;margin-bottom:.8rem">
          CAUSAL + UPLIFT MODEL OUTPUT
        </div>
        <p style="color:{C['text']};line-height:1.7">
          "Customer X has <strong style="color:#f1f5f9">78% churn probability</strong> AND
          an <strong style="color:#f1f5f9">uplift score of +0.29</strong>.<br>
          They belong to the 'Persuadable' segment.<br>
          ➜ Offer an annual contract discount."
        </p>
        <div style="background:#0f2a1e;border-left:3px solid {C['retain']};border-radius:8px;
                    padding:.8rem;margin-top:1rem;font-size:.88rem;color:#6ee7b7">
          ✅ <strong>Result:</strong> We know this customer responds positively to contract-based
          interventions, targeting them yields positive expected ROI.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PSM results ───────────────────────────────────────────────────────────
    section("Propensity Score Matching: Causal Effect Estimates")
    st.markdown(
        '<p style="font-size:.92rem;color:#475569;line-height:1.6;max-width:720px;margin-bottom:1rem">'
        'PSM creates matched pairs of similar customers who differ only on one treatment variable. '
        'The difference in churn rates between matched pairs is the <strong>causal effect</strong>: '
        'not a correlation, but an estimate of what would happen if we changed that one factor.'
        '</p>', unsafe_allow_html=True
    )

    psm_data = pd.DataFrame({
        "Treatment": [
            "Fiber optic internet",
            "Month-to-month contract",
            "High monthly charges",
            "Low tenure (≤6 months)",
        ],
        "Causal Effect (pp)": [36.1, 30.4, 21.3, 21.2],
        "Confounder Controls": [
            "MonthlyCharges, tenure, Contract, SeniorCitizen, NumServices",
            "Partner, tenure, MonthlyCharges, Dependents, SeniorCitizen, NumServices",
            "Partner, tenure, Contract, Dependents, SeniorCitizen, NumServices",
            "Partner, MonthlyCharges, Contract, Dependents, SeniorCitizen",
        ]
    })

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        colors = [C["churn"], C["churn"], C["warning"], C["warning"]]
        fig.add_trace(go.Bar(
            x=psm_data["Causal Effect (pp)"],
            y=psm_data["Treatment"],
            orientation="h",
            marker_color=colors,
            text=[f"+{v:.1f}pp" for v in psm_data["Causal Effect (pp)"]],
            textposition="outside",
        ))
        fig.update_layout(
            **LAYOUT, height=300,
            xaxis_title="Causal increase in churn probability (percentage points)",
            xaxis_range=[0, 45],
            title="Holding all else equal, each treatment causally increases churn by:",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style="background:{C['card2']};border:1px solid {C['border']};
                    border-radius:12px;padding:1.2rem;margin-top:.5rem">
          <div style="font-weight:700;color:#e2e8f0;margin-bottom:.8rem">
            How PSM Works
          </div>
          <ol style="color:{C['muted']};font-size:.88rem;line-height:1.9;padding-left:1.2rem;margin:0">
            <li>Fit a propensity model to predict treatment assignment from confounders</li>
            <li>Match each treated customer to the most similar control customer</li>
            <li>Compare churn rates in matched pairs only</li>
            <li>The remaining difference is causally attributed to the treatment</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(insight(
        "<strong>Fiber optic + month-to-month is the highest-risk combination.</strong> "
        "Fiber optic causally adds +36.1pp to churn; month-to-month adds +30.4pp, independently, "
        "after controlling for price, tenure, demographics, and service bundle.",
        "red"
    ), unsafe_allow_html=True)

    # ── Uplift distribution ───────────────────────────────────────────────────
    section("Individual Treatment Effects: Uplift Score Distribution")
    if risk_df is not None and "uplift_X_T_MonthToMonth" in risk_df.columns:
        uplift_clipped = risk_df["uplift_X_T_MonthToMonth"].clip(-2, 5)
        fig = px.histogram(
            risk_df.assign(uplift_clipped=uplift_clipped),
            x="uplift_clipped",
            color="segment" if "segment" in risk_df.columns else None,
            nbins=60,
            color_discrete_map={
                "Persuadable":   C["secondary"],
                "Sure Thing":    C["retain"],
                "Sleeping Dog":  C["warning"],
                "Lost Cause":    C["churn"],
            },
            labels={"uplift_clipped": "Uplift Score (clipped to [-2, 5])"},
            barmode="overlay",
            opacity=0.75,
        )
        fig.add_vline(x=0, line_dash="dash", line_color=C["text"], line_width=1.5,
                      annotation_text="  uplift = 0", annotation_font_size=11)
        fig.update_layout(**LAYOUT, height=340,
                          title="Distribution of Individual Uplift Scores by Segment",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    # ── Segment strategy ─────────────────────────────────────────────────────
    section("The 4-Segment Intervention Strategy")
    seg_table = pd.DataFrame([
        ("Persuadable",  "High", "Positive", "Intervene: highest ROI",                    C["secondary"]),
        ("Lost Cause",   "High", "Negative", "Skip: won't respond to treatment",           C["churn"]),
        ("Sleeping Dog", "Low",  "Negative", "Do NOT contact: intervention accelerates churn", C["warning"]),
        ("Sure Thing",   "Low",  "Positive", "Monitor only: will stay without help",       C["retain"]),
    ], columns=["Segment", "Churn Risk", "Treatment Uplift", "Strategy", "_color"])

    for _, row in seg_table.iterrows():
        st.markdown(
            f'<div style="background:{C["card"]};border:1px solid {C["border"]};'
            f'border-left:5px solid {row["_color"]};border-radius:8px;'
            f'padding:.9rem 1.2rem;margin:.4rem 0;display:flex;gap:1rem;align-items:center">'
            f'<strong style="color:#f1f5f9;min-width:120px">{row["Segment"]}</strong>'
            f'<span style="color:{C["muted"]};font-size:.88rem;min-width:80px">Risk: {row["Churn Risk"]}</span>'
            f'<span style="color:{C["muted"]};font-size:.88rem;min-width:130px">Uplift: {row["Treatment Uplift"]}</span>'
            f'<span style="color:{C["text"]};font-size:.92rem">{row["Strategy"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(insight(
        "<strong>The Sleeping Dog insight is the most counterintuitive finding.</strong> "
        "These customers have negative treatment uplift: contacting them about their contract "
        "appears to <em>remind them they could leave</em>. Any blanket 'contact all at-risk' strategy "
        "would inadvertently trigger churn in this 3,151-customer group.",
        "red"
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        '<p style="font-size:1rem;color:#475569;line-height:1.7;max-width:760px">'
        'The causal segmentation translates directly into business strategy. '
        'Rather than targeting all high-risk customers (expensive, harmful for Sleeping Dogs), '
        'we focus every dollar on the <strong>Persuadable</strong> segment where intervention '
        'causally improves retention.'
        '</p>', unsafe_allow_html=True
    )

    if risk_df is not None:
        # ── Segment overview ──────────────────────────────────────────────────
        section("Customer Segmentation Summary")
        seg_counts = risk_df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]

        col1, col2 = st.columns([1, 1])
        with col1:
            seg_color = {
                "Persuadable":  C["secondary"],
                "Lost Cause":   C["churn"],
                "Sleeping Dog": C["warning"],
                "Sure Thing":   C["retain"],
            }
            fig = px.pie(
                seg_counts, names="Segment", values="Customers",
                color="Segment",
                color_discrete_map=seg_color,
                hole=0.5,
            )
            fig.update_traces(textinfo="percent+label", textposition="outside",
                              marker_line_width=2, marker_line_color="white")
            fig.update_layout(**LAYOUT, height=320, showlegend=False,
                              title="Customer Segments by Uplift + Risk")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Churn_Probability" in risk_df.columns:
                seg_stats = risk_df.groupby("segment").agg(
                    Customers=("segment", "count"),
                    Avg_Churn_Prob=("Churn_Probability", "mean"),
                    Avg_Monthly=("MonthlyCharges", "mean"),
                ).round(3).reset_index()

                seg_color_map = {
                    "Persuadable":  C["secondary"],
                    "Lost Cause":   C["churn"],
                    "Sleeping Dog": C["warning"],
                    "Sure Thing":   C["retain"],
                }
                rows_html = "".join(
                    f"""<tr>
                      <td style="padding:.6rem .9rem;border-bottom:1px solid {C['border']};
                                 font-weight:600;color:{seg_color_map.get(r['segment'], C['text'])}">{r['segment']}</td>
                      <td style="padding:.6rem .9rem;border-bottom:1px solid {C['border']};
                                 color:{C['text']};text-align:right">{int(r['Customers']):,}</td>
                      <td style="padding:.6rem .9rem;border-bottom:1px solid {C['border']};
                                 color:{C['text']};text-align:right">{r['Avg_Churn_Prob']:.1%}</td>
                      <td style="padding:.6rem .9rem;border-bottom:1px solid {C['border']};
                                 color:{C['text']};text-align:right">${r['Avg_Monthly']:.2f}</td>
                    </tr>"""
                    for _, r in seg_stats.iterrows()
                )
                st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;background:{C['card']};
                              border:1px solid {C['border']};border-radius:10px;overflow:hidden;font-size:.88rem">
                  <thead><tr style="background:{C['card2']}">
                    {"".join(f'<th style="text-align:{"right" if i>0 else "left"};padding:.6rem .9rem;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:{C["muted"]};border-bottom:1px solid {C["border"]}">{h}</th>' for i,h in enumerate(["Segment","Customers","Avg Churn Prob","Avg Monthly $"]))}
                  </tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

        # ── ROI analysis ──────────────────────────────────────────────────────
        section("ROI Analysis by Intervention Target")
        if roi_df is not None:
            roi = roi_df.copy()
            if "segment" not in roi.columns:
                roi = roi.reset_index().rename(columns={roi.index.name or "index": "segment"})

            roi_plot = roi.copy()
            roi_plot["ROI Color"] = roi_plot["ROI_%"].apply(
                lambda x: C["retain"] if x > 0 else C["churn"]
            )
            roi_plot["ROI Label"] = roi_plot["ROI_%"].apply(
                lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=roi_plot["segment"],
                y=roi_plot["ROI_%"],
                marker_color=roi_plot["ROI Color"],
                text=roi_plot["ROI Label"],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_width=1.5, line_color=C["text"])
            fig.update_layout(
                **LAYOUT, height=360,
                yaxis_ticksuffix="%",
                title="ROI by Segment: only Persuadable customers generate positive returns",
                yaxis_range=[min(roi_plot["ROI_%"]) * 1.2, max(roi_plot["ROI_%"]) * 1.35],
            )
            st.plotly_chart(fig, use_container_width=True)

            # ROI table
            roi_headers = ["Segment", "Customers", "Avg Churn Prob", "Avg Monthly $",
                           "Intervention Cost", "Net Value", "ROI"]
            def roi_row_html(r):
                roi_val = float(r["ROI_%"])
                net_val = float(r["net_value"])
                roi_color = C["retain"] if roi_val > 0 else C["churn"]
                net_color = C["retain"] if net_val > 0 else C["churn"]
                seg_c = seg_color_map.get(r["segment"], C["text"])
                cells = [
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};font-weight:600;color:{seg_c}">{r["segment"]}</td>',
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};color:{C["text"]};text-align:right">{int(r["n_customers"]):,}</td>',
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};color:{C["text"]};text-align:right">{float(r["avg_churn_prob"]):.1%}</td>',
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};color:{C["text"]};text-align:right">${float(r["avg_monthly_charge"]):.2f}</td>',
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};color:{C["muted"]};text-align:right">${float(r["intervention_cost"]):,.0f}</td>',
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};color:{net_color};font-weight:600;text-align:right">{"+" if net_val>0 else ""}${net_val:,.0f}</td>',
                    f'<td style="padding:.6rem .9rem;border-bottom:1px solid {C["border"]};color:{roi_color};font-weight:700;text-align:right">{"+" if roi_val>0 else ""}{roi_val:.1f}%</td>',
                ]
                return "<tr>" + "".join(cells) + "</tr>"

            hdr = "".join(
                f'<th style="text-align:{"right" if i>0 else "left"};padding:.6rem .9rem;font-size:.7rem;'
                f'font-weight:700;text-transform:uppercase;letter-spacing:.06em;'
                f'color:{C["muted"]};border-bottom:1px solid {C["border"]}">{h}</th>'
                for i, h in enumerate(roi_headers)
            )
            rows_html = "".join(roi_row_html(r) for _, r in roi.iterrows())
            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;background:{C['card']};
                          border:1px solid {C['border']};border-radius:10px;overflow:hidden;font-size:.88rem">
              <thead><tr style="background:{C['card2']}">{hdr}</tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            # Highlight Persuadable
            if "Persuadable" in roi["segment"].values:
                p = roi[roi["segment"] == "Persuadable"].iloc[0]
                st.markdown(insight(
                    f"💰 <strong>Targeting only the {int(p['n_customers']):,} Persuadable customers</strong> "
                    f"at ${float(p['avg_monthly_charge']):.2f}/month avg charge: "
                    f"<strong>${float(p['net_value']):,.0f} net value recovered</strong> "
                    f"at {float(p['ROI_%']):.1f}% ROI. "
                    f"Intervention cost: ${float(p['intervention_cost']):,.0f}.",
                    "green"
                ), unsafe_allow_html=True)

        # ── Priority customer list ─────────────────────────────────────────────
        section("Priority Intervention List: Persuadable Customers")
        if "segment" in risk_df.columns:
            persuadable = risk_df[risk_df["segment"] == "Persuadable"].copy()


            display_cols = [c for c in
                ["customerID", "tenure", "MonthlyCharges", "Contract",
                 "InternetService", "Churn_Probability", "Risk_Level",
                 "uplift_X_T_MonthToMonth"]
                if c in persuadable.columns]

            disp = persuadable[display_cols].sort_values(
                "Churn_Probability", ascending=False
            ).head(50).reset_index(drop=True)

            st.markdown(
                f'<p style="color:{C["muted"]};font-size:.88rem;margin-bottom:.8rem">'
                f'Showing top <strong style="color:{C["text"]}">{len(disp)}</strong> of '
                f'<strong style="color:{C["text"]}">{len(persuadable):,}</strong> Persuadable customers '
                f'in selected tenure range, sorted by churn probability.</p>',
                unsafe_allow_html=True
            )

            col_labels = {
                "customerID": "Customer ID", "tenure": "Tenure", "MonthlyCharges": "Monthly $",
                "Contract": "Contract", "InternetService": "Internet",
                "Churn_Probability": "Churn Prob", "Risk_Level": "Risk",
                "uplift_X_T_MonthToMonth": "Uplift",
            }
            hdr = "".join(
                f'<th style="text-align:left;padding:.55rem .85rem;font-size:.7rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.06em;color:{C["muted"]};'
                f'border-bottom:1px solid {C["border"]};white-space:nowrap">{col_labels.get(c,c)}</th>'
                for c in display_cols
            )
            risk_colors = {"High": C["churn"], "Medium": C["warning"], "Low": C["retain"]}

            def _cell(col, val):
                if col == "Churn_Probability":
                    prob = float(val)
                    clr  = C["churn"] if prob >= 0.6 else C["warning"] if prob >= 0.3 else C["retain"]
                    return f'<td style="padding:.5rem .85rem;border-bottom:1px solid {C["border"]};color:{clr};font-weight:600">{prob:.1%}</td>'
                if col == "Risk_Level":
                    clr = risk_colors.get(str(val), C["muted"])
                    return f'<td style="padding:.5rem .85rem;border-bottom:1px solid {C["border"]};color:{clr};font-weight:600">{val}</td>'
                if col == "uplift_X_T_MonthToMonth":
                    v = float(val)
                    clr = C["retain"] if v > 0 else C["churn"]
                    return f'<td style="padding:.5rem .85rem;border-bottom:1px solid {C["border"]};color:{clr}">{v:+.3f}</td>'
                if col == "MonthlyCharges":
                    return f'<td style="padding:.5rem .85rem;border-bottom:1px solid {C["border"]};color:{C["muted"]}">${float(val):.2f}</td>'
                return f'<td style="padding:.5rem .85rem;border-bottom:1px solid {C["border"]};color:{C["text"]};font-size:.85rem">{val}</td>'

            rows_html = "".join(
                "<tr>" + "".join(_cell(c, row[c]) for c in display_cols) + "</tr>"
                for _, row in disp.iterrows()
            )
            st.markdown(f"""
            <div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;background:{C['card']};
                          border:1px solid {C['border']};border-radius:10px;overflow:hidden;font-size:.86rem">
              <thead><tr style="background:{C['card2']}">{hdr}</tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Risk results not found. Ensure `churn_risk_results.csv` is in the data directory.")

    # ── Final summary ─────────────────────────────────────────────────────────
    section("Summary: What Makes This Approach Different")
    summary_points = [
        (C["secondary"], "Beyond Prediction", "Most churn tools stop at a risk score. We go further, using causal inference to identify which customers will actually respond to intervention."),
        (C["retain"],    "No Wasted Spend",   "The Sleeping Dog segment (3,151 customers) has negative uplift. Contacting them increases churn. A standard model would target them."),
        (C["purple"],    "Quantified ROI",    "Every recommendation comes with an expected return, making it easy to prioritise budget allocation across customer segments."),
        (C["warning"],   "Honest Uncertainty","Causal estimates rely on PSM assumptions. We report effect sizes alongside confounder sets and acknowledge untestable assumptions."),
    ]
    col1, col2 = st.columns(2)
    for i, (color, title, desc) in enumerate(summary_points):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(
                f'<div style="background:{C["card"]};border:1px solid {C["border"]};'
                f'border-top:4px solid {color};border-radius:10px;padding:1.2rem;margin:.5rem 0">'
                f'<strong style="color:#f1f5f9">{title}</strong>'
                f'<p style="color:{C["muted"]};font-size:.9rem;margin:.5rem 0 0;line-height:1.6">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
