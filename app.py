
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import io
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-main:       #0f1117;
    --bg-card:       #1a1d27;
    --bg-card2:      #20243a;
    --accent:        #4f8ef7;
    --accent2:       #a78bfa;
    --accent3:       #34d399;
    --text-primary:  #f0f2f8;
    --text-secondary:#a8b0cc;
    --text-muted:    #6b7399;
    --border:        #2a2f45;
    --success:       #34d399;
    --warning:       #fbbf24;
    --danger:        #f87171;
    --sidebar-bg:    #13162a;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    padding: 6px 0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Main content area ── */
[data-testid="stMain"] {
    background-color: var(--bg-main) !important;
}
.block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── All text elements ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
p, span, div, label {
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #3a6fd8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79,142,247,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,142,247,0.4) !important;
    background: linear-gradient(135deg, #5e9bff, var(--accent)) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent3), #059669) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(52,211,153,0.25) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(52,211,153,0.4) !important;
}

/* ── Selectbox & Sliders ── */
.stSelectbox div[data-baseweb="select"] > div,
.stSlider div[data-baseweb="slider"] {
    background-color: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
.stSelectbox label, .stSlider label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* ── DataFrames / Tables ── */
.stDataFrame, .stDataFrame * {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
.stDataFrame table {
    border-collapse: collapse !important;
}
.stDataFrame th {
    background-color: var(--bg-card2) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid var(--border) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stDataFrame td {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: 0.9rem !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    color: var(--accent3) !important;
}

/* ── Info / Warning / Success boxes ── */
.stInfo {
    background-color: rgba(79,142,247,0.12) !important;
    border: 1px solid rgba(79,142,247,0.35) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
.stInfo p { color: var(--text-primary) !important; }
.stWarning {
    background-color: rgba(251,191,36,0.12) !important;
    border: 1px solid rgba(251,191,36,0.35) !important;
    border-radius: 10px !important;
}
.stSuccess {
    background-color: rgba(52,211,153,0.12) !important;
    border: 1px solid rgba(52,211,153,0.35) !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--accent) !important;
    color: #ffffff !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
}
.streamlit-expanderContent {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    color: var(--text-primary) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)



def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1d27 0%, #20243a 100%);
        border: 1px solid #2a2f45;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: 0; right: 0;
            width: 300px; height: 100%;
            background: radial-gradient(circle at 80% 50%, rgba(79,142,247,0.08), transparent 70%);
        "></div>
        <h1 style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem; font-weight: 700;
            color: #f0f2f8 !important;
            margin: 0 0 6px 0;
            letter-spacing: -0.01em;
        ">{title}</h1>
        <p style="color: #a8b0cc !important; margin: 0; font-size: 1rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def info_card(title: str, body: str, accent: str = "#4f8ef7"):
    st.markdown(f"""
    <div style="
        background: #1a1d27;
        border: 1px solid #2a2f45;
        border-left: 4px solid {accent};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    ">
        <p style="color: {accent} !important; font-weight: 700;
                  text-transform: uppercase; letter-spacing: 0.07em;
                  font-size: 0.78rem; margin: 0 0 8px 0;">{title}</p>
        <p style="color: #d0d5ea !important; margin: 0; font-size: 0.95rem;
                  line-height: 1.6;">{body}</p>
    </div>
    """, unsafe_allow_html=True)


def section_title(text: str, icon: str = ""):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px;
                margin: 28px 0 16px 0; padding-bottom: 10px;
                border-bottom: 1px solid #2a2f45;">
        <span style="font-size: 1.25rem;">{icon}</span>
        <h2 style="font-family:'Space Grotesk',sans-serif;
                   font-size: 1.2rem; font-weight: 700;
                   color: #f0f2f8 !important; margin: 0;
                   letter-spacing: -0.01em;">{text}</h2>
    </div>
    """, unsafe_allow_html=True)


def cluster_badge(cluster_id: int, color: str, name: str, desc: str):
    return f"""
    <div style="
        background: #1a1d27; border: 1px solid #2a2f45;
        border-top: 3px solid {color}; border-radius: 12px;
        padding: 18px 20px; margin-bottom: 12px;
    ">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="width:14px; height:14px; border-radius:50%;
                        background:{color}; flex-shrink:0;"></div>
            <span style="color:#f0f2f8 !important; font-weight:700;
                         font-size:1rem;">{name}</span>
        </div>
        <p style="color:#a8b0cc !important; margin:0; font-size:0.88rem;
                  line-height:1.55;">{desc}</p>
    </div>
    """



@st.cache_data
def load_data(uploaded=None):
    """Load mall customer data from upload or bundled CSV."""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif os.path.exists("mall_customers.csv"):
        df = pd.read_csv("mall_customers.csv")
    else:
        st.error("⚠️  No dataset found. Please upload mall_customers.csv")
        st.stop()
    return df

def preprocess(df: pd.DataFrame, features: list):
    """
    Scale selected features with StandardScaler.
    Returns the scaled array and the fitted scaler.
    """
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def compute_inertia(X_scaled, k_range):
    """Run K-Means for each k and collect inertia values."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


def plot_elbow(k_range, inertias, optimal_k):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    ax.plot(k_range, inertias, "o-", color="#4f8ef7", linewidth=2.5,
            markersize=8, markerfacecolor="#f0f2f8", markeredgecolor="#4f8ef7",
            markeredgewidth=1.5)

    # Highlight optimal k
    ax.axvline(x=optimal_k, color="#34d399", linestyle="--", linewidth=1.5,
               label=f"Optimal k = {optimal_k}")
    ax.scatter([optimal_k], [inertias[optimal_k - k_range[0]]],
               color="#34d399", s=150, zorder=5)

    ax.set_xlabel("Number of Clusters (k)", color="#a8b0cc", fontsize=11)
    ax.set_ylabel("Inertia (WCSS)", color="#a8b0cc", fontsize=11)
    ax.set_title("Elbow Method — Optimal K Selection",
                 color="#f0f2f8", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#a8b0cc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f45")
    ax.legend(facecolor="#20243a", edgecolor="#2a2f45",
              labelcolor="#f0f2f8", fontsize=10)
    ax.grid(color="#2a2f45", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig



def train_kmeans(X_scaled, n_clusters):
    """Fit KMeans and return model + labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels



PALETTE = ["#4f8ef7", "#a78bfa", "#34d399", "#fbbf24",
           "#f87171", "#38bdf8", "#fb923c", "#e879f9"]



def _style_axes(ax, title=""):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#a8b0cc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f45")
    ax.grid(color="#2a2f45", linestyle="--", linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color="#f0f2f8", fontsize=12,
                     fontweight="bold", pad=12)
    ax.xaxis.label.set_color("#a8b0cc")
    ax.yaxis.label.set_color("#a8b0cc")


def plot_scatter_income_spending(df, labels, n_clusters):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1d27")
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(df.loc[mask, "Annual Income (k$)"],
                   df.loc[mask, "Spending Score (1-100)"],
                   color=PALETTE[c % len(PALETTE)], alpha=0.78,
                   s=60, edgecolors="#0f1117", linewidths=0.5,
                   label=f"Cluster {c+1}")
    _style_axes(ax, "Annual Income vs Spending Score by Cluster")
    ax.set_xlabel("Annual Income (k$)", fontsize=11)
    ax.set_ylabel("Spending Score (1–100)", fontsize=11)
    ax.legend(facecolor="#20243a", edgecolor="#2a2f45",
              labelcolor="#f0f2f8", fontsize=9, markerscale=1.2)
    fig.tight_layout()
    return fig


def plot_scatter_age_spending(df, labels, n_clusters):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1d27")
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(df.loc[mask, "Age"],
                   df.loc[mask, "Spending Score (1-100)"],
                   color=PALETTE[c % len(PALETTE)], alpha=0.78,
                   s=60, edgecolors="#0f1117", linewidths=0.5,
                   label=f"Cluster {c+1}")
    _style_axes(ax, "Age vs Spending Score by Cluster")
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("Spending Score (1–100)", fontsize=11)
    ax.legend(facecolor="#20243a", edgecolor="#2a2f45",
              labelcolor="#f0f2f8", fontsize=9, markerscale=1.2)
    fig.tight_layout()
    return fig


def plot_cluster_distribution(labels, n_clusters):
    counts = [np.sum(labels == c) for c in range(n_clusters)]
    cluster_names = [f"Cluster {c+1}" for c in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1d27")
    bars = ax.bar(cluster_names, counts,
                  color=[PALETTE[c % len(PALETTE)] for c in range(n_clusters)],
                  edgecolor="#0f1117", linewidth=0.7, width=0.55)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(count), ha="center", va="bottom",
                color="#f0f2f8", fontsize=10, fontweight="600")
    _style_axes(ax, "Customer Count per Cluster")
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=11)
    fig.tight_layout()
    return fig


def plot_3d_scatter(df, labels, n_clusters):
    """3-D scatter: Age × Income × Spending Score."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1d27")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1d27")

    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(df.loc[mask, "Age"],
                   df.loc[mask, "Annual Income (k$)"],
                   df.loc[mask, "Spending Score (1-100)"],
                   color=PALETTE[c % len(PALETTE)], alpha=0.75,
                   s=50, label=f"Cluster {c+1}")

    ax.set_xlabel("Age", color="#a8b0cc", fontsize=9)
    ax.set_ylabel("Annual Income (k$)", color="#a8b0cc", fontsize=9)
    ax.set_zlabel("Spending Score", color="#a8b0cc", fontsize=9)
    ax.tick_params(colors="#a8b0cc", labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#2a2f45")
    ax.yaxis.pane.set_edgecolor("#2a2f45")
    ax.zaxis.pane.set_edgecolor("#2a2f45")
    ax.set_title("3-D Cluster View", color="#f0f2f8",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#20243a", edgecolor="#2a2f45",
              labelcolor="#f0f2f8", fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_heatmap(df, features):
    """Correlation heatmap for numeric features."""
    corr = df[features + ["Spending Score (1-100)"]].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, linecolor="#2a2f45",
                ax=ax, annot_kws={"size": 10, "color": "#f0f2f8"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap",
                 color="#f0f2f8", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#a8b0cc", labelsize=9)
    plt.xticks(color="#a8b0cc")
    plt.yticks(color="#a8b0cc", rotation=0)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# CLUSTER INSIGHT LABELS
# (Heuristic based on cluster centroids)
# ─────────────────────────────────────────────
SEGMENT_PRESETS = {
    5: [
        ("🛒 Budget Shoppers", "Young, low income, high spending — impulsive buyers driven by trends.",    "#4f8ef7"),
        ("💎 Premium Loyalists", "Middle-aged, high income, high spending — best target for loyalty programs.", "#a78bfa"),
        ("💼 Cautious Wealthy",  "Older, high income, low spending — value quality over quantity.",           "#34d399"),
        ("🌱 Frugal Starters",   "Young, low income, low spending — price-sensitive, growth segment.",        "#fbbf24"),
        ("⚖️  Balanced Spenders", "Mid-career, average income & spending — steady, predictable buyers.",       "#f87171"),
    ]
}


def get_segment_labels(n_clusters, df_clustered):
    """
    Auto-label clusters by ranking on income & spending score centroids.
    Falls back to generic labels if n != 5.
    """
    centroids = (df_clustered
                 .groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)", "Age"]]
                 .mean())
    labels_map = {}
    icons = ["🔵","🟣","🟢","🟡","🔴","🩵","🟠","🟤"]
    for c in centroids.index:
        inc  = centroids.loc[c, "Annual Income (k$)"]
        sco  = centroids.loc[c, "Spending Score (1-100)"]
        age  = centroids.loc[c, "Age"]
        inc_tag  = "High Income"  if inc  > 65 else ("Medium Income"  if inc  > 40 else "Low Income")
        sco_tag  = "High Spender" if sco  > 65 else ("Medium Spender" if sco  > 40 else "Low Spender")
        age_tag  = "Older"        if age  > 45 else ("Middle-Aged"    if age  > 32 else "Young")
        name = f"{icons[c % len(icons)]}  {age_tag} · {inc_tag} · {sco_tag}"
        desc = (f"Avg Age: {age:.0f}  |  Avg Income: ${inc:.0f}k  "
                f"|  Avg Spending Score: {sco:.0f}")
        labels_map[c] = (name, desc, PALETTE[c % len(PALETTE)])
    return labels_map


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:2.8rem;">🛍️</div>
            <h1 style="font-family:'Space Grotesk',sans-serif;
                       font-size:1.1rem; font-weight:700;
                       color:#f0f2f8 !important; margin:8px 0 2px 0;">
                Customer Segmentation
            </h1>
            <p style="color:#6b7399 !important; font-size:0.78rem;
                      margin:0; letter-spacing:0.08em;">
                ML DASHBOARD
            </p>
        </div>
        <hr style="border-color:#2a2f45; margin:12px 0 20px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["📊 Overview", "🔍 Dataset", "⚙️  Preprocessing",
             "📈 Elbow Method", "🤖 Train Model",
             "🎯 Cluster Analysis", "💡 Insights", "📚 About"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:#2a2f45; margin:20px 0;'>",
                    unsafe_allow_html=True)

        st.markdown("""
        <p style="color:#6b7399 !important; font-size:0.75rem;
                  text-transform:uppercase; letter-spacing:0.08em;
                  margin-bottom:10px;">Settings</p>
        """, unsafe_allow_html=True)

        n_clusters = st.slider("Number of Clusters (k)", 2, 8, 5)
        features = st.multiselect(
            "Clustering Features",
            ["Annual Income (k$)", "Spending Score (1-100)", "Age"],
            default=["Annual Income (k$)", "Spending Score (1-100)", "Age"]
        )

        st.markdown("<hr style='border-color:#2a2f45; margin:20px 0;'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#6b7399 !important; font-size:0.75rem;
                  text-transform:uppercase; letter-spacing:0.08em;
                  margin-bottom:8px;">Upload Dataset</p>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv"],
                                    label_visibility="collapsed")

        st.markdown("""
        <div style="margin-top:30px; padding:14px 16px;
                    background:#0f1117; border-radius:10px;
                    border:1px solid #2a2f45; text-align:center;">
            <p style="color:#6b7399 !important; font-size:0.75rem;
                      margin:0; line-height:1.7;">
                scikit-learn · Pandas · Seaborn<br>Matplotlib · Streamlit
            </p>
        </div>
        """, unsafe_allow_html=True)

    return page, n_clusters, features, uploaded


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

def page_overview(df):
    page_header(
        "Customer Segmentation Dashboard",
        "Unsupervised Machine Learning with K-Means Clustering"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Features",        f"{df.shape[1]}")
    c3.metric("Avg Income",      f"${df['Annual Income (k$)'].mean():.0f}k")
    c4.metric("Avg Spend Score", f"{df['Spending Score (1-100)'].mean():.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        section_title("What is Customer Segmentation?", "🎯")
        info_card("Business Context",
                  "Customer segmentation divides a customer base into groups "
                  "that share common characteristics — enabling businesses to "
                  "tailor marketing, pricing, and product strategies to each segment.",
                  "#4f8ef7")
        info_card("K-Means Clustering",
                  "K-Means is an unsupervised ML algorithm that partitions customers "
                  "into K distinct clusters by minimising the within-cluster sum of "
                  "squared distances (inertia). Each cluster centroid represents "
                  "the 'average customer' in that group.",
                  "#a78bfa")
        info_card("The Elbow Method",
                  "Run K-Means for k = 1 … N and plot inertia. The 'elbow' — where "
                  "adding more clusters yields diminishing returns — is the optimal k.",
                  "#34d399")

    with col2:
        section_title("Gender Distribution", "👥")
        gender_counts = df["Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1a1d27")
        wedges, texts, autotexts = ax.pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            colors=["#4f8ef7", "#a78bfa"],
            startangle=90,
            wedgeprops={"edgecolor": "#1a1d27", "linewidth": 2}
        )
        for t in texts:
            t.set_color("#f0f2f8")
            t.set_fontsize(11)
        for at in autotexts:
            at.set_color("#0f1117")
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax.set_facecolor("#1a1d27")
        fig.tight_layout()
        st.pyplot(fig)

        section_title("Age Distribution", "📊")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig2.patch.set_facecolor("#1a1d27")
        ax2.hist(df["Age"], bins=18, color="#4f8ef7",
                 edgecolor="#0f1117", linewidth=0.5, alpha=0.9)
        _style_axes(ax2)
        ax2.set_xlabel("Age", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        fig2.tight_layout()
        st.pyplot(fig2)


def page_dataset(df):
    page_header("Dataset Preview", "Raw mall customer data — 200 records")

    tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📊 Statistics", "🔍 Missing Values"])

    with tab1:
        st.dataframe(df, use_container_width=True, height=420)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        st.download_button("⬇  Download CSV", buf.getvalue(),
                           "mall_customers.csv", "text/csv")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Feature", "Missing Values"]
        missing["Status"] = missing["Missing Values"].apply(
            lambda x: "✅ No Missing" if x == 0 else f"⚠️  {x} Missing")
        st.dataframe(missing, use_container_width=True)


def page_preprocessing(df, features):
    page_header("Data Preprocessing", "Feature scaling with StandardScaler")

    if not features:
        st.warning("Please select at least one feature in the sidebar.")
        return

    section_title("Why Feature Scaling?", "⚖️")
    info_card("StandardScaler",
              "K-Means uses Euclidean distance, so features on different scales "
              "(e.g. Income in $k vs Score 1–100) would bias the algorithm. "
              "StandardScaler transforms each feature to mean=0, std=1.",
              "#4f8ef7")

    X, scaler = preprocess(df, features)
    scaled_df = pd.DataFrame(X, columns=features)

    col1, col2 = st.columns(2)
    with col1:
        section_title("Before Scaling", "📋")
        st.dataframe(df[features].head(10).round(2), use_container_width=True)
    with col2:
        section_title("After Scaling", "✅")
        st.dataframe(scaled_df.head(10).round(4), use_container_width=True)

    section_title("Correlation Heatmap", "🌡️")
    num_features = [f for f in features if f in df.select_dtypes(include=np.number).columns]
    if len(num_features) >= 2:
        st.pyplot(plot_heatmap(df, num_features))


def page_elbow(df, features, n_clusters):
    page_header("Elbow Method", "Determine the optimal number of clusters")

    if not features:
        st.warning("Please select features in the sidebar.")
        return

    info_card("How to Read the Elbow Chart",
              "Each point represents the total inertia (WCSS) for that k. "
              "The 'elbow' is where the curve bends sharply — further clusters "
              "add little improvement. The green dashed line marks your chosen k.",
              "#34d399")

    X, _ = preprocess(df, features)
    k_range = range(2, 11)
    inertias = compute_inertia(X, k_range)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(plot_elbow(list(k_range), inertias, n_clusters))
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1a1d27; border:1px solid #2a2f45;
                    border-radius:12px; padding:20px;">
            <p style="color:#a78bfa !important; font-weight:700;
                      text-transform:uppercase; letter-spacing:0.07em;
                      font-size:0.78rem; margin:0 0 12px 0;">K vs Inertia</p>
        """, unsafe_allow_html=True)
        for k, inr in zip(k_range, inertias):
            marker = "◀ optimal" if k == n_clusters else ""
            color  = "#34d399"  if k == n_clusters else "#6b7399"
            st.markdown(
                f"<p style='color:{color} !important; margin:4px 0; font-size:0.88rem;'>"
                f"k={k}  →  {inr:,.0f}  <em style='color:{color}'>{marker}</em></p>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_train(df, features, n_clusters):
    page_header("Train K-Means Model", f"Clustering customers into {n_clusters} segments")

    if not features:
        st.warning("Please select features in the sidebar.")
        return

    if st.button(f"🚀  Train K-Means  (k={n_clusters})"):
        with st.spinner("Training model…"):
            X, scaler = preprocess(df, features)
            model, labels = train_kmeans(X, n_clusters)

            # Store in session state
            df_clustered            = df.copy()
            df_clustered["Cluster"] = labels + 1  # 1-indexed
            st.session_state["model"]        = model
            st.session_state["scaler"]       = scaler
            st.session_state["labels"]       = labels
            st.session_state["df_clustered"] = df_clustered
            st.session_state["n_clusters"]   = n_clusters
            st.session_state["features"]     = features

        st.success(f"✅  Model trained successfully! {n_clusters} clusters found.")

        # Metrics
        sil = silhouette_score(X, labels)
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters",           str(n_clusters))
        c2.metric("Silhouette Score",   f"{sil:.4f}")
        c3.metric("Inertia (WCSS)",     f"{model.inertia_:,.0f}")

        info_card("Silhouette Score",
                  f"Score = {sil:.4f} — ranges from -1 (poor) to +1 (perfect). "
                  "Values > 0.4 indicate reasonable cluster separation.",
                  "#a78bfa" if sil >= 0.4 else "#fbbf24")

        # Save model
        model_bytes = pickle.dumps({"model": model, "scaler": scaler,
                                    "features": features})
        st.download_button("⬇  Download Trained Model (.pkl)",
                           model_bytes, "kmeans_model.pkl",
                           "application/octet-stream")

    elif "model" in st.session_state:
        st.info("ℹ️  A trained model is already loaded. Adjust settings and retrain if needed.")


def page_cluster_analysis(df):
    page_header("Cluster Analysis", "Visualise and explore customer segments")

    if "df_clustered" not in st.session_state:
        st.info("👈  Train the model first (sidebar → Train Model).")
        return

    df_c      = st.session_state["df_clustered"]
    labels    = st.session_state["labels"]
    n_clusters = st.session_state["n_clusters"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Income vs Spending",
        "🎂 Age vs Spending",
        "📊 Distribution",
        "🌐 3-D View"
    ])

    with tab1:
        st.pyplot(plot_scatter_income_spending(df_c, labels, n_clusters))
    with tab2:
        st.pyplot(plot_scatter_age_spending(df_c, labels, n_clusters))
    with tab3:
        st.pyplot(plot_cluster_distribution(labels, n_clusters))
    with tab4:
        st.pyplot(plot_3d_scatter(df_c, labels, n_clusters))

    section_title("Clustered Dataset", "📋")
    st.dataframe(df_c, use_container_width=True, height=340)

    buf = io.BytesIO()
    df_c.to_csv(buf, index=False)
    st.download_button("⬇  Download Clustered Dataset",
                       buf.getvalue(), "customers_clustered.csv", "text/csv")


def page_insights(df):
    page_header("Cluster Insights", "Business intelligence per customer segment")

    if "df_clustered" not in st.session_state:
        st.info("👈  Train the model first.")
        return

    df_c       = st.session_state["df_clustered"]
    n_clusters = st.session_state["n_clusters"]
    seg_map    = get_segment_labels(n_clusters, df_c)

    section_title("Segment Profiles", "🎯")
    badges_html = ""
    for c in range(n_clusters):
        name, desc, color = seg_map.get(c, (f"Cluster {c+1}", "", PALETTE[c]))
        badges_html += cluster_badge(c + 1, color, f"Cluster {c+1} — {name}", desc)
    st.markdown(badges_html, unsafe_allow_html=True)

    section_title("Cluster-Wise Statistics", "📊")
    stats = (df_c.groupby("Cluster")[
                 ["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
             .agg(["mean", "std", "count"])
             .round(1))
    st.dataframe(stats, use_container_width=True)

    section_title("Business Recommendations", "💡")
    recs = [
        ("📣 Targeted Promotions",
         "Identify high-income, high-spending clusters for premium product launches.",
         "#4f8ef7"),
        ("💰 Loyalty Programmes",
         "Reward mid-income, consistent spenders to increase lifetime value.",
         "#a78bfa"),
        ("📧 Re-engagement Campaigns",
         "Target low-spending clusters with discount-driven email campaigns.",
         "#34d399"),
        ("📦 Product Bundles",
         "Offer bundles to young, low-income segments who respond to value deals.",
         "#fbbf24"),
    ]
    col1, col2 = st.columns(2)
    for i, (title, body, acc) in enumerate(recs):
        (col1 if i % 2 == 0 else col2).markdown(
            f"""<div style="background:#1a1d27; border:1px solid #2a2f45;
                border-left:4px solid {acc}; border-radius:12px;
                padding:18px 20px; margin-bottom:14px;">
                <p style="color:{acc} !important; font-weight:700;
                           font-size:0.95rem; margin:0 0 6px 0;">{title}</p>
                <p style="color:#d0d5ea !important; margin:0;
                           font-size:0.88rem; line-height:1.55;">{body}</p>
            </div>""", unsafe_allow_html=True)


def page_about():
    page_header("About This Project", "Customer Segmentation using K-Means Clustering")

    st.markdown("""
    <div style="background:#1a1d27; border:1px solid #2a2f45;
                border-radius:14px; padding:28px 32px; margin-bottom:20px;">
        <h2 style="font-family:'Space Grotesk',sans-serif;
                   color:#4f8ef7 !important; font-size:1.15rem;
                   margin:0 0 14px 0;">🛠  Technologies Used</h2>
        <div style="display:grid; grid-template-columns: repeat(3,1fr); gap:12px;">
    """ + "".join([
        f"""<div style="background:#20243a; border-radius:8px;
                        padding:12px 16px; border:1px solid #2a2f45;">
                <p style="color:#a78bfa !important; font-weight:700;
                           font-size:0.8rem; margin:0 0 4px 0;
                           text-transform:uppercase; letter-spacing:0.05em;">{t}</p>
                <p style="color:#d0d5ea !important; margin:0;
                           font-size:0.85rem;">{d}</p>
            </div>"""
        for t, d in [
            ("Python 3.10+",     "Core language"),
            ("Streamlit",        "Web application framework"),
            ("scikit-learn",     "K-Means & StandardScaler"),
            ("Pandas",           "Data manipulation"),
            ("NumPy",            "Numerical computing"),
            ("Matplotlib",       "Static visualisations"),
            ("Seaborn",          "Statistical charts"),
            ("Pickle",           "Model serialisation"),
        ]
    ]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        info_card("K-Means Algorithm",
                  "1. Initialise k centroids randomly.<br>"
                  "2. Assign each point to the nearest centroid.<br>"
                  "3. Recompute centroids as cluster means.<br>"
                  "4. Repeat 2–3 until convergence.",
                  "#4f8ef7")
    with col2:
        info_card("Elbow Method",
                  "Plot WCSS for k = 1…N. The 'elbow' is where "
                  "the reduction in WCSS starts to slow — that k "
                  "balances model complexity and fit quality.",
                  "#a78bfa")

    info_card("Why Segment Customers?",
              "Segmentation lets businesses allocate marketing budgets efficiently, "
              "personalise offers, predict churn, and identify the most valuable "
              "customer groups — directly improving ROI and customer satisfaction.",
              "#34d399")

    st.markdown("""
    <div style="background:#1a1d27; border:1px solid #2a2f45;
                border-radius:14px; padding:24px 28px; margin-top:8px;">
        <p style="color:#6b7399 !important; font-size:0.82rem;
                  text-align:center; margin:0; line-height:1.8;">
            Built for internship applications · GitHub portfolio · LinkedIn showcase<br>
            <strong style="color:#4f8ef7 !important;">
                streamlit run app.py
            </strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    page, n_clusters, features, uploaded = build_sidebar()
    df = load_data(uploaded)

    # Route pages
    if   page == "📊 Overview":         page_overview(df)
    elif page == "🔍 Dataset":           page_dataset(df)
    elif page == "⚙️  Preprocessing":    page_preprocessing(df, features)
    elif page == "📈 Elbow Method":      page_elbow(df, features, n_clusters)
    elif page == "🤖 Train Model":       page_train(df, features, n_clusters)
    elif page == "🎯 Cluster Analysis":  page_cluster_analysis(df)
    elif page == "💡 Insights":          page_insights(df)
    elif page == "📚 About":             page_about()


if __name__ == "__main__":
    main()
