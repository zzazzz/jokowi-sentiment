from __future__ import annotations

import re
import math
import warnings
from io import BytesIO
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Persepsi Publik · Jokowi 2019–2024",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME / CSS (diperbarui dengan card-style untuk konten tab)
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:wght@500;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background: radial-gradient(circle at top, #111827 0%, #0b1020 40%, #070b14 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #090f1c 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Tabs styling */
div[data-testid="stTabs"] {
    width: 100%;
}

.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: stretch;
    gap: 0;
    background: rgba(15,23,42,0.52);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px 20px 0 0;
    padding: 6px 8px 0 8px;
    backdrop-filter: blur(4px);
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    flex: 1 1 0;
    min-width: 0;
    color: #cbd5e1;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 10px 12px;
    border-radius: 40px;
    transition: all 0.2s ease;

    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"] > div {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.stTabs [data-baseweb="tab"] p {
    margin: 0;
    text-align: center;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    background: linear-gradient(135deg, rgba(96,165,250,0.2), rgba(34,197,94,0.1)) !important;
    border-bottom: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05);
    color: #e2e8f0;
}

/* Card shell untuk setiap tab content */
.tab-content-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.7));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 20px 24px;
    margin-top: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.hero-wrap {
    padding: 10px 0 22px 0;
}

.hero-eyebrow {
    color: #93c5fd;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 700;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 4vw, 3.9rem);
    line-height: 1.02;
    font-weight: 900;
    color: #f8fafc;
    margin: 0;
}

.hero-subtitle {
    margin-top: 10px;
    color: #cbd5e1;
    font-size: 1rem;
    max-width: 980px;
    line-height: 1.7;
}

.accent-line {
    margin-top: 18px;
    width: 120px;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, #60a5fa, #22c55e, #f59e0b);
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
}

.section-note {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 4px;
    line-height: 1.55;
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.09);
    margin: 10px 0 18px 0;
}

.kpi-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.76));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 18px 18px 16px 18px;
    min-height: 126px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 26px rgba(0,0,0,0.16);
}

.kpi-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(59,130,246,0.14), transparent 46%);
    pointer-events: none;
}

.kpi-label {
    color: #94a3b8;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    font-weight: 700;
    margin-bottom: 10px;
}

.kpi-value {
    font-family: 'Playfair Display', serif;
    color: #f8fafc;
    font-size: 2rem;
    line-height: 1;
    font-weight: 900;
    margin-bottom: 8px;
}

.kpi-foot {
    color: #cbd5e1;
    font-size: 0.82rem;
}

.badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.badge-pos { background: rgba(34,197,94,0.12); color: #86efac; }
.badge-neu { background: rgba(148,163,184,0.12); color: #cbd5e1; }
.badge-neg { background: rgba(248,113,113,0.12); color: #fca5a5; }
.badge-info { background: rgba(96,165,250,0.12); color: #93c5fd; }
.badge-warn { background: rgba(245,158,11,0.12); color: #fcd34d; }

.info-box {
    background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.68));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.info-box h4 {
    margin: 0 0 6px 0;
    color: #f8fafc;
    font-size: 0.98rem;
}

.info-box p {
    margin: 0;
    color: #cbd5e1;
    line-height: 1.6;
    font-size: 0.9rem;
}

.news-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.75));
    border: 1px solid rgba(255,255,255,0.08);
    border-left-width: 4px;
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.14);
}

.news-pos { border-left-color: rgba(34,197,94,0.85); }
.news-neu { border-left-color: rgba(148,163,184,0.85); }
.news-neg { border-left-color: rgba(248,113,113,0.85); }

.news-title {
    color: #f8fafc;
    font-weight: 700;
    font-size: 0.96rem;
    line-height: 1.55;
    margin-bottom: 8px;
}

.news-meta {
    color: #94a3b8;
    font-size: 0.79rem;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.84rem;
    line-height: 1.55;
}

.media-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.8));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 12px;
    text-align: center;
    transition: transform 0.2s;
}
.media-card:hover {
    transform: translateY(-4px);
    border-color: rgba(96,165,250,0.4);
}
.media-name {
    font-weight: 700;
    color: #f8fafc;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.media-count {
    font-size: 1.2rem;
    font-weight: 800;
    color: #60a5fa;
}

.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
}

[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# =============================================================================
COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#f87171",
    "info": "#60a5fa",
    "warn": "#f59e0b",
    "bg": "#0b1020",
    "panel": "#111827",
    "panel2": "#0f172a",
    "grid": "rgba(255,255,255,0.08)",
}

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_LABEL = {
    "positive": "Positif",
    "neutral": "Netral",
    "negative": "Negatif",
}
COLOR_MAP = {
    "positive": COLORS["positive"],
    "neutral": COLORS["neutral"],
    "negative": COLORS["negative"],
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#dbe4f0", size=12),
    margin=dict(t=56, l=12, r=12, b=18),
    xaxis=dict(
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        tickfont=dict(color="#cbd5e1"),
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        tickfont=dict(color="#cbd5e1"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
    ),
)

# =============================================================================
# HELPERS
# =============================================================================
def rgba_from_hex(hex_color: str, alpha: float = 0.18) -> str:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        return f"rgba(255,255,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def human_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def safe_lower(x):
    if pd.isna(x):
        return ""
    return str(x).lower()

def pick_data_file():
    candidates = [
        Path("publik persepsi.csv"),
        Path("publik_persepsi.csv"),
        Path("/mnt/data/publik persepsi.csv"),
        Path("/mnt/data/publik_persepsi.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("File CSV tidak ditemukan. Pastikan nama file 'publik persepsi.csv' tersedia.")

def norm_sentiment(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    s = (
        s.replace("positif", "positive")
         .replace("negatif", "negative")
         .replace("netral", "neutral")
    )
    return s if s in SENTIMENT_ORDER else np.nan

def clean_text(text):
    text = safe_lower(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    tokens = [t for t in tokens if not t.isdigit()]
    return " ".join(tokens)

def tokenize_text(text):
    text = clean_text(text)
    return [t for t in text.split() if t]

def hhi_from_shares(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    shares = (counts / total) * 100
    return float((shares ** 2).sum())

def top_ngrams_from_texts(texts, n=2, top_n=20):
    token_stream = []
    for t in texts:
        token_stream.extend(tokenize_text(t))
    if len(token_stream) < n:
        return []
    grams = [" ".join(token_stream[i:i+n]) for i in range(len(token_stream) - n + 1)]
    return Counter(grams).most_common(top_n)

def ngram_df_from_series(series, n=2, top_n=20, sentiment=None):
    rows = []
    for phrase, freq in top_ngrams_from_texts(series, n=n, top_n=top_n):
        rows.append({"phrase": phrase, "freq": int(freq), "ngram": n, "sentiment": sentiment})
    return pd.DataFrame(rows)

def make_plotly_layout(height=350, xaxis_updates=None, yaxis_updates=None, legend_updates=None):
    layout = {k: (v.copy() if isinstance(v, dict) else v) for k, v in PLOT_LAYOUT.items()}
    if xaxis_updates:
        layout["xaxis"] = {**layout["xaxis"], **xaxis_updates}
    if yaxis_updates:
        layout["yaxis"] = {**layout["yaxis"], **yaxis_updates}
    if legend_updates:
        layout["legend"] = {**layout["legend"], **legend_updates}
    layout["height"] = height
    return layout

def render_wordcloud(text, bg_color, cmap, max_words=120):
    text = clean_text(text)
    if not text.strip():
        return None
    try:
        wc = WordCloud(
            width=1000,
            height=560,
            background_color=bg_color,
            max_words=max_words,
            prefer_horizontal=0.86,
            collocations=False,
            colormap=cmap,
            min_font_size=8,
        ).generate(text)
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=bg_color)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None

def style_sentiment_badge(s):
    cls = {"positive": "badge-pos", "neutral": "badge-neu", "negative": "badge-neg"}.get(s, "badge-neu")
    return f'<span class="badge {cls}">{SENTIMENT_LABEL.get(s, s)}</span>'

def download_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def plot_blank_message(msg: str):
    st.info(msg)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(path_str):
    path = Path(path_str)
    df = pd.read_csv(path)

    df["Waktu Terbit"] = pd.to_datetime(df["Waktu Terbit"], errors="coerce")
    df["sentiment"] = df["sentiment"].apply(norm_sentiment)
    df = df.dropna(subset=["Waktu Terbit", "sentiment"]).copy()

    for col in ["Judul Berita", "Nama Media", "Link Berita", "Clean Text", "Detected Language", "stopword"]:
        if col not in df.columns:
            df[col] = ""
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "text_word_count" not in df.columns:
        df["text_word_count"] = np.nan
    if "word_count" not in df.columns:
        df["word_count"] = np.nan

    df["Judul Berita"] = df["Judul Berita"].fillna("(tanpa judul)").astype(str)
    df["Nama Media"] = df["Nama Media"].fillna("Unknown").astype(str)
    df["Link Berita"] = df["Link Berita"].fillna("").astype(str)
    df["Clean Text"] = df["Clean Text"].fillna("").astype(str)
    df["Detected Language"] = df["Detected Language"].fillna("").astype(str).str.lower()
    df["stopword"] = df["stopword"].fillna("").astype(str)

    for col in ["confidence", "text_word_count", "word_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = df["Waktu Terbit"].dt.year
    df["month"] = df["Waktu Terbit"].dt.month
    df["year_month"] = df["Waktu Terbit"].dt.strftime("%Y-%m")
    df["year_month_dt"] = pd.to_datetime(df["year_month"] + "-01")
    df["quarter"] = df["Waktu Terbit"].dt.to_period("Q").astype(str)
    df["day_name"] = df["Waktu Terbit"].dt.day_name()
    df["week"] = df["Waktu Terbit"].dt.isocalendar().week.astype(int)
    df["sent_score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1}).fillna(0)

    if df["stopword"].str.len().sum() > 0:
        df["processed_text"] = df["stopword"].apply(clean_text)
    else:
        df["processed_text"] = df["Clean Text"].apply(clean_text)

    df["title_word_count"] = df["Judul Berita"].str.split().apply(len)
    df["title_len"] = df["Judul Berita"].str.len()
    return df

@st.cache_data(show_spinner=False)
def compute_overview(df):
    total = len(df)
    counts = df["sentiment"].value_counts().to_dict()
    pos = int(counts.get("positive", 0))
    neu = int(counts.get("neutral", 0))
    neg = int(counts.get("negative", 0))
    media_n = int(df["Nama Media"].nunique())
    lang_n = int(df["Detected Language"].nunique())

    pos_share = pos / total * 100 if total else 0
    neg_share = neg / total * 100 if total else 0
    neu_share = neu / total * 100 if total else 0
    sentiment_balance = (pos - neg) / total * 100 if total else 0
    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns and total else np.nan
    avg_words = float(df["text_word_count"].mean()) if "text_word_count" in df.columns and total else np.nan
    avg_title_words = float(df["title_word_count"].mean()) if total else np.nan
    dominant = max([(pos, "positive"), (neu, "neutral"), (neg, "negative")], key=lambda x: x[0])[1] if total else None
    return {
        "total": total,
        "pos": pos,
        "neu": neu,
        "neg": neg,
        "media_n": media_n,
        "lang_n": lang_n,
        "pos_share": pos_share,
        "neg_share": neg_share,
        "neu_share": neu_share,
        "sentiment_balance": sentiment_balance,
        "dominant": dominant,
        "avg_conf": avg_conf,
        "avg_words": avg_words,
        "avg_title_words": avg_title_words,
    }

@st.cache_data(show_spinner=False)
def compute_monthly(df):
    monthly = (
        df.groupby(["year_month", "sentiment"])
        .size().reset_index(name="count")
        .pivot(index="year_month", columns="sentiment", values="count")
        .fillna(0).reset_index()
    )
    for s in SENTIMENT_ORDER:
        if s not in monthly.columns:
            monthly[s] = 0
    monthly["year_month_dt"] = pd.to_datetime(monthly["year_month"] + "-01")
    monthly = monthly.sort_values("year_month_dt").reset_index(drop=True)
    monthly["total"] = monthly[SENTIMENT_ORDER].sum(axis=1)
    monthly["pos_share"] = np.where(monthly["total"] > 0, monthly["positive"] / monthly["total"] * 100, 0)
    monthly["neg_share"] = np.where(monthly["total"] > 0, monthly["negative"] / monthly["total"] * 100, 0)
    monthly["net_score"] = np.where(monthly["total"] > 0, (monthly["positive"] - monthly["negative"]) / monthly["total"] * 100, 0)
    monthly["ma3_total"] = monthly["total"].rolling(3, min_periods=1, center=True).mean()
    monthly["ma3_score"] = monthly["net_score"].rolling(3, min_periods=1, center=True).mean()
    return monthly

@st.cache_data(show_spinner=False)
def compute_yearly(df):
    yearly = (
        df.groupby(["year", "sentiment"])
        .size().reset_index(name="count")
        .pivot(index="year", columns="sentiment", values="count")
        .fillna(0).reset_index()
    )
    for s in SENTIMENT_ORDER:
        if s not in yearly.columns:
            yearly[s] = 0
    yearly["total"] = yearly[SENTIMENT_ORDER].sum(axis=1)
    yearly["pos_share"] = np.where(yearly["total"] > 0, yearly["positive"] / yearly["total"] * 100, 0)
    yearly["neg_share"] = np.where(yearly["total"] > 0, yearly["negative"] / yearly["total"] * 100, 0)
    yearly["net_score"] = np.where(yearly["total"] > 0, (yearly["positive"] - yearly["negative"]) / yearly["total"] * 100, 0)
    yearly["change_total"] = yearly["total"].diff()
    yearly["change_pos"] = yearly["positive"].diff()
    yearly["change_neg"] = yearly["negative"].diff()
    return yearly.sort_values("year")

@st.cache_data(show_spinner=False)
def compute_media_summary(df, top_n=10):
    media_counts = df["Nama Media"].value_counts()
    top_media = media_counts.head(top_n).reset_index()
    top_media.columns = ["Nama Media", "count"]
    return top_media

@st.cache_data(show_spinner=False)
def compute_anomalies(df):
    monthly_vol = df.groupby("year_month")["sentiment"].size().reset_index(name="volume")
    monthly_score = df.groupby("year_month")["sent_score"].mean().reset_index(name="sent_score")
    merged = monthly_vol.merge(monthly_score, on="year_month", how="left").sort_values("year_month")
    merged["z_volume"] = (merged["volume"] - merged["volume"].mean()) / (merged["volume"].std(ddof=0) + 1e-9)
    merged["z_score"] = (merged["sent_score"] - merged["sent_score"].mean()) / (merged["sent_score"].std(ddof=0) + 1e-9)
    merged["flag_volume"] = merged["z_volume"].abs() >= 2.0
    merged["flag_score"] = merged["z_score"].abs() >= 2.0
    return merged

@st.cache_data(show_spinner=False)
def compute_ratio_by_time(df, granularity="month"):
    """Return DataFrame with time column and positive/negative ratio."""
    if granularity == "month":
        df["time_period"] = df["year_month"]
        sort_key = "year_month_dt"
        df["sort_key"] = pd.to_datetime(df["year_month"] + "-01")
    elif granularity == "quarter":
        df["time_period"] = df["quarter"]
        # Extract year-quarter for sorting
        df["sort_key"] = pd.to_datetime(df["quarter"].str.replace("Q", "").str.split().str[1] + "-" +
                                        (df["quarter"].str.extract(r"Q(\d)")[0].astype(int)*3).astype(str) + "-01")
    else:  # year
        df["time_period"] = df["year"].astype(str)
        df["sort_key"] = pd.to_datetime(df["year"].astype(str) + "-01-01")

    ratio = df.groupby(["time_period", "sentiment"]).size().unstack(fill_value=0)
    ratio["positive_ratio"] = ratio["positive"] / (ratio["positive"] + ratio["negative"] + 1e-9) * 100
    ratio["negative_ratio"] = ratio["negative"] / (ratio["positive"] + ratio["negative"] + 1e-9) * 100
    ratio = ratio.reset_index()
    # Attach sort_key
    sort_df = df[["time_period", "sort_key"]].drop_duplicates()
    ratio = ratio.merge(sort_df, on="time_period", how="left")
    ratio = ratio.sort_values("sort_key")
    return ratio

# =============================================================================
# LOAD DATA
# =============================================================================
data_path = pick_data_file()
df = load_data(str(data_path))

# =============================================================================
# SIDEBAR (hanya filter global penting)
# =============================================================================
with st.sidebar:
    st.markdown("### 🔎 Filter Global")
    st.markdown("<div class='small-muted'>Filter ini berlaku untuk semua tab. Setiap tab juga memiliki filter tambahan sendiri.</div>", unsafe_allow_html=True)
    st.markdown("---")

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    year_range = st.slider("Rentang tahun", year_min, year_max, (year_min, year_max))

    sentiments_sel = st.multiselect(
        "Sentimen",
        options=SENTIMENT_ORDER,
        default=SENTIMENT_ORDER,
        format_func=lambda x: {"positive": "✅ Positif", "neutral": "⬜ Netral", "negative": "❌ Negatif"}[x],
    )

    language_options = sorted([x for x in df["Detected Language"].dropna().unique().tolist() if x and x != "nan"])
    lang_sel = st.multiselect(
        "Bahasa",
        options=language_options,
        default=[x for x in ["id", "en"] if x in language_options] or language_options[:2],
        format_func=lambda x: "🇮🇩 Indonesia" if x == "id" else ("🇬🇧 Inggris" if x == "en" else str(x)),
    )

    st.markdown("---")
    st.markdown(
        f"""
<div class="small-muted">
<b>Sumber:</b> {data_path.name}<br>
<b>Baris data:</b> {len(df):,}<br>
<b>Media unik:</b> {df["Nama Media"].nunique():,}<br>
<b>Periode:</b> {df["year"].min()}–{df["year"].max()}
</div>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# FILTER DATA GLOBAL
# =============================================================================
dff = df.copy()
dff = dff[dff["year"].between(year_range[0], year_range[1])]
if sentiments_sel:
    dff = dff[dff["sentiment"].isin(sentiments_sel)]
if lang_sel:
    dff = dff[dff["Detected Language"].isin(lang_sel)]

overview = compute_overview(dff)
monthly = compute_monthly(dff) if len(dff) else pd.DataFrame()
yearly = compute_yearly(dff) if len(dff) else pd.DataFrame()
top_media_df = compute_media_summary(dff, top_n=10) if len(dff) else pd.DataFrame()
anomalies = compute_anomalies(dff) if len(dff) else pd.DataFrame()

# =============================================================================
# HERO
# =============================================================================
st.markdown(
    f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">DASHBOARD ANALISIS SENTIMEN · MEDIA MONITORING</div>
    <h1 class="hero-title">Persepsi Publik terhadap Jokowi<br>2019–2024</h1>
    <div class="hero-subtitle">
        Dashboard ini merangkum sentimen pemberitaan, dominasi media, pola waktu, dan konteks frasa sampai tiga kata
        dari <strong style="color:#93c5fd">{len(dff):,} berita</strong> yang lolos filter saat ini.
    </div>
    <div class="accent-line"></div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# KPI ROW
# =============================================================================
k1, k2, k3, k4, k5= st.columns(5)
with k1:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Total Berita</div>
  <div class="kpi-value">{human_int(overview["total"])}</div>
  <div class="kpi-foot">{year_range[0]}–{year_range[1]}</div>
</div>
""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Sentimen Positif</div>
  <div class="kpi-value" style="color:{COLORS['positive']}">{overview["pos_share"]:.1f}%</div>
  <div class="kpi-foot">{human_int(overview["pos"])} berita</div>
</div>
""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Sentimen Negatif</div>
  <div class="kpi-value" style="color:{COLORS['negative']}">{overview["neg_share"]:.1f}%</div>
  <div class="kpi-foot">{human_int(overview["neg"])} berita</div>
</div>
""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Netral</div>
  <div class="kpi-value" style="color:{COLORS['neutral']}">{overview["neu_share"]:.1f}%</div>
  <div class="kpi-foot">{human_int(overview["neu"])} berita</div>
</div>
""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Media Unik</div>
  <div class="kpi-value">{human_int(overview["media_n"])}</div>
  <div class="kpi-foot">{overview["lang_n"]} bahasa terdeteksi</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# =============================================================================
# NAVIGATION TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Tren & Dinamika",
    "🗞️ Analisis Media",
    "☁️ N-gram & Topik",
    "📊 Rasio Sentimen",
    "📰 Jelajahi Berita",
    "💾 Export Data"
])

# =============================================================================
# FUNGSI BANTU UNTUK MEMBUNGKUS KONTEN TAB DENGAN CARD
# =============================================================================
def tab_content(func):
    """Decorator to wrap tab content in a card shell."""
    def wrapper(*args, **kwargs):
        with st.container():
            st.markdown('<div class="tab-content-card">', unsafe_allow_html=True)
            func(*args, **kwargs)
            st.markdown('</div>', unsafe_allow_html=True)
    return wrapper

# =============================================================================
# TAB 1 — TREN & DINAMIKA
# =============================================================================
with tab1:
    with st.container():
        st.markdown('<div class="tab-content-card">', unsafe_allow_html=True)
        st.markdown("### Tren Sentimen Bulanan")
        st.markdown("<div class='section-note'>Dinamika berita positif, netral, dan negatif dari waktu ke waktu.</div>", unsafe_allow_html=True)

        if len(monthly):
            fig_trend = go.Figure()
            for sent in SENTIMENT_ORDER:
                if sent in monthly.columns:
                    c = COLOR_MAP[sent]
                    fig_trend.add_trace(
                        go.Scatter(
                            x=monthly["year_month_dt"],
                            y=monthly[sent],
                            name=SENTIMENT_LABEL[sent],
                            mode="lines",
                            line=dict(color=c, width=2.8),
                            fill="tozeroy",
                            fillcolor=rgba_from_hex(c, 0.12),
                            hovertemplate="<b>%{x|%b %Y}</b><br>%{fullData.name}: %{y:,}<extra></extra>",
                        )
                    )
            fig_trend.update_layout(
                **PLOT_LAYOUT,
                height=360,
                hovermode="x unified",
                xaxis_title=None,
                yaxis_title="Jumlah berita",
            )
            fig_trend.update_xaxes(tickformat="%b\n%Y", tickangle=0)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            plot_blank_message("Tidak ada data untuk ditampilkan pada tren bulanan.")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Distribusi Sentimen")
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            s = dff["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).reset_index()
            s.columns = ["sentiment", "count"]
            fig_donut = go.Figure(go.Pie(
                labels=[SENTIMENT_LABEL[x] for x in s["sentiment"]],
                values=s["count"],
                hole=0.62,
                marker=dict(colors=[COLOR_MAP[x] for x in s["sentiment"]], line=dict(color="#0b1020", width=3)),
                hovertemplate="<b>%{label}</b><br>%{value:,} berita<br>%{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                **make_plotly_layout(height=300),
                annotations=[dict(text=f"<b>{human_int(overview['total'])}</b><br>berita", x=0.5, y=0.5, showarrow=False, font=dict(color="#f8fafc", size=14))]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.markdown("### Volume & Komposisi per Tahun")
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            if len(yearly):
                fig_year = px.bar(
                    yearly,
                    x="year",
                    y="total",
                    color="net_score",
                    color_continuous_scale=[[0, COLORS["negative"]], [0.5, COLORS["neutral"]], [1, COLORS["positive"]]],
                    labels={"year": "Tahun", "total": "Jumlah berita", "net_score": "Net score"},
                )
                fig_year.update_traces(marker_line_width=0)
                layout = make_plotly_layout(height=300, xaxis_updates={"tickmode": "linear", "dtick": 1})
                fig_year.update_layout(**layout)
                fig_year.update_coloraxes(showscale=False)
                st.plotly_chart(fig_year, use_container_width=True)
            else:
                st.info("Tidak ada data tahunan.")

        st.markdown("### Skor Sentimen Agregat Bulanan (dengan MA-3)")
        st.markdown("<div class='section-note'>Skor rata-rata (+1 positif, 0 netral, −1 negatif).</div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        if len(monthly):
            fig_score = go.Figure()
            fig_score.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dash", line_width=1)
            fig_score.add_trace(go.Bar(
                x=monthly["year_month_dt"],
                y=monthly["net_score"],
                marker_color=monthly["net_score"].apply(lambda x: COLORS["positive"] if x > 0.05 else (COLORS["negative"] if x < -0.05 else COLORS["neutral"])),
                marker_line_width=0,
                hovertemplate="<b>%{x|%b %Y}</b><br>Skor: %{y:.2f}<extra></extra>",
                name="Skor",
            ))
            fig_score.add_trace(go.Scatter(
                x=monthly["year_month_dt"],
                y=monthly["ma3_score"],
                mode="lines",
                name="MA-3",
                line=dict(color="#f59e0b", width=2.5, dash="dot"),
                hovertemplate="MA-3: %{y:.2f}<extra></extra>",
            ))
            fig_score.update_layout(**make_plotly_layout(height=280), yaxis_title="Skor sentimen")
            st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("### Insight Tambahan")
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown("#### Distribusi Hari Publikasi")
        if len(dff):
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow_id = {"Monday":"Sen","Tuesday":"Sel","Wednesday":"Rab","Thursday":"Kam","Friday":"Jum","Saturday":"Sab","Sunday":"Min"}
            dow = dff["day_name"].value_counts().reindex(dow_order).fillna(0).reset_index()
            dow.columns = ["Hari", "Jumlah"]
            dow["Hari_ID"] = dow["Hari"].map(dow_id)
            fig_dow = px.bar(dow, x="Hari_ID", y="Jumlah", color="Jumlah", color_continuous_scale=[[0, "#111827"], [1, COLORS["info"]]])
            fig_dow.update_coloraxes(showscale=False)
            fig_dow.update_layout(**make_plotly_layout(height=300), xaxis_title=None, yaxis_title="Jumlah berita")
            fig_dow.update_traces(marker_line_width=0)
            st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.info("Tidak ada data.")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Perbandingan Panjang Judul")
            if len(dff):
                fig_v = go.Figure()
                for sent in SENTIMENT_ORDER:
                    vals = dff.loc[dff["sentiment"] == sent, "title_word_count"]
                    if len(vals) > 0:
                        fig_v.add_trace(go.Violin(
                            y=vals,
                            name=SENTIMENT_LABEL[sent],
                            fillcolor=rgba_from_hex(COLOR_MAP[sent], 0.2),
                            line_color=COLOR_MAP[sent],
                            meanline_visible=True,
                            box_visible=True,
                            opacity=0.8,
                        ))
                fig_v.update_layout(**make_plotly_layout(height=400), yaxis_title="Jumlah kata", violinmode="overlay")
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.info("Tidak ada data.")

        with col_right:
            st.markdown("#### Distribusi Panjang Judul")
            if len(dff):
                title_len_filtered = dff[dff["title_word_count"] <= 25]["title_word_count"]
                fig_title = px.histogram(title_len_filtered, x="title_word_count", nbins=24, color_discrete_sequence=[COLORS["neutral"]])
                fig_title.update_layout(**make_plotly_layout(height=400), xaxis_title="Jumlah kata", yaxis_title="Frekuensi")
                fig_title.update_traces(marker_line_width=0)
                st.plotly_chart(fig_title, use_container_width=True)
            else:
                st.info("Tidak ada data.")

        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2 — ANALISIS MEDIA
# =============================================================================
with tab2:
    st.markdown("### Top 10 Media Paling Sering Muncul")
    if len(dff):
        top_media_counts = dff["Nama Media"].value_counts().head(10).reset_index()
        top_media_counts.columns = ["Nama Media", "count"]
        top_media_counts = top_media_counts.sort_values("count", ascending=True)
        fig_top = go.Figure(go.Bar(
            x=top_media_counts["count"],
            y=top_media_counts["Nama Media"],
            orientation="h",
            marker=dict(color=top_media_counts["count"], colorscale=[[0, "#1f2937"], [1, COLORS["info"]]]),
            text=top_media_counts["count"].map(lambda x: f"{x:,}"),
            textposition="outside",
        ))
        fig_top.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Jumlah berita", yaxis_title=None)
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("---")
        st.markdown("### Proporsi Sentimen per Media (pilih media dari daftar)")
        # Daftar media diurutkan dari yang paling banyak beritanya
        media_list = dff["Nama Media"].value_counts().index.tolist()
        selected_media = st.selectbox("Pilih media:", media_list, key="media_select")
        if selected_media:
            media_df = dff[dff["Nama Media"] == selected_media]
            sent_counts = media_df["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0)
            fig_pie = go.Figure(go.Pie(
                labels=[SENTIMENT_LABEL[s] for s in sent_counts.index],
                values=sent_counts.values,
                hole=0.5,
                marker=dict(colors=[COLOR_MAP[s] for s in sent_counts.index]),
                hovertemplate="%{label}: %{value} berita (%{percent})<extra></extra>"
            ))
            fig_pie.update_layout(**PLOT_LAYOUT, height=350, title=f"Proporsi sentimen untuk {selected_media}")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Tombol untuk melihat berita per sentimen
            if "clicked_sentiment_media" not in st.session_state:
                st.session_state.clicked_sentiment_media = None
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📰 Lihat berita Positif ({sent_counts.get('positive', 0)})", key="btn_pos_media"):
                    st.session_state.clicked_sentiment_media = "positive"
            with col2:
                if st.button(f"📰 Lihat berita Netral ({sent_counts.get('neutral', 0)})", key="btn_neu_media"):
                    st.session_state.clicked_sentiment_media = "neutral"
            with col3:
                if st.button(f"📰 Lihat berita Negatif ({sent_counts.get('negative', 0)})", key="btn_neg_media"):
                    st.session_state.clicked_sentiment_media = "negative"

            if st.session_state.clicked_sentiment_media:
                sent_filter = st.session_state.clicked_sentiment_media
                news_df = media_df[media_df["sentiment"] == sent_filter].sort_values("Waktu Terbit", ascending=False).head(20)
                st.markdown(f"#### Berita dengan sentimen **{SENTIMENT_LABEL[sent_filter]}** dari **{selected_media}**")
                if len(news_df) == 0:
                    st.info("Tidak ada berita dengan sentimen tersebut.")
                else:
                    for _, row in news_df.iterrows():
                        sent = row["sentiment"]
                        dt = row["Waktu Terbit"].strftime("%Y-%m-%d") if not pd.isna(row["Waktu Terbit"]) else "-"
                        st.markdown(f"""
<div class="news-card news-{sent[:3]}">
  <div class="news-title">{style_sentiment_badge(sent)} &nbsp; {row["Judul Berita"]}</div>
  <div class="news-meta">
    <span>📅 {dt}</span>
    <span>🌐 {row["Detected Language"]}</span>
    {"<span>🔗 <a href='" + row["Link Berita"] + "' target='_blank' style='color:#93c5fd;text-decoration:none;'>Buka sumber</a></span>" if str(row["Link Berita"]).strip() else ""}
  </div>
</div>
""", unsafe_allow_html=True)
                if st.button("Tutup daftar berita", key="reset_news_media"):
                    st.session_state.clicked_sentiment_media = None
                    st.rerun()
    else:
        plot_blank_message("Tidak ada data media untuk ditampilkan.")

# =============================================================================
# TAB 3 — N-GRAM & WORDCLOUD
# =============================================================================
with tab3:
    with st.container():
        st.markdown('<div class="tab-content-card">', unsafe_allow_html=True)
        st.markdown("### Wordcloud per Sentimen (Unigram)")
        st.markdown("<div class='section-note'>Visualisasi kata tunggal yang paling sering muncul pada setiap sentimen.</div>", unsafe_allow_html=True)
        wc_cols = st.columns(3)
        wc_cfg = {
            "positive": ("Positif", "Greens", "#08130d"),
            "neutral": ("Netral", "Blues", "#0d1320"),
            "negative": ("Negatif", "Reds", "#160b0b"),
        }
        for i, sent in enumerate(SENTIMENT_ORDER):
            if sent not in sentiments_sel:
                continue
            with wc_cols[i]:
                st.markdown(f"<span class='badge {('badge-pos' if sent=='positive' else 'badge-neu' if sent=='neutral' else 'badge-neg')}'>{wc_cfg[sent][0]}</span>", unsafe_allow_html=True)
                text_data = " ".join(dff.loc[dff["sentiment"] == sent, "processed_text"].dropna().astype(str).tolist())
                img = render_wordcloud(text_data, bg_color=wc_cfg[sent][2], cmap=wc_cfg[sent][1], max_words=120)
                if img is None:
                    st.info("Tidak cukup teks untuk wordcloud.")
                else:
                    st.image(img, use_column_width=True)

        st.markdown("---")
        st.markdown("### Analisis N-Gram 1–3 Kata (dengan eksplorasi berita)")
        st.markdown(
            "<div class='section-note'>Pilih level n-gram dan sentimen. Gunakan dropdown di bawah bar chart untuk memilih frasa dan melihat berita yang mengandung frasa tersebut dengan sentimen yang sama.</div>",
            unsafe_allow_html=True,
        )

        col_ng1, col_ng2 = st.columns(2)
        with col_ng1:
            ngram_level = st.radio(
                "Level konteks",
                [1, 2, 3],
                horizontal=True,
                format_func=lambda x: {1: "Unigram", 2: "Bigram", 3: "Trigram"}[x],
                key="ngram_level_tab3"
            )
        with col_ng2:
            top_k_ngram_local = st.slider("Jumlah n-gram tertinggi", 5, 30, 15, key="topk_ngram_tab3")

        focus_sent = st.selectbox(
            "Fokus sentimen",
            sentiments_sel if sentiments_sel else SENTIMENT_ORDER,
            format_func=lambda x: SENTIMENT_LABEL[x],
            key="ngram_focus_sent_tab3",
        )
        ngram_df_local = ngram_df_from_series(dff.loc[dff["sentiment"] == focus_sent, "processed_text"], n=ngram_level, top_n=top_k_ngram_local, sentiment=focus_sent)
        if len(ngram_df_local):
            fig_ng = px.bar(
                ngram_df_local.sort_values("freq", ascending=True),
                x="freq",
                y="phrase",
                orientation="h",
                color="sentiment",
                color_discrete_map=COLOR_MAP,
                text="freq",
                labels={"freq": "Frekuensi", "phrase": "Frasa"},
            )
            fig_ng.update_traces(textposition="outside")
            fig_ng.update_layout(**PLOT_LAYOUT, height=max(400, 34 * len(ngram_df_local) + 100), showlegend=False, xaxis_title="Frekuensi", yaxis_title=None)
            st.plotly_chart(fig_ng, use_container_width=True)

            # Dropdown untuk memilih frasa dan melihat berita (dengan sentimen yang sama)
            st.markdown("#### 🔍 Telusuri berita berdasarkan frasa (hanya dengan sentimen yang sama)")
            phrase_list = ngram_df_local["phrase"].tolist()
            selected_phrase = st.selectbox("Pilih frasa:", phrase_list, key="phrase_select_ngram")
            if selected_phrase:
                phrase_lower = selected_phrase.lower()
                # Cari berita dengan sentimen focus_sent dan mengandung frasa
                news_with_phrase = dff[(dff["sentiment"] == focus_sent) & (dff["processed_text"].str.contains(phrase_lower, na=False, regex=False))]
                if len(news_with_phrase) == 0:
                    st.info(f"Tidak ditemukan berita dengan sentimen {SENTIMENT_LABEL[focus_sent]} yang mengandung frasa '{selected_phrase}'.")
                else:
                    st.markdown(f"**{len(news_with_phrase)}** berita ditemukan dengan sentimen **{SENTIMENT_LABEL[focus_sent]}** dan mengandung frasa **'{selected_phrase}'**.")
                    for _, row in news_with_phrase.head(10).iterrows():
                        sent = row["sentiment"]
                        dt = row["Waktu Terbit"].strftime("%Y-%m-%d") if not pd.isna(row["Waktu Terbit"]) else "-"
                        st.markdown(f"""
<div class="news-card news-{sent[:3]}">
  <div class="news-title">{style_sentiment_badge(sent)} &nbsp; {row["Judul Berita"]}</div>
  <div class="news-meta">
    <span>📰 {row["Nama Media"]}</span>
    <span>📅 {dt}</span>
    <span>🌐 {row["Detected Language"]}</span>
    {"<span>🔗 <a href='" + row["Link Berita"] + "' target='_blank' style='color:#93c5fd;text-decoration:none;'>Buka sumber</a></span>" if str(row["Link Berita"]).strip() else ""}
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            plot_blank_message("Belum cukup data untuk membentuk n-gram pada filter ini.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 4 — RASIO SENTIMEN
# =============================================================================
with tab4:
    with st.container():
        st.markdown('<div class="tab-content-card">', unsafe_allow_html=True)
        st.markdown("### Rasio Positif vs Negatif")
        st.markdown("<div class='section-note'>Pilih granularitas waktu untuk melihat keseimbangan antara berita positif dan negatif.</div>", unsafe_allow_html=True)

        granularity = st.selectbox("Tampilkan berdasarkan:", ["Bulan", "Kuartal", "Tahun"], index=0)
        if granularity == "Bulan":
            time_col = "month"
        elif granularity == "Kuartal":
            time_col = "quarter"
        else:
            time_col = "year"

        ratio_df = compute_ratio_by_time(dff, granularity=time_col)
        if len(ratio_df):
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=ratio_df["time_period"],
                y=ratio_df["positive_ratio"],
                mode="lines+markers",
                name="Positif (%)",
                line=dict(color=COLORS["positive"], width=3),
                marker=dict(size=6),
            ))
            fig_ratio.add_trace(go.Scatter(
                x=ratio_df["time_period"],
                y=ratio_df["negative_ratio"],
                mode="lines+markers",
                name="Negatif (%)",
                line=dict(color=COLORS["negative"], width=3),
                marker=dict(size=6),
            ))
            fig_ratio.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Seimbang")
            fig_ratio.update_layout(
                **PLOT_LAYOUT,
                height=400,
                xaxis_title=None,
                yaxis_title="Persentase (%)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

            st.markdown("#### Data Rasio")
            display_ratio = ratio_df[["time_period", "positive_ratio", "negative_ratio"]].copy()
            display_ratio.columns = ["Periode", "Positif (%)", "Negatif (%)"]
            st.dataframe(display_ratio, use_container_width=True, hide_index=True)
        else:
            plot_blank_message("Tidak cukup data untuk menghitung rasio.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 5 — JELAJAHI BERITA
# =============================================================================
with tab5:
    with st.container():
        st.markdown('<div class="tab-content-card">', unsafe_allow_html=True)
        st.markdown("### Jelajahi Berita")
        st.markdown("<div class='section-note'>Filter tambahan untuk mencari dan menyortir berita.</div>", unsafe_allow_html=True)

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])
        with col_f1:
            sent_filter_tab5 = st.selectbox(
                "Sentimen",
                ["Semua", "positive", "neutral", "negative"],
                format_func=lambda x: {"Semua": "Semua", "positive": "✅ Positif", "neutral": "⬜ Netral", "negative": "❌ Negatif"}[x],
                key="sent_filter_tab5"
            )
        with col_f2:
            year_filter_tab5 = st.selectbox("Tahun", ["Semua"] + sorted(dff["year"].dropna().astype(int).unique().tolist(), reverse=True), key="year_filter_tab5")
        with col_f3:
            keyword_tab5 = st.text_input("Cari judul", placeholder="misalnya: ekonomi, infrastruktur", key="keyword_tab5")
        with col_f4:
            n_show_tab5 = st.number_input("Jumlah", min_value=5, max_value=100, value=20, step=5, key="n_show_tab5")

        news_df_tab5 = dff.copy()
        if sent_filter_tab5 != "Semua":
            news_df_tab5 = news_df_tab5[news_df_tab5["sentiment"] == sent_filter_tab5]
        if year_filter_tab5 != "Semua":
            news_df_tab5 = news_df_tab5[news_df_tab5["year"] == int(year_filter_tab5)]
        if keyword_tab5.strip():
            news_df_tab5 = news_df_tab5[news_df_tab5["Judul Berita"].str.contains(keyword_tab5.strip(), case=False, na=False)]
        news_df_tab5 = news_df_tab5.sort_values("Waktu Terbit", ascending=False).head(int(n_show_tab5))

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Berita ditemukan", human_int(len(news_df_tab5)))
        with col_m2:
            st.metric("Positif", human_int((news_df_tab5["sentiment"] == "positive").sum()))
        with col_m3:
            st.metric("Negatif", human_int((news_df_tab5["sentiment"] == "negative").sum()))

        if len(news_df_tab5) == 0:
            st.info("Tidak ada berita yang cocok dengan filter.")
        else:
            for _, row in news_df_tab5.iterrows():
                sent = row["sentiment"]
                dt = row["Waktu Terbit"].strftime("%Y-%m-%d") if not pd.isna(row["Waktu Terbit"]) else "-"
                st.markdown(f"""
<div class="news-card news-{sent[:3]}">
  <div class="news-title">{style_sentiment_badge(sent)} &nbsp; {row["Judul Berita"]}</div>
  <div class="news-meta">
    <span>📰 {row["Nama Media"]}</span>
    <span>📅 {dt}</span>
    <span>🌐 {row["Detected Language"]}</span>
    {"<span>🔗 <a href='" + row["Link Berita"] + "' target='_blank' style='color:#93c5fd;text-decoration:none;'>Buka sumber</a></span>" if str(row["Link Berita"]).strip() else ""}
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("### Data Terfilter (CSV Preview)")
        st.dataframe(
            news_df_tab5[["Waktu Terbit", "Judul Berita", "Nama Media", "sentiment", "Detected Language", "Link Berita"]].sort_values("Waktu Terbit", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 6 — EXPORT
# =============================================================================
with tab6:
    with st.container():
        st.markdown('<div class="tab-content-card">', unsafe_allow_html=True)
        st.markdown("### Export Data")
        st.markdown("<div class='section-note'>Unduh data yang sudah difilter (global) untuk analisis lanjutan.</div>", unsafe_allow_html=True)
        if len(dff):
            export_cols = ["Judul Berita", "Nama Media", "Waktu Terbit", "Link Berita", "Detected Language", "confidence", "text_word_count", "sentiment", "year", "month", "year_month", "quarter"]
            export_cols = [c for c in export_cols if c in dff.columns]
            export_df = dff[export_cols].copy()
            st.dataframe(export_df.head(100), use_container_width=True, hide_index=True)
            csv_bytes = download_csv(export_df)
            st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_publik_persepsi.csv", mime="text/csv")
        else:
            st.info("Tidak ada data untuk diekspor.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div style="text-align:center;padding:22px 0 12px 0;border-top:1px solid rgba(255,255,255,0.08);">
  <div style="font-family:'Playfair Display',serif;font-size:1rem;color:#e2e8f0;letter-spacing:0.18em;">
    PERSEPSI PUBLIK · JOKOWI 2019–2024
  </div>
  <div style="font-size:0.65rem;color:#94a3b8;margin-top:6px;letter-spacing:0.12em;">
    DASHBOARD ANALISIS SENTIMEN MEDIA · STREAMLIT + PLOTLY
  </div>
</div>
""", unsafe_allow_html=True)