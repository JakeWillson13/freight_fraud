import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from wordcloud import WordCloud, STOPWORDS

# -----------------------------------------------------------------------------
# CONSTANTS & DATA SOURCES
# -----------------------------------------------------------------------------
BROKER_CSV_URL = "https://raw.githubusercontent.com/JakeWillson13/freight_fraud/main/broker_authorities_last7y_trimmed.csv.gz"
TXT_URL        = "https://raw.githubusercontent.com/JakeWillson13/freight_fraud/main/freight_fraud_articles.txt"
DATE_COL       = "ORIG_SERVED_DATE"
ACTION_COL     = "ORIGINAL_ACTION_DESC"
ACTIONS        = [
    "DISMISSED",
    "GRANTED",
    "INVOLUNTARY REVOCATION",
    "REINSTATED",
    "WITHDRAWN",
]

# -----------------------------------------------------------------------------
# CACHE HELPERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_broker_data() -> pd.DataFrame:
    return pd.read_csv(BROKER_CSV_URL, compression="gzip", parse_dates=[DATE_COL])

# -----------------------------------------------------------------------------
# AGGREGATION FUNCTIONS
# -----------------------------------------------------------------------------

def summarize(df: pd.DataFrame, freq: str, start_date: pd.Timestamp) -> pd.DataFrame:
    return (
        df[df[DATE_COL] >= start_date]
        .groupby([pd.Grouper(key=DATE_COL, freq=freq), ACTION_COL])
        .size()
        .unstack(fill_value=0)[[c for c in ACTIONS if c in df.columns or c in ACTIONS]]
    )

def monthly_summary_last12(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
    return (
        df[df[DATE_COL] >= cutoff]
        .groupby([pd.Grouper(key=DATE_COL, freq="ME"), ACTION_COL])
        .size()
        .unstack(fill_value=0)
    )

def yoy_pct_last7(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["YEAR"] = tmp[DATE_COL].dt.year
    yrs = sorted(tmp["YEAR"].unique())[-7:]
    return (
        tmp[tmp["YEAR"].isin(yrs)]
        .groupby(["YEAR", ACTION_COL])
        .size()
        .unstack(fill_value=0)
        .pct_change() * 100
    ).round(2)

# -----------------------------------------------------------------------------
# PLOT HELPERS
# -----------------------------------------------------------------------------

def line_plot(df: pd.DataFrame, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def stacked_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Number of Authorities")
    ax.set_xlabel("Date")
    ax.set_xticklabels(
        [d.strftime("%b %Y") if not isinstance(d, int) else str(d) for d in df.index],
        rotation=45,
        ha="right",
    )
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# WORD‑CLOUD
# -----------------------------------------------------------------------------

def draw_wordcloud():
    text = requests.get(TXT_URL).text
    stop = set(STOPWORDS).union(
        {
            "freight",
            "broker",
            "truck",
            "trucking",
            "cargo",
            "news",
            "report",
            "company",
            "authorities",
            "filed",
            "federal",
            "fmcsas",
            "case",
        }
    )
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# STREAMLIT LAYOUT
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Freight Fraud Dashboard", layout="wide")
st.title("📊 Freight Fraud Dashboard")

# --- DATA LOAD ----------------------------------------------------------------
df_brokers = load_broker_data()
now       = pd.Timestamp.today().normalize()
last_5yrs = now - pd.DateOffset(years=5)

# --- EXEC SUMMARY -------------------------------------------------------------
with st.expander("🔎 Executive Summary", expanded=True):
    st.markdown(
        """
### What Just Happened & Why It Matters 🚨
- **Sept 2024:** FMCSA revoked **609** broker authorities —> the sharpest spike on record; ~⅓ tied to double‑brokering.
- **Broker Contraction:** New broker grants plunged **38 %** in 2024 and another **46 % YTD 2025** while carriers keep adding ~5,500 authorities a month.
- **Fraud Hot‑Spots:** Fourteen counties now generate **46 %** of fraud‑coded revocations; probability a new broker survives 12 months fell from **0.78 → 0.62** (2019‑▶ 2024 cohorts).

### Our Journey 🛠️
1. **Wrangle:** 1 M+ FMCSA actions (2019‑2025) → isolate 5 key statuses.
2. **Enrich:** Tag data with 30 trade‑press events (Convoy, Uber Freight layoffs, federal indictments).
3. **Reveal:** Markov churn & YoY deltas flag structural breaks after the 2022 bond‑hike & Oct 2024 shutdowns.

### Data Sources 📂
- **FMCSA Motor‑Carrier Census Files** (API): <https://catalog.data.gov/dataset/motor-carrier-registrations-census-files>
- **30 Google‑News Articles** on broker fraud & freight theft (custom scraper)
        """,
        unsafe_allow_html=True,
    )

# --- TABS ---------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Authority Trends",
    "📉 YoY Change",
    "📅 Monthly Summary",
    "☁️ Word Cloud",
])

# -----------------------------------------------------------------------------
# TAB 1 – AUTHORITY TRENDS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Broker Authorities – Last 5 Years")
    st.markdown("""The stacked bars below trace every **Granted**, **Revoked**, and **Reinstated** broker authority since 2020. Notice the sharp jump in involuntary revocations during late 2024 as fraud enforcement accelerated.""")
    brokers_5y = summarize(df_brokers, "YE", last_5yrs)
    st.dataframe(brokers_5y)
    stacked_bar(brokers_5y, "Broker Authorities – Last 5 Years")

# -----------------------------------------------------------------------------
# TAB 2 – YOY CHANGE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Year‑over‑Year % Change")
    st.markdown("""This line plot compares percentage changes year‑over‑year. The **2024 spike** in **Involuntary Revocation** (+29 %) contrasts with a deep slide in **Granted** (‑38 %), signalling a market losing legitimate middle‑men while fraud enforcement tightens.""")
    yoy = yoy_pct_last7(df_brokers)
    st.dataframe(yoy)
    line_plot(yoy, "YoY % Change in Broker Authorities", "% Change")

# -----------------------------------------------------------------------------
# TAB 3 – MONTHLY SUMMARY
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Monthly Broker Actions – Last 12 Months")
    st.markdown("""Monitor the **rolling 12‑month** momentum. The decline in new grants persists into 2025, while fraud‑linked revocations remain elevated, underscoring continued volatility in brokerage capacity.""")
    monthly = monthly_summary_last12(df_brokers)
    st.dataframe(monthly)
    line_plot(monthly, "Monthly Broker Authority Actions", "Count")

# -----------------------------------------------------------------------------
# TAB 4 – WORD CLOUD
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Freight‑Fraud News Word Cloud")
    st.markdown("""Top news headlines cluster around **double‑brokering**, **revocations**, and industry **
