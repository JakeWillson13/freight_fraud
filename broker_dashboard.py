import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from wordcloud import WordCloud, STOPWORDS

# -----------------------------------------------------------------------------
# DATA SOURCES
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
# DATA HELPERS  (cached for fast reloads)
# -----------------------------------------------------------------------------
@st.cache_data
def load_broker_data() -> pd.DataFrame:
    df = pd.read_csv(BROKER_CSV_URL, compression="gzip", parse_dates=[DATE_COL])
    return df

# -----------------------------------------------------------------------------
# SUMMARY + VISUAL FUNCTIONS
# -----------------------------------------------------------------------------

def summarize(df: pd.DataFrame, freq: str, start_date: pd.Timestamp) -> pd.DataFrame:
    df_filt = df[df[DATE_COL] >= start_date]
    return (
        df_filt
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
    df = df.copy()
    df["YEAR"] = df[DATE_COL].dt.year
    years = sorted(df["YEAR"].unique())[-7:]
    df = df[df["YEAR"].isin(years)]
    return (
        df.groupby(["YEAR", ACTION_COL]).size().unstack(fill_value=0).pct_change() * 100
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

# --- Executive Summary -------------------------------------------------------
with st.expander("🔎 Executive Summary", expanded=True):
    st.markdown(
        """
### What Just Happened?
- **Sept 2024 Surge:** FMCSA revoked **609** broker authorities—>2× pre‑COVID norm; ~⅓ tied to double‑brokering.
- **Broker Contraction:** New broker grants fell **38 %** in 2024 and another **46 % YTD 2025** while carrier counts kept climbing.
- **Fraud Concentration:** *Fourteen counties* now account for **46 %** of fraud‑coded revocations; double‑brokering flags have **quadrupled** since 2021.
- **Early Enforcement Works:** Revoking within 60 days of complaint averts ≈ **$52 k** in unpaid invoices.

### Call to Action
- **Shippers & Carriers:** Verify MC numbers daily, require insurance APIs, adopt live load‑tracking.
- **Brokers:** Invest in identity‑proofing & value‑added services.
- **Regulators / Load‑boards:** Share fraud telemetry in real time; audit brokers showing 90‑day churn.
        """,
        unsafe_allow_html=True,
    )

# --- DATA --------------------------------------------------------------------
df_brokers = load_broker_data()
now       = pd.Timestamp.today().normalize()
last_5yrs = now - pd.DateOffset(years=5)

# --- Tabs --------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Authority Trends",
    "📉 YoY Change",
    "📅 Monthly Summary",
    "☁️ Word Cloud",
])

with tab1:
    st.subheader("Broker Authorities – Last 5 Years")
    brokers_5y = summarize(df_brokers, "YE", last_5yrs)
    st.dataframe(brokers_5y)
    stacked_bar(brokers_5y, "Broker Authorities – Last 5 Years")

with tab2:
    st.subheader("Year‑over‑Year % Change (Last 7 Years)")
    yoy = yoy_pct_last7(df_brokers)
    st.dataframe(yoy)
    line_plot(yoy, "YoY % Change in Broker Authorities", "% Change")

with tab3:
    st.subheader("Monthly Broker Actions (Last 12 Months)")
    monthly = monthly_summary_last12(df_brokers)
    st.dataframe(monthly)
    line_plot(monthly, "Monthly Broker Authority Actions", "Count")

with tab4:
    st.subheader("Freight‑Fraud News Word Cloud")
    draw_wordcloud()
