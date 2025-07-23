import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from wordcloud import WordCloud, STOPWORDS

# --- Constants ---
BROKER_CSV_URL = "https://raw.githubusercontent.com/JakeWillson13/freight_fraud/main/broker_authorities_last7y_trimmed.csv.gz"
TXT_URL = "https://raw.githubusercontent.com/JakeWillson13/freight_fraud/main/freight_fraud_articles.txt"
DATE_COL = "ORIG_SERVED_DATE"
ACTION_COL = "ORIGINAL_ACTION_DESC"
ACTIONS = ['DISMISSED', 'GRANTED', 'INVOLUNTARY REVOCATION', 'REINSTATED', 'WITHDRAWN']

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(BROKER_CSV_URL, compression="gzip", parse_dates=[DATE_COL])
    return df

def summarize(df, freq, start_date):
    df_filtered = df[df[DATE_COL] >= start_date]
    summary = df_filtered.groupby([pd.Grouper(key=DATE_COL, freq=freq), ACTION_COL]).size().unstack(fill_value=0)
    return summary[[col for col in ACTIONS if col in summary.columns]]

def monthly_summary_last12(df):
    recent_cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
    df = df[df[DATE_COL] >= recent_cutoff]
    return df.groupby([pd.Grouper(key=DATE_COL, freq='ME'), ACTION_COL]).size().unstack(fill_value=0)

def yoy_percentage_change_last7(df):
    df = df.copy()
    df['YEAR'] = df[DATE_COL].dt.year
    recent_years = sorted(df['YEAR'].dropna().unique())[-7:]
    df = df[df['YEAR'].isin(recent_years)]
    summary = df.groupby(['YEAR', ACTION_COL]).size().unstack(fill_value=0)
    return summary.pct_change().multiply(100).round(2)

# --- Plotting ---
def plot_bar_chart(df, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Number of Authorities")
    ax.set_xlabel("Date")
    ax.set_xticklabels([d.strftime('%b %Y') if not isinstance(d, int) else str(d) for d in df.index], rotation=45, ha='right')
    st.pyplot(fig)

def plot_line_chart(df, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], marker='o', label=col)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def generate_wordcloud():
    text = requests.get(TXT_URL).text
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["freight", "broker", "truck", "trucking", "cargo", "news", "report", "company", "authorities", "filed", "federal", "fmcsas", "case"])
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=custom_stopwords).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- Streamlit App ---
st.title("📊 Freight Fraud Dashboard")
st.markdown("""
Visual analysis of broker authority actions and freight fraud-related news.
""")

# Load and process data
df_brokers = load_data()
today = pd.Timestamp.today().normalize()
last_5 = today - pd.DateOffset(years=5)

# Tabs for navigation
tabs = st.tabs(["📈 Authority Trends", "📉 YoY Change", "📅 Monthly Summary", "☁️ Word Cloud"])

# --- Tab 1 ---
with tabs[0]:
    st.header("Broker Authorities - Last 5 Years")
    brokers_5y = summarize(df_brokers, 'YE', last_5)
    st.dataframe(brokers_5y)
    plot_bar_chart(brokers_5y, "Broker Authorities - Last 5 Years")

# --- Tab 2 ---
with tabs[1]:
    st.header("Year-over-Year % Change")
    yoy = yoy_percentage_change_last7(df_brokers)
    st.dataframe(yoy)
    plot_line_chart(yoy, "YoY % Change in Broker Authorities", "% Change")

# --- Tab 3 ---
with tabs[2]:
    st.header("Monthly Authority Actions (Last 12 Months)")
    monthly = monthly_summary_last12(df_brokers)
    st.dataframe(monthly)
    plot_line_chart(monthly, "Monthly Broker Actions (Last 12 Months)", "Count")

# --- Tab 4 ---
with tabs[3]:
    st.header("Freight Fraud Word Cloud")
    generate_wordcloud()
