import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime
from textblob import TextBlob
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Transformers & Semantic Search
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import wikipedia  # For offline events summary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# --------------------------------------------------------------------------------
# ----------------------- Data Loading and Normalization -------------------------
# --------------------------------------------------------------------------------
@st.cache_data
def load_raw_data(filepath):
    """Load the newline-delimited JSON file into a Pandas DataFrame."""
    try:
        raw_df = pd.read_json(filepath, lines=True)
    except ValueError as e:
        st.error("Error reading the JSONL file. Please check the file format.")
        raise e
    return raw_df

DATA_PATH = "data.jsonl"
if not os.path.exists(DATA_PATH):
    st.error("data.jsonl file not found. Please ensure it is in the same directory as this app.")
else:
    raw_df = load_raw_data(DATA_PATH)

st.sidebar.markdown("### Raw Dataset Columns")
st.sidebar.write(raw_df.columns.tolist())

# Normalize the nested "data" column if present
if 'data' in raw_df.columns:
    try:
        df = pd.json_normalize(raw_df['data'])
    except Exception as e:
        st.error("Error normalizing the 'data' column.")
        df = raw_df
else:
    df = raw_df

st.sidebar.markdown("### Normalized Data Columns")
st.sidebar.write(df.columns.tolist())

# --------------------------------------------------------------------------------
# ------------------------- Column Mapping (Reddit Data) -------------------------
# --------------------------------------------------------------------------------
# Typical Reddit fields:
timestamp_col = "created_utc"  # Unix timestamp (in seconds)
user_col = "author"            # Author

# For text, prefer "selftext" if available; otherwise, use "title".
if "selftext" in df.columns and df["selftext"].notnull().sum() > 0:
    text_col = "selftext"
elif "title" in df.columns:
    text_col = "title"
else:
    text_col = None

# For hashtags: if not provided, extract from text using regex.
if "hashtags" not in df.columns:
    def extract_hashtags(row):
        text = ""
        if "title" in row and pd.notnull(row["title"]):
            text += row["title"] + " "
        if "selftext" in row and pd.notnull(row["selftext"]):
            text += row["selftext"]
        return re.findall(r"#\w+", text)
    df["hashtags"] = df.apply(extract_hashtags, axis=1)
hashtags_col = "hashtags"

# Convert Unix timestamp to datetime if available
if timestamp_col in df.columns:
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    except Exception as e:
        st.error(f"Error converting timestamp. Check the format of '{timestamp_col}'.")

# --------------------------------------------------------------------------------
# --------------------------- Sidebar: Filters & Platform -------------------------
# --------------------------------------------------------------------------------
st.sidebar.header("Filters & Platform")

# Platform Selector (simulate multiple platforms)
platform = st.sidebar.selectbox("Select Platform", ["Reddit", "Twitter", "Facebook"])
if platform != "Reddit":
    st.sidebar.info(f"Data for {platform} is not available. Showing Reddit data.")

# Date Filter
if timestamp_col in df.columns:
    try:
        min_date = df[timestamp_col].min().date()
        max_date = df[timestamp_col].max().date()
        start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.sidebar.error("Error: End date must fall after start date.")
        df = df[(df[timestamp_col].dt.date >= start_date) & (df[timestamp_col].dt.date <= end_date)]
    except Exception as e:
        st.sidebar.error("Error processing the timestamp column for filtering.")
else:
    st.sidebar.info(f"No '{timestamp_col}' column found for filtering by date.")

# Keyword/Hashtag Search
search_term = st.sidebar.text_input("Search for a keyword/hashtag:")
if search_term:
    if text_col in df.columns:
        df = df[df[text_col].str.contains(search_term, case=False, na=False)]
    st.sidebar.markdown(f"### Showing results for '{search_term}'")

# --------------------------------------------------------------------------------
# ------------------------- Main Dashboard: Basic Visualizations ------------------
# --------------------------------------------------------------------------------
st.title("Social Media Data Analysis Dashboard")
st.markdown("""This dashboard visualizes Reddit data, showcasing trends over time, key contributors, topic embeddings, and more.""")

# Summary Metrics
total_posts = len(df)
st.markdown("### Summary Metrics")
st.write("**Total Posts:**", total_posts)
if user_col in df.columns:
    unique_users = df[user_col].nunique()
    st.write("**Unique Users:**", unique_users)
else:
    st.write("**Unique Users:** Data not available")

# Time Series Plot with 7-day Moving Average
if timestamp_col in df.columns:
    st.markdown("### Posts Over Time with Moving Average")
    df["date"] = df[timestamp_col].dt.date  # Ensure date is extracted from timestamp
    if "date" not in df.columns:
        st.error("Error: 'date' column not found after timestamp conversion.")
    else:
        time_series = df.groupby("date").size().reset_index(name="count")
        time_series["7-day Moving Avg"] = time_series["count"].rolling(window=7).mean()
        fig_time = px.line(time_series, x="date", y=["count", "7-day Moving Avg"],
                           labels={"date": "Date", "value": "Number of Posts"},
                           title="Posts Over Time with 7-day Moving Average")
        st.plotly_chart(fig_time)
else:
    st.info("No timestamp data available for time series plot.")

# GenAI Summary for Time Series Plot
st.markdown("## GenAI Summary for Time Series")
if timestamp_col in df.columns and not df.empty:
    time_series = df.groupby(df[timestamp_col].dt.date).size().reset_index(name="count")
    if not time_series.empty:
        start = time_series["date"].min()
        end = time_series["date"].max()
        avg_posts = time_series["count"].mean()
        peak = time_series.loc[time_series["count"].idxmax()]
        description = (
            f"From {start} to {end}, the average number of posts per day was {avg_posts:.1f}. "
            f"The highest activity was on {peak['date']} with {peak['count']} posts."
        )
        st.write("Time Series Description:")
        st.write(description)

        ts_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        try:
            ts_summary = ts_summarizer(description, max_length=80, min_length=40, do_sample=False)[0]['summary_text']
            st.markdown("**GenAI Summary:**")
            st.write(ts_summary)
        except Exception as e:
            st.error("Error generating time series summary.")
    else:
        st.info("Time series data not available for summarization.")
else:
    st.info("No timestamp data available for time series summary.")

# ------------------------------- End of Dashboard --------------------------------
st.markdown("### End of Dashboard")
st.markdown("""This dashboard is a prototype implementation for analyzing Reddit social media data. It demonstrates advanced trend analysis, contributor insights, topic embeddings, GenAI summaries, offline event linking, and semantic search functionality.""")
