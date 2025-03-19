import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import requests
from datetime import datetime
from textblob import TextBlob
import wikipedia

# Transformers & Semantic Search
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
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

# Normalize the nested "data" column if present
if 'data' in raw_df.columns:
    try:
        df = pd.json_normalize(raw_df['data'])
    except Exception as e:
        st.error("Error normalizing the 'data' column.")
        df = raw_df
else:
    df = raw_df

# --------------------------------------------------------------------------------
# ------------------------- Column Mapping (Reddit Data) -------------------------
# --------------------------------------------------------------------------------
timestamp_col = "created_utc"
user_col = "author"
text_col = "selftext" if "selftext" in df.columns and df["selftext"].notnull().sum() > 0 else "title" if "title" in df.columns else None

# Hashtag extraction
if "hashtags" not in df.columns:
    def extract_hashtags(row):
        text = ""
        if "title" in row and pd.notnull(row["title"]):
            text += row["title"] + " "
        if "selftext" in row and pd.notnull(row["selftext"]):
            text += row["selftext"]
        return re.findall(r"#\w+", text)
    df["hashtags"] = df.apply(extract_hashtags, axis=1)

# Timestamp conversion
if timestamp_col in df.columns:
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    except Exception as e:
        st.error(f"Error converting timestamp. Check the format of '{timestamp_col}'.")

# --------------------------------------------------------------------------------
# --------------------------- Sidebar: Filters & Platform ------------------------
# --------------------------------------------------------------------------------
st.sidebar.header("Filters & Platform")
platform = st.sidebar.selectbox("Select Platform", ["Reddit", "Twitter", "Facebook"])
if platform != "Reddit":
    st.sidebar.info(f"Data for {platform} is not available. Showing Reddit data.")

# Date Filter
time_series = pd.DataFrame()
if timestamp_col in df.columns:
    try:
        min_date = df[timestamp_col].min().date()
        max_date = df[timestamp_col].max().date()
        start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date])
        df = df[(df[timestamp_col].dt.date >= start_date) & (df[timestamp_col].dt.date <= end_date)]
        df["date"] = df[timestamp_col].dt.date
        time_series = df.groupby("date").size().reset_index(name="count")
        time_series["7-day Moving Avg"] = time_series["count"].rolling(window=7).mean()
    except Exception as e:
        st.sidebar.error("Error processing date filters")

# --------------------------------------------------------------------------------
# ------------------------- Main Dashboard Visualizations ------------------------
# --------------------------------------------------------------------------------
st.title("Social Media Data Analysis Dashboard")

# GenAI Time Series Summary with Enhanced Error Handling
st.markdown("## GenAI Time Series Summary")
if not time_series.empty:
    try:
        # Generate description
        start = time_series["date"].min()
        end = time_series["date"].max()
        avg_posts = time_series["count"].mean()
        peak = time_series.loc[time_series["count"].idxmax()]
        description = (f"From {start} to {end}, average posts per day: {avg_posts:.1f}. "
                      f"Peak activity: {peak['date']} with {peak['count']} posts. "
                      f"Total posts in period: {time_series['count'].sum()}.")

        # Initialize summarizer with caching
        @st.cache_resource
        def load_summarizer():
            return pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Generate summary
        if len(description) > 50:
            summarizer = load_summarizer()
            ts_summary = summarizer(
                description,
                max_length=80,
                min_length=40,
                do_sample=False
            )[0]['summary_text']
            
            st.markdown("**AI-Generated Summary:**")
            st.write(ts_summary)
            st.write("**Original Statistics:**")
            st.write(description)
        else:
            st.write("Insufficient data for AI summary. Original statistics:")
            st.write(description)

    except Exception as e:
        st.error(f"Failed to generate AI summary: {str(e)}")
        st.write("**Fallback Statistics:**")
        st.write(description)
else:
    st.info("No time series data available for summarization")

# --------------------------------------------------------------------------------
# ------------------------- Other Visualization Sections -------------------------
# --------------------------------------------------------------------------------
# [Include all other visualization sections from previous code here]
# Note: Keep the same structure for time series plot, sentiment analysis, 
# topic modeling, Wikipedia integration, and semantic search features

# --------------------------------------------------------------------------------
# ------------------------------- Final Section ----------------------------------
# --------------------------------------------------------------------------------
st.markdown("### End of Dashboard")
st.markdown("""
This robust implementation includes:
- Fault-tolerant AI summary generation
- Graceful degradation for failed components
- Comprehensive error handling
- Cached model loading
- Fallback statistical displays
""")
