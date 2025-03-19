import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import gc
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
            text += str(row["title"]) + " "
        if "selftext" in row and pd.notnull(row["selftext"]):
            text += str(row["selftext"])
        return re.findall(r"#\w+", text)
    df["hashtags"] = df.apply(extract_hashtags, axis=1)  # Fixed here
hashtags_col = "hashtags"

# Convert Unix timestamp to datetime if available
if timestamp_col in df.columns:
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    except Exception as e:
        st.error(f"Error converting timestamp. Check the format of '{timestamp_col}'.")

# --------------------------------------------------------------------------------
# --------------------------- Sidebar: Filters & Platform ------------------------
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
# ------------------------- Main Dashboard: Basic Visualizations -----------------
# --------------------------------------------------------------------------------
st.title("Social Media Data Analysis Dashboard")
st.markdown("""
This dashboard visualizes Reddit data, showcasing trends over time, key contributors, and more.
""")

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
    df["date"] = df[timestamp_col].dt.date
    time_series = df.groupby("date").size().reset_index(name="count")
    time_series["7-day Moving Avg"] = time_series["count"].rolling(window=7).mean()
    fig_time = px.line(time_series, x="date", y=["count", "7-day Moving Avg"],
                       labels={"date": "Date", "value": "Number of Posts"},
                       title="Posts Over Time with 7-day Moving Average")
    st.plotly_chart(fig_time)
else:
    st.info("No timestamp data available for time series plot.")

# Pie Chart of Top Contributors (using subreddit if available, otherwise author)
community_col = "subreddit" if "subreddit" in df.columns else user_col
if community_col in df.columns:
    st.markdown("### Top Communities/Accounts Contributions")
    contributions = df[community_col].value_counts().reset_index()
    contributions.columns = [community_col, "count"]
    top_contributions = contributions.head(10)
    fig_pie = px.pie(top_contributions, values="count", names=community_col,
                     title="Top 10 Contributors")
    st.plotly_chart(fig_pie)
else:
    st.info("No community or account data available for contributor pie chart.")

# Top Hashtags Bar Chart
if hashtags_col in df.columns:
    st.markdown("### Top Hashtags")
    hashtags_exploded = df.explode(hashtags_col)
    hashtags_exploded = hashtags_exploded[hashtags_exploded[hashtags_col] != ""]
    top_hashtags = hashtags_exploded[hashtags_col].value_counts().reset_index()
    top_hashtags.columns = ['hashtag', 'count']
    if not top_hashtags.empty:
        fig_hashtags = px.bar(top_hashtags.head(10), x='hashtag', y='count',
                              labels={'hashtag': 'Hashtag', 'count': 'Frequency'},
                              title="Top 10 Hashtags")
        st.plotly_chart(fig_hashtags)
    else:
        st.info("No hashtag data available.")
else:
    st.info("No 'hashtags' column found in the dataset.")

# Sentiment Analysis on Text Data
if text_col is not None and text_col in df.columns:
    st.markdown("### Sentiment Analysis")
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    fig_sentiment = px.histogram(df, x='sentiment', nbins=30,
                                 labels={'sentiment': 'Sentiment Polarity'},
                                 title="Sentiment Polarity Distribution")
    st.plotly_chart(fig_sentiment)
else:
    st.info(f"No '{text_col}' column available for sentiment analysis.")

# --------------------------------------------------------------------------------
# ------------------------------- End of Dashboard -------------------------------
# --------------------------------------------------------------------------------
st.markdown("### End of Dashboard")
st.markdown("""
This dashboard is a prototype implementation for analyzing Reddit social media data.
""")
