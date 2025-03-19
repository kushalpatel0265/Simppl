import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime
from textblob import TextBlob
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
timestamp_col = "created_utc"  # Unix timestamp (in seconds)
user_col = "author"            # Author

# Text column handling
if "selftext" in df.columns and df["selftext"].notnull().sum() > 0:
    text_col = "selftext"
elif "title" in df.columns:
    text_col = "title"
else:
    text_col = None

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
hashtags_col = "hashtags"

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

# Platform selector
platform = st.sidebar.selectbox("Select Platform", ["Reddit", "Twitter", "Facebook"])
if platform != "Reddit":
    st.sidebar.info(f"Data for {platform} is not available. Showing Reddit data.")

# Date filter
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

# Keyword search
search_term = st.sidebar.text_input("Search for a keyword/hashtag:")
if search_term:
    if text_col in df.columns:
        df = df[df[text_col].str.contains(search_term, case=False, na=False)]
    st.sidebar.markdown(f"### Showing results for '{search_term}'")

# --------------------------------------------------------------------------------
# ------------------------- Main Dashboard Visualizations ------------------------
# --------------------------------------------------------------------------------
st.title("Social Media Data Analysis Dashboard")
st.markdown("This dashboard visualizes Reddit data with core analytical features.")

# Summary metrics
total_posts = len(df)
st.markdown("### Summary Metrics")
st.write("**Total Posts:**", total_posts)
if user_col in df.columns:
    unique_users = df[user_col].nunique()
    st.write("**Unique Users:**", unique_users)

# Time series plot
if timestamp_col in df.columns:
    st.markdown("### Posts Over Time with Moving Average")
    df["date"] = df[timestamp_col].dt.date
    time_series = df.groupby("date").size().reset_index(name="count")
    time_series["7-day Moving Avg"] = time_series["count"].rolling(window=7).mean()
    fig_time = px.line(time_series, x="date", y=["count", "7-day Moving Avg"],
                       labels={"date": "Date", "value": "Posts"},
                       title="Posts Over Time with 7-day Moving Average")
    st.plotly_chart(fig_time)

# Top contributors pie chart
community_col = "subreddit" if "subreddit" in df.columns else user_col
if community_col in df.columns:
    st.markdown("### Top Communities/Accounts")
    contributions = df[community_col].value_counts().reset_index()
    contributions.columns = [community_col, "count"]
    fig_pie = px.pie(contributions.head(10), values="count", names=community_col,
                     title="Top 10 Contributors")
    st.plotly_chart(fig_pie)

# Hashtag analysis
if hashtags_col in df.columns:
    st.markdown("### Top Hashtags")
    hashtags_exploded = df.explode(hashtags_col)
    hashtags_exploded = hashtags_exploded[hashtags_exploded[hashtags_col] != ""]
    top_hashtags = hashtags_exploded[hashtags_col].value_counts().reset_index()
    top_hashtags.columns = ['hashtag', 'count']
    if not top_hashtags.empty:
        fig_hashtags = px.bar(top_hashtags.head(10), x='hashtag', y='count',
                              title="Top 10 Hashtags")
        st.plotly_chart(fig_hashtags)

# Sentiment analysis
if text_col is not None and text_col in df.columns:
    st.markdown("### Sentiment Analysis")
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    fig_sentiment = px.histogram(df, x='sentiment', nbins=30,
                                 title="Sentiment Distribution")
    st.plotly_chart(fig_sentiment)

# Topic modeling visualization
# if text_col in df.columns:
#     st.markdown("## Topic Embedding Visualization")
#     texts = df[text_col].dropna().sample(n=min(500, len(df)), random_state=42).tolist()
#     vectorizer = CountVectorizer(stop_words='english', max_features=1000)
#     X = vectorizer.fit_transform(texts)
#     lda = LatentDirichletAllocation(n_components=5, random_state=42)
#     topic_matrix = lda.fit_transform(X)
#     dominant_topic = topic_matrix.argmax(axis=1)
#     tsne_model = TSNE(n_components=2, random_state=42)
#     tsne_values = tsne_model.fit_transform(topic_matrix)
#     tsne_df = pd.DataFrame(tsne_values, columns=["x", "y"])
#     tsne_df["Dominant Topic"] = dominant_topic.astype(str)
#     fig_topics = px.scatter(tsne_df, x="x", y="y", color="Dominant Topic",
#                             title="Topic Clusters")
#     st.plotly_chart(fig_topics)

st.markdown("### End of Dashboard")
