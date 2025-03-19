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
import wikipedia
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
# ------------------------- Column Mapping (Reddit Data) --------------------------
# --------------------------------------------------------------------------------
timestamp_col = "created_utc"
user_col = "author"

if "selftext" in df.columns and df["selftext"].notnull().sum() > 0:
    text_col = "selftext"
elif "title" in df.columns:
    text_col = "title"
else:
    text_col = None

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

if timestamp_col in df.columns:
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    except Exception as e:
        st.error(f"Error converting timestamp. Check the format of '{timestamp_col}'.")

# --------------------------------------------------------------------------------
# --------------------------- Sidebar: Filters & Platform -------------------------
# --------------------------------------------------------------------------------
st.sidebar.header("Filters & Platform")
platform = st.sidebar.selectbox("Select Platform", ["Reddit", "Twitter", "Facebook"])
if platform != "Reddit":
    st.sidebar.info(f"Data for {platform} is not available. Showing Reddit data.")

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
        st.sidebar.error("Error processing timestamp column.")
else:
    st.sidebar.info(f"No '{timestamp_col}' column found.")

search_term = st.sidebar.text_input("Search for a keyword/hashtag:")
if search_term and text_col in df.columns:
    df = df[df[text_col].str.contains(search_term, case=False, na=False)]
    st.sidebar.markdown(f"### Showing results for '{search_term}'")

# --------------------------------------------------------------------------------
# ------------------------- Main Dashboard: Visualizations ------------------------
# --------------------------------------------------------------------------------
st.title("Social Media Data Analysis Dashboard")
st.markdown("Analyzing social media trends with advanced NLP and network analysis.")

total_posts = len(df)
st.markdown("### Summary Metrics")
st.write("**Total Posts:**", total_posts)
if user_col in df.columns:
    st.write("**Unique Users:**", df[user_col].nunique())

if timestamp_col in df.columns:
    st.markdown("### Posts Over Time")
    df["date"] = df[timestamp_col].dt.date
    time_series = df.groupby("date").size().reset_index(name="count")
    time_series["7-day MA"] = time_series["count"].rolling(7).mean()
    fig_time = px.line(time_series, x="date", y=["count", "7-day MA"], 
                      title="Post Activity with Moving Average")
    st.plotly_chart(fig_time)

community_col = "subreddit" if "subreddit" in df.columns else user_col
if community_col in df.columns:
    st.markdown("### Top Contributors")
    top_contributors = df[community_col].value_counts().head(10)
    fig_contrib = px.pie(top_contributors, names=top_contributors.index, values=top_contributors.values)
    st.plotly_chart(fig_contrib)

if hashtags_col in df.columns:
    st.markdown("### Trending Hashtags")
    hashtags = df.explode(hashtags_col)[hashtags_col].value_counts().head(10)
    fig_hashtags = px.bar(hashtags, x=hashtags.index, y=hashtags.values)
    st.plotly_chart(fig_hashtags)

if text_col in df.columns:
    st.markdown("### Sentiment Analysis")
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    fig_sent = px.histogram(df, x='sentiment', title="Sentiment Distribution")
    st.plotly_chart(fig_sent)

# --------------------------------------------------------------------------------
# ---------------------------- Advanced Features ---------------------------------
# --------------------------------------------------------------------------------
st.markdown("## Topic Modeling (LDA)")
if text_col in df.columns:
    texts = df[text_col].dropna().sample(min(500, len(df))).tolist()
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(lda.transform(X))
    tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    tsne_df['topic'] = lda.transform(X).argmax(axis=1).astype(str)
    fig_tsne = px.scatter(tsne_df, x='x', y='y', color='topic', 
                         title="t-SNE Visualization of Topics")
    st.plotly_chart(fig_tsne)

st.markdown("## Wikipedia Event Context")
wiki_query = st.text_input("Enter event/person name for Wikipedia context:")
if wiki_query:
    try:
        page = wikipedia.page(wiki_query, auto_suggest=True)
        summary = wikipedia.summary(wiki_query, sentences=3, auto_suggest=True)
        st.markdown(f"**{page.title}**")
        st.write(summary)
        st.markdown(f"[Read more]({page.url})")
    except wikipedia.DisambiguationError as e:
        st.error("Multiple matches found. Please be more specific:")
        for option in e.options[:5]:
            st.write(f"- {option}")
    except wikipedia.PageError:
        st.error("No Wikipedia page found for this query.")
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

st.markdown("## Semantic Search")
search_query = st.text_input("Enter search query:")
if search_query and text_col in df.columns:
    @st.cache_data
    def get_embeddings():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(df[text_col].dropna().tolist(), convert_to_tensor=True)
    
    embeddings = get_embeddings()
    query_embed = SentenceTransformer('all-MiniLM-L6-v2').encode(search_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embed, embeddings)[0]
    top_matches = scores.topk(5)
    
    st.markdown("### Top Matching Posts")
    for score, idx in zip(top_matches[0], top_matches[1]):
        st.write(f"**Score:** {score:.3f}")
        st.write(df[text_col].iloc[idx.item()])
        st.write("---")

# --------------------------------------------------------------------------------
# ------------------------------- Footer -----------------------------------------
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown("**Social Media Analytics Dashboard** | Built with Streamlit")
