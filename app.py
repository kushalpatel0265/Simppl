# Fix for Streamlit event loop issues
import sys
import asyncio
if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime
from textblob import TextBlob

# Core NLP imports
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import wikipedia

# Machine learning imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# Set page config first
st.set_page_config(
    page_title="Social Media Analyzer",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------------------------------------
# ----------------------- Data Loading with Caching ------------------------------
# --------------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data with error handling"""
    try:
        raw_df = pd.read_json("data.jsonl", lines=True)
        if 'data' in raw_df.columns:
            df = pd.json_normalize(raw_df['data'])
        else:
            df = raw_df
            
        # Preprocess timestamps
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
            
        # Extract hashtags
        if 'hashtags' not in df.columns:
            df['hashtags'] = df.apply(
                lambda x: re.findall(r"#\w+", str(x.get('title', '')) + ' ' + str(x.get('selftext', ''))),
                axis=1
            )
            
        return df
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# --------------------------- Main Application ------------------------------------
# --------------------------------------------------------------------------------
def main():
    st.title("📊 Social Media Analytics Dashboard")
    st.write("Analyze social media trends with NLP and machine learning")
    
    # Load data
    df = load_and_preprocess_data()
    
    if df.empty:
        st.warning("No data loaded - check your data.jsonl file")
        return
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # Date filter
    if 'created_utc' in df.columns:
        min_date = df['created_utc'].min().date()
        max_date = df['created_utc'].max().date()
        date_range = st.sidebar.date_input(
            "Select date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        df = df[(df['created_utc'].dt.date >= date_range[0]) & 
                (df['created_utc'].dt.date <= date_range[1])]
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Posts", len(df))
    with col2:
        st.metric("Unique Users", df['author'].nunique() if 'author' in df else "N/A")
    with col3:
        st.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}" if 'sentiment' in df else "N/A")
    
    # Visualization Section
    st.header("Trend Analysis")
    
    # Time series chart
    if 'created_utc' in df.columns:
        time_series = df.resample('D', on='created_utc').size()
        fig = px.line(time_series, title="Posts Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Analysis
    st.header("Sentiment Insights")
    if 'text' in df.columns:
        df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        fig = px.histogram(df, x='sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Features
    st.header("Advanced Analytics")
    
    # Topic Modeling
    with st.expander("Topic Modeling"):
        if 'text' in df.columns:
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(df['text'].dropna())
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(dtm)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_features = tsne.fit_transform(lda.transform(dtm))
            fig = px.scatter(
                x=tsne_features[:,0], 
                y=tsne_features[:,1],
                color=lda.transform(dtm).argmax(axis=1),
                title="Topic Clustering"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Wikipedia Context
    with st.expander("Wikipedia Context Search"):
        wiki_query = st.text_input("Search Wikipedia for context:")
        if wiki_query:
            try:
                wiki_summary = wikipedia.summary(wiki_query, sentences=3, auto_suggest=True)
                st.write(wiki_summary)
            except wikipedia.DisambiguationError as e:
                st.error(f"Ambiguous term! Try one of these: {', '.join(e.options[:5])}")
            except wikipedia.PageError:
                st.error("No Wikipedia page found")
    
    # Semantic Search
    with st.expander("Semantic Search"):
        search_query = st.text_input("Enter semantic search query:")
        if search_query and 'text' in df.columns:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(df['text'].dropna().tolist())
            query_embedding = model.encode(search_query)
            scores = util.cos_sim(query_embedding, embeddings)[0]
            top_results = scores.topk(5)
            
            st.write("Top matching posts:")
            for score, idx in zip(top_results.values, top_results.indices):
                st.write(f"Score: {score:.3f}")
                st.write(df['text'].iloc[idx])
                st.divider()

# --------------------------------------------------------------------------------
# --------------------------- Run Application -------------------------------------
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
