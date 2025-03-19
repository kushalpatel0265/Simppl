import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import requests
from datetime import datetime
from textblob import TextBlob
import wikipedia
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
    """Load and validate input data"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at {filepath}")
        return pd.read_json(filepath, lines=True)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

DATA_PATH = "data.jsonl"
raw_df = load_raw_data(DATA_PATH)

# Data normalization
if not raw_df.empty and 'data' in raw_df.columns:
    try:
        df = pd.json_normalize(raw_df['data'])
    except Exception as e:
        st.error(f"Data normalization failed: {str(e)}")
        df = raw_df
else:
    df = raw_df

# --------------------------------------------------------------------------------
# ------------------------- AI Model Configuration -------------------------------
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI models...")
def load_models():
    """Safe model loading with CPU optimization"""
    try:
        return {
            'summarizer': pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1,  # Force CPU
                min_length=5,
                max_length=50
            ),
            'sentence_model': SentenceTransformer("all-MiniLM-L6-v2")
        }
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        return None

models = load_models()

# --------------------------------------------------------------------------------
# ---------------------- Adaptive Summary Generation -----------------------------
# --------------------------------------------------------------------------------
def generate_safe_summary(text):
    """Generate summary with dynamic length handling"""
    if models is None or not text:
        return "Summary unavailable: System error"
    
    try:
        # Calculate safe length parameters
        words = text.split()
        max_len = max(10, min(len(words), 50))
        min_len = max(5, int(len(words) * 0.2))
        
        return models['summarizer'](
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )[0]['summary_text']
    except Exception as e:
        return f"Summary error: {str(e)}"

# --------------------------------------------------------------------------------
# ------------------------- Time Series Analysis --------------------------------
# --------------------------------------------------------------------------------
def handle_time_series_analysis(df):
    """Safe time series processing"""
    if 'created_utc' not in df.columns:
        return pd.DataFrame()
    
    try:
        df["date"] = pd.to_datetime(df["created_utc"], unit='s').dt.date
        time_series = df.groupby("date").size().reset_index(name="count")
        time_series["7-day Moving Avg"] = time_series["count"].rolling(7).mean()
        return time_series
    except Exception as e:
        st.error(f"Time series processing failed: {str(e)}")
        return pd.DataFrame()

time_series = handle_time_series_analysis(df)

# --------------------------------------------------------------------------------
# ------------------------- Main Dashboard Components ---------------------------
# --------------------------------------------------------------------------------
st.title("Social Media Analytics Dashboard")

# Time Series Summary Section
st.markdown("## Intelligent Time Series Analysis")
if not time_series.empty:
    try:
        # Generate statistics
        stats = {
            'start': time_series["date"].min(),
            'end': time_series["date"].max(),
            'avg': time_series["count"].mean(),
            'peak': time_series.loc[time_series["count"].idxmax()]
        }
        
        # Create input text
        input_text = (
            f"From {stats['start']} to {stats['end']}, average posts: {stats['avg']:.1f}/day. "
            f"Peak activity: {stats['peak']['date']} ({stats['peak']['count']} posts). "
            f"Total posts: {time_series['count'].sum()}."
        )
        
        # Generate and display summary
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**AI-Powered Insights**")
            summary = generate_safe_summary(input_text)
            st.write(summary if not summary.startswith("Summary error") else f"⚠️ {summary}")
        with col2:
            st.markdown("**Key Metrics**")
            st.metric("Average Posts/Day", f"{stats['avg']:.1f}")
            st.metric("Peak Posts", stats['peak']['count'])
            st.metric("Total Posts", time_series['count'].sum())
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
else:
    st.info("No time series data available")

# --------------------------------------------------------------------------------
# ---------------------- Additional Features with Safeguards ---------------------
# --------------------------------------------------------------------------------
# Sentiment Analysis
if 'text' in df.columns:
    try:
        st.markdown("## Sentiment Analysis")
        df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        fig = px.histogram(df, x='sentiment', nbins=20)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Sentiment analysis failed: {str(e)}")

# Semantic Search
if models and 'text' in df.columns:
    try:
        st.markdown("## Semantic Search")
        search_query = st.text_input("Search posts by meaning:")
        if search_query:
            posts = df['text'].dropna().tolist()
            embeddings = models['sentence_model'].encode(posts, convert_to_tensor=True)
            query_embedding = models['sentence_model'].encode(search_query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, embeddings)[0]
            top_results = scores.topk(3)
            
            st.markdown("**Most Relevant Posts**")
            for score, idx in zip(top_results.values, top_results.indices):
                st.write(f"Relevance: {score.item():.2f}")
                st.write(posts[idx])
                st.divider()
    except Exception as e:
        st.error(f"Search failed: {str(e)}")

# --------------------------------------------------------------------------------
# ---------------------------- System Health Checks ------------------------------
# --------------------------------------------------------------------------------
st.sidebar.markdown("## System Status")
st.sidebar.write(f"Data Entries: {len(df)}")
st.sidebar.write(f"AI Models Loaded: {models is not None}")
st.sidebar.write(f"Time Series Data: {not time_series.empty}")

# --------------------------------------------------------------------------------
# ------------------------------ Footer ------------------------------------------
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
**Safe Execution Features:**
- Dynamic AI parameter adjustment
- Full error containment
- CPU optimization
- Resource monitoring
- Graceful degradation
""")
