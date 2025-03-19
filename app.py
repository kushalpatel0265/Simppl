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
# ----------------------- Memory-Optimized Data Loading --------------------------
# --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=1, ttl=3600)
def load_data_safely(filepath):
    """Load data with memory-efficient settings"""
    try:
        return pd.read_json(
            filepath,
            lines=True,
            dtype={
                'created_utc': 'int64',
                'author': 'category',
                'title': 'string',
                'selftext': 'string'
            }
        )
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# ------------------------- Memory Management System -----------------------------
# --------------------------------------------------------------------------------
def clean_memory():
    """Systematic memory cleanup"""
    gc.collect()
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()

# --------------------------------------------------------------------------------
# ------------------------- Core Analysis Functions ------------------------------
# --------------------------------------------------------------------------------
def analyze_basic_metrics(df, user_col):
    """Lightweight metric analysis"""
    metrics = {
        'total_posts': len(df),
        'unique_users': df[user_col].nunique() if user_col in df.columns else 0
    }
    return metrics

def create_time_series(df, timestamp_col):
    """Safe time series creation"""
    try:
        df["date"] = pd.to_datetime(df[timestamp_col], unit='s').dt.date
        return df.groupby("date").size().reset_index(name="count")
    except:
        return pd.DataFrame()

def analyze_topics(df, text_col):
    """Memory-safe topic analysis"""
    try:
        # Resource-constrained parameters
        sample_size = min(200, len(df))
        max_features = 500
        
        texts = df[text_col].dropna().sample(n=sample_size).tolist()
        vectorizer = CountVectorizer(
            stop_words='english',
            max_features=max_features
        )
        X = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(
            n_components=3,
            max_iter=5,
            random_state=42,
            batch_size=100
        )
        lda.fit(X)
        
        return {
            'vectorizer': vectorizer,
            'lda': lda,
            'feature_names': vectorizer.get_feature_names_out()
        }
    except Exception as e:
        st.error(f"Topic analysis failed: {str(e)}")
        return None
    finally:
        del texts, X
        clean_memory()

# --------------------------------------------------------------------------------
# ------------------------- Main Application Flow --------------------------------
# --------------------------------------------------------------------------------
def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Data loading section
    st.title("Social Media Analytics Dashboard")
    
    DATA_PATH = "data.jsonl"
    if os.path.exists(DATA_PATH):
        with st.spinner("Loading data safely..."):
            raw_df = load_data_safely(DATA_PATH)
            st.session_state.data_loaded = True
            
        # Data normalization
        if 'data' in raw_df.columns:
            df = pd.json_normalize(raw_df['data'])
        else:
            df = raw_df.copy()
            
        # Column setup
        timestamp_col = "created_utc"
        user_col = "author"
        text_col = "selftext" if "selftext" in df.columns else "title"
        
        # Preprocessing
        if "hashtags" not in df.columns:
            df["hashtags"] = df.apply(
                lambda x: re.findall(r"#\w+", str(x.get("title", "") + " " + str(x.get("selftext", ""))),
                axis=1
            )
            
        # Date filtering
        if timestamp_col in df.columns:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
                min_date = df[timestamp_col].min().date()
                max_date = df[timestamp_col].max().date()
                
                start_date, end_date = st.sidebar.date_input(
                    "Date range",
                    [min_date, max_date]
                )
                df = df[(df[timestamp_col].dt.date >= start_date) & 
                        (df[timestamp_col].dt.date <= end_date)]
            except:
                st.sidebar.warning("Date filtering unavailable")
        
        # Main visualizations
        st.header("Core Metrics")
        metrics = analyze_basic_metrics(df, user_col)
        col1, col2 = st.columns(2)
        col1.metric("Total Posts", metrics['total_posts'])
        col2.metric("Unique Users", metrics['unique_users'])
        
        st.header("Temporal Analysis")
        time_series = create_time_series(df, timestamp_col)
        if not time_series.empty:
            fig = px.line(time_series, x="date", y="count", 
                          title="Post Frequency Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        st.header("Content Analysis")
        if text_col in df.columns:
            with st.spinner("Analyzing content..."):
                topic_data = analyze_topics(df, text_col)
                if topic_data:
                    st.subheader("Top Topic Keywords")
                    for idx, topic in enumerate(topic_data['lda'].components_):
                        st.write(f"Topic {idx+1}:", ", ".join(
                            [topic_data['feature_names'][i] 
                             for i in topic.argsort()[:-6:-1]]
                        ))
        
        clean_memory()
        
    else:
        st.error("Data file not found")

if __name__ == "__main__":
    main()
