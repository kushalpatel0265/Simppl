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
# ----------------------- Optimized Data Loading ----------------------------------
# --------------------------------------------------------------------------------
@st.cache_data(max_entries=1, show_spinner=False)
def load_and_preprocess_data(filepath):
    """Load data with memory optimization"""
    try:
        # Load in chunks if file is large
        chunk_reader = pd.read_json(filepath, lines=True, chunksize=1000)
        raw_df = pd.concat(chunk for chunk in chunk_reader)
        
        # Optimize memory usage
        for col in raw_df.columns:
            if raw_df[col].dtype == 'object':
                raw_df[col] = raw_df[col].astype('category')
        
        # Normalize nested data
        if 'data' in raw_df.columns:
            df = pd.json_normalize(raw_df['data'])
            raw_df = pd.concat([raw_df.drop('data', axis=1), df], axis=1)
            
        return raw_df
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# ----------------------- Main App with Resource Management ----------------------
# --------------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_icon="📊")
    
    # Load data
    DATA_PATH = "data.jsonl"
    if not os.path.exists(DATA_PATH):
        st.error("Missing data.jsonl file")
        return
    
    with st.spinner("Optimizing data loading..."):
        df = load_and_preprocess_data(DATA_PATH)
    
    # ----------------------------------------------------------------------------
    # ------------------------- Safe Data Filtering ------------------------------
    # ----------------------------------------------------------------------------
    st.sidebar.header("Filters")
    
    # Date filter
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors='coerce', unit='s')
        df = df.dropna(subset=["created_utc"])
        
        min_date = df["created_utc"].min().date()
        max_date = df["created_utc"].max().date()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[
                (df["created_utc"].dt.date >= date_range[0]) & 
                (df["created_utc"].dt.date <= date_range[1])
            ]

    # Text filter
    search_term = st.sidebar.text_input("Search posts:")
    if search_term:
        df = df[df["selftext"].str.contains(search_term, case=False, na=False)]

    # ----------------------------------------------------------------------------
    # -------------------- Memory-Optimized Visualizations -----------------------
    # ----------------------------------------------------------------------------
    st.title("Social Media Analytics Dashboard")
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Posts", len(df))
        
    with col2:
        if "author" in df.columns:
            st.metric("Unique Users", df["author"].nunique())

    # Time series plot
    if "created_utc" in df.columns:
        with st.expander("Posts Over Time", expanded=True):
            time_series = df.set_index("created_utc").resample('D').size()
            fig = px.line(time_series, title="Daily Post Count")
            st.plotly_chart(fig, use_container_width=True)

    # Sentiment analysis
    if "selftext" in df.columns:
        with st.expander("Sentiment Analysis", expanded=True):
            # Process in chunks
            sentiment_chunks = []
            chunk_size = 500
            
            for i in range(0, len(df), chunk_size):
                chunk = df["selftext"].iloc[i:i+chunk_size]
                sentiment_chunks.extend([
                    TextBlob(text).sentiment.polarity 
                    for text in chunk if isinstance(text, str)
                ])
                
            fig = px.histogram(
                x=sentiment_chunks, 
                nbins=30, 
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Topic modeling (safer implementation)
    if "selftext" in df.columns:
        with st.expander("Topic Modeling", expanded=False):
            # Limit to 1000 posts for stability
            sample_texts = df["selftext"].dropna().sample(
                n=min(1000, len(df)), 
                random_state=42
            ).tolist()
            
            with st.spinner("Processing topics (this may take a minute)..."):
                try:
                    vectorizer = CountVectorizer(
                        stop_words='english', 
                        max_features=500  # Reduced features
                    )
                    X = vectorizer.fit_transform(sample_texts)
                    
                    lda = LatentDirichletAllocation(
                        n_components=3,  # Fewer components
                        learning_method='online',  # Memory-efficient
                        random_state=42
                    )
                    lda.fit(X)
                    
                    # Reduced t-SNE dimensions
                    tsne = TSNE(n_components=2, perplexity=15)
                    embeddings = tsne.fit_transform(lda.transform(X))
                    
                    fig = px.scatter(
                        x=embeddings[:, 0], 
                        y=embeddings[:, 1],
                        title="Topic Clusters"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Topic modeling failed: {str(e)}")

if __name__ == "__main__":
    main()
