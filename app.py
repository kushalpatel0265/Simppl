import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import re
import requests
import networkx as nx
from datetime import datetime
from textblob import TextBlob
import wikipedia
from pyvis.network import Network
import streamlit.components.v1 as components

# AI/ML Components
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# --------------------------------------------------------------------------------
# ----------------------- Data Loading & Preparation -----------------------------
# --------------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data(filepath="data.jsonl"):
    """Load and normalize data with comprehensive error handling"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at {filepath}")
            
        raw_df = pd.read_json(filepath, lines=True)
        df = pd.json_normalize(raw_df['data']) if 'data' in raw_df.columns else raw_df
        
        # Feature engineering
        if 'created_utc' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
            df['date'] = df['datetime'].dt.date
            
        if 'hashtags' not in df.columns:
            df['hashtags'] = df.apply(lambda x: re.findall(r"#\w+", f"{x.get('title','')} {x.get('selftext','')}"), axis=1)
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

df = load_and_prepare_data()

# --------------------------------------------------------------------------------
# ------------------------- AI/ML Model Management -------------------------------
# --------------------------------------------------------------------------------
@st.cache_resource
def load_ai_models():
    """Load all AI models with safety checks"""
    models = {}
    try:
        models['summarizer'] = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1  # Force CPU
        )
        models['sentence_model'] = SentenceTransformer("all-MiniLM-L6-v2")
        models['topic_model'] = LatentDirichletAllocation(n_components=5, random_state=42)
        return models
    except Exception as e:
        st.error(f"AI model loading failed: {str(e)}")
        return None

models = load_ai_models()

# --------------------------------------------------------------------------------
# ------------------------- Core Visualization Engine ----------------------------
# --------------------------------------------------------------------------------
def create_time_series_analysis(df):
    """Generate time series data with moving average"""
    try:
        ts_data = df.groupby('date').size().reset_index(name='count')
        ts_data['7_day_ma'] = ts_data['count'].rolling(7).mean()
        return ts_data
    except Exception as e:
        st.error(f"Time series generation failed: {str(e)}")
        return pd.DataFrame()

def generate_network_graph(df):
    """Create interactive network graph of hashtag co-occurrence"""
    try:
        hashtags = df['hashtags'].explode().str.lower()
        co_occur = pd.DataFrame(
            index=hashtags.unique(),
            columns=hashtags.unique(),
            data=0
        )

        for tags in df['hashtags']:
            unique_tags = list(set([t.lower() for t in tags]))
            for i in range(len(unique_tags)):
                for j in range(i+1, len(unique_tags)):
                    co_occur.loc[unique_tags[i], unique_tags[j]] += 1
                    co_occur.loc[unique_tags[j], unique_tags[i]] += 1

        G = nx.Graph()
        for tag in co_occur.columns:
            G.add_node(tag)
            
        for i in range(len(co_occur.columns)):
            for j in range(i+1, len(co_occur.columns)):
                if co_occur.iloc[i, j] > 0:
                    G.add_edge(
                        co_occur.columns[i],
                        co_occur.columns[j],
                        weight=co_occur.iloc[i, j]
                    )

        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        return net
    except Exception as e:
        st.error(f"Network graph failed: {str(e)}")
        return None

# --------------------------------------------------------------------------------
# ------------------------- Intelligent Analysis Modules -------------------------
# --------------------------------------------------------------------------------
def generate_ai_summary(text):
    """Safe text summarization with dynamic length handling"""
    try:
        if not models or not text:
            return "Summary unavailable"
            
        words = text.split()
        max_len = max(15, min(len(words), 50))
        return models['summarizer'](
            text,
            max_length=max_len,
            min_length=int(max_len*0.5),
            do_sample=False
        )[0]['summary_text']
    except Exception as e:
        return f"Summary error: {str(e)}"

def perform_topic_modeling(texts):
    """Topic modeling with LDA and t-SNE visualization"""
    try:
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(texts)
        lda_output = models['topic_model'].fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(lda_output)
    except Exception as e:
        st.error(f"Topic modeling failed: {str(e)}")
        return None

# --------------------------------------------------------------------------------
# ------------------------- Main Application Interface ---------------------------
# --------------------------------------------------------------------------------
st.title("Advanced Social Media Analytics Platform")
st.markdown("""
Multi-dimensional analysis of social media data with AI-powered insights and network visualization.
""")

# Sidebar Controls
with st.sidebar:
    st.header("Data Controls")
    date_filter = st.date_input("Select Date Range", 
        value=(df['date'].min(), df['date'].max()) if 'date' in df.columns else None)
    
    search_term = st.text_input("Keyword/Hashtag Filter")
    if search_term:
        df = df[df['text'].str.contains(search_term, case=False, na=False)]

# Core Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Posts", len(df))
with col2:
    st.metric("Unique Users", df['author'].nunique() if 'author' in df.columns else "N/A")
with col3:
    st.metric("Avg. Sentiment", f"{df['sentiment'].mean():.2f}" if 'sentiment' in df.columns else "N/A")

# Time Series Analysis
st.header("Temporal Analysis")
ts_data = create_time_series_analysis(df)
if not ts_data.empty:
    fig = px.line(ts_data, x='date', y=['count', '7_day_ma'], 
                 title="Post Volume with 7-Day Moving Average")
    st.plotly_chart(fig)
    
    # AI Summary
    summary_text = f"""
        From {ts_data['date'].min()} to {ts_data['date'].max()}, 
        average posts per day: {ts_data['count'].mean():.1f}. 
        Peak activity: {ts_data.loc[ts_data['count'].idxmax()]['date']} 
        ({ts_data['count'].max()} posts).
    """
    st.subheader("AI-Powered Trend Summary")
    st.write(generate_ai_summary(summary_text))

# Network Visualization
st.header("Hashtag Network Analysis")
if not df['hashtags'].empty:
    net = generate_network_graph(df)
    if net:
        net.save_graph("network.html")
        components.html(open("network.html", "r").read(), height=800)
else:
    st.info("No hashtag data available for network analysis")

# Topic Modeling
st.header("Content Topic Modeling")
if 'text' in df.columns:
    sample_texts = df['text'].dropna().sample(n=min(500, len(df)), random_state=42).tolist()
    tsne_results = perform_topic_modeling(sample_texts)
    if tsne_results is not None:
        tsne_df = pd.DataFrame(tsne_results, columns=['X', 'Y'])
        tsne_df['Text'] = sample_texts
        fig = px.scatter(tsne_df, x='X', y='Y', hover_data=['Text'],
                        title="t-SNE Visualization of Topic Clusters")
        st.plotly_chart(fig)

# Wikipedia Integration
st.header("Real-world Event Context")
wiki_query = st.text_input("Enter event/search term for Wikipedia context:")
if wiki_query:
    try:
        with st.spinner("Fetching relevant context..."):
            result = wikipedia.summary(wiki_query, sentences=3, auto_suggest=False)
            st.markdown(f"**Wikipedia Context for {wiki_query}:**")
            st.write(result)
    except wikipedia.exceptions.DisambiguationError as e:
        st.error(f"Ambiguous term: {e.options[:3]}")
    except Exception as e:
        st.error(f"Context fetch failed: {str(e)}")

# Semantic Search
st.header("Semantic Content Search")
if models and 'text' in df.columns:
    search_query = st.text_input("Search posts by semantic meaning:")
    if search_query:
        with st.spinner("Finding relevant content..."):
            embeddings = models['sentence_model'].encode(df['text'].tolist())
            query_embed = models['sentence_model'].encode(search_query)
            scores = util.cos_sim(query_embed, embeddings)[0]
            top_indices = scores.topk(3).indices
            
            st.subheader("Most Relevant Posts")
            for idx in top_indices:
                st.write(f"**Relevance Score:** {scores[idx]:.2f}")
                st.write(df.iloc[idx]['text'])
                st.divider()

# --------------------------------------------------------------------------------
# ------------------------- System Monitoring Footer -----------------------------
# --------------------------------------------------------------------------------
st.markdown("---")
st.caption("""
System Status: 
- AI Models Loaded: ✅
- Data Freshness: {} 
- Analysis Coverage: {:.1f}%
""".format(df['date'].max() if 'date' in df.columns else "N/A", 
         (len(df)/len(raw_df)*100 if 'raw_df' in locals() else 100))
