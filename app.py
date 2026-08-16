import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.preprocessing import load_iris_data, FEATURE_NAMES, SPECIES_NAMES
from src.predict import predict_species, load_saved_model_and_scaler
from src.train_model import get_classifiers, prepare_data

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nature/Iris Inspired Theme (Deep Purple, Soft Lavender, Modern Glassmorphism & Cards)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #fcfbfe;
    }
    
    /* Title Banner */
    .header-banner {
        background: linear-gradient(135deg, #4A154B 0%, #6B21A8 50%, #9333EA 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(107, 33, 168, 0.25);
    }
    
    .header-banner h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-banner p {
        color: #e9d5ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .stat-card {
        background: white;
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(147, 51, 234, 0.15);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #6B21A8;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Result Box */
    .prediction-card {
        background: linear-gradient(135deg, #FAF5FF 0%, #F3E8FF 100%);
        border: 2px solid #C084FC;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    .species-badge {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        color: #581C87;
        background: #ffffff;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        box-shadow: 0 4px 12px rgba(147, 51, 234, 0.2);
        margin: 1rem 0;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7E22CE;
    }
    
    /* Section Headings */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #3B0764;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def get_data():
    return load_iris_data()

# Cache model evaluation
@st.cache_data
def get_cached_evaluation():
    from src.train_model import train_and_evaluate_all
    return train_and_evaluate_all()

df, target_names = get_data()

# Header Banner
st.markdown("""
<div class="header-banner">
    <h1>🌸 Iris Flower Classification System</h1>
    <p>Predict Iris flower species (Setosa, Versicolor, Virginica) using Machine Learning algorithms.</p>
</div>
""", unsafe_allow_html=True)

# Key Dataset Statistics Summary Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-number">150</div><div class="stat-label">Total Samples</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-number">4</div><div class="stat-label">Physical Features</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-number">3</div><div class="stat-label">Species Classes</div></div>', unsafe_allow_html=True)
with c4:
    try:
        model, _, meta = load_saved_model_and_scaler()
        best_name = meta.get('best_model_name', 'Trained Classifier')
    except Exception:
        best_name = 'Logistic Regression'
    st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:1.3rem; padding-top:0.4rem">{best_name}</div><div class="stat-label">Active Classifier</div></div>', unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_predict, tab_analytics, tab_models, tab_dataset = st.tabs([
    "🔮 Species Predictor", 
    "📊 Data Visualizations", 
    "🤖 Model Performance & Comparison", 
    "ℹ️ Dataset Explorer"
])

# TAB 1: SPECIES PREDICTOR
with tab_predict:
    st.markdown('<div class="section-title">🌸 Enter Flower Measurements (in cm)</div>', unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.info("Adjust the sliders or type measurements below to classify the flower.")
        
        # Input controls
        sepal_length = st.slider(
            "Sepal Length (cm)", 
            min_value=4.0, max_value=8.0, value=5.1, step=0.1,
            help="Length of the flower sepal in centimeters."
        )
        sepal_width = st.slider(
            "Sepal Width (cm)", 
            min_value=2.0, max_value=4.5, value=3.5, step=0.1,
            help="Width of the flower sepal in centimeters."
        )
        petal_length = st.slider(
            "Petal Length (cm)", 
            min_value=1.0, max_value=7.0, value=1.4, step=0.1,
            help="Length of the flower petal in centimeters."
        )
        petal_width = st.slider(
            "Petal Width (cm)", 
            min_value=0.1, max_value=2.5, value=0.2, step=0.1,
            help="Width of the flower petal in centimeters."
        )
        
        predict_btn = st.button("✨ Predict Species", type="primary", width="stretch")

    with col_result:
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        try:
            res = predict_species(sepal_length, sepal_width, petal_length, petal_width)
            species = res['species_title']
            confidence = res['confidence']
            probs = res['probabilities']
            model_used = res['model_used']
            
            # Species icon map
            species_icons = {
                'Setosa': '🌺',
                'Versicolor': '🪻',
                'Virginica': '🌷'
            }
            icon = species_icons.get(species, '🌸')
            
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color: #64748b; font-weight: 600; font-size: 0.95rem;">PREDICTED SPECIES</div>
                <div class="species-badge">{icon} {species}</div>
                <div class="confidence-text">Prediction Confidence: {confidence:.2f}%</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">Classified using {model_used}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("**Species Probability Breakdown:**")
            for sp_name, prob_val in probs.items():
                st.write(f"**{sp_name}:** {prob_val:.2f}%")
                st.progress(prob_val / 100.0)
                
        except Exception as e:
            st.error(f"⚠ Prediction Error: {str(e)}")

# TAB 2: DATA VISUALIZATIONS
with tab_analytics:
    st.markdown('<div class="section-title">📊 Exploratory Data Visualizations</div>', unsafe_allow_html=True)
    
    viz_choice = st.radio(
        "Select Visualization:", 
        ["Feature Distributions (Boxplots)", "Scatter Matrix / Pairplot", "Correlation Heatmap"],
        horizontal=True
    )
    
    if viz_choice == "Feature Distributions (Boxplots)":
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        feature_cols = FEATURE_NAMES
        for idx, col in enumerate(feature_cols):
            ax = axes[idx // 2, idx % 2]
            sns.boxplot(data=df, x='species_name', y=col, ax=ax, hue='species_name', palette='Set2', legend=False)
            ax.set_title(f'{col.capitalize()} by Species', fontweight='bold')
            ax.set_xlabel('Species')
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_choice == "Scatter Matrix / Pairplot":
        g = sns.pairplot(df, hue='species_name', palette='Dark2', markers=['o', 's', 'D'])
        g.fig.suptitle("Pairwise Relationships across Iris Features", y=1.02, fontweight='bold')
        st.pyplot(g.fig)
        
    elif viz_choice == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df[FEATURE_NAMES].corr(), annot=True, cmap='Purples', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title("Iris Feature Correlation Matrix", fontweight='bold')
        st.pyplot(fig)

# TAB 3: MODEL PERFORMANCE & COMPARISON
with tab_models:
    st.markdown('<div class="section-title">🤖 Classifier Performance & Comparison</div>', unsafe_allow_html=True)
    
    try:
        # Load cached comparison
        results_df, trained_models, evaluation_details, meta_info = get_cached_evaluation()
        
        col_tbl, col_chart = st.columns([1, 1])
        
        with col_tbl:
            st.markdown("### Model Comparison Metrics")
            st.dataframe(
                results_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}'
                }).highlight_max(subset=['Accuracy', 'F1-Score'], color='#e9d5ff'),
                width="stretch"
            )
            st.success(f"🏆 Best Model: **{meta_info['best_model_name']}** based on test performance.")
            
        with col_chart:
            st.markdown("### Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Model', palette='Purples_r', legend=False, ax=ax)
            ax.set_ylim(0.8, 1.05)
            for idx, row in results_df.iterrows():
                ax.text(idx, row['Accuracy'] + 0.01, f"{row['Accuracy']*100:.1f}%", ha='center', fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig)
            
        st.markdown("---")
        st.markdown("### Confusion Matrices")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for idx, (m_name, m_details) in enumerate(evaluation_details.items()):
            ax = axes[idx // 2, idx % 2]
            sns.heatmap(m_details['confusion_matrix'], annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax,
                        xticklabels=SPECIES_NAMES, yticklabels=SPECIES_NAMES)
            ax.set_title(m_name, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Failed to render model metrics: {e}")

# TAB 4: DATASET EXPLORER
with tab_dataset:
    st.markdown('<div class="section-title">ℹ️ Iris Dataset Overview</div>', unsafe_allow_html=True)
    st.write("Explore raw dataset records, descriptive statistics, and missing value checks.")
    
    st.dataframe(df, width="stretch")
    
    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(), width="stretch")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Iris Flower Classification Application | Built with Streamlit, scikit-learn & Python"
    "</div>", 
    unsafe_allow_html=True
)


