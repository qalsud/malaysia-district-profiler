import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Malaysian District Socioeconomic Profiler",
    page_icon="🇲🇾",
    layout="wide"
)

# ---------------------------------------------------------
# 1. DATA LOADING & CACHING
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    """Fetches data from OpenDOSM API and returns raw & scaled DataFrames."""
    url = "https://api.data.gov.my/data-catalogue?id=hies_district&limit=500"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("Failed to fetch data from OpenDOSM API.")
        return None, None, None

    data = response.json()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    # Filter for latest survey
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()

    features = ['income_median', 'income_mean', 'expenditure_mean', 'gini', 'poverty']
    df_latest[features] = df_latest[features].fillna(df_latest[features].median())

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_latest[features])

    return df_latest, X_scaled, features

# Load data
df_latest, X_scaled, features = load_and_preprocess_data()

if df_latest is not None:
    # ---------------------------------------------------------
    # 2. SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.title("🇲🇾 Model Settings")
    st.sidebar.markdown("Configure parameters for the Unsupervised Clustering pipeline.")

    # Select K clusters
    chosen_k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=6, value=3, step=1)

    # ---------------------------------------------------------
    # 3. MACHINE LEARNING PIPELINE
    # ---------------------------------------------------------
    # K-Means Clustering
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df_latest['cluster'] = kmeans.fit_predict(X_scaled)

    # Business Labels (for default K=3)
    if chosen_k == 3:
        cluster_names = {
            1: "High-Income Metro Hubs",
            0: "Middle-Income Towns & Suburbs",
            2: "Rural & High-Poverty Districts"
        }
        df_latest['cluster_name'] = df_latest['cluster'].map(cluster_names)
    else:
        df_latest['cluster_name'] = df_latest['cluster'].apply(lambda x: f"Cluster {x}")

    # PCA 2D Dimensionality Reduction
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_scaled)
    df_latest['PC1'] = pca_transformed[:, 0]
    df_latest['PC2'] = pca_transformed[:, 1]
    var_explained = sum(pca.explained_variance_ratio_) * 100

    # ---------------------------------------------------------
    # 4. APP INTERFACE
    # ---------------------------------------------------------
    st.title("🇲🇾 Malaysian District Socioeconomic Profiler")
    st.markdown("""
    This web application uses **Unsupervised Machine Learning (K-Means & PCA)** on official census metrics 
    sourced directly from the **OpenDOSM API** to group administrative districts based on their economic DNA.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Districts Analyzed", len(df_latest))
    col2.metric("Selected Clusters (K)", chosen_k)
    col3.metric("PCA Variance Retained", f"{var_explained:.1f}%")

    st.markdown("---")

    # SECTION 1: PCA Plot
    st.subheader("1. Interactive District Cluster Map (2D PCA Projection)")
    
    fig = px.scatter(
        df_latest,
        x='PC1',
        y='PC2',
        color='cluster_name',
        hover_name='district',
        hover_data=['state', 'income_median', 'expenditure_mean', 'poverty'],
        labels={
            'PC1': 'Principal Component 1 (Economic Prosperity)', 
            'PC2': 'Principal Component 2 (Inequality & Spending Variance)',
            'cluster_name': 'District Profile'
        },
        template='plotly_white',
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    # SECTION 2: Cluster Profiles
    st.subheader("2. Cluster Profile Summary")
    
    profile_df = df_latest.groupby('cluster_name')[
        ['income_median', 'income_mean', 'expenditure_mean', 'gini', 'poverty']
    ].mean().reset_index()

    # Format numbers cleanly
    profile_df['income_median'] = profile_df['income_median'].apply(lambda x: f"RM {x:,.0f}")
    profile_df['income_mean'] = profile_df['income_mean'].apply(lambda x: f"RM {x:,.0f}")
    profile_df['expenditure_mean'] = profile_df['expenditure_mean'].apply(lambda x: f"RM {x:,.0f}")
    profile_df['poverty'] = profile_df['poverty'].apply(lambda x: f"{x:.2f}%")
    profile_df['gini'] = profile_df['gini'].apply(lambda x: f"{x:.3f}")

    st.dataframe(profile_df, use_container_width=True)

    # SECTION 3: Twin District Finder (CASCADING DROPDOWNS: STATE -> DISTRICT)
    st.markdown("---")
    st.subheader("3. 🔍 Twin District Finder")
    st.write("Select a State and District to find its top 3 most economically similar 'twins' in Malaysia based on Euclidean distance in scaled space.")

    # Two columns for cleaner layout
    select_col1, select_col2 = st.columns(2)

    with select_col1:
        state_list = sorted(df_latest['state'].unique())
        selected_state = st.selectbox("1. Select a State:", state_list)

    # Filter districts by the chosen state
    districts_in_state = sorted(df_latest[df_latest['state'] == selected_state]['district'].unique())

    with select_col2:
        selected_district = st.selectbox("2. Select a District:", districts_in_state)

    if selected_district:
        # Get target district index
        target_idx = df_latest[
            (df_latest['state'] == selected_state) & (df_latest['district'] == selected_district)
        ].index[0]
        
        row_pos = df_latest.index.get_loc(target_idx)
        
        # Calculate pairwise distance
        distances = pairwise_distances([X_scaled[row_pos]], X_scaled)[0]
        
        df_temp = df_latest.copy()
        df_temp['distance'] = distances
        
        # Get top 3 nearest matches (excluding itself)
        twins = df_temp[df_temp.index != target_idx].nsmallest(3, 'distance')
        
        st.write(f"**Top 3 Twin Districts for '{selected_district}, {selected_state}':**")
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(twins.iterrows()):
            with cols[i]:
                st.info(f"**#{i+1}: {row['district']} ({row['state']})**")
                st.write(f"**Profile:** {row['cluster_name']}")
                st.write(f"**Median Income:** RM {row['income_median']:,}")
                st.write(f"**Poverty Rate:** {row['poverty']:.1f}%")