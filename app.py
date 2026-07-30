import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

st.set_page_config(
    page_title="Malaysian District Socioeconomic Profiler",
    page_icon=":material/map:",
    layout="wide"
)

# ---------------------------------------------------------
# 1. DATA LOADING & CACHING
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    url_hies = "https://api.data.gov.my/data-catalogue?id=hies_district&limit=500"
    response = requests.get(url_hies)
    if response.status_code != 200:
        st.error("Failed to fetch HIES data from OpenDOSM API.")
        return None, None, None, None

    df_hies = pd.DataFrame(response.json())
    df_hies['date'] = pd.to_datetime(df_hies['date'])

    url_lfs = "https://api.data.gov.my/data-catalogue?id=lfs_district&limit=500"
    response = requests.get(url_lfs)
    if response.status_code != 200:
        st.error("Failed to fetch LFS data from OpenDOSM API.")
        return None, None, None, None

    df_lfs = pd.DataFrame(response.json())
    df_lfs['date'] = pd.to_datetime(df_lfs['date'])

    df = pd.merge(
        df_hies,
        df_lfs[['date', 'state', 'district', 'u_rate', 'p_rate']],
        on=['date', 'state', 'district'],
        how='left'
    )

    features = ['income_median', 'income_mean', 'expenditure_mean', 'gini', 'poverty', 'u_rate', 'p_rate']
    df[features] = df[features].fillna(df[features].median())

    available_years = sorted(df['date'].dt.year.unique(), reverse=True)

    url_geo = "https://raw.githubusercontent.com/atifmustaffa/malaysia-geojson/refs/heads/master/malaysia.district.min.geojson"
    response = requests.get(url_geo)
    geo_json = response.json() if response.status_code == 200 else None

    return df, available_years, features, geo_json


df_all, available_years, features, geo_json = load_and_preprocess_data()

if df_all is not None:
    # ---------------------------------------------------------
    # 2. SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.title(":material/map: Model Settings")
    st.sidebar.markdown("Configure parameters for the Unsupervised Clustering pipeline.")

    selected_year = st.sidebar.selectbox("Survey Year", available_years)
    chosen_k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=6, value=3, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### :material/download: Export")

    # Filter to selected year
    df_latest = df_all[df_all['date'].dt.year == selected_year].copy()

    # ---------------------------------------------------------
    # 3. MACHINE LEARNING PIPELINE
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_latest[features])

    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df_latest['cluster'] = kmeans.fit_predict(X_scaled)

    if chosen_k == 3:
        cluster_names = {
            1: "High-Income Metro Hubs",
            0: "Middle-Income Towns & Suburbs",
            2: "Rural & High-Poverty Districts"
        }
        df_latest['cluster_name'] = df_latest['cluster'].map(cluster_names)
    else:
        df_latest['cluster_name'] = df_latest['cluster'].apply(lambda x: f"Cluster {x}")

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_scaled)
    df_latest['PC1'] = pca_transformed[:, 0]
    df_latest['PC2'] = pca_transformed[:, 1]
    var_explained = sum(pca.explained_variance_ratio_) * 100

    metric_labels = {
        'income_median': 'Median Income (RM)',
        'income_mean': 'Mean Income (RM)',
        'expenditure_mean': 'Mean Expenditure (RM)',
        'gini': 'Gini Coefficient',
        'poverty': 'Poverty Rate (%)',
        'u_rate': 'Unemployment Rate (%)',
        'p_rate': 'Participation Rate (%)'
    }

    csv_all = df_latest[['state', 'district', 'cluster', 'cluster_name'] + features].to_csv(index=False)
    st.sidebar.download_button(":material/download: Download All Districts", csv_all, "all_districts.csv")

    DISTRICT_NAME_MAP = {
        "Larut & Matang": "Larut Dan Matang",
        "S.P. Selatan": "Seberang Perai Selatan",
        "S.P.Tengah": "Seberang Perai Tengah",
        "S.P.Utara": "Seberang Perai Utara",
        "Hulu": "Hulu Terengganu",
        "Lubok antu": "Lubok Antu",
    }

    # ---------------------------------------------------------
    # 4. APP INTERFACE
    # ---------------------------------------------------------
    st.title(":material/map: Malaysian District Socioeconomic Profiler")
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
    st.markdown("Each point is a district. Districts closer together have similar economic profiles. The two axes compress 7 economic indicators into 2 dimensions for visualization.")

    fig = px.scatter(
        df_latest,
        x='PC1',
        y='PC2',
        color='cluster_name',
        hover_name='district',
        hover_data=['state', 'income_median', 'expenditure_mean', 'poverty', 'u_rate', 'p_rate'],
        labels={
            'PC1': 'PC1: Economic Prosperity (Income & Spending)',
            'PC2': 'PC2: Labour Market Health (Unemployment & Participation)',
            'cluster_name': 'District Profile'
        },
        template='plotly_white',
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    # SECTION 2: Correlation Heatmap
    st.markdown("---")
    st.subheader("2. Feature Correlation Matrix")
    st.markdown("Shows how strongly pairs of indicators move together. Darker red means a strong positive correlation, darker blue means a strong negative one. Use this to understand trade-offs between metrics.")

    corr_matrix = df_latest[features].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # SECTION 3: Cluster Profiles
    st.markdown("---")
    st.subheader("3. Cluster Profile Summary")
    st.markdown("Average values of each indicator within each cluster. Compare how the archetypes differ across income, poverty, unemployment, and more.")

    profile_df = df_latest.groupby('cluster_name')[
        ['income_median', 'income_mean', 'expenditure_mean', 'gini', 'poverty', 'u_rate', 'p_rate']
    ].mean().reset_index()

    profile_df['income_median'] = profile_df['income_median'].apply(lambda x: f"RM {x:,.0f}")
    profile_df['income_mean'] = profile_df['income_mean'].apply(lambda x: f"RM {x:,.0f}")
    profile_df['expenditure_mean'] = profile_df['expenditure_mean'].apply(lambda x: f"RM {x:,.0f}")
    profile_df['poverty'] = profile_df['poverty'].apply(lambda x: f"{x:.2f}%")
    profile_df['gini'] = profile_df['gini'].apply(lambda x: f"{x:.3f}")
    profile_df['u_rate'] = profile_df['u_rate'].apply(lambda x: f"{x:.1f}%")
    profile_df['p_rate'] = profile_df['p_rate'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(profile_df, use_container_width=True)
    csv_profile = profile_df.to_csv(index=False)
    st.download_button(":material/download: Download Profile Summary", csv_profile, "cluster_profiles.csv")

    with st.expander("District Radar Profile"):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            state_list = sorted(df_latest['state'].unique())
            radar_state = st.selectbox("Select State", state_list, key="radar_state")
        districts_in_state = sorted(df_latest[df_latest['state'] == radar_state]['district'].unique())
        with col_r2:
            radar_district = st.selectbox("Select District", districts_in_state, key="radar_district")

        if radar_district:
            idx = df_latest[(df_latest['state'] == radar_state) & (df_latest['district'] == radar_district)].index[0]
            row_pos = df_latest.index.get_loc(idx)

            district_vals = X_scaled[row_pos]
            cluster_label = df_latest.loc[idx, 'cluster']
            centroid_vals = kmeans.cluster_centers_[cluster_label]
            cluster_name = df_latest.loc[idx, 'cluster_name']

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=district_vals,
                theta=[metric_labels[f] for f in features],
                fill='toself',
                name=f'{radar_district}'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=centroid_vals,
                theta=[metric_labels[f] for f in features],
                fill='toself',
                name=f'{cluster_name} avg',
                line=dict(dash='dash')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                template='plotly_white',
                height=500,
                title=f'{radar_district}, {radar_state} vs {cluster_name} Average'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Values shown are standardized z-scores. Positive = above district average, negative = below.")

    with st.expander("Compare Two Districts"):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            state_a = st.selectbox("District A — Select State", sorted(df_latest['state'].unique()), key="state_a")
        dists_a = sorted(df_latest[df_latest['state'] == state_a]['district'].unique())
        with col_c2:
            dist_a = st.selectbox("District A — Select District", dists_a, key="dist_a")

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            state_b = st.selectbox("District B — Select State", sorted(df_latest['state'].unique()), key="state_b")
        dists_b = sorted(df_latest[df_latest['state'] == state_b]['district'].unique())
        with col_c4:
            dist_b = st.selectbox("District B — Select District", dists_b, key="dist_b")

        if dist_a and dist_b:
            if state_a == state_b and dist_a == dist_b:
                st.warning("Select two different districts to compare.")
            else:
                idx_a = df_latest[(df_latest['state'] == state_a) & (df_latest['district'] == dist_a)].index[0]
                idx_b = df_latest[(df_latest['state'] == state_b) & (df_latest['district'] == dist_b)].index[0]
                row_a = df_latest.index.get_loc(idx_a)
                row_b = df_latest.index.get_loc(idx_b)

                vals_a = X_scaled[row_a]
                vals_b = X_scaled[row_b]
                cluster_a = df_latest.loc[idx_a, 'cluster_name']
                cluster_b = df_latest.loc[idx_b, 'cluster_name']
                centroid_a = kmeans.cluster_centers_[df_latest.loc[idx_a, 'cluster']]
                centroid_b = kmeans.cluster_centers_[df_latest.loc[idx_b, 'cluster']]

                theta_labels = [metric_labels[f] for f in features]
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatterpolar(
                    r=vals_a, theta=theta_labels, fill='toself',
                    name=f'{dist_a} ({cluster_a})'
                ))
                fig_compare.add_trace(go.Scatterpolar(
                    r=centroid_a, theta=theta_labels, fill='none',
                    name=f'{cluster_a} avg', line=dict(dash='dash')
                ))
                fig_compare.add_trace(go.Scatterpolar(
                    r=vals_b, theta=theta_labels, fill='toself',
                    name=f'{dist_b} ({cluster_b})'
                ))
                fig_compare.add_trace(go.Scatterpolar(
                    r=centroid_b, theta=theta_labels, fill='none',
                    name=f'{cluster_b} avg', line=dict(dash='dash')
                ))
                fig_compare.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    template='plotly_white', height=500,
                    title=f'{dist_a} vs {dist_b}'
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                row_a_raw = df_latest.loc[idx_a]
                row_b_raw = df_latest.loc[idx_b]
                compare_data = []
                for f in features:
                    label = metric_labels[f]
                    val_a = row_a_raw[f]
                    val_b = row_b_raw[f]
                    if f in ('income_median', 'income_mean', 'expenditure_mean'):
                        compare_data.append([label, f'RM {val_a:,.0f}', f'RM {val_b:,.0f}'])
                    elif f in ('poverty', 'u_rate', 'p_rate'):
                        compare_data.append([label, f'{val_a:.1f}%', f'{val_b:.1f}%'])
                    elif f == 'gini':
                        compare_data.append([label, f'{val_a:.3f}', f'{val_b:.3f}'])
                compare_df = pd.DataFrame(compare_data, columns=['Metric', dist_a, dist_b])
                st.dataframe(compare_df, use_container_width=True)

    # SECTION 4: Twin District Finder
    st.markdown("---")
    st.subheader("4. :material/search_globe: Twin District Finder")
    st.write("Select a State and District to find its top 3 most economically similar 'twins' in Malaysia based on Euclidean distance in scaled space.")

    select_col1, select_col2 = st.columns(2)

    with select_col1:
        state_list = sorted(df_latest['state'].unique())
        selected_state = st.selectbox("1. Select a State:", state_list)

    districts_in_state = sorted(df_latest[df_latest['state'] == selected_state]['district'].unique())

    with select_col2:
        selected_district = st.selectbox("2. Select a District:", districts_in_state)

    if selected_district:
        target_idx = df_latest[
            (df_latest['state'] == selected_state) & (df_latest['district'] == selected_district)
        ].index[0]

        row_pos = df_latest.index.get_loc(target_idx)

        distances = pairwise_distances([X_scaled[row_pos]], X_scaled)[0]

        df_temp = df_latest.copy()
        df_temp['distance'] = distances

        twins = df_temp[df_temp.index != target_idx].nsmallest(3, 'distance')

        st.write(f"**Top 3 Twin Districts for '{selected_district}, {selected_state}':**")

        cols = st.columns(3)
        for i, (_, row) in enumerate(twins.iterrows()):
            with cols[i]:
                st.info(f"**#{i+1}: {row['district']} ({row['state']})**")
                st.write(f"**Profile:** {row['cluster_name']}")
                st.write(f"**Median Income:** RM {row['income_median']:,.0f}")
                st.write(f"**Poverty Rate:** {row['poverty']:.1f}%")
                st.write(f"**Unemployment:** {row['u_rate']:.1f}%")
                st.write(f"**Participation:** {row['p_rate']:.1f}%")

        twin_cols = ['district', 'state', 'cluster_name', 'income_median', 'poverty', 'u_rate', 'p_rate']
        csv_twins = twins[twin_cols].to_csv(index=False)
        st.download_button(":material/download: Download Twin Results", csv_twins, "twin_results.csv")

    # SECTION 5: Trends Over Time
    st.markdown("---")
    st.subheader("5. :material/trending_up: Trends Over Time")
    st.markdown("See how economic indicators have changed between survey years across states. Select states and a metric to compare trajectories.")

    trend_df = df_all.copy()
    trend_df['year'] = trend_df['date'].dt.year
    trend_agg = trend_df.groupby(['state', 'year'])[features].mean().reset_index()

    state_options = sorted(trend_agg['state'].unique())
    selected_states = st.multiselect(
        "Select States to Display",
        state_options,
        default=state_options[:5]
    )

    trend_metric = st.selectbox(
        "Select Metric",
        options=list(metric_labels.keys()),
        format_func=lambda k: metric_labels[k],
        index=0
    )

    if selected_states:
        filtered = trend_agg[trend_agg['state'].isin(selected_states)]
        fig_trend = px.line(
            filtered,
            x='year',
            y=trend_metric,
            color='state',
            markers=True,
            template='plotly_white',
            height=500,
            labels={'year': 'Year', trend_metric: metric_labels[trend_metric], 'state': 'State'}
        )
        fig_trend.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Select at least one state to view trends.")

    st.markdown("---")
    st.subheader("6. :material/map: Malaysia District Map (Choropleth)")
    st.markdown("Districts are colored by their cluster assignment. Hover over a district to see its key metrics.")

    if geo_json is not None:
        plot_data = df_latest.copy()
        plot_data['geo_name'] = plot_data['district'].map(DISTRICT_NAME_MAP).fillna(plot_data['district'])

        geo_names = set(f['properties']['name'] for f in geo_json['features'])
        matched_count = plot_data['geo_name'].isin(geo_names).sum()
        unmatched_count = len(plot_data) - matched_count

        fig_map = px.choropleth_map(
            plot_data, geojson=geo_json,
            locations='geo_name', featureidkey='properties.name',
            color='cluster_name', hover_name='district',
            hover_data=['state', 'income_median', 'poverty'],
            map_style="open-street-map",
            zoom=4.5, center={"lat": 4.5, "lon": 109.0},
            opacity=0.6, height=650,
            labels={'cluster_name': 'District Profile'}
        )
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

        if unmatched_count > 0:
            st.caption(f"{unmatched_count} district(s) could not be mapped (not found in boundary data).")
    else:
        st.warning("GeoJSON boundary data could not be loaded. Map unavailable.")

    st.markdown("---")
    st.markdown(
        "**Data sources:** "
        "[HIES District](https://api.data.gov.my/data-catalogue?id=hies_district) | "
        "[LFS District](https://api.data.gov.my/data-catalogue?id=lfs_district) | "
        "[District Boundaries](https://github.com/atifmustaffa/malaysia-geojson) | "
        "[OpenDOSM](https://open.dosm.gov.my)"
    )
