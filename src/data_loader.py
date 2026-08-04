import streamlit as st
import pandas as pd
import requests

from .constants import (
    CLUSTER_FEATURES, LFS_EXTRA_FEATURES, AMENITY_FEATURES, ALL_FEATURES,
    URL_HIES, URL_LFS, URL_AMENITIES, URL_GEOJSON,
)


@st.cache_data
def load_and_preprocess_data():
    response = requests.get(URL_HIES)
    if response.status_code != 200:
        st.error("Failed to fetch HIES data from OpenDOSM API.")
        return None, None, None

    df_hies = pd.DataFrame(response.json())
    df_hies['date'] = pd.to_datetime(df_hies['date'])

    response = requests.get(URL_LFS)
    if response.status_code != 200:
        st.error("Failed to fetch LFS data from OpenDOSM API.")
        return None, None, None

    df_lfs = pd.DataFrame(response.json())
    df_lfs['date'] = pd.to_datetime(df_lfs['date'])

    lfs_merge_cols = ['date', 'state', 'district', 'u_rate', 'p_rate'] + LFS_EXTRA_FEATURES
    df = pd.merge(
        df_hies,
        df_lfs[lfs_merge_cols],
        on=['date', 'state', 'district'],
        how='left'
    )

    all_numeric = CLUSTER_FEATURES + LFS_EXTRA_FEATURES
    df[all_numeric] = df[all_numeric].fillna(df[all_numeric].median())

    response = requests.get(URL_AMENITIES)
    if response.status_code == 200:
        df_amenities = pd.DataFrame(response.json())
        df_amenities['date'] = pd.to_datetime(df_amenities['date'])
        df_amenities = df_amenities[df_amenities['district'] != 'All Districts']
        df_amenities = df_amenities[['date', 'state', 'district'] + AMENITY_FEATURES]

        df = pd.merge(
            df,
            df_amenities,
            on=['date', 'state', 'district'],
            how='left'
        )
        df[AMENITY_FEATURES] = df[AMENITY_FEATURES].fillna(df[AMENITY_FEATURES].median())
    else:
        for col in AMENITY_FEATURES:
            df[col] = float('nan')

    available_years = sorted(df['date'].dt.year.unique(), reverse=True)

    response = requests.get(URL_GEOJSON)
    geo_json = response.json() if response.status_code == 200 else None

    return df, available_years, geo_json


def get_features_for_year(df_all, selected_year):
    df_year = df_all[df_all['date'].dt.year == selected_year].copy()

    present_features = [f for f in ALL_FEATURES if f in df_year.columns]
    df_year = df_year.dropna(subset=present_features)

    return df_year, present_features
