# Malaysian District Socioeconomic Profiler

An interactive web dashboard and unsupervised machine learning pipeline that categorizes ~160 administrative districts across Malaysia into socioeconomic archetypes using census data from OpenDOSM.

Live Demo: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## Overview

Analyzing regional economic variation usually involves digging through fragmented spreadsheets. This project automates district-level demographic profiling by ingesting live government census data and applying K-Means clustering and Principal Component Analysis (PCA) to discover underlying economic groupings without manual target labels.

The app also includes a site-selection feature ("Twin District Finder") that calculates vector distances between districts to identify regions with similar household income, spending, and poverty profiles.

---

## Features

- **Live Data Ingestion:** Fetches district-level Household Income and Expenditure Survey (HIES) records directly from the `data.gov.my` REST API.
- **Feature Normalization & Clustering:** Standardizes multi-variable features (`StandardScaler`) and groups districts into 3 distinct economic tiers via `K-Means`.
- **Dimensionality Reduction:** Compresses 5 economic variables into 2 principal components (`PCA`) to display clusters on an interactive 2D Plotly scatter plot.
- **Twin District Finder:** Uses pairwise Euclidean distance in scaled space to recommend the top 3 most economically similar districts for any chosen region (filtered by state and district).

---

## Tech Stack

- **Language:** Python 3.9+
- **Frontend / Dashboard:** Streamlit, Plotly Express
- **Machine Learning & Stats:** Scikit-Learn (`StandardScaler`, `KMeans`, `PCA`), NumPy
- **Data Engineering:** Pandas, Requests (REST API fetching)

---

## Methodology & Pipeline
