# Malaysian District Socioeconomic Profiler

A dashboard that profiles Malaysian districts using unsupervised and supervised ML on live OpenDOSM census data. Groups ~160 districts into economic archetypes with K-Means, and predicts poverty, unemployment, and income brackets with regression and classification models.

Live Demo: https://malaysia-district-profiler-irvqp4zbpzyq99vk8fecp3.streamlit.app/

---

## What it does

The app fetches district-level Household Income & Expenditure Survey (HIES), Labour Force Survey (LFS), and Basic Amenities data from data.gov.my. **Two pages:**

1. **Main Dashboard** — K-Means clustering (7 indicators), PCA projection, correlation matrix, radar charts, district comparison, twin finder, time-series trends, and choropleth map.
2. **ML Lab** — Supervised ML models for poverty prediction, unemployment prediction, income bracket classification, economic vulnerability scoring, and feature importance analysis.

---

## File Structure

```
├── app.py                            (main dashboard: clustering + exploration)
├── pages/
│   └── 1_Machine_Learning.py          (ML Lab: prediction models)
├── src/
│   ├── constants.py                  (feature lists, labels, API URLs)
│   ├── data_loader.py                (API fetching, caching, merging)
│   ├── unsupervised.py               (K-Means, PCA, scaling)
│   └── supervised.py                 (regression, classification, vulnerability)
├── requirements.txt
└── README.md
```

---

## Features

### Main Dashboard
- **Live data** — pulls straight from the OpenDOSM API. No manual downloads.
- **12 indicators** — income, expenditure, inequality, poverty, labour force, employment-population ratio, and basic amenities.
- **K-Means clustering** — adjustable K (2-6), with elbow and silhouette charts.
- **PCA projection** — interactive 2D scatter plot colored by cluster.
- **Correlation matrix** — shows how indicators relate to each other.
- **Cluster profiles** — average values per cluster with CSV export.
- **Radar chart** — compare a district against its cluster average.
- **Compare mode** — pick two districts and see side-by-side profiles.
- **Twin finder** — top 3 most economically similar districts.
- **Trends over time** — state-level changes between survey years.
- **Choropleth map** — cluster assignments on a Malaysia district map.

### ML Lab
- **Poverty Rate Predictor** (Random Forest / Gradient Boosting / Linear Regression) — R², MAE, RMSE + actual vs predicted plot + feature importance + over/under-predicted districts.
- **Unemployment Rate Predictor** — same regression framework for unemployment.
- **Income Bracket Classifier** — classifies districts into Low (<RM4k), Middle (RM4k-8k), High (>RM8k) with confusion matrix and feature importance.
- **Economic Vulnerability Score** — composite risk index from model residuals; ranks all 160 districts from most to least vulnerable.
- **Feature Explorer** — global and state-level feature importance comparisons, plus per-district explanation of predictions.

---

## How to run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech

- Python 3.9+
- Streamlit (frontend + multi-page)
- Scikit-learn (K-Means, PCA, RandomForest, GradientBoosting, LogisticRegression)
- Plotly (charts and map)
- Pandas, NumPy
- Requests (API fetching)

Data sourced from the Department of Statistics Malaysia via the OpenDOSM API. District boundary data from the DOSM open data repository.
