# Malaysian District Socioeconomic Profiler

A dashboard that groups Malaysian districts by their economic profile. It pulls census data from the OpenDOSM API — household income, spending, poverty, inequality, and labour force stats — then runs K-Means clustering to find patterns across ~160 districts.

Live Demo: https://malaysia-district-profiler-irvqp4zbpzyq99vk8fecp3.streamlit.app/

---

## What it does

The app fetches district-level Household Income & Expenditure Survey (HIES) and Labour Force Survey (LFS) data from data.gov.my. Seven economic indicators go into a K-Means model that sorts districts into clusters. PCA reduces everything to 2D so you can see the groupings on a scatter plot.

There's also a "Twin District Finder" that measures how similar any two districts are, a radar chart to compare a district against its cluster average, and a map that shows where each cluster sits geographically.

---

## Features

- **Live data** — pulls straight from the OpenDOSM API. No manual downloads.
- **7 indicators** — median and mean income, mean expenditure, Gini coefficient, poverty rate, unemployment rate, and labour force participation.
- **K-Means clustering** — adjustable K (2-6), with elbow and silhouette charts to help you pick the right number.
- **PCA projection** — interactive 2D scatter plot colored by cluster.
- **Correlation matrix** — shows how the 7 indicators relate to each other.
- **Cluster profiles** — average values per cluster with download as CSV.
- **Radar chart** — pick a district and see its standardized scores vs its cluster average.
- **Compare mode** — pick two districts and see their profiles side by side.
- **Twin finder** — pick any district and get its top 3 most similar matches.
- **Trends over time** — compare state-level changes between survey years (2022 vs 2024).
- **Choropleth map** — shows cluster assignments on a map of Malaysia.
- **Export** — download district assignments, cluster profiles, and twin results as CSV.

---

## How to run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech

- Python 3.9+
- Streamlit (frontend)
- Scikit-learn (K-Means, PCA, StandardScaler)
- Plotly (charts and map)
- Pandas, NumPy
- Requests (API fetching)

Data sourced from the Department of Statistics Malaysia via the OpenDOSM API. District boundary data from the DOSM open data repository.
