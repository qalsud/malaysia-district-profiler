CLUSTER_FEATURES = [
    'income_median', 'income_mean', 'expenditure_mean',
    'gini', 'poverty', 'u_rate', 'p_rate'
]

LFS_EXTRA_FEATURES = ['ep_ratio', 'lf_outside']

AMENITY_FEATURES = ['electricity', 'piped_water', 'sanitation']

ALL_FEATURES = CLUSTER_FEATURES + LFS_EXTRA_FEATURES

METRIC_LABELS = {
    'income_median': 'Median Income (RM)',
    'income_mean': 'Mean Income (RM)',
    'expenditure_mean': 'Mean Expenditure (RM)',
    'gini': 'Gini Coefficient',
    'poverty': 'Poverty Rate (%)',
    'u_rate': 'Unemployment Rate (%)',
    'p_rate': 'Participation Rate (%)',
    'ep_ratio': 'Employment-Population Ratio',
    'lf_outside': 'Labour Force Outside (thousands)',
    'electricity': 'Electricity Access (%)',
    'piped_water': 'Piped Water Access (%)',
    'sanitation': 'Sanitation Access (%)',
}

DISTRICT_NAME_MAP = {
    "Larut & Matang": "Larut Dan Matang",
    "S.P. Selatan": "Seberang Perai Selatan",
    "S.P.Tengah": "Seberang Perai Tengah",
    "S.P.Utara": "Seberang Perai Utara",
    "Hulu": "Hulu Terengganu",
    "Lubok antu": "Lubok Antu",
}

CLUSTER_NAMES_K3 = {
    1: "High-Income Metro Hubs",
    0: "Middle-Income Towns & Suburbs",
    2: "Rural & High-Poverty Districts",
}

URL_HIES = "https://api.data.gov.my/data-catalogue?id=hies_district&limit=500"
URL_LFS = "https://api.data.gov.my/data-catalogue?id=lfs_district&limit=500"
URL_AMENITIES = "https://api.data.gov.my/data-catalogue?id=hh_access_amenities&limit=500"
URL_GEOJSON = "https://raw.githubusercontent.com/atifmustaffa/malaysia-geojson/refs/heads/master/malaysia.district.min.geojson"

INCOME_BRACKETS = [
    ("Low", 0, 4000),
    ("Middle", 4000, 8000),
    ("High", 8000, float("inf")),
]
