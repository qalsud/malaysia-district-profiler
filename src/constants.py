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

# District names are not consistent across the HIES, LFS and amenities
# datasets. Normalise them before merging so the same district is not
# silently treated as missing.
DISTRICT_ALIASES = {
    "Larut Dan Matang": "Larut & Matang",
    "Larut dan Matang": "Larut & Matang",
    "S.P. Selatan": "Seberang Perai Selatan",
    "S.P.Tengah": "Seberang Perai Tengah",
    "S.P.Utara": "Seberang Perai Utara",
    "Hulu": "Hulu Terengganu",
    "Lubok antu": "Lubok Antu",
    "Ulu Langat": "Hulu Langat",
    "Ulu Selangor": "Hulu Selangor",
    "Tanjong Manis": "Tanjung Manis",
}

# Maps the normalised names to the names used by the GeoJSON file.
DISTRICT_NAME_MAP = {
    "Larut & Matang": "Larut Dan Matang",
    "Hulu Langat": "Ulu Langat",
    "Hulu Selangor": "Ulu Selangor",
}

# The LFS and amenities endpoints contain more than 500 rows, so a limit of
# 500 returns only part of the data and leaves most districts missing.
URL_HIES = "https://api.data.gov.my/data-catalogue?id=hies_district&limit=5000"
URL_LFS = "https://api.data.gov.my/data-catalogue?id=lfs_district&limit=5000"
URL_AMENITIES = "https://api.data.gov.my/data-catalogue?id=hh_access_amenities&limit=5000"
URL_GEOJSON = "https://raw.githubusercontent.com/atifmustaffa/malaysia-geojson/refs/heads/master/malaysia.district.min.geojson"

INCOME_BRACKETS = [
    ("Low", 0, 4000),
    ("Middle", 4000, 8000),
    ("High", 8000, float("inf")),
]
