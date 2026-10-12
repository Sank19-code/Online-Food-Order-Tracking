"""
Shared data pipeline for the Online Food Order Tracking capstone.

`clean_and_engineer`, `build_preprocessor` and the column lists are copied
unchanged from the notebooks (regression / classification / clustering), so the
GUI and the training script use exactly the same features the models were
evaluated on. `make_order_frame` turns one order typed into the GUI into the
same engineered columns.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DELAY_THRESHOLD_MIN = 25.0          # SLA: an order taking > 25 min counts as "delayed"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CANDIDATES = [os.path.join(ROOT, 'data', 'train.csv'), os.path.join(ROOT, 'train.csv'),
                   'data/train.csv', 'train.csv', '../data/train.csv']

NUMERICAL_COLS = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 'prep_time_min',
                  'order_hour', 'Vehicle_condition', 'multiple_deliveries', 'Festival_binary',
                  'is_rush_hour', 'is_weekend']
CATEGORICAL_COLS = ['Weatherconditions', 'Road_traffic_density', 'City', 'Type_of_vehicle']
FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS
RUSH_HOURS = [12, 13, 14, 19, 20, 21]


def find_data_file():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'train.csv not found in any of {DATA_CANDIDATES}')


def load_raw_data():
    """Load train.csv from data/ or the repo root."""
    return pd.read_csv(find_data_file())


def haversine_dist(lon1, lat1, lon2, lat2):
    """Vectorised great-circle distance in km between two sets of coordinates."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean_and_engineer(df_raw, verbose=True):
    """Review 1 cleaning + feature-engineering pipeline (identical to the notebooks)."""
    df = df_raw.copy()

    # 1. Targets: numeric delivery time (regression) and SLA-breach flag (classification)
    df['Time_taken'] = df['Time_taken(min)'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['Is_Delayed'] = (df['Time_taken'] > DELAY_THRESHOLD_MIN).astype(int)

    # 2. Strip whitespace and turn hidden 'NaN' string tokens into real missing values
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().replace({'NaN': np.nan, 'nan': np.nan, '': np.nan})
    df['Weatherconditions'] = (df['Weatherconditions'].str.replace('conditions ', '', regex=False)
                               .replace({'NaN': np.nan, 'nan': np.nan}))

    # 3. Numeric casting and rating outliers (ratings > 5 are data-entry errors)
    for col in ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    n_rating_outliers = int((df['Delivery_person_Ratings'] > 5.0).sum())
    df.loc[df['Delivery_person_Ratings'] > 5.0, 'Delivery_person_Ratings'] = 5.0

    # 4. Haversine distance; invalid (near-zero) coordinates replaced by the median distance
    lat1, lon1 = df['Restaurant_latitude'].abs(), df['Restaurant_longitude'].abs()
    lat2, lon2 = df['Delivery_location_latitude'].abs(), df['Delivery_location_longitude'].abs()
    df['distance_km'] = haversine_dist(lon1, lat1, lon2, lat2)
    valid = (lat1 > 1.0) & (lon1 > 1.0) & (lat2 > 1.0) & (lon2 > 1.0)
    df.loc[~valid, 'distance_km'] = df.loc[valid, 'distance_km'].median()

    # 5. Kitchen preparation time (order placed -> order picked up), overnight wrap fixed
    t_order = pd.to_datetime(df['Order_Date'] + ' ' + df['Time_Orderd'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    t_picked = pd.to_datetime(df['Order_Date'] + ' ' + df['Time_Order_picked'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    prep = (t_picked - t_order).dt.total_seconds() / 60.0
    prep = prep.where(prep >= 0, prep + 1440.0)
    df['prep_time_min'] = prep.fillna(prep.median())

    # 6. Temporal features
    df['order_hour'] = t_order.dt.hour.fillna(17)
    df['is_rush_hour'] = df['order_hour'].isin(RUSH_HOURS).astype(int)
    order_date = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
    df['day_of_week'] = order_date.dt.dayofweek.fillna(2)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['Festival_binary'] = (df['Festival'] == 'Yes').astype(int)

    # 7. Ordinal traffic level (used by the clustering track)
    df['traffic_level'] = df['Road_traffic_density'].map({'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3})

    if verbose:
        print(f'Rows: {len(df):,} | rating outliers capped: {n_rating_outliers} | '
              f'coordinate anomalies fixed: {int((~valid).sum())} | duplicate IDs: {int(df["ID"].duplicated().sum())}')
    return df


def build_preprocessor():
    """Median-impute + scale numerics; mode-impute + one-hot categoricals. Fit on TRAIN only."""
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, NUMERICAL_COLS), ('cat', cat_pipe, CATEGORICAL_COLS)])


def get_feature_names(preprocessor):
    cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(CATEGORICAL_COLS)
    return NUMERICAL_COLS + list(cat_names)


def make_order_frame(rider_age, rider_rating, distance_km, prep_time_min, order_hour,
                     vehicle_condition, multiple_deliveries, festival, is_weekend,
                     weather, traffic, city_type, vehicle_type):
    """One GUI order -> a 1-row DataFrame with the same engineered columns as training."""
    row = {
        'Delivery_person_Age': float(rider_age),
        'Delivery_person_Ratings': min(float(rider_rating), 5.0),
        'distance_km': float(distance_km),
        'prep_time_min': float(prep_time_min),
        'order_hour': float(order_hour),
        'Vehicle_condition': int(vehicle_condition),
        'multiple_deliveries': float(multiple_deliveries),
        'Festival_binary': int(bool(festival)),
        'is_rush_hour': int(int(order_hour) in RUSH_HOURS),
        'is_weekend': int(bool(is_weekend)),
        'Weatherconditions': weather,
        'Road_traffic_density': traffic,
        'City': city_type,
        'Type_of_vehicle': vehicle_type,
    }
    return pd.DataFrame([row], columns=FEATURES)
