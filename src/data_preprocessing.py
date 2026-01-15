import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import config

def fetch_raw_data():
    """
    Loads the standard French Motor Third-Party Liability datasets.
    """
    print("Downloading dataset from OpenML...")
    freq = fetch_openml(data_id=config.OPENML_FREQUENCY_DATA_ID, as_frame=True, parser='auto').frame
    sev = fetch_openml(data_id=config.OPENML_SEVERITY_DATA_ID, as_frame=True, parser='auto').frame
    
    # Standardise ID columns
    freq['IDpol'] = freq['IDpol'].astype(int)
    sev['IDpol'] = sev['IDpol'].astype(int)

    return freq, sev

def preprocess_data(df_freq, df_sev):
    """
    Merges Frequency and Severity data and performs basic cleaning.
    """
    # 1. Aggregate Severity
    sev_agg = df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index()
    sev_agg.rename(columns={'ClaimAmount': 'TotalLoss'}, inplace=True)

    # 2. Merge with Frequency Data
    df = pd.merge(df_freq, sev_agg, on='IDpol', how='left')

    # 3. RENAME COLUMNS (The Fix)
    # The raw data uses 'DrivAge', but our scripts expect 'DriverAge'
    df.rename(columns={'DrivAge': 'DriverAge'}, inplace=True)

    # 4. Fill Missing Values
    df['TotalLoss'] = df['TotalLoss'].fillna(0)
    
    # Calculate Average Severity
    df['AvgSeverity'] = np.where(
        df['ClaimNb'] > 0, 
        df['TotalLoss'] / df['ClaimNb'], 
        0
    )

    # 5. Standard Filtering
    df = df[df['Exposure'] > config.MIN_EXPOSURE].copy()
    df['Exposure'] = df['Exposure'].clip(upper=config.MAX_EXPOSURE)
    df = df[df['ClaimNb'] < config.MAX_CLAIM_NB].copy()

    return df

def bin_features(df):
    """
    Applies standard actuarial binning.
    """
    df = df.copy()

    # --- 1. Driver Age ---
    # Now this will work because we renamed 'DrivAge' to 'DriverAge' above
    df['DriverAge_Bin'] = pd.cut(
        df['DriverAge'], 
        bins=config.DRIVER_AGE_BINS, 
        labels=config.DRIVER_AGE_LABELS,
        include_lowest=True
    )

    # --- 2. Vehicle Age ---
    df['VehAge_Bin'] = pd.cut(
        df['VehAge'], 
        bins=config.VEHICLE_AGE_BINS, 
        labels=config.VEHICLE_AGE_LABELS
    )

    # --- 3. Vehicle Power ---
    df['VehPower_Bin'] = df['VehPower'].apply(lambda x: f'{x}' if x < config.VEHICLE_POWER_THRESHOLD else '10+')

    # --- 4. Convert Categoricals to Strings ---
    categorical_cols = ['VehBrand', 'VehGas', 'Region', 'Area']
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # ... inside bin_features ...

    # --- 5. Group Small Brands (Force B12 to disappear) ---
    top_brands = df['VehBrand'].value_counts().nlargest(config.TOP_BRANDS_COUNT).index
    
    # Logic: Keep it ONLY if it's in Top N AND it is NOT the excluded brand
    df['VehBrand_Bin'] = df['VehBrand'].apply(
        lambda x: x if (x in top_brands and x != config.EXCLUDED_BRAND) else 'Other'
    )
    
    # Use 'VehBrand_Bin' in your models instead of 'VehBrand'

    return df