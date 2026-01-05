"""
Module for fetching and preprocessing French Motor Third-Party Liability data.
Follows a Frequency-Severity framework for insurance modeling.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def fetch_raw_data():
    """
    Fetch the French Motor datasets from OpenML.

    Returns:
        tuple: (df_freq, df_sev) containing the raw pandas DataFrames.
    """
    df_freq = fetch_openml(data_id=41214, as_frame=True, parser='pandas').data
    df_sev = fetch_openml(data_id=41215, as_frame=True, parser='pandas').data
    return df_freq, df_sev


def preprocess_data(df_freq, df_sev):
    """
    Cleans, joins, and engineers features for Frequency-Severity modeling.

    Args:
        df_freq (pd.DataFrame): Frequency data containing exposure and counts.
        df_sev (pd.DataFrame): Severity data containing claim amounts.

    Returns:
        pd.DataFrame: A processed dataframe ready for GLM modeling.
    """
    # 1. Aggregate Severity & Join
    claims_agg = df_sev.groupby('IDpol').agg(
        TotalLoss=('ClaimAmount', 'sum')
    ).reset_index()

    df = pd.merge(df_freq, claims_agg, on='IDpol', how='left')
    df['TotalLoss'] = df['TotalLoss'].fillna(0)

    # 2. Data Cleaning
    # Validating exposure and capping counts to remove data errors/outliers
    df['Exposure'] = df['Exposure'].clip(lower=0.001, upper=1.0)
    df['ClaimNb'] = df['ClaimNb'].clip(upper=4)

    # 3. Categorical Encoding
    cat_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))

    # 4. Target Engineering
    df['Frequency'] = df['ClaimNb'] / df['Exposure']
    df['AvgSeverity'] = np.where(
        df['ClaimNb'] > 0,
        df['TotalLoss'] / df['ClaimNb'],
        np.nan
    )

    return df


def main():
    """Execute the data pipeline and print summary statistics."""
    df_f, df_s = fetch_raw_data()
    processed_df = preprocess_data(df_f, df_s)

    train, test = train_test_split(
        processed_df, test_size=0.2, random_state=42
    )

    print(f"Preprocessing Complete. Training: {len(train)}, Test: {len(test)}")


if __name__ == "__main__":
    main()
