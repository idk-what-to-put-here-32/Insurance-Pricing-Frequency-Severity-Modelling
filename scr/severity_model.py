import pandas as pd
import numpy as np
from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Note: No need to import data_preprocessing here; the main engine passes the data in.

def train_severity_model(df_train):
    """
    Fits a Gamma GLM for Severity (Cost per Claim).
    Uses a Pipeline to handle categorical variables (One-Hot Encoding) automatically.
    """
    
    # --- 1. Filter and Cap Data (The Actuarial Adjustments) ---
    
    # Filter: We only want claims that actually cost money
    df_nonzero = df_train[
        (df_train['ClaimNb'] > 0) & (df_train['AvgSeverity'] > 0)
    ].copy()

    # Capping: Cap large losses at the 99.9th percentile to stabilize training
    # (Note: This only caps the training data. We still test on raw data.)
    cap = df_nonzero['AvgSeverity'].quantile(0.999)
    df_nonzero['AvgSeverity'] = df_nonzero['AvgSeverity'].clip(upper=cap)
    print(f"Severity Capping Threshold (99.9%): €{cap:,.2f}")

    # --- 2. Define Features (Using the BINNED versions) ---
    
    # We use the "_Bin" columns we created in data_preprocessing
    categorical_features = [
        'VehBrand_Bin', 
        'VehGas', 
        'Region', 
        'Area', 
        'VehPower_Bin',   # Using the grouped Power
        'VehAge_Bin',     # Using the grouped Vehicle Age
        'DriverAge_Bin'   # Using the grouped Driver Age
    ]
    
    # If you have continuous features (e.g. Density), list them here
    numerical_features = [] # e.g. ['Density']

    # --- 3. Build the Pipeline ---
    
    # This transformer automatically converts the text categories into 0/1 columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    # The Model: Gamma Regressor (Standard for Severity)
    # alpha=0 is a standard GLM. Small alpha (1e-4) adds slight regularization.
    glm = GammaRegressor(alpha=0.001, max_iter=1000, solver='newton-cholesky')

    # Combine into a single object
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', glm)
    ])

    # --- 4. Train the Model ---
    
    print("Fitting Severity GLM...")
    
    # We fit the pipeline
    # Note: We must separate Weights from the dataframe
    model_pipeline.fit(
        df_nonzero[categorical_features + numerical_features], 
        df_nonzero['AvgSeverity'],
        regressor__sample_weight=df_nonzero['ClaimNb'] # Weight by number of claims
    )

    # Return the pipeline object (it contains both the encoder and the model)
    # We also return the list of feature names we expected
    return model_pipeline, categorical_features + numerical_features