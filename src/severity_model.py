import pandas as pd
import numpy as np
from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import config

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

    # Capping: Cap large losses at the specified percentile to stabilize training
    # (Note: This only caps the training data. We still test on raw data.)
    cap = df_nonzero['AvgSeverity'].quantile(config.SEVERITY_CAP_PERCENTILE)
    df_nonzero['AvgSeverity'] = df_nonzero['AvgSeverity'].clip(upper=cap)
    print(f"Severity Capping Threshold ({config.SEVERITY_CAP_PERCENTILE*100}%): €{cap:,.2f}")

    # --- 2. Define Features (Using the BINNED versions) ---
    
    # We use the "_Bin" columns we created in data_preprocessing
    categorical_features = config.CATEGORICAL_FEATURES.copy()
    numerical_features = config.NUMERICAL_FEATURES.copy()

    # --- 3. Build the Pipeline ---
    
    # This transformer automatically converts the text categories into 0/1 columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    # The Model: Gamma Regressor (Standard for Severity)
    # Small alpha adds slight regularization to prevent overfitting
    glm = GammaRegressor(alpha=config.GLM_ALPHA, max_iter=config.MAX_ITER, solver=config.GLM_SOLVER)

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