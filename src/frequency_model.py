import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import config

def train_frequency_model(df_train):
    """
    Fits a Poisson GLM for Frequency (Claims per Year).
    Uses a Pipeline to handle categorical variables (One-Hot Encoding) automatically.
    """
    
    # --- 1. Define Features (Using the BINNED versions) ---
    # We use the "_Bin" columns we created in data_preprocessing
    categorical_features = config.CATEGORICAL_FEATURES.copy()
    numerical_features = config.NUMERICAL_FEATURES.copy() 

    # --- 2. Build the Pipeline ---
    
    # This transformer automatically converts the text categories into 0/1 columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    # The Model: Poisson Regressor (Standard for Frequency)
    # alpha adds slight regularization to prevent overfitting on rare categories
    glm = PoissonRegressor(alpha=config.GLM_ALPHA, max_iter=config.MAX_ITER, solver=config.GLM_SOLVER)

    # Combine into a single object
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', glm)
    ])

    # --- 3. Prepare Targets ---
    # For Poisson, the target is the RATE (Claims / Exposure)
    # And we weight by Exposure
    y_train = df_train['ClaimNb'] / df_train['Exposure']

    # --- 4. Train the Model ---
    print("Fitting Frequency GLM...")
    
    model_pipeline.fit(
        df_train[categorical_features + numerical_features], 
        y_train,
        regressor__sample_weight=df_train['Exposure'] # Weight by Exposure
    )

    # Return the pipeline object (it contains both the encoder and the model)
    # We also return the list of feature names we expected
    return model_pipeline, categorical_features + numerical_features