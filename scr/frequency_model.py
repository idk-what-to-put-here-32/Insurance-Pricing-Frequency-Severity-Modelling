import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def train_frequency_model(df_train):
    """
    Fits a Poisson GLM for Frequency (Claims per Year).
    Uses a Pipeline to handle categorical variables (One-Hot Encoding) automatically.
    """
    
    # --- 1. Define Features (Using the BINNED versions) ---
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
    numerical_features = [] 

    # --- 2. Build the Pipeline ---
    
    # This transformer automatically converts the text categories into 0/1 columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    # The Model: Poisson Regressor (Standard for Frequency)
    # alpha=1e-4 adds slight regularization to prevent overfitting on rare categories
    glm = PoissonRegressor(alpha=0.001, max_iter=1000, solver='newton-cholesky')

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