import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

# Import your custom modules
import data_preprocessing
import frequency_model
import severity_model

# --- HELPER: CALIBRATION ---
def train_calibrator(predictions, actuals, exposure, cap_threshold=15000):
    """
    Fits an Isotonic Regression model with capped actuals.
    """
    iso_reg = IsotonicRegression(y_min=0, out_of_bounds='clip')
    
    # Cap the actuals BEFORE calculating the rate
    capped_actuals = actuals.clip(upper=cap_threshold)
    
    # Calculate Rate using CAPPED losses
    actual_rate = np.divide(
        capped_actuals, exposure, 
        out=np.zeros_like(capped_actuals), 
        where=exposure!=0
    )
    
    iso_reg.fit(predictions, actual_rate, sample_weight=exposure)
    return iso_reg

# --- HELPER FUNCTION: ONE-WAY ANALYSIS ---
def plot_one_way(df, feature, actual_col='TotalLoss', pred_col='PurePremium', exposure_col='Exposure'):
    df_plot = df.copy()

    # Binning logic
    if pd.api.types.is_numeric_dtype(df_plot[feature]) and df_plot[feature].nunique() > 20:
        df_plot['group_col'] = pd.qcut(df_plot[feature], 10, duplicates='drop')
    else:
        df_plot['group_col'] = df_plot[feature]

    df_plot['ExpectedLoss'] = df_plot[pred_col] * df_plot[exposure_col]
    
    agg = df_plot.groupby('group_col', observed=False).agg({
        actual_col: 'sum',
        'ExpectedLoss': 'sum',
        exposure_col: 'sum'
    }).reset_index()

    agg['Actual_Rate'] = agg[actual_col] / agg[exposure_col]
    agg['Predicted_Rate'] = agg['ExpectedLoss'] / agg[exposure_col]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax2.bar(agg['group_col'].astype(str), agg[exposure_col], alpha=0.3, color='gray', label='Exposure')
    ax2.set_ylabel('Exposure (Volume)')
    
    ax1.plot(agg['group_col'].astype(str), agg['Actual_Rate'], color='#1f77b4', marker='o', linewidth=2, label='Actual Cost')
    
    # UPDATED: Changed to Green to match the final model style
    ax1.plot(agg['group_col'].astype(str), agg['Predicted_Rate'], color='green', marker='o', linewidth=2, linestyle='-', label='Calibrated Prediction')
    
    ax1.set_ylabel('Pure Premium (€)')
    ax1.set_title(f'One-Way Analysis: {feature}')
    ax1.set_xticklabels(agg['group_col'].astype(str), rotation=45)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

# --- MAIN ENGINE ---
def main():
    # 1. Data Loading & Preprocessing
    print("Loading data...")
    df_f, df_s = data_preprocessing.fetch_raw_data()
    df = data_preprocessing.preprocess_data(df_f, df_s)
    print("Binning features...")
    df = data_preprocessing.bin_features(df)

    # 2. Train/Test Split
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Train Base Models (Remember to keep alpha=0.0001 in your model files!)
    print("Training Frequency Model...")
    freq_m, feat_f = frequency_model.train_frequency_model(train)
    
    print("Training Severity Model...")
    sev_m, feat_s = severity_model.train_severity_model(train)

    # 4. Generate Training Predictions
    print("Generating Training Predictions for Calibration...")
    train['PredFreq'] = freq_m.predict(train[feat_f])
    train['PredSev'] = sev_m.predict(train[feat_s])
    train['PurePremium_Raw'] = train['PredFreq'] * train['PredSev']

    # 5. Off-Balance Adjustment
    actual_total_train = train['TotalLoss'].sum()
    predicted_total_train = (train['PurePremium_Raw'] * train['Exposure']).sum()
    off_balance_factor = actual_total_train / predicted_total_train
    
    print(f"Off-Balance Factor: {off_balance_factor:.4f}")
    train['PurePremium_OB'] = train['PurePremium_Raw'] * off_balance_factor

    # 6. TRAIN CALIBRATION LAYER
    print("Fitting Isotonic Calibration Layer (With Capping)...")
    calibrator = train_calibrator(
        predictions=train['PurePremium_OB'], 
        actuals=train['TotalLoss'], 
        exposure=train['Exposure'],
        cap_threshold=15000 
    )

    # 7. Apply to TEST Set
    print("Applying to Test Set...")
    test_results = test.copy()
    test_results['PredFreq'] = freq_m.predict(test[feat_f])
    test_results['PredSev'] = sev_m.predict(test[feat_s])
    
    test_results['PurePremium_Raw'] = test_results['PredFreq'] * test_results['PredSev']
    test_results['PurePremium_OB'] = test_results['PurePremium_Raw'] * off_balance_factor
    test_results['PurePremium_Calibrated'] = calibrator.transform(test_results['PurePremium_OB'])

    # 8. LIFT CHART GENERATION (Cleaned Up)
    visual_cap = 15000
    test_results['TotalLoss_Visual'] = test_results['TotalLoss'].clip(upper=visual_cap)
    
    test_results['RiskBucket'] = pd.qcut(
        test_results['PurePremium_Calibrated'], 10, labels=False, duplicates='drop'
    )
    
    # Calculate weighted sums per bucket
    lift_agg = test_results.groupby('RiskBucket').agg({
        'Exposure': 'sum'
    })

    # Calculate Rates manually to be safe
    # Rate = Sum(Prem * Exp) / Sum(Exp) -> Effectively just Weighted Average
    lift_agg['Actual_Rate'] = test_results.groupby('RiskBucket').apply(
        lambda x: x['TotalLoss_Visual'].sum() / x['Exposure'].sum()
    )
    
    lift_agg['Calibrated_Rate'] = test_results.groupby('RiskBucket').apply(
        lambda x: (x['PurePremium_Calibrated'] * x['Exposure']).sum() / x['Exposure'].sum()
    )
    
    lift_agg = lift_agg.reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.bar(lift_agg['RiskBucket'], lift_agg['Actual_Rate'], alpha=0.6, label='Actuals', color='#1f77b4')
    
    # UPDATED: Removed the Red Line, only showing Final Model
    plt.plot(lift_agg['RiskBucket'], lift_agg['Calibrated_Rate'], 
             color='green', marker='o', linewidth=2.5, label='Final Model (Calibrated)')
    
    plt.title('Lift Chart: Final Model Performance')
    plt.xlabel('Risk Decile')
    plt.ylabel('Pure Premium (€)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    # 9. ONE-WAY ANALYSIS (Updated to use Calibrated Price)
    print("Generating One-Way Plots...")
    features_to_check = ['VehPower_Bin', 'VehAge_Bin', 'DriverAge_Bin'] 
    
    for feat in features_to_check:
        try:
            # We pass the Calibrated column and the Capped Actuals for consistency
            plot_one_way(
                test_results, 
                feat, 
                pred_col='PurePremium_Calibrated', 
                actual_col='TotalLoss_Visual'
            )
        except Exception as e:
            print(f"Skipping {feat}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()