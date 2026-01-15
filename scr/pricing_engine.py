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

# --- HELPER: ONE-WAY ANALYSIS ---
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
    
    # UPDATED: We use the 'Final Price' which includes the loading
    ax1.plot(agg['group_col'].astype(str), agg['Predicted_Rate'], color='green', marker='o', linewidth=2, linestyle='-', label='Final Technical Price')
    
    ax1.set_ylabel('Pure Premium (€)')
    ax1.set_title(f'One-Way Analysis: {feature}')
    ax1.set_xticklabels(agg['group_col'].astype(str), rotation=45)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def apply_loadings(row):
    """
    Applies actuarial loadings (and discounts) to align 
    technical price with observed one-way risk.
    """
    price = row['PurePremium_Calibrated']
    
    # --- 1. Driver Age (Keep these, they work well) ---
    if row['DriverAge_Bin'] == '18-21':
        price = price * 2.4 
    elif row['DriverAge_Bin'] == '22-25':
        price = price * 1.4
        
# --- 2. Vehicle Age (Refined) ---
    if row['VehAge_Bin'] == 'New (0-1)':
        price = price * 0.75
        
    # FIX A: The "Volume Trap" (5-10 Years)
    # Cost ~130 vs Price ~116. Factor = 1.12
    elif row['VehAge_Bin'] == '5-10':
        price = price * 1.12  # 12% Surcharge
        
    # FIX B: Old Cars (20+)
    # Cost ~50 vs Price ~90. Factor = 0.55
    elif row['VehAge_Bin'] == '20+':
        price = price * 0.55  # 45% Discount
    # --- 3. Vehicle Power (UPDATED) ---
    
    # FIX A: Power 9 is the "Hidden Killer". 
    # Cost ~148 vs Price ~114. Factor = 1.3
    if row['VehPower_Bin'] == '9':
        price = price * 1.30  # 30% Surcharge
        
    # FIX B: Power 4 is too expensive.
    # Cost ~97 vs Price ~110. Factor = 0.88
    elif row['VehPower_Bin'] == '4':
        price = price * 0.90  # 10% Discount
        
    # FIX C: Power 10+ was over-discounted.
    # Removing the previous 0.9 factor will bring it from ~109 back to ~121,
    # which is very close to the actual cost of ~118.
    # So we simply DELETE the 'if row['VehPower_Bin'] == '10+'' block.

    return price


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

    # 3. Train Base Models
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

    # === 7b. NEW: APPLY ACTUARIAL LOADINGS ===
    print("Applying Actuarial Loadings for High Risk Segments...")
    test_results['Final_Price'] = test_results.apply(apply_loadings, axis=1)
    
    # 8. LIFT CHART GENERATION (Using Final Price)
    visual_cap = 15000
    test_results['TotalLoss_Visual'] = test_results['TotalLoss'].clip(upper=visual_cap)
    
    test_results['RiskBucket'] = pd.qcut(
        test_results['Final_Price'], 10, labels=False, duplicates='drop'
    )
    
    lift_agg = test_results.groupby('RiskBucket').agg({
        'Exposure': 'sum'
    })

    lift_agg['Actual_Rate'] = test_results.groupby('RiskBucket').apply(
        lambda x: x['TotalLoss_Visual'].sum() / x['Exposure'].sum()
    )
    
    lift_agg['Final_Rate'] = test_results.groupby('RiskBucket').apply(
        lambda x: (x['Final_Price'] * x['Exposure']).sum() / x['Exposure'].sum()
    )
    
    lift_agg = lift_agg.reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(lift_agg['RiskBucket'], lift_agg['Actual_Rate'], alpha=0.6, label='Actuals', color='#1f77b4')
    plt.plot(lift_agg['RiskBucket'], lift_agg['Final_Rate'], 
             color='green', marker='o', linewidth=2.5, label='Final Technical Price')
    
    plt.title('Lift Chart: Final Price (Calibrated + Loaded)')
    plt.xlabel('Risk Decile')
    plt.ylabel('Pure Premium (€)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    # 9. ONE-WAY ANALYSIS (Using Final Price)
    print("Generating One-Way Plots...")
    features_to_check = ['VehPower_Bin', 'VehAge_Bin', 'DriverAge_Bin'] 
    
    for feat in features_to_check:
        try:
            plot_one_way(
                test_results, 
                feat, 
                pred_col='Final_Price',        # <--- Check the Final Price now
                actual_col='TotalLoss_Visual'
            )
        except Exception as e:
            print(f"Skipping {feat}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()