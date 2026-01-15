# Context Audit: Insurance Pricing Frequency-Severity Modelling

## Tech Stack & Environment

### Core Languages
- **Python 3.x** (exact version not specified in codebase)

### Key Dependencies
Based on imports found in the codebase:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** (`sklearn`) - Machine learning framework
  - `sklearn.linear_model.PoissonRegressor` - Frequency modeling
  - `sklearn.linear_model.GammaRegressor` - Severity modeling
  - `sklearn.pipeline.Pipeline` - Model pipeline construction
  - `sklearn.compose.ColumnTransformer` - Feature preprocessing
  - `sklearn.preprocessing.OneHotEncoder` - Categorical encoding
  - `sklearn.model_selection.train_test_split` - Data splitting
  - `sklearn.isotonic.IsotonicRegression` - Model calibration
  - `sklearn.datasets.fetch_openml` - Data loading
- **matplotlib** - Visualization and plotting

### Build Tools & Configuration
- No explicit dependency management file found (`requirements.txt`, `setup.py`, `pyproject.toml`, etc.)
- No build configuration files detected
- Project appears to be a standalone Python script collection

### Data Source
- Uses OpenML datasets (IDs: 41214 for frequency, 41215 for severity)
- French Motor Third-Party Liability insurance data

---

## Project Architecture

### Design Pattern
**Modular Pipeline Architecture** - The project follows a sequential, modular design where each component handles a specific aspect of the insurance pricing workflow:

1. **Data Layer** (`data_preprocessing.py`) - Data fetching, cleaning, and feature engineering
2. **Model Layer** (`frequency_model.py`, `severity_model.py`) - Separate GLM models for frequency and severity
3. **Orchestration Layer** (`pricing_engine.py`) - Coordinates the entire pipeline, calibration, and validation

### Data Flow
```
Raw Data (OpenML)
    ↓
Data Preprocessing (fetch, merge, clean, bin)
    ↓
Train/Test Split
    ↓
┌─────────────────┬─────────────────┐
│ Frequency Model │ Severity Model  │
│ (Poisson GLM)   │ (Gamma GLM)     │
└─────────────────┴─────────────────┘
    ↓                    ↓
    └────────┬───────────┘
             ↓
    Pure Premium = Frequency × Severity
             ↓
    Off-Balance Adjustment
             ↓
    Isotonic Calibration
             ↓
    Final Price (with optional loadings)
             ↓
    Validation (Gini, Lift Charts, One-Way Analysis)
```

### State Management
- **Stateless Models** - Models are trained and serialized implicitly through scikit-learn pipelines
- **Data-Driven** - All state is maintained in pandas DataFrames passed between functions
- **No Persistent Storage** - Models are not explicitly saved/loaded (trained fresh each run)

---

## Directory Structure

```
Insurance-Pricing-Frequency-Severity-Modelling/
│
├── README.md                    # Project description
│
└── scr/                         # Source code directory (note: "scr" not "src")
    ├── __pycache__/            # Python bytecode cache
    ├── data_preprocessing.py   # Data loading, cleaning, binning
    ├── frequency_model.py      # Poisson GLM for claim frequency
    ├── severity_model.py       # Gamma GLM for claim severity
    └── pricing_engine.py       # Main orchestration script
```

### Module Responsibilities

- **`data_preprocessing.py`**
  - `fetch_raw_data()` - Downloads frequency and severity datasets from OpenML
  - `preprocess_data()` - Merges datasets, handles missing values, calculates derived features
  - `bin_features()` - Applies actuarial binning to continuous variables (age, vehicle age, power)

- **`frequency_model.py`**
  - `train_frequency_model()` - Trains Poisson GLM with one-hot encoding pipeline
  - Returns trained pipeline and feature list

- **`severity_model.py`**
  - `train_severity_model()` - Trains Gamma GLM on non-zero claims with capping
  - Returns trained pipeline and feature list

- **`pricing_engine.py`**
  - `main()` - Entry point orchestrating entire workflow
  - `train_calibrator()` - Fits isotonic regression for calibration
  - `apply_loadings()` - Applies actuarial loadings/discounts (currently disabled)
  - `calculate_gini()` - Computes Gini coefficient for model validation
  - `plot_one_way()` - Generates one-way analysis plots
  - Visualization functions for lift charts and risk analysis

---

## Key Entry Points

### Main Entry Point
**`scr/pricing_engine.py`** - Contains the `main()` function executed when script is run directly:
```python
if __name__ == "__main__":
    main()
```

### Execution Flow
1. **Data Loading** → `data_preprocessing.fetch_raw_data()`
2. **Preprocessing** → `data_preprocessing.preprocess_data()` → `data_preprocessing.bin_features()`
3. **Model Training** → `frequency_model.train_frequency_model()` → `severity_model.train_severity_model()`
4. **Prediction & Calibration** → Off-balance adjustment → Isotonic calibration
5. **Validation** → Gini calculation → Lift charts → One-way analysis

### Configuration
- **No external config files** - All parameters are hardcoded in functions
- **Key Parameters**:
  - Test split: 20% (`test_size=0.2`, `random_state=42`)
  - Severity cap: 99.9th percentile
  - Visual cap: €15,000 for plotting
  - Regularization: `alpha=0.01` for both GLMs
  - Risk buckets: 10 deciles for lift charts

---

## Core Domain Logic

### Insurance Pricing Methodology
The project implements a **Tweedie GLM approach** (decomposed into Poisson-Gamma) for motor insurance pricing:

1. **Frequency Modeling** (Claims per Year)
   - Uses Poisson GLM with log link
   - Target: `ClaimNb / Exposure` (rate)
   - Weighted by `Exposure`
   - Features: Vehicle brand, gas type, region, area, vehicle power, vehicle age, driver age (all binned)

2. **Severity Modeling** (Cost per Claim)
   - Uses Gamma GLM with log link
   - Trained only on policies with claims (`ClaimNb > 0`)
   - Target: `AvgSeverity` (capped at 99.9th percentile)
   - Weighted by `ClaimNb`
   - Same feature set as frequency model

3. **Pure Premium Calculation**
   - `PurePremium = PredictedFrequency × PredictedSeverity`
   - Off-balance adjustment to match total actual losses
   - Isotonic calibration for non-parametric refinement

4. **Actuarial Adjustments** (Currently Disabled)
   - Manual loadings/discounts for high-risk segments:
     - Driver age: 18-21 (2.4x), 22-25 (1.4x)
     - Vehicle age: New (0.75x), 5-10 years (1.12x), 20+ (0.55x)
     - Vehicle power: Power 9 (1.30x), Power 4 (0.90x)
   - Code exists but is commented out; calibrated price used directly

5. **Model Validation**
   - **Gini Coefficient**: Measures risk differentiation ability
   - **Lift Charts**: Compares actual vs predicted rates across risk deciles
   - **One-Way Analysis**: Validates pricing by individual risk factors

### Key Actuarial Practices
- **Exposure Weighting**: Models account for policy exposure duration
- **Loss Capping**: Large losses capped to stabilize training (99.9th percentile)
- **Feature Binning**: Continuous variables grouped into actuarial categories
- **Off-Balance Adjustment**: Ensures total predicted losses match actuals
- **Calibration**: Isotonic regression ensures monotonic risk ordering

---

## Code Conventions

### Styling
- **PEP 8 compliant** - Standard Python naming conventions
- **Function naming**: snake_case (e.g., `train_frequency_model`, `calculate_gini`)
- **Variable naming**: camelCase for DataFrames (`df_train`, `test_results`), descriptive names
- **Module imports**: Relative imports within `scr/` directory (e.g., `import data_preprocessing`)

### Typing
- **No type hints** - Functions do not use Python type annotations
- **Dynamic typing** - Relies on pandas/sklearn type inference

### Code Organization
- **Functional approach** - Each module exports functions rather than classes
- **Pipeline pattern** - Uses scikit-learn `Pipeline` for preprocessing + modeling
- **Separation of concerns** - Clear boundaries between data, modeling, and orchestration

### Testing Patterns
- **No test files detected** - No `test/`, `tests/`, or `*_test.py` files found
- **No unit tests** - Validation appears to be done through final metrics (Gini, lift charts)

### Documentation
- **Docstrings**: Present for main functions (e.g., `train_frequency_model`, `calculate_gini`)
- **Inline comments**: Extensive comments explaining actuarial logic and fixes
- **README**: Minimal - single line description

### Error Handling
- **Minimal error handling** - Basic try/except in one-way plotting loop
- **Assumptions**: Code assumes data availability and correct formats

### Data Handling
- **Pandas-centric** - Heavy reliance on pandas DataFrames
- **In-place operations**: Some operations modify DataFrames in place (e.g., `rename`, `clip`)
- **Copy patterns**: Uses `.copy()` when needed to avoid SettingWithCopyWarning

---

## Notable Implementation Details

### Feature Engineering
- **Binning Strategy**:
  - Driver Age: 8 bins (18-21, 22-25, 26-30, 31-40, 41-50, 51-60, 61-75, 75+)
  - Vehicle Age: 5 bins (New 0-1, 2-4, 5-10, 11-20, 20+)
  - Vehicle Power: Individual values <10, "10+" for higher
  - Vehicle Brand: Top 5 brands kept, others grouped as "Other" (excludes B12)

### Model Configuration
- **Solver**: `newton-cholesky` for both GLMs
- **Max iterations**: 1000
- **Regularization**: L2 with `alpha=0.01`
- **One-hot encoding**: `drop='first'` (dummy variable trap prevention)
- **Unknown categories**: `handle_unknown='ignore'` for production robustness

### Calibration Approach
- **Isotonic Regression**: Non-parametric, monotonic calibration
- **Capping**: Actual losses capped at €15,000 before calibration
- **Weighting**: Calibration weighted by exposure
- **Bounds**: `y_min=0`, `out_of_bounds='clip'`

---

## Potential Improvements & Observations

### Missing Components
- No dependency management (`requirements.txt`)
- No model persistence/serialization (models retrained each run)
- No configuration management (hardcoded parameters)
- No logging framework (uses `print()` statements)
- No unit tests or integration tests

### Code Quality Notes
- Directory name `scr/` instead of standard `src/`
- Manual loadings code exists but is disabled (commented out)
- Some hardcoded magic numbers (e.g., `cap_threshold=15000`, `test_size=0.2`)
- Limited error handling for edge cases

### Scalability Considerations
- Single-threaded execution
- No distributed training support
- Data loaded entirely into memory
- No incremental learning capabilities

---

## Quick Start Guide for New Developers

1. **Install Dependencies** (inferred):
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Run the Pipeline**:
   ```bash
   cd scr
   python pricing_engine.py
   ```

3. **Expected Output**:
   - Model training progress messages
   - Off-balance factor
   - Gini coefficient
   - Lift chart visualization
   - One-way analysis plots for key features

4. **Key Files to Understand**:
   - Start with `pricing_engine.py` → `main()` function
   - Review `data_preprocessing.py` for data transformations
   - Examine `frequency_model.py` and `severity_model.py` for modeling logic

---

*Generated: Context Audit for Insurance Pricing Frequency-Severity Modelling Project*
