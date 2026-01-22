# Insurance Pricing Engine & Dashboard
### End-to-End Frequency-Severity GLM Modelling & Commercial Dislocation Analysis Tool.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

## **Project Overview**
This project is an end-to-end actuarial pricing pipeline and interactive simulation tool for French Motor Third-Party Liability (MTPL) insurance. It bridges the gap between technical risk modelling and commercial decision-making by allowing users to adjust actuarial loadings and visualise the impact on portfolio KPIs.

The project utilizes the industry-standard French Motor datasets (ID 41214/41215) to demonstrate a complete ratemaking cycle.

## Key Features

### 1. Technical Risk Modelling
The engine implements a decoupled Frequency-Severity GLM structure to determine the Technical Pure Premium.

* **Frequency (Poisson GLM):** * Utilises a log-link function to ensure non-negative arrival rates.
    * Incorporates **Exposure duration** as a weight to account for varying policy lengths.
* **Severity (Gamma GLM):** * Models the average cost per claim for policies with $ClaimNb > 0$.
    * Includes **Actuarial Capping** at the 99.9th percentile to remove the volatility of catastrophic "large losses" from attritional pricing.
    
* **Technical Pure Premium:** * Calculated using the fundamental identity: $PurePremium = E[Frequency] \times E[Severity]$.

* **Isotonic Calibration:** Applies a non-parametric mapping to ensure that predicted risk remains monotonic

### 2. Commercial Analysis (Streamlit Dashboard)
An interactive UI allows for real-time stress testing of commercial pricing rules:
* **Commercial Levers:** Real-time sliders to adjust loadings for high-risk segments (e.g., Young Drivers aged 18-21) and discounts for preferred segments (e.g., Vehicle Power < 4).
* **Dislocation Analysis:** Histograms measuring the deviation between the Technical Price and the Final Commercial Price.
* **Performance Metrics:** Real-time calculation of Technical vs. Commercial **Gini Coefficients** to ensure commercial adjustments do not degrade risk differentiation.

### 3. Visualisation
* **Lift Charts:** Decile-based analysis comparing Predicted vs. Actual loss costs.
* **Lorenz Curves:** Visual representation of the model's ability to sort risk.
* **One-Way Analysis:** Dual-axis plots showing average loss cost and exposure volume across rating factors (Driver Age, Vehicle Age, etc.).