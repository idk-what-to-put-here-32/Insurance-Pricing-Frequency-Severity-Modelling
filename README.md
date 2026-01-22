# Insurance Pricing Engine & Dashboard
### End-to-End Frequency-Severity GLM Modelling & Commercial Dislocation Analysis Tool.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

## **Project Overview**

This project is an end-to-end actuarial pricing pipeline and interactive simulation tool for French Motor Third-Party Liability (MTPL) insurance. It implements a Frequency-Severity GLM architecture, Isotonic Calibration, and a Streamlit dashboard for Commercial Dislocation Analysis.

The tool bridges the gap between technical modelling and commercial strategy by allowing users to adjust actuarial loadings and visualise the impact on the model's Technical and Commercial Prices via Lift Charts, Lorenz Curves, and Dislocation Histograms.

The project utilises the industry-standard French Motor datasets (ID 41214/41215) to demonstrate the full Technical Pricing Cycle: Data Preprocessing $\rightarrow$ GLM Modelling $\rightarrow$ Calibration $\rightarrow$ Commercial Analysis.

## **Key Features**

### **1. Technical Risk Modelling**
The engine implements a decoupled Frequency-Severity GLM structure to determine the Technical Pure Premium.

* **Frequency (Poisson GLM):** Utilises a log-link function to model claim counts, ensuring non-negative arrival rates.
* **Severity (Gamma GLM):** Models the average cost per claim for policies where $ClaimNb > 0$. Includes Actuarial Capping at the 99.9th percentile to remove the volatility of catastrophic "large losses" from attritional pricing.
* **Technical Pure Premium:** Calculated using the fundamental identity:
    $$PurePremium = E[Frequency] \times E[Severity]$$
* **Isotonic Calibration:** Applies a non-parametric mapping to the raw model output to ensure that predicted risk probabilities remain monotonically increasing with actual experience.

### **2. Commercial Analysis (Streamlit Dashboard)**
An interactive UI allows for real-time stress testing of commercial pricing rules:

* **Commercial Levers:** Real-time sliders to adjust loadings for high-risk segments (e.g., Young Drivers aged 18-21) and discounts for preferred segments (e.g., Vehicle Power < 4).
* **Performance Metrics:** Real-time calculation of Technical vs. Commercial **Gini Coefficients** to ensure commercial adjustments do not degrade risk differentiation.

### **3. Visualisation**
* **Lift Charts:** Decile-based analysis comparing Predicted vs. Actual loss costs to validate model performance across risk bands.
* **Lorenz Curves:** Visual representation of the model's ability to sort risk (Gini Coefficient).
* **One-Way Analysis:** Dual-axis plots showing average loss cost and exposure volume across rating factors (Driver Age, Vehicle Age, etc.).
* **Dislocation Histograms:** Visualises the distribution of price changes between the Technical Price and the Final Commercial Price, quantifying the impact of actuarial loadings.