# Smart Irrigation Prediction System

A CRISP-DM machine learning and deep learning pipeline for precision drip irrigation scheduling, applied to IoT sensor data from a tomato greenhouse in Parma, Italy.

---

## Overview

This project predicts irrigation volume (m³) at 10-minute resolution using multi-sensor IoT data. Twenty-three ML and DL models are benchmarked, with CatBoost achieving the best regression performance (R² = 0.790, MAE = 2.15 m³). SHAP explanations identify cumulative daily irrigation and CO₂ concentration as the dominant predictors.

The work supports **SDG 2** (Zero Hunger) and **SDG 12** (Responsible Consumption and Production) by enabling data-driven irrigation scheduling that reduces water waste.

---

## Dataset

Source: Belli et al. (2025) — LoRaWAN IoT monitoring of tomato (*Solanum lycopersicum* cv. HEINZ 1301), June 28 – September 13, 2023.

Three irrigation treatment lines: 100%, 60%, and 30% of standard water recommendation.

| File | Device | Signals |
|------|--------|---------|
| `stuard_environmental_data.csv` | Milesight EM500-CO2 | CO₂ (ppm), air temp (°C), humidity (%RH), pressure (hPa) |
| `stuard_soil_data.csv` | Milesight EM500-SMTC | Soil EC (μS/cm), moisture (%RH), temperature (°C) @ 20 cm depth |
| `stuard_water_meter_data.csv` | Talkpool OY1310 | Cumulative water volume (m³) per line |
| `indicators.csv` | Agriware platform | Growing Degree Days (GDD), Ontario Heat Units (daily) |

All sensors sample at 10-minute intervals. Coordinates: Parma, Italy (lat: 44.809, lon: 10.273).

---

## Pipeline

The notebook `Smart_Irrigation_CRISP_DM.ipynb` implements the full CRISP-DM workflow:

1. **Data loading & merging** — four CSV sources merged on 10-minute timestamp bins
2. **Preprocessing** — IQR outlier detection, median imputation, duplicate removal
3. **Feature engineering** — 24 features: temporal (cyclic encoding), VPD, rolling sensor means, change rates, cumulative daily irrigation, agronomic indicators
4. **Train/test split** — temporal 70/30 (7,662 train / 3,288 test), chronological order preserved
5. **Model training** — 23 models across 4 families (see Models section)
6. **Hyperparameter tuning** — Optuna Bayesian search, 75 trials total
7. **Evaluation** — MAE, RMSE, R²; classification metrics via optimal-threshold binarisation
8. **Explainability** — SHAP TreeExplainer on CatBoost; permutation importance
9. **Serialisation** — all models saved to `outputs/models/`

---

## Models

| Family | Models |
|--------|--------|
| ML Baseline | Linear Regression, Ridge, Lasso |
| ML Ensemble | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| ML Other | SVR, KNN |
| Deep Learning | ANN, RNN, LSTM, BiLSTM, CNN-1D, CNN-LSTM |
| Optuna-tuned | RF, XGBoost, SVR, ANN, LSTM |

Deep learning models use a 12-step look-back window (2-hour sequence), trained with EarlyStopping (patience = 15).

---

## Results

### Regression (top models)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **CatBoost** | **2.15** | **8.77** | **0.790** |
| Extra Trees | 2.57 | 8.78 | 0.790 |
| XGBoost | 3.24 | 9.03 | 0.777 |
| Gradient Boosting | 2.42 | 9.21 | 0.769 |
| RNN | 3.14 | 10.91 | 0.675 |
| Linear Regression | 15.53 | 21.20 | −0.226 |

### Classification (irrigation event detection)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Decision Tree** | **0.9894** | 0.7442 | 0.9796 | **0.8458** |
| KNN | 0.9903 | **0.8929** | 0.7653 | 0.8242 |
| Random Forest | 0.9316 | 0.3009 | 0.9796 | 0.4604 |
| CatBoost | 0.8756 | 0.1921 | 0.9898 | 0.3217 |

### Top SHAP features (CatBoost)

1. `cum_daily_irr` — cumulative irrigation applied today (dominant)
2. `roll_co2_1h` — 1-hour rolling CO₂ concentration
3. `air_temperature`
4. `sm_change` — soil moisture change rate
5. `vpd` — vapour pressure deficit

---

## Project Structure

```
.
├── Smart_Irrigation_CRISP_DM.ipynb   # Main pipeline notebook
├── indicators.csv                     # Daily agronomic indicators
├── stuard_environmental_data.csv      # Environmental sensor data
├── stuard_soil_data.csv               # Soil sensor data
├── stuard_water_meter_data.csv        # Water meter data
└── outputs/
    ├── images/                        # EDA, training curves, SHAP plots
    ├── metrics/
    │   ├── model_comparison.csv       # Regression metrics for all 23 models
    │   └── classification_metrics.csv # Classification metrics for all models
    ├── models/                        # Serialised models (.pkl, .keras)
    └── reports/
        ├── final_report.md
        └── final_report.pdf
```

---

## Requirements

Python 3.12. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap \
            tensorflow keras matplotlib seaborn joblib
```

---

## Citation

Dataset: Belli, A. et al. (2025). IoT-based multi-sensor dataset for smart irrigation of tomato crops. *Zenodo*. https://doi.org/10.5281/zenodo.14901438
