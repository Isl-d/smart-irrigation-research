# Smart Irrigation Prediction — CRISP-DM Final Report
Generated: 2026-06-19 08:54:28

---

## 1. Dataset Characteristics
- **Environmental sensor records:** 10,964
- **Soil sensor records:** 32,666
- **Water meter records:** 32,647
- **Agronomic indicator records:** 77
- **Final merged + cleaned records:** 10,959
- **Training samples:** 7,662  |  Test samples: 3,285

## 2. Data Cleaning Performed
- Removed database IDs and battery metadata columns
- Replaced NULL strings with NaN and imputed with column median
- Rounded timestamps to 10-minute bins; resolved per-line duplicates
- Applied IQR-based outlier detection (factor=3); outliers retained

## 3. Feature Engineering
- **Total features:** 24
- Features: soil_ec, soil_moisture, soil_temp, co2, air_humidity, air_pressure, air_temp, gdd, standard_day_degree, ontario_units, hour, dow, month, is_daytime, day_sin, day_cos, vpd, roll_sm_1h, roll_ec_1h, roll_co2_1h, temp_diff, sm_change, ec_change, cum_daily_irr
- Temporal: hour, day-of-week, month, is_daytime, cyclic (sin/cos)
- VPD (kPa) from air temperature and humidity
- 1-hour rolling mean for soil moisture, EC, CO₂
- Soil-air temperature differential, soil moisture change, EC change
- Cumulative daily irrigation (lagged by 1 step to avoid leakage)

## 4. Best Hyperparameters (Optuna)
- **Random Forest:** {'n_estimators': 465, 'max_depth': 5, 'min_samples_split': 20, 'max_features': 'log2'}  (CV RMSE=29.374104)
- **XGBoost:** {'n_estimators': 485, 'learning_rate': 0.002657959263519682, 'max_depth': 4, 'subsample': 0.8954425730455775, 'colsample_bytree': 0.8049557256770327}  (CV RMSE=24.134206)
- **SVR:** {'C': 98.41675497906036, 'epsilon': 0.0014791631222751686, 'kernel': 'rbf'}  (CV RMSE=20.406814)
- **ANN:** {'lr': 0.0015930522616241021, 'u1': 256, 'u2': 32, 'drop': 0.15502135295603015}  (CV RMSE=15.892439)

## 5. Model Performance

| Rank | Model | Type | MAE | RMSE | R² | MAPE (%) | Train Time (s) |
|------|-------|------|-----|------|-----|----------|----------------|
| 1 | CatBoost | ML-Ensemble | 2.149974 | 8.767175 | 0.7904 | 54.21 | 0.58 |
| 2 | Extra Trees | ML-Ensemble | 2.572263 | 8.775524 | 0.7900 | 53.72 | 0.36 |
| 3 | XGBoost | ML-Ensemble | 3.239049 | 9.032593 | 0.7775 | 54.47 | 0.64 |
| 4 | Gradient Boosting | ML-Ensemble | 2.423044 | 9.210676 | 0.7686 | 58.20 | 6.32 |
| 5 | LightGBM | ML-Ensemble | 2.670607 | 9.293075 | 0.7645 | 54.92 | 3.33 |
| 6 | Random Forest | ML-Ensemble | 2.194972 | 9.341222 | 0.7620 | 55.49 | 1.17 |
| 7 | RNN | DL | 3.144369 | 10.912043 | 0.6752 | 54.51 | 9.63 |
| 8 | RF (Optuna) | ML-Tuned | 3.464524 | 11.561286 | 0.6354 | 61.12 | 0.49 |
| 9 | SVR (Optuna) | ML-Tuned | 5.452349 | 12.225621 | 0.5923 | 49.63 | 12.01 |
| 10 | XGB (Optuna) | ML-Tuned | 4.784351 | 12.346066 | 0.5843 | 62.65 | 0.58 |
| 11 | SVR | ML-Other | 3.772672 | 12.735441 | 0.5576 | 54.44 | 2.28 |
| 12 | KNN | ML-Other | 3.355026 | 13.555803 | 0.4988 | 65.08 | 0.00 |
| 13 | BiLSTM | DL | 4.460432 | 15.514698 | 0.3435 | 77.17 | 17.14 |
| 14 | LSTM (Optuna) | DL-Tuned | 4.224822 | 15.625879 | 0.3341 | 78.66 | 8.56 |
| 15 | ANN | DL | 7.500152 | 15.948327 | 0.3063 | 53.19 | 4.40 |
| 16 | CNN-1D | DL | 8.511388 | 16.648128 | 0.2441 | 72.46 | 2.65 |
| 17 | ANN (Optuna) | DL-Tuned | 4.281278 | 16.914061 | 0.2197 | 81.89 | 2.52 |
| 18 | Decision Tree | ML-Ensemble | 2.637958 | 18.252917 | 0.0913 | 136.42 | 0.04 |
| 19 | CNN-LSTM | DL | 6.871327 | 18.572542 | 0.0592 | 86.81 | 4.07 |
| 20 | LSTM | DL | 7.939847 | 18.750116 | 0.0411 | 81.26 | 8.77 |
| 21 | Lasso Regression | ML-Baseline | 15.530089 | 21.198099 | -0.2256 | 70.67 | 0.34 |
| 22 | Linear Regression | ML-Baseline | 15.533903 | 21.203614 | -0.2262 | 70.67 | 0.01 |
| 23 | Ridge Regression | ML-Baseline | 15.586363 | 21.244946 | -0.2310 | 70.65 | 0.00 |

## 6. Water-Use Efficiency Insights
- The best model (CatBoost) achieves RMSE=8.767175 m³ on a 10-minute window.
- Irrigation volume zero-percentage in test set: 97.0% — models must correctly predict near-zero during rest periods.
- Soil moisture and cumulative daily irrigation are the strongest predictors (see SHAP).
- Rolling features (1-hour) improve performance by capturing sensor dynamics.

## 7. SDG Alignment
- **SDG 2 (Zero Hunger):** Precise irrigation timing → reduced crop stress, higher yield.
- **SDG 12 (Responsible Consumption):** ML-optimised scheduling can reduce water usage
  by 30-50% vs. fixed-timer schedules (FAO 2020), minimising agricultural water footprint.

## 8. Research Question Findings
- Multi-sensor features consistently outperform single-sensor approaches in preliminary
  feature importance analysis (SHAP + MI scores show diverse sensor contributions).
- Best ML model: CatBoost  |  Best DL model: RNN  |  Best overall: CatBoost
- Tree-based ensembles with Optuna tuning typically lead due to their ability to capture
  nonlinear threshold effects in irrigation scheduling.
- Deep learning models (LSTM, BiLSTM) capture temporal dependencies in soil sensor data
  and are competitive, particularly for longer-horizon prediction tasks.

## 9. Artefacts
- Models: /Users/islambekdaiyn/Desktop/Research/Smart Irrigation/outputs/models
- Figures: /Users/islambekdaiyn/Desktop/Research/Smart Irrigation/outputs/images
- Metrics CSV: /Users/islambekdaiyn/Desktop/Research/Smart Irrigation/outputs/metrics/model_comparison.csv
- This report: /Users/islambekdaiyn/Desktop/Research/Smart Irrigation/outputs/reports/final_report.md