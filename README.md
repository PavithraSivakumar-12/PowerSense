# ⚡ PowerSense Household Electricity Consumption ML System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end Machine Learning system that predicts household electricity consumption, detects anomalies, classifies usage patterns, and estimates energy costs — all through an interactive Streamlit dashboard.

---

## 🎯 Project Goals

| Goal | Algorithm | Result |
|------|-----------|--------|
| Predict power consumption | XGBoost Regressor | R² = 0.871 |
| Detect unusual usage events | Isolation Forest | 2,010 anomalies detected |
| Classify usage patterns | Random Forest | 91% accuracy |
| Estimate energy cost | Time-based tariff model | ~9.4% potential saving |

---

## 📊 Dataset

**UCI Individual Household Electric Power Consumption**
- 2,049,280 readings at 1-minute intervals
- Duration: December 2006 – November 2010
- Features: Global active power, voltage, current, sub-metering readings

> Dataset not included in this repo due to size (124 MB).
> Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

---

## 🧠 ML Pipeline

### 1. Data Preprocessing
- Parsed datetime index from separate Date and Time columns
- Converted all columns to numeric, handled missing values
- Dropped 25,979 null rows (~1.3% of data)

### 2. Feature Engineering
```python
df['hours']       = df.index.hour        # 0–23
df['month']       = df.index.month       # 1–12
df['Day_of_week'] = df.index.dayofweek   # 0=Mon, 6=Sun
df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
```

### 3. Data Leakage Detection & Fix
Initial model scored R² = 0.998 — suspiciously perfect.

**Root cause:** `Global_intensity` (current in amps) and `Voltage` directly calculate active power via `P = V × I`. The model was doing physics, not learning.

**Fix:** Removed `Global_intensity`, `Voltage`, and `Global_reactive_power` from features.

**Result after fix:** R² = 0.792 (honest baseline)

### 4. Train/Test Split
Time-based split (not random) to prevent future data leaking into training:
- **Train:** December 2006 → February 2010 (1,639,424 rows)
- **Test:** February 2010 → November 2010 (409,856 rows)

---

## 📈 Model Results

### Regression — Power Consumption Prediction

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| Linear Regression | 0.792 | 0.310 kW | 0.409 kW |
| **XGBoost** | **0.871** | **–** | **–** |

### Classification — Usage Pattern (Low / Medium / High)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 🌙 Low (< 1 kW) | 0.94 | 0.94 | 0.94 |
| ⚡ Medium (1–3 kW) | 0.88 | 0.88 | 0.88 |
| 🔥 High (> 3 kW) | 0.73 | 0.75 | 0.74 |
| **Overall accuracy** | | | **0.91** |

### Anomaly Detection — Isolation Forest
- **2,010 anomalies** detected in test period (0.5% of readings)
- Notable finding: August 2010 shows near-zero consumption — household holiday

---

## 💰 Cost Optimisation

Time-of-use tariff model:

| Zone | Hours | Rate |
|------|-------|------|
| 🟢 Off-Peak | 11pm – 7am | 0.08 /kWh |
| 🟡 Standard | 7am – 5pm | 0.15 /kWh |
| 🔴 Peak | 5pm – 11pm | 0.25 /kWh |

Shifting high-usage appliances (>3 kW) from peak to off-peak hours yields ~**9.4% cost saving** with no reduction in consumption.

---

## 🖥️ Dashboard

Built with Streamlit — 4 interactive pages:

- **⚡ Predict My Usage** — Select appliances and time, get instant kWh + cost forecast with smart tips
- **📈 Model Performance** — Actual vs predicted scatter plot, R²/MAE/RMSE metrics
- **🚨 Anomaly Detection** — Full timeline with anomaly markers
- **📊 Usage Patterns** — Distribution of Low/Medium/High usage classifications

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/PavithraSivakumar-12/PowerSense.git
cd PowerSense

# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost matplotlib pickle5

# Download dataset from UCI and place in this folder
# Then run
streamlit run Dashboard.py
```

---

## 📁 Repository Structure

```
PowerSense/
│
├── Dashboard.py              # Streamlit web dashboard
├── power_consumption.ipynb   # Full ML pipeline notebook
├── xgb_model.sav             # Trained XGBoost model
├── iso_model.sav             # Trained Isolation Forest model
├── .gitignore                # Excludes large files
└── README.md                 # This file
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data processing
- **Scikit-learn** — Linear Regression, Random Forest, Isolation Forest
- **XGBoost** — primary regression model
- **Matplotlib** — visualisations
- **Streamlit** — interactive dashboard
- **Pickle** — model serialisation

---

## 📌 Key Learnings

- **Data leakage** can make a model look perfect (R²=0.998) while being completely useless in practice
- **Time-based train/test splits** are essential for sequential data — random splits inflate results
- **Feature importance** reveals that Number of Customers drives 60%+ of restaurant revenue patterns
- **Isolation Forest** is highly effective for unsupervised anomaly detection on time-series data

---

## 👩‍💻 Author

**Pavithra Sivakumar**
- GitHub: [@PavithraSivakumar-12](https://github.com/PavithraSivakumar-12)

---

*Dataset: UCI Machine Learning Repository — Individual Household Electric Power Consumption*
