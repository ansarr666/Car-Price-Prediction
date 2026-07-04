# 🚗 Car Price Prediction — End-to-End ML Project

**Problem Statement:** Build an ML regression pipeline to predict vehicle MSRP based on brand, engine specs, and vehicle characteristics — enabling automated dealership pricing and protecting profit margins.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Car Features and MSRP](https://www.kaggle.com/CooperUnion/cardataset) |
| **Records** | 11,914 cars (11,199 after deduplication) |
| **Features** | 16 original features |
| **Target** | MSRP (Manufacturer Suggested Retail Price) |

---

## 🔬 Project Pipeline

```
Data Loading → Cleaning → EDA → Feature Engineering → Train-Test Split → Model Training → Evaluation → Model Saving
```

### 1. Data Cleaning
- Standardized column names and text formatting
- Removed 715 duplicate records
- Imputed missing values (median for numeric, mode for categorical)

### 2. Feature Engineering
- **`age`** — Derived from `2017 - year` to capture vehicle depreciation
- **`hp_per_cylinder`** — Engine efficiency metric (`engine_hp / engine_cylinders`)
- **Log-transformed MSRP** — Reduced skewness from 11.61 → −0.93 for Linear Regression

### 3. EDA Highlights
- MSRP is heavily right-skewed; log transformation normalizes the distribution
- `engine_hp` and `engine_cylinders` are the strongest price predictors
- Dropped `highway_mpg`, `year`, `popularity`, and `market_category` to reduce multicollinearity

### 4. Models Trained

| Model | Target Scale | Preprocessing |
|---|---|---|
| Linear Regression | Log-transformed | StandardScaler + OneHotEncoder |
| Random Forest | Raw dollars | Passthrough + OneHotEncoder |
| XGBoost | Raw dollars | Passthrough + OneHotEncoder |

---

## 📈 Results

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| **Random Forest** 🏆 | **3,518** | **11,578** | **7.71** | **0.9534** |
| XGBoost | 4,604 | 11,964 | 15.00 | 0.9503 |
| Linear Regression | 4,096 | 13,759 | 10.81 | 0.9342 |

**Winner: Random Forest** — Lowest MAPE (7.71%) and highest R² (0.9534), making it the most accurate model for pricing mainstream consumer vehicles.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** & **NumPy** — Data manipulation
- **Matplotlib** & **Seaborn** — Visualization
- **scikit-learn** — Pipelines, preprocessing, Linear Regression, Random Forest
- **XGBoost** — Gradient boosting regressor
- **joblib** — Model serialization

---

## 📂 Project Structure

```
Car-Price-Prediction/
├── data/
│   └── car_price_prediction.csv    # Raw dataset
├── models/
│   └── car_price_rf_model.pkl      # Saved Random Forest model
├── car_price_prediction.ipynb      # Main notebook (full analysis)
├── car_price_prediction.py         # Python script version
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/Car-Price-Prediction.git
cd Car-Price-Prediction

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook car_price_prediction.ipynb
```

---

## 🔑 Key Takeaways

- **Engine power dominates pricing** — `engine_hp` and `engine_cylinders` account for 63% of the Random Forest's decision-making
- **Log transformation is essential** for Linear Regression to avoid outlier bias from luxury supercars
- **Random Forest excels at mainstream vehicles** (under $150k) but struggles with rare exotics — in production, luxury cars would be routed to manual valuation

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
