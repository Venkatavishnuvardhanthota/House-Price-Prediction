# 🏠 House Price Prediction

A machine learning project for predicting house prices based on property characteristics such as area, bedrooms, bathrooms, stories, amenities, parking, preferred area, and furnishing status.

This project covers the complete machine learning workflow from data loading, cleaning, categorical encoding, and feature engineering to model training, hyperparameter tuning, evaluation, model saving, and visualization.

---

## 📌 Project Overview

House prices depend on several property characteristics. The goal of this project is to build regression models that estimate the price of a house using the available property features.

Two regression models were developed and evaluated:

- **Linear Regression**
- **Random Forest Regressor**

The models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

The project also includes exploratory and diagnostic visualizations to examine price distributions, feature correlations, feature importances, and model predictions.

---

## 📊 Dataset

The project uses the **Housing Prices Dataset** from Kaggle.

### Source

[Housing Prices Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

### Dataset Details

| Property | Details |
|---|---|
| **Rows** | 545 |
| **Original Columns** | 13 |
| **Processed Columns** | 17 (after one-hot encoding and feature engineering) |
| **Target Variable** | `price` |

### Original Features

| Feature | Description |
|---|---|
| `price` | House price (in dataset units; currency is not specified in the Kaggle source) |
| `area` | Area of the property (in dataset units; measurement unit is not specified in the Kaggle source) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the property is connected to the main road (`yes` / `no`) |
| `guestroom` | Whether the property has a guest room (`yes` / `no`) |
| `basement` | Whether the property has a basement (`yes` / `no`) |
| `hotwaterheating` | Whether hot water heating is available (`yes` / `no`) |
| `airconditioning` | Whether air conditioning is available (`yes` / `no`) |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the property is located in a preferred area (`yes` / `no`) |
| `furnishingstatus` | Furnishing status of the property (`furnished`, `semi-furnished`, `unfurnished`) |

---

## 🔄 Project Workflow

The workflow below matches the exact execution order in `notebooks/analysis.ipynb`:

### 1. Data Loading

The raw `data/raw/Housing.csv` dataset is loaded and inspected:

- First 10 records (`df.head(10)`)
- Dataset shape (`545` rows, `13` columns)
- Column names and data types
- Basic statistical summary (`df.describe()`)
- Missing value counts per column (`0` missing values)

---

### 2. Data Cleaning & Categorical Encoding

The dataset is checked and encoded into numeric features for machine learning:

- Checking for duplicate rows (`0` duplicate rows)
- Inspecting unique values across all categorical columns
- Mapping binary `yes`/`no` columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) to `1` and `0`
- One-hot encoding `furnishingstatus` (`drop_first=True`) into `furnishingstatus_semi-furnished` and `furnishingstatus_unfurnished`
- Verifying that all resulting columns are numeric (`14` columns, `0` missing values)

---

### 3. Feature Engineering

Three additional features are engineered from the existing property attributes:

- `bed_bath_count`: Total bedrooms and bathrooms (`bedrooms + bathrooms`)
- `area_per_bed_bath`: Property area divided by total bedrooms and bathrooms (`area / bed_bath_count`)
- `luxury_score`: Sum of five binary indicators (`airconditioning + hotwaterheating + prefarea + guestroom + basement`)

The cleaned and engineered dataset (`545` rows, `17` columns) is saved to `data/processed/Housing_cleaned.csv`.

---

### 4. Train-Test Split

The dataset is separated into `16` features (`X`) and the target variable `price` (`y`), then split into:

- **80% training data** (`436` rows)
- **20% test data** (`109` rows)

A `random_state` of **42** is used for reproducibility.

---

### 5. Feature Scaling

`StandardScaler` is fitted on the training features (`X_train`) and used to transform both `X_train` and `X_test` for the **Linear Regression** model.

Random Forest is trained on the unscaled features because tree-based splits are invariant to monotonic feature scaling.

---

### 6. Model Building & Hyperparameter Tuning

Two regression models are trained and compared:

#### Linear Regression

Trained on the standardized training features (`X_train_scaled`) as a baseline linear model.

#### Random Forest Regressor

Tuned using `GridSearchCV` across **108 hyperparameter combinations** with **5-fold cross-validation** (`540` total fits):

- `n_estimators`: `[100, 200, 300]`
- `max_depth`: `[3, 5, 7, None]`
- `min_samples_split`: `[2, 5, 10]`
- `min_samples_leaf`: `[1, 2, 4]`

Best parameters found: `{'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}` (best 5-fold cross-validated R² = **0.6351**).

---

### 7. Model Evaluation

Both models are evaluated on the held-out test set (`109` rows) using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

### 8. Model Saving

The trained Linear Regression model, tuned Random Forest model, and fitted `StandardScaler` are serialized to `models/` using **Joblib**.

---

### 9. Exploratory & Evaluation Visualizations

Five visualizations are generated and saved to `charts/`:

- Raw and log-transformed (`log(1 + Price)`) house price distributions
- Lower-triangle feature correlation heatmap across all 17 numeric columns
- Actual vs. predicted test-set prices for both models
- Random Forest feature importance bar chart
- House price vs. area scatter plot colored by furnishing status

---

## 📈 Model Results

Both models were evaluated on the same held-out test dataset (`109` rows). Because the Kaggle source dataset does not specify a currency unit for `price`, error metrics are reported in raw dataset price units.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| **Linear Regression** | **966,320** | **1,314,226** | **0.6583** |
| Random Forest | 1,049,055 | 1,426,042 | 0.5977 |

### Best Model

Based on the test-set results, **Linear Regression** achieved the higher R² score and lower error metrics among the two evaluated models:

- **R² Score:** 0.6583
- **MAE:** 966,320
- **RMSE:** 1,314,226

The R² score of **0.6583** indicates that the Linear Regression model explains approximately **65.83% of the variance in house prices** in the test dataset.

The MAE of **966,320** indicates that the model's predictions differ from actual test-set house prices by **966,320 units on average**.

---

## 🔍 Key Findings

- **`area`** shows the strongest positive linear correlation with `price` ($r = 0.54$) and is the top contributor in Random Forest feature importance ($0.462$).
- **`bed_bath_count` and `bathrooms`** are strongly associated with house price ($r = 0.51$ and $r = 0.52$; Random Forest importances of $0.100$ and $0.085$, respectively).
- **`luxury_score`** (combining `airconditioning`, `hotwaterheating`, `prefarea`, `guestroom`, and `basement`) correlates strongly with `price` ($r = 0.51$) and ranks fourth in Random Forest importance ($0.081$).
- **Linear Regression generalized better than Random Forest** on this 545-row dataset and 80/20 split (Test R² of `0.6583` vs. `0.5977`).

> **Note:** These relationships describe associations observed in the dataset and should not be interpreted as proof of causation.

---

## 💼 Practical Interpretation

### For Sellers and Buyers

Within this dataset, properties with larger `area`, more `bathrooms`, a higher `luxury_score` (air conditioning, preferred area, basement, guest room, hot water heating), and furnished status are associated with higher prices.

### Important Considerations

Any real-world property valuation must also account for factors not captured in this dataset:

- Exact geographic location and neighborhood
- Currency and measurement units
- Property age and physical condition
- Construction quality
- Macroeconomic and local market conditions

---

## 📁 Project Structure

```text
House Price Prediction/
│
├── data/
│   ├── raw/
│   │   └── Housing.csv
│   │
│   └── processed/
│       └── Housing_cleaned.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── charts/
│   ├── chart1_price_distribution.png
│   ├── chart2_correlation_heatmap.png
│   ├── chart3_actual_vs_predicted.png
│   ├── chart4_feature_importance.png
│   └── chart5_price_vs_area_furnishing.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

- **Python 3.13**
- **Pandas** (`3.0.3`)
- **NumPy** (`2.5.1`)
- **Matplotlib** (`3.11.0`)
- **Seaborn** (`0.13.2`)
- **Scikit-learn** (`1.9.0`)
- **Joblib** (`1.5.3`)
- **Jupyter Notebook**

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "House Price Prediction"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Open `notebooks/analysis.ipynb` in **Jupyter Notebook** or **VS Code**. Because the notebook resolves paths dynamically using `PROJECT_ROOT`, it can be launched from either the project root directory or the `notebooks/` directory.

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells from top to bottom to reproduce:

1. **Data loading and summary statistics**
2. **Data cleaning and categorical encoding**
3. **Feature engineering and saving `data/processed/Housing_cleaned.csv`**
4. **Train-test split and `StandardScaler` preprocessing**
5. **Linear Regression training and test-set evaluation**
6. **Random Forest `GridSearchCV` tuning and test-set evaluation**
7. **Saving models and scaler to `models/`**
8. **Generating and saving all 5 charts to `charts/`**

---

## 📊 Visualizations

### 1. Price Distribution

Displays the raw house price distribution (with mean `4.77M` and median `4.34M` reference lines) alongside the log-transformed `log(1 + Price)` distribution.

- **File:** `charts/chart1_price_distribution.png`

### 2. Feature Correlation Heatmap

Displays Pearson correlation coefficients across all 17 numeric features (`Light Teal/Green = Positive, Dark Purple = Negative`).

- **File:** `charts/chart2_correlation_heatmap.png`

### 3. Actual vs. Predicted Prices

Compares actual vs. predicted test-set prices for both Linear Regression (`R² = 0.6583`) and Random Forest (`R² = 0.5977`).

- **File:** `charts/chart3_actual_vs_predicted.png`

### 4. Random Forest Feature Importance

Ranks all 16 features by their relative importance in the tuned Random Forest model.

- **File:** `charts/chart4_feature_importance.png`

### 5. Price vs. Area by Furnishing Status

Illustrates the relationship between property `area` and `price` across `Furnished`, `Semi-Furnished`, and `Unfurnished` properties.

- **File:** `charts/chart5_price_vs_area_furnishing.png`

---

## 💾 Saved Models

The trained models and preprocessing scaler are saved using **Joblib**:

- **Linear Regression Model:** `models/linear_regression_model.pkl`
- **Random Forest Model:** `models/random_forest_model.pkl`
- **Feature Scaler:** `models/scaler.pkl`

> **Environment Note:** These `.pkl` files were serialized using **Python 3.13.1**, **scikit-learn 1.9.0**, **numpy 2.5.1**, and **joblib 1.5.3** (pinned in `requirements.txt`). Use matching package versions when loading these artifacts to avoid version mismatch warnings or deserialization issues.

---

## 📏 Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual house prices. Lower values indicate better average accuracy.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error, penalizing large errors more heavily than MAE.

### R² Score

Measures the proportion of variance in the target variable (`price`) explained by the model. Higher values (closer to `1.0`) indicate better explanatory power.

---

## ⚠️ Limitations

This project is based on a single dataset and has several important limitations:

- The dataset is small (`545` rows total; `436` training rows and `109` test rows).
- The Kaggle source dataset does not document the currency unit for `price` or the measurement unit for `area`.
- Exact geographic coordinates, city/region, and neighborhood data are not included.
- Property age, architectural style, and construction quality are not included.
- Temporal and market-condition variables (e.g., year sold, interest rates) are not included.
- Model evaluation is based on a single 80/20 train-test split (`random_state=42`).

Accordingly, this project is intended as an **educational machine learning and data analysis portfolio project** rather than a production property appraisal system.

---

## 🏁 Conclusion

This project demonstrates an end-to-end regression workflow on structured housing data—covering data inspection, categorical encoding, feature engineering, scaling, baseline and ensemble modeling, cross-validated hyperparameter tuning, model serialization, and visual diagnostics.

On the held-out test set, **Linear Regression achieved the stronger performance (`R² = 0.6583`, `MAE = 966,320`, `RMSE = 1,314,226`)** compared to the tuned **Random Forest Regressor (`R² = 0.5977`, `MAE = 1,049,055`, `RMSE = 1,426,042`)**.

---

## 📬 Contact

**Thota Venkata Vishnu Vardhan**

Email: [venkatavishnuvardhanthota@gmail.com](mailto:venkatavishnuvardhanthota@gmail.com)

GitHub: [https://github.com/Venkatavishnuvardhanthota](https://github.com/Venkatavishnuvardhanthota)

LinkedIn: [https://www.linkedin.com/in/venkata-vishnu-vardhan-thota/](https://www.linkedin.com/in/venkata-vishnu-vardhan-thota/)