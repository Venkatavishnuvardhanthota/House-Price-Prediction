# 🏠 House Price Prediction

A machine learning project for predicting house prices based on property characteristics such as area, bedrooms, bathrooms, stories, amenities, parking, preferred area, and furnishing status.

This project covers the complete machine learning workflow from data cleaning and exploratory data analysis to feature engineering, model training, evaluation, visualization, and model saving.

---

## 📌 Project Overview

House prices depend on several property characteristics. The goal of this project is to build regression models that can estimate the price of a house using the available property features.

Two regression models were developed and evaluated:

- **Linear Regression**
- **Random Forest Regressor**

The models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

The project also includes exploratory visualizations to understand relationships between property characteristics and house prices.

---

## 📊 Dataset

The project uses the **Housing Prices Dataset** from Kaggle.

### Dataset

[Housing Prices Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

### Dataset Details

| Property | Details |
|---|---|
| **Rows** | 545 |
| **Original Columns** | 13 |
| **Target Variable** | `price` |

### Features

| Feature | Description |
|---|---|
| `price` | House price |
| `area` | Area of the property in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the property is connected to the main road |
| `guestroom` | Whether the property has a guest room |
| `basement` | Whether the property has a basement |
| `hotwaterheating` | Whether hot water heating is available |
| `airconditioning` | Whether air conditioning is available |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the property is located in a preferred area |
| `furnishingstatus` | Furnishing status of the property |

---

## 🔄 Project Workflow

### 1. Data Loading

The raw `Housing.csv` dataset is loaded and inspected.

Initial checks include:

- Dataset shape
- First few records
- Column names
- Data types
- Missing values
- Duplicate rows
- Basic statistical information

---

### 2. Data Cleaning

The dataset is checked and prepared for machine learning.

The cleaning process includes:

- Checking for missing values
- Checking for duplicate rows
- Checking categorical values
- Converting categorical variables into numerical representations
- Checking data types
- Investigating numerical distributions and potential outliers

The dataset contains **no missing values or duplicate rows**.

---

### 3. Exploratory Data Analysis

Exploratory analysis is performed to understand the distribution of house prices and relationships between the available features.

The analysis includes:

- Price distribution
- Numerical feature relationships
- Correlation analysis
- Categorical feature analysis
- Feature importance analysis

---

### 4. Feature Engineering

Additional features are created from the available property characteristics.

The notebook creates:

- `bed_bath_count`
- `area_per_bed_bath`
- `luxury_score`

The processed dataset is saved for further modeling.

---

### 5. Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% test data**

A `random_state` of **42** is used to make the results reproducible.

---

### 6. Feature Scaling

`StandardScaler` is used for the **Linear Regression** model.

The scaler is fitted using the training data and then applied to the training and test datasets.

Random Forest does not require feature scaling because tree-based models are not affected by differences in feature magnitude in the same way as distance- or coefficient-based models.

---

### 7. Model Building

Two regression models are developed.

#### Linear Regression

Linear Regression is used as a baseline regression model for predicting house prices.

#### Random Forest Regressor

A Random Forest Regressor is trained to capture non-linear relationships between the property features and house prices.

`GridSearchCV` is used to tune the Random Forest hyperparameters.

A total of **108 parameter combinations** are evaluated using **5-fold cross-validation**.

---

### 8. Model Evaluation

The models are evaluated on the test dataset using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

### 9. Model Saving

The trained models and scaler are saved using **Joblib** so that they can be reused for future predictions.

---

## 📈 Model Results

The models were evaluated on the same test dataset.

| Model | MAE | R² Score |
|---|---:|---:|
| **Linear Regression** | **₹966,320** | **0.6583** |
| Random Forest | ₹1,049,055 | 0.5977 |

### Best Model

Based on the test-set results, **Linear Regression** achieved the higher R² score among the two evaluated models.

- **R² Score:** 0.6583
- **MAE:** ₹966,320

The R² score of **0.6583** means that the model explains approximately **65.83% of the variation in house prices** in the test dataset.

The MAE of **₹966,320** means that the model's predictions differ from the actual house prices by approximately **₹966,320 on average** on the test dataset.

For this dataset and test split, Linear Regression achieved a higher test-set R² score than the tuned Random Forest model.

---

## 🔍 Key Findings

The analysis shows noticeable relationships between house prices and several property characteristics.

Some of the features associated with house prices include:

- Area
- Bathrooms
- Bedrooms
- Furnishing status
- Parking
- Property amenities
- Preferred area

The Random Forest feature importance analysis provides an additional view of the relative contribution of the available features to the model's predictions.

> **Note:** These relationships describe patterns observed in the dataset and should not be interpreted as proof of causation.

---

## 💼 Business Interpretation

### For Sellers

Property characteristics such as larger area, additional bathrooms, furnishing status, and available amenities are associated with higher house prices in this dataset.

### For Buyers

The model can be used as a supporting tool to estimate an expected house price based on the characteristics of a property.

### Important Considerations

The model should be considered alongside other real-world factors such as:

- Exact location
- Market conditions
- Property age
- Construction quality
- Neighborhood characteristics
- Current demand

---

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── data/
│   ├── raw/
│   │   └── Housing.csv
│   │
│   └── processed/
│       └── Housing_cleaned.csv
│
├── notebooks/
│   └── House_Price_Prediction.ipynb
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
├── predictions/
│   └── sample_predictions.csv
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Jupyter Notebook**

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd House_Price_Prediction
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

Open the notebook located in the `notebooks/` directory using **Jupyter Notebook** or **VS Code**.

Run the notebook cells from top to bottom to reproduce the complete machine learning workflow:

1. **Data loading**
2. **Data cleaning**
3. **Exploratory data analysis**
4. **Feature engineering**
5. **Train-test split**
6. **Model training**
7. **Hyperparameter tuning**
8. **Model evaluation**
9. **Visualizations**
10. **Model saving**
11. **Sample predictions**

### Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/House_Price_Prediction.ipynb
```

---

## 📊 Visualizations

The project includes several visualizations to analyze the dataset and model performance.

### Price Distribution

The distribution of house prices is visualized to understand the overall price range and distribution.

**File:**

```text
charts/chart1_price_distribution.png
```

### Correlation Heatmap

A correlation heatmap is used to understand relationships between numerical features and house prices.

**File:**

```text
charts/chart2_correlation_heatmap.png
```

### Actual vs Predicted Prices

This visualization compares the actual house prices with the prices predicted by the model.

**File:**

```text
charts/chart3_actual_vs_predicted.png
```

### Feature Importance

The Random Forest model's feature importance is visualized to understand the relative contribution of different features to the predictions.

**File:**

```text
charts/chart4_feature_importance.png
```

### Price vs Area and Furnishing

A visualization is used to explore the relationship between property area, furnishing status, and house prices.

**File:**

```text
charts/chart5_price_vs_area_furnishing.png
```

---

## 💾 Saved Models

The trained machine learning models and preprocessing scaler are saved using **Joblib**.

### Linear Regression Model

```text
models/linear_regression_model.pkl
```

### Random Forest Model

```text
models/random_forest_model.pkl
```

### Feature Scaler

```text
models/scaler.pkl
```

These saved files can be loaded later to make predictions without retraining the models.

---

## 🔮 Sample Predictions

Sample model predictions are saved in:

```text
predictions/sample_predictions.csv
```

The predictions can be used to compare the model's estimated house prices with the actual values from the test dataset.

---

## 📏 Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the predicted and actual house prices.

A lower MAE indicates that the predictions are, on average, closer to the actual values.

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared difference between predicted and actual values.

Because errors are squared before averaging, RMSE gives more weight to larger prediction errors.

### R² Score

R² measures the proportion of variance in the target variable that is explained by the model.

A higher R² generally indicates that the model explains more of the variation in the target variable.

---

## ⚠️ Limitations

This project is based on a single housing dataset and has several limitations:

- The dataset contains a relatively small number of observations.
- The available features do not represent every factor that can influence house prices.
- Exact location information is not available.
- Property age is not included.
- Construction quality is not included.
- Neighborhood characteristics are not included.
- Current market conditions are not included.
- Model performance is based on the selected train-test split.
- The results may not generalize to other housing markets or future market conditions.

Therefore, the model should be considered an **educational and analytical prediction tool** rather than a complete property valuation system.

---

## 🏁 Conclusion

This project demonstrates a complete machine learning workflow for a house price regression problem.

The project covers:

- Data cleaning
- Exploratory data analysis
- Feature engineering
- Categorical encoding
- Feature scaling
- Model building
- Hyperparameter tuning
- Model evaluation
- Data visualization
- Model saving

Among the two evaluated models, **Linear Regression achieved the higher test-set R² score of 0.6583**, with an MAE of approximately **₹966,320**.

The project demonstrates how machine learning can be applied to estimate house prices from structured property data while highlighting the importance of model evaluation and understanding dataset limitations.

---

## Contact

**Thota Venkata Vishnu Vardhan**

📧 Email: [venkatavishnuvardhanthota@gmail.com](mailto:venkatavishnuvardhanthota@gmail.com)

💻 GitHub: https://github.com/Venkatavishnuvardhanthota

🔗 LinkedIn: https://www.linkedin.com/in/venkata-vishnu-vardhan-thota/