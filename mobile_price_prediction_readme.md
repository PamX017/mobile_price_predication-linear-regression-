# 📱 Mobile Phone Price Prediction

A Machine Learning project using **Linear Regression** to predict mobile phone prices based on hardware specifications and brand metadata. The pipeline features automated data cleaning, regular expression feature extraction, categorical encoding via `scikit-learn` Pipelines, and performance evaluation.

---

## 📌 Project Overview

This model processes unstructured metadata from modern mobile phone listings (such as multi-camera setup strings and formatted currency values) to build an end-to-end regression pipeline.

### Key Highlights:
- **Data Preprocessing**: Regular expressions to extract numerical values from strings (RAM, Storage, Screen Size, Camera Megapixels, and Price).
- **Feature Engineering**: Standardized extraction of primary camera specs from multi-lens configurations (e.g., `12 + 12 + 12`).
- **Scikit-Learn Pipeline**: Integrated `ColumnTransformer` with `OneHotEncoder` for categorical features (`Brand`) and `LinearRegression` estimation.
- **Evaluation**: Evaluated using $R^2$ Score, MAE, and RMSE metrics along with Actual vs. Predicted visualizations.

---

## 📊 Model Performance

Evaluated on a 20% hold-out test set ($80/20$ train-test split):

| Metric | Score / Value |
| :--- | :--- |
| **$R^2$ Score** | **0.7895** (78.95% variance explained) |
| **Mean Absolute Error (MAE)** | **\$89.88** |
| **Root Mean Squared Error (RMSE)** | **\$138.69** |

---

## 📁 Dataset & Features

The dataset (`Mobile phone price.csv`) consists of hardware specifications and pricing details across multiple smartphone brands.

### Features Used:
- **`Brand`** *(Categorical)*: Manufacturer name (One-Hot Encoded).
- **`Storage`** *(Numerical, GB)*: Internal storage capacity extracted as float.
- **`RAM`** *(Numerical, GB)*: Memory capacity extracted as float.
- **`Screen Size (inches)`** *(Numerical)*: Display diagonal size in inches.
- **`Primary_Camera_MP`** *(Numerical)*: Primary sensor resolution extracted from raw camera specs.
- **`Battery Capacity (mAh)`** *(Numerical)*: Battery power capacity.

### Target Variable:
- **`Price ($)`** *(Numerical)*: Retail price in USD cleaned from string formatting.

---

## 🛠️ Technology Stack

- **Python 3.x**
- **Pandas** & **NumPy** for data manipulation and regular expression extraction.
- **Scikit-Learn** for modeling, preprocessing pipelines, and evaluation metrics.
- **Matplotlib** & **Seaborn** for data visualization.

---

## ⚙️ Installation & Usage

### 1. Requirements
Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Dataset Setup
Place the `Mobile phone price.csv` file in the root directory alongside the notebook or script.

### 3. Execution Pipeline
Run the code cells sequentially or execute via script:

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 1. Load Data
df = pd.read_csv('Mobile phone price.csv')
df.columns = df.columns.str.strip()

# 2. Data Cleaning & Feature Extraction
df['Storage'] = df['Storage'].astype(str).str.extract(r'(\d+)').astype(float)
df['RAM'] = df['RAM'].astype(str).str.extract(r'(\d+)').astype(float)
df['Screen Size (inches)'] = df['Screen Size (inches)'].astype(str).str.extract(r'([\d.]+)').astype(float)
df['Price ($)'] = df['Price ($)'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

def get_primary_camera(val):
    nums = re.findall(r'[\d.]+', str(val))
    return float(nums[0]) if nums else np.nan

df['Primary_Camera_MP'] = df['Camera (MP)'].apply(get_primary_camera)
df_clean = df.dropna(subset=['Storage', 'RAM', 'Screen Size (inches)', 'Primary_Camera_MP', 'Battery Capacity (mAh)', 'Price ($)'])

# 3. Features & Target
X = df_clean[['Brand', 'Storage', 'RAM', 'Screen Size (inches)', 'Primary_Camera_MP', 'Battery Capacity (mAh)']]
y = df_clean['Price ($)']

# 4. Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Storage', 'RAM', 'Screen Size (inches)', 'Primary_Camera_MP', 'Battery Capacity (mAh)']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Brand'])
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
```

---

## 🔮 Making Predictions

Predict prices for new device specifications using custom pandas DataFrames:

```python
sample_device = pd.DataFrame([{
    'Brand': 'Samsung',
    'Storage': 128.0,
    'RAM': 8.0,
    'Screen Size (inches)': 6.5,
    'Primary_Camera_MP': 48.0,
    'Battery Capacity (mAh)': 5000
}])

predicted_price = model.predict(sample_device)
print(f"Estimated Price: ${predicted_price[0]:.2f}")
# Output: Estimated Price for Sample Phone: $572.24
```