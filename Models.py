#This script here is the code for the random forest ML model. It involves training the ML model with 80% of the datset entries while reseving the other 20% for testing. Some charts/graphs are also included here to better help visualize the accuracy of this model, we will also use R^2, MAE and RMSE as metrics to measure this. We could definetly move the charts and graphs to the evaluation file though.

# import commands

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv("car_sales_data.csv")

# Drop duplicates + handle missing values
df = df.drop_duplicates()
df = df.dropna()   # or df.fillna(method='ffill') depending on your dataset

# Separate features from the target
X = df.drop("Price", axis=1)
y = df["Price"]

# 4. Detect categorical vs numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Build the Random Forest pipeline
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_split=2
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
