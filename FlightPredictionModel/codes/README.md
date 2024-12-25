<!-- # Flight Price Prediction Machine Learning Model

## Overview

This project demonstrates the process of building machine learning models to predict flight prices using German air fare data. We explore and compare the performance of three different algorithms: Random Forest Regressor, Gradient Boosting Regressor, and Random Forest Classifier.

## Goal

This project demonstrates how to build machine learning models to predict flight prices. It explores three different models:
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Random Forest Classifier**

The workflow of the project includes:
1. Data loading and exploration
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Hyperparameter tuning (optional)
5. Prediction using the trained model


## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Data Processing](#data-processing)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Authors](#authors)

## Authors

1. Ba Viet Anh (Henry) Nguyen


## How to Run

### 1. Configure the Project Environment

1. Create a new conda environment:

   ```bash
   conda create -n flight_price_env python=3.10.9
   ```

2. Activate the environment:

   ```bash
   conda activate flight_price_env
   ```

3. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn datetime scikit-optimize statsmodels mlxtend
   ```


### Data Processing
Before training the model, the dataset must be prepared through data preprocessing. This step ensures that the data is clean, features are encoded correctly, and the dataset is split into training and testing sets.

### Step 1: Load the Dataset
Ensure that the dataset is in the correct directory. Use the following code to load the data:

import pandas as pd

# Load the dataset
df = pd.read_csv('german_flight_data.csv')

# Preview the first few rows
df.head()

### Step 2: Data Preprocessing
To prepare the data for model training, we need to:
  Drop irrelevant columns.
  Encode categorical variables: Convert categorical features into numerical values.
# Drop columns that aren't needed
df = df.drop(['column_to_drop'], axis=1)

# Encode categorical columns
df['category_column'] = df['category_column'].astype('category').cat.codes

### Step3: Step 3: Split the Dataset
Next, we split the dataset into training and test sets. This allows us to train the model on one portion and evaluate it on another to measure performance:

from sklearn.model_selection import train_test_split

# Define features and target variable
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### MODEL TRAINING
This section covers training the machine learning models. The example below shows how to train a Random Forest Regressor.

### Step 1: Initialize and Train the Model
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

### Step 2: Evaluate the Model
Once the model is trained, you can evaluate its performance by predicting the target values on the test set:
# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

### Step 3: Calculate Performance Metrics
You can calculate the mean squared error to assess how well the model performed:

from sklearn.metrics import mean_squared_error

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

### USING THE MODEL FOR PREDICTION
Once the model has been trained and evaluated, you can use it to predict flight prices for new data. Here is a step-by-step guide to using the model for predictions.

### Step 1: Prepare the Input Data for Prediction
Make sure that any new input data for prediction is preprocessed in the same way as the training data. This includes applying any categorical encoding and removing irrelevant columns.

### Step 2: Use the Trained Model to Make Predictions
Once the new data is preprocessed, you can use the trained model (rf_regressor) to make predictions on this data.
# Load new flight data for prediction
new_data = pd.read_csv('new_flight_data.csv')

# Predict flight prices for the new data
predictions = rf_regressor.predict(new_data)

# Display the predictions
print(predictions)

### MAKING PREDICTIONS
Once the model has been trained and evaluated, you can use it to predict flight prices for new data.

### Step 1: Prepare the Input Data
Ensure that the input data is preprocessed in the same way as the training data (e.g., encoding, dropping irrelevant columns).

# Load new data
new_data = pd.read_csv('new_flight_data.csv')

# Preprocess the new data
new_data['category_column'] = new_data['category_column'].astype('category').cat.codes
new_data = new_data.drop(['column_to_drop'], axis=1)

### Step 2: Make Predictions
Use the trained model to predict prices for the new data:

# Predict flight prices
predictions = rf_regressor.predict(new_data)

# Display predictions
print(predictions)

### Step 3: Save Predictions (Optional)
If you want to save the predictions for further use:

# Save predictions to a CSV file
new_data['predicted_price'] = predictions
new_data.to_csv('predicted_flight_prices.csv', index=False) -->

# Flight Price Prediction Machine Learning Model

## Overview

This project demonstrates the process of building machine learning models to predict flight prices using German air fare data. We explore and compare the performance of three different algorithms: Random Forest Regressor, Gradient Boosting Regressor, and Random Forest Classifier.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Processing](#data-processing)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Making Predictions](#making-predictions)

## Environment Setup

1. Create a new conda environment:

   ```bash
   conda create -n flight_price_env python=3.10.9
   ```

2. Activate the environment:

   ```bash
   conda activate flight_price_env
   ```

3. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn datetime scikit-optimize statsmodels mlxtend
   ```

## Data Processing

### Step 1: Load the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('German Air Fares.csv')
print(df.head(5))
```

### Step 2: Data Cleaning and Preprocessing

```python
# Remove null values and duplicates
df = df.dropna().drop_duplicates()

#strip whitespace from string column 
object_column = df.select_dtypes(include="object").columns
df[object_column] = df[object_column].apply(lambda x: x.str.strip())

# Convert price to float
df["price (€)"] = df["price (€)"].str.replace(",", "").str.replace("€", "").str.strip().astype(float)
df.rename(columns={'price (€)': 'price'}, inplace=True)

# Convert date columns
df["departure_date"] = pd.to_datetime(df["departure_date"], format='%d.%m.%Y')
df['scrape_date'] = pd.to_datetime(df["scrape_date"], format='%d.%m.%Y')

# Convert stops to numerical
df["stops"] = df["stops"].replace({"direct": 0, "(1 Stopp)": 1, "(1 stop)": 1, "(2 Stopps)": 2})

# Standardize time formats
def standardize_time(time_str):
    # Implementation of standardize_time function
    pass

df["arrival_time"] = df["arrival_time"].apply(standardize_time)
df["departure_time"] = df["departure_time"].apply(standardize_time)

#convert arrival_time and departure_time to datetime datatype
df["arrival_time"] = pd.to_datetime(df["arrival_time"], format='%H:%M').dt.time
df["departure_time"] = pd.to_datetime(df["departure_time"], format='%H:%M').dt.time

# Convert departure_date_distance
def convert_number_date_distance(time_str):
    # Implementation of convert_number_date_distance function
    pass

df["departure_date_distance"] = df["departure_date_distance"].apply(convert_number_date_distance)
```

### Step 3: Feature Engineering

```python
# function to convert time to minutes
def times_to_minute (time_obj):
   # Implementation of times_to_minute function
   pass
# Create new features
df["flight_duration_in_minutes"] = df["arrival_time"].apply(times_to_minute) - df["departure_time"].apply(times_to_minute)
df.loc[(df['flight_duration_in_minutes'] < 150) & (df['stops'] != 0), 'flight_duration_in_minutes'] = (1440 - df['departure_time'].apply(times_to_minute)) + df['arrival_time'].apply(times_to_minute)
df["departure_time_in_minutes_from_midnight"] = df["departure_time"].apply(times_to_minute)
df["day_of_week"] = df["departure_date"].dt.weekday
df["day_of_month"] = df["departure_date"].dt.day
df["month"] = df["departure_date"].dt.month
df["year"] = df["departure_date"].dt.year

# Categorize prices
df['price_category'] = pd.cut(df['price'], bins=[-float('inf'), 200, 500, float('inf')], labels=['budget', 'moderate', 'expensive'])

# Drop irrelevant columns
df.drop(columns=["scrape_date", "departure_time", "arrival_time", "departure_date"], inplace=True, axis=1)
```

### Step 4: Handle Outliers
- Use IQR method to detect outliers
```python
def detect_outliers(df, cols) -> dict:
    outliers: dict = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)  # 1st quartile
        Q3 = df[col].quantile(0.75)  # 3rd quartile
        IQR = Q3 - Q1  # interquartile range
        lower_bound = Q1 - 1.5 * IQR  # lower bound
        upper_bound = Q3 + 1.5 * IQR  # upper bound
        outliers[col] = df.loc[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]  #count of outliers in each column (column: count of rows)
    return outliers
```
- Use capping method to handle outliers
```python
for col in numerical_cols:
    lower_bound = df[col].quantile(0.01)  # 1st percentile
    upper_bound = df[col].quantile(0.99)  # 99th percentile
    df[col] = df[col].clip(lower_bound, upper_bound)
```

### Step 5: Feature Selection

```python
def stepwise_regression(X, y, significance_level_in=0.05, significance_level_out=0.05):
    # Implementation of stepwise_regression function
    pass

def feature_selection_ramdomforestclassifier(X, y):
    # Implementation of feature_selection_ramdomforestclassifier function
    pass
# For regression
final_features_rg = stepwise_regression(X_train_preprocessed_rg, y_train_rg)
X_train_final_rg = X_train_preprocessed_rg[final_features_rg]
X_test_final_rg = X_test_preprocessed_rg[final_features_rg]

# For classification
selected_features = feature_selection_ramdomforestclassifier(X_train_preprocessed_cl, z_train_cl)
X_train_final_cl = X_train_preprocessed_cl[selected_features]
X_test_final_cl = X_test_preprocessed_cl[selected_features]
```

## Model Training and Model Evaluation

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

model1 = RandomForestRegressor(random_state=9214)
model1.fit(X_train_final_rg, y_train_rg)
```

### Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor

model2 = GradientBoostingRegressor(random_state=9214)
model2.fit(X_train_final_rg, y_train_rg)
```

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(random_state=9214)
model3.fit(X_train_final_cl, z_train_cl)
```

## Model Evaluation

```python
# function to call the model with numeric scorer (evaluation metric)
def model_train(model_name, X_train, X_test, y_train, y_test):
    model = model_name.fit(X_train, y_train)
    print(f"Training Score: {model.score(X_train, y_train)}")
    print(f"Test Score: {model.score(X_test, y_test)}")
    predict_value = model.predict(X_test)
    print(f"R^2 Score: {r2_score(y_test, predict_value)}")
    print(f"Mean Square Error: {mean_squared_error(y_test, predict_value)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predict_value)}")

# function to call the classification model (evaluation metric)
def model_train_classification(model_name, X_train, X_test, y_train, y_test):
    print(f"Classification Model: {model_name}")
    model = model_name.fit(X_train, y_train)  # fit the training data into model
    predict_value = model.predict(X_test)  # predict the result of the test set
    print(f"Accuracy: {accuracy_score(y_test, predict_value)}")
    print(f"Precision: {precision_score(y_test, predict_value, average='weighted')}")
    print(f"Recall: {recall_score(y_test, predict_value, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, predict_value, average='weighted')}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predict_value)}")

# Evaluate regression models
model_train(model1, X_train_final_rg, X_test_final_rg, y_train_rg, y_test_rg)
model_train(model2, X_train_final_rg, X_test_final_rg, y_train_rg, y_test_rg)

# Evaluate classification model
model_train_classification(model3, X_train_final_cl, X_test_final_cl, z_train_cl, z_test_cl)
```

### Cross validation
```python
# the data use to train the regression model also use to validation (should be the whole data not the splitted data because will use kfold to valid the model)
X_cross_val_rg = df_regression.drop(columns=['price'], axis=1)
y_cross_val_rg = df_regression['price']
#validation for the regression model
models = [model1, model2]  #list of models
for i, model in enumerate(models, start=1):
    cv_scores = cross_val_score(model, X_cross_val_rg, y_cross_val_rg, cv=5)  # 5-fold cross-validation

    print(f"Model {i} Cross-validation scores: {cv_scores}")
    print(f"Model {i} Mean CV score: {cv_scores.mean():.4f}")
    print(f"Model {i} Standard deviation of CV score: {cv_scores.std():.4f}")

# the data use to train the classification model also use to validation (should be the whole data not the splitted data because will use kfold to valid the model)
X_cross_val_cl = df_classification.drop(columns=['price_category'], axis=1)
y_cross_val_cl = df_classification['price_category']

# validation for the classification model
models = [model3] #list of models

# Define the metrics you want to use
metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

# Perform cross-validation for each model
for i, model in enumerate(models, 1):
    print(f"\nModel {i}:")
    for metric_name, metric_func in metrics.items():
        scores = cross_val_score(model, X_cross_val_cl, y_cross_val_cl, cv=5, scoring=make_scorer(metric_func))
        print(f"{metric_name}:")
        print(f"  Scores: {scores}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std. Dev: {scores.std():.4f}")
```

## Hyperparameter Tuning

### Random Forest Regressor

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [500, 600],
    'max_depth': [35, 40],
    'min_samples_split': [15, 20],
    'min_samples_leaf': [3, 4],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

ran_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=9214),
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=custom_score
)

ran_search.fit(X_train_final_rg, y_train_rg)
best_rf = RandomForestRegressor(**ran_search.best_params_, random_state=9214)
best_rf.fit(X_train_final_rg, y_train_rg)
```

### Gradient Boosting Regressor

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'subsample': [0.8],
    'max_features': ['sqrt', None]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=9214),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=custom_score
)

grid_search.fit(X_train_final_rg, y_train_rg)
best_gb = GradientBoostingRegressor(**grid_search.best_params_, random_state=9214)
best_gb.fit(X_train_final_rg, y_train_rg)
```

## Making Predictions

To use the trained models for prediction:

```python
# Load and preprocess new data
new_data = pd.read_csv('new_flight_data.csv')
# Apply the same preprocessing steps as done for the training data

# For regression (price prediction)
rf_predictions = best_rf.predict(new_data)
gb_predictions = best_gb.predict(new_data)

# For classification (price category prediction)
clf_predictions = model3.predict(new_data)

print("Random Forest Predictions:", rf_predictions)
print("Gradient Boosting Predictions:", gb_predictions)
print("Classification Predictions:", clf_predictions)
```

Note: Ensure that the new data is preprocessed and features are selected in the same way as the training data before making predictions.
