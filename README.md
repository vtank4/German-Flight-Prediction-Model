# Civil Aviation - Flight Price Prediction and Classification

## Project Overview
A machine learning project that predicts and categorizes flight prices for the German domestic air travel market using Random Forest techniques. The model provides both precise price predictions and price range categorizations (budget, moderate, expensive) to help travelers make informed booking decisions.

## Data Source
- Dataset: "German Domestic Air Fares" from Mendeley Data
- Size: 63,000 data points covering 84 routes
- Features: Departure/arrival cities, dates, times, airlines, and prices
- Time span: 6-month period of domestic German flights
- Source credibility: Published by Zeppelin Universität

## Machine Learning Models
- **Random Forest Regressor**
  - Accuracy: R² score of 0.8410 after tuning
  - Mean Absolute Error: 35.29 EUR
  - Cross-validation score: 0.8332 (std: 0.0199)

- **Random Forest Classifier**
  - Accuracy: 92.72%
  - Precision: 0.9212
  - Recall: 0.9213
  - F1 Score: 0.9212

## Data Processing Pipeline
1. **Data Cleaning**
   - Handling missing values
   - Removing duplicate entries (804 exact duplicates removed)
   - Standardizing time formats and categorical variables

2. **Feature Engineering**
   - Flight duration calculation
   - Time-based feature extraction (day of week, month, etc.)
   - Price category creation

3. **Preprocessing**
   - Outlier detection using IQR method
   - MinMax scaling for numerical features
   - One-Hot Encoding for categorical variables

## Model Optimization
- Hyperparameter tuning using RandomizedSearchCV
- 5-fold cross-validation for performance validation
- Feature importance analysis for selection
- Overfitting reduction: Training score improved from 0.9139 to 0.8581

## Performance Metrics
### Random Forest Regressor
- Initial R² Score: 0.8268
- Optimized R² Score: 0.8410
- Mean Squared Error: 3579.84
- Mean Absolute Error: 35.29 EUR

### Random Forest Classifier
- Accuracy: 92.72%
- Consistent performance across all price categories
- Balanced metrics (precision, recall, F1 all ~0.92)

## Requirements
- pandas
- numpy
- scikit-learn
- datetime
- matplotlib
- seaborn
- statsmodels

## Project Structure
```
├── data/
│   └── german_domestic_airfares.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── exploratory_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   └── evaluation/
├── requirements.txt
└── README.md
```

## Future Improvements
- Expand feature engineering for temporal patterns
- Implement more complex hyperparameter optimization
- Explore deep learning approaches
- Incorporate additional external factors (events, weather, etc.)

## Citation
Dataset source:
```
German Domestic Air Fares
Published: January 6, 2021
DOI: [dataset DOI]
Contributor: Frederick F
Institution: Zeppelin Universität
```

## License
[Project License]

