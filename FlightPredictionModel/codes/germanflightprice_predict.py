# %% [markdown]
# # Flight Price Prediction Machine Learning Model
#
# ## Overview
# This notebook demonstrates the process of building machine learning models to predict flight prices using German air fare data. We will explore and compare the performance of three different algorithms:
#
# 1. Random Forest Regressor
# 2. Gradient Boosting Regressor
# 3. Random Forest Classifier
#
# ## Goal
# Our goal is to develop accurate models for predicting flight prices and gain insights into the factors that influence ticket costs in the German air travel market.
#
# ## Process
# The notebook covers the following steps:
# 1. Data loading and exploration
# 2. Data preprocessing and feature engineering
# 3. Model training and evaluation
# 4. Hyperparameter tuning
# 5. Model comparison and interpretation
#
# ## Author
# Ba Viet Anh (Henry) Nguyen


# %% [markdown]
# ## Initialize a Virtual Environment using `Conda` (If you have your own `Virutal Environment` your then ignore this !)

# %%
# To activate the conda environment, copy the below content to `conda-config.sh` and run the following command in the terminal:
# sh conda-config.sh

'''conda-config.sh

# Create a new conda environment
conda create -n cos30049_env python=3.10.9
conda activate cos30049_env

# Check current environment
conda info --envs

#Check current python version
python --version

'''

# %% [markdown]
# ## Install required libraries that are used for processing data and training model

# %%
# Install required libraries using requirements.txt
from typing_extensions import Annotated, Union, Doc
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %pip install - r requirements.txt
###

# OR if you prefer installing libraries manually
# %pip install numpy
# %pip install matplotlib
# %pip install pandas
# %pip install scikit-learn
# %pip install scipy
# %pip install statsmodels
# %pip install mlxtend
# %pip install seaborn
# %pip install scikit-optimize
###


# %% [markdown]
# ##  Import necessary libraries for data analysis, visualization, and machine learning
#
# - `pandas` and `numpy` for data manipulation
# - `matplotlib` and `seaborn` for data visualization
# - `sklearn` for machine learning models and preprocessing
# - `statsmodels` for statistical models
# - `datetime` for handling date and time data
# - Other helper functions from sklearn for model evaluation and optimization
#

# %%
# import library


# %% [markdown]
# ## Load the **German Air Fares** dataset taken from https://data.mendeley.com/datasets/gz75x2pzr7/2

# %%
# load the csv file
df = pd.read_csv('German Air Fares.csv')
print(df.head(5))

# %% [markdown]
# <h1>Exploratory Data Analysis</h1>

# %% [markdown]
# ### Display statistics of the datasets
# - Display the shape (x,y) of the dataset using `shape()` function.
# - Display the type of the column using `dtype()` function.
# - Describe the total number, the unique number, the top value, the frequency statistic of the data using `describe()` function

# %%
# descriptive statistic
print("\n Dataset shape: ", df.shape)
print("\n Column Type: \n", df.dtypes)
print("\nSummary statistics:\n", df.describe())

# %% [markdown]
# ### Display duplications and inconsistences
# - Display total missing values for each column using `isnull()` function.
# - Display the total duplicated rows using `duplicated()` function.
# - Display the total unique value of each column using `nunique()` function.

# %%
# data quality assesment
print("\nMissing value:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nUnique value in each column:")
for col in df.columns:
    print(col, ":", df[col].nunique())

# %% [markdown]
# ### Display number of each categorized variables for each columns

# %%
# analyze categorical variables:
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nValue counts for {col}:")  # print the name of the column
    print(df[col].value_counts())  # print the value counts for each category


# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### `standardize_time()` helper function
# `standardize_time()` helper function to convert various time formats to a standardized 24-hour format string.
#
# **Input**
# - `time_str` (str): A time-like string in various formats (e.g., "12:30 PM", "12:30 uhr", "14:30")
#
# **Output**
# - Returns a standardized time string in 24-hour format (e.g., "14:30")
#
# **Functionalitites**
# 1. Handles AM/PM format
# 2. Handles "uhr" format (German time)
# 3. Handles 24-hour format with or without minutes
# 4. Strips whitespace and converts to lowercase for consistency
# 5. Returns None if unable to parse the time string
#

# %%
# Convert the arrival and departure time to same format  (consistently in 24 hour format)
def standardize_time(time_str: Annotated[str, Doc("The time-like string to be converted")]) -> Annotated[str, Doc("The standardized time string")]:
    """
    Convert a time-like string to a standardized 24-hour format string.

    Args:
        - time_str (str): The time-like string to be converted.

    Returns:
        - str: The standardized time string.

    Examples:
        >>> standardize_time("12:30 PM") # "12:30"
        >>> standardize_time("12:30 uhr") # "12:30"
    """
    # remove leading/trailing whitespace
    time_str = time_str.strip().lower()
    try:
        # handle 'am' or 'pm' format to 12 hour format (convert to datetime datatype)
        if 'am' in time_str or 'pm' in time_str:
            time_obj = datetime.strptime(time_str, '%I:%M%p')
        # handle 'uhr'
        elif 'uhr' in time_str:
            time_str = time_str.replace("uhr", "").strip()
            if ":" in time_str:
                time_obj = datetime.strptime(time_str, '%H:%M')
            else:
                time_obj = datetime.strptime(time_str, '%H')
        # handle 24-hour format
        else:
            if ":" in time_str:
                time_obj = datetime.strptime(time_str, '%H:%M')
            else:
                time_obj = datetime.strptime(time_str, '%H')
        # convert to 24 hour format
        return time_obj.strftime('%H:%M')
    except ValueError:
        print(f"Unable to parse time: {time_str}")
        return None

# %% [markdown]
# ### `convert_number_date_distance()` helper function
#
# `convert_number_date_distance()` helper function to convert a time-like string representing the departure date distance (how far in advance the flight was booked) to a number of days.
#
# **Input**
# - `time_str` (str): A time-like string in various formats (e.g., "1 day", "2 weeks", "3 months")/
#
# **Output**
# - Number of days as an integer (e.g., "2 weeks" -> 14)
#
# **Functionalities:**
# 1. Handles various time units: days, weeks, months, and years
# 2. Converts the input to lowercase and splits it for easier parsing
# 3. Returns the number of days as an integer
# 4. Returns None if the time string cannot be parsed
#
#
# Note: This function assumes 30 days per month and 365 days per year for simplicity.
#

# %%
# convert the departure_date_distance format to day format (how long)  // def: departure_date_distance: How far in advance the flight was booked


def convert_number_date_distance(time_str: Annotated[str, Doc("The time-like string to be converted")]) -> Annotated[int, Doc("The number of days")]:
    """
    Convert a time-like string to a number of days.

    Args:
        - time_str (str): The time-like string to be converted.

    Returns:
        - int: The number of days.

    Examples:
        >>> convert_number_date_distance("1 day") # 1
        >>> convert_number_date_distance("2 weeks") # 14
        >>> convert_number_date_distance("3 months") # 90
    """
    time_str = time_str.strip().lower().split(" ")
    if "day" in time_str:
        return int(time_str[0])
    elif "week" in time_str or "weeks" in time_str:
        return int(time_str[0]) * 7
    elif "month" in time_str or "months" in time_str:
        return int(time_str[0]) * 30
    elif "year" in time_str or "year" in time_str:
        return int(time_str[0]) * 365
    else:
        return None

# %% [markdown]
#

# %% [markdown]
# ### Data Cleaning
# This section performs data cleaning and preprocessing steps on the DataFrame `df` based on the procedures below:
#
# 1. Removes null and duplicate rows
# 2. Strips whitespace from string columns
# 3. Casts the price columns to float and rename the column
# 4. Casts date columns to datetime format
# 5. Casts the 'stops' column to numerical values
# 6. Uses the helper function `standardize_time()` to standardize and convert arrival and departure times
# 7. Uses the helper function `convert_number_date_distance()` function to the 'departure_date_distance' column
#
#


# %%
# clean data
# drop the null value ( decide to remove the row because the dataset actually has only one row with null value and this row only have price column, remaining columns are null)
df = df.dropna()

# total rows before removing duplicates
print("\nTotal rows before removing duplicates: ", df.shape[0])

# remove the exact duplicate rows
df = df.drop_duplicates()

# total rows after removing duplicates
print("\nTotal rows after removing duplicates: ", df.shape[0])

# strip whitespace from string column
object_column = df.select_dtypes(include="object").columns
df[object_column] = df[object_column].apply(lambda x: x.str.strip())

# convert price column to float
# remove the comma in some prices to make sure data is consistent
df["price (€)"] = df["price (€)"].str.replace(
    ",", "").str.replace("€", "").str.strip()
df["price (€)"] = df["price (€)"].astype(float)
# rename the column for better readability
df.rename(columns={'price (€)': 'price'}, inplace=True)

# convert date columns to datetime
df["departure_date"] = pd.to_datetime(df["departure_date"], format='%d.%m.%Y')
df['scrape_date'] = pd.to_datetime(df["scrape_date"], format='%d.%m.%Y')

# convert the stops column to numerical value
df["stops"] = df["stops"].replace("direct", 0)
df["stops"] = df["stops"].replace("(1 Stopp)", 1)
df["stops"] = df["stops"].replace("(1 stop)", 1)
df["stops"] = df["stops"].replace("(2 Stopps)", 2)


# apply convert for arrival_time and departure time to standardize format
df["arrival_time"] = df["arrival_time"].apply(standardize_time)
df["departure_time"] = df["departure_time"].apply(standardize_time)

# convert arrival_time and departure_time to datetime datatype
df["arrival_time"] = pd.to_datetime(df["arrival_time"], format='%H:%M').dt.time
df["departure_time"] = pd.to_datetime(
    df["departure_time"], format='%H:%M').dt.time

df["departure_date_distance"] = df["departure_date_distance"].apply(
    convert_number_date_distance)

# %% [markdown]
# ### `times_to_minute()` helper function
# `times_to_minute()` helper function to convert a datetime object, specifically the hour object (HH:mm), to number of minutes.
#
# **Input**
# - `time_obj` (datetime): The time object to be converted in `datetime` format.
#
# **Output**
# - Number of minutes as an integer (e.g., "2:30" -> 150)
#
# **Functionalities:**
# 1. Convert to number of minutes by taking the number of hours times 60 and sum with the number of minutes
#

# %%
# function to convert time to minutes


def times_to_minute(time_obj: Annotated[datetime, Doc("The time object to be converted")]) -> Annotated[int, Doc("The number of minutes")]:
    """
    Convert a time object to the number of minutes.

    Args:
        - time_obj (datetime): The time object to be converted.

    Returns:
        - int: The number of minutes.
    """
    return time_obj.hour * 60 + time_obj.minute  # convert time to minutes

# %% [markdown]
# ### Feature Engineering
#
# This section focuses on preprocessing the data and creating new features to enhance our analysis and model performance.
#
# #### New Features Created:
#
# 1. `flight_duration_in_minutes`: Calculates the duration of each flight in minutes.
#    - For flights with stops, it accounts for potential overnight flights.
#
# 2. `departure_time_in_minutes_from_midnight`: Converts departure time to minutes from midnight.
#    - This feature can help capture time-of-day effects on flight prices.
#
# 3. `day_of_week`: Extracts the day of the week from the departure date (0 for Monday, 6 for Sunday).
#
# 4. `day_of_month`: Extracts the day of the month from the departure date.
#
# 5. `month`: Extracts the month from the departure date.
#
# 6. `year`: Extracts the year from the departure date.
#
# 7. `price_category`: Categorizes flights into 'budget', 'moderate', or 'expensive' based on price ranges.
#    - Budget: < $200
#    - Moderate: $200 - $500
#    - Expensive: > $500
#
#


# %%
# create a new column for flight duration in minutes
# condition for flight duration (if stops more than 1 so the duration cannot be lower than 150)
df["flight_duration_in_minutes"] = df["arrival_time"].apply(
    times_to_minute) - df["departure_time"].apply(times_to_minute)
df.loc[(df['flight_duration_in_minutes'] < 150) & (df['stops'] != 0), 'flight_duration_in_minutes'] = (
    1440 - df['departure_time'].apply(times_to_minute)) + df['arrival_time'].apply(times_to_minute)

# create departure_time_in_minutes_from_midnight      => the flight price is apparently affected by the departure_time
df["departure_time_in_minutes_from_midnight"] = df["departure_time"].apply(
    times_to_minute)

# create a new column for the day of the week of the departure date (day_number) (monday is 0, sunday is 6)
df["day_of_week"] = df["departure_date"].dt.weekday

# create a new column for the day of the month of the departure date (day_number)
df["day_of_month"] = df["departure_date"].dt.day

# create a new column for the month of the departure date (month_number)
df["month"] = df["departure_date"].dt.month


# create new column for the year of the departure date
df["year"] = df["departure_date"].dt.year

# create new column call 'price_category' to categorize the price into 3 categories: budget, moderate, expensive
# categorize the price into 3 categories based on the price range (0-200: budget, 200-500: moderate, 500+: expensive)
df['price_category'] = pd.cut(df['price'], bins=[-float('inf'), 200,
                              500, float('inf')], labels=['budget', 'moderate', 'expensive'])


# %% [markdown]
# ### Drop irrelevant columns

# %%
# drop the column that is obviously irrelevant
# this is just the data collection artifact
df.drop(columns=["scrape_date"], inplace=True, axis=1)

# drop the time stamp column
# already have the departure_time_in_minutes_from_midnight and durations
df.drop(columns=["departure_time", "arrival_time",
        "departure_date"], inplace=True, axis=1)

# drop the airline named

# %% [markdown]
# ### Show the pre-processed datasets

# %%
df.info()

# %% [markdown]
# ### Expect the `dropna()` function runs correctly by looking at the number of null values

# %%
df.isnull().sum()  # double check whether drop successfull or not

# %% [markdown]
# ### Show the dataset after being processed

# %%
df.describe()

# %% [markdown]
# ## Handle Outliers Of Data

# %% [markdown]
# ### `detect_outliers()` helper function
# `detect_outliers` helper function is used for detecting the outliers in the specified columns of the DataFrame using the **Interquartile** method.
#
# **Input**
# - `df` (pd.DataFrame): The DataFrame to be detected
# - `cols` (list[str]): The list of columns to be detected.
#
# **Output**
# - A dictionary with column names as keys and the count of outliers as values.
#
# **Functionalities:**
# 1. Find the first quartile (Q1) and third quartile (Q3) for each specified column.
# 2. Computes the Interquartile Range (IQR) as Q3 - Q1.
# 3. Determines the lower and upper bounds for outliers using the formula:
#    - Lower bound = Q1 - 1.5 * IQR
#    - Upper bound = Q3 + 1.5 * IQR
# 4. Identifies data points falling below the lower bound or above the upper bound as outliers.
# 5. Counts the number of outliers for each column.
# 6. Returns a dictionary containing the count of outliers for each specified column.
#

# %%
# detect outliers function by using IQR method


def detect_outliers(df: Annotated[pd.DataFrame, Doc("The DataFrame to be detected")], cols: Annotated[list[str], Doc("The columns to be detected")]) -> dict:
    """
    Detect outliers in specified columns of the DataFrame using the IQR method.

    Args:
        - df (pd.DataFrame): The DataFrame containing the data.
        - cols (list[str]): The list of columns to be detected.

    Returns:
        - dict: A dictionary with column names as keys and the count of outliers as values.
    """
    outliers: dict = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)  # 1st quartile
        Q3 = df[col].quantile(0.75)  # 3rd quartile
        IQR = Q3 - Q1  # interquartile range
        lower_bound = Q1 - 1.5 * IQR  # lower bound
        upper_bound = Q3 + 1.5 * IQR  # upper bound
        # count of outliers in each column (column: count of rows)
        outliers[col] = df.loc[(df[col] < lower_bound)
                               | (df[col] > upper_bound)].shape[0]
    return outliers


# %% [markdown]
# ### `plot_boxplots()` helper function
# `plot_boxplots()` helper function is used for plotting box plots for specified columns in a processed dataset.
#
# **Input**
# - `df` (pd.DataFrame): The DataFrame to be used for plotting
# - `cols` (list[str]): The list of columns to be used for plotting
#
# **Output**
# - A box plot
#

# %%
# boxplot to visualize outliers
def plot_boxplots(df: Annotated[pd.DataFrame, Doc("The DataFrame to be plotted")], columns: Annotated[list[str], Doc("The columns to be plotted")]) -> None:
    """
    Plot boxplots for specified columns in the DataFrame.

    Args:
        - df (pd.DataFrame): The DataFrame containing the data.
        - columns (list[str]): The list of columns to be plotted.

    Returns:
        - None
    """
    fig, axes = plt.subplots(len(columns), 1, figsize=(
        10, 5*len(columns)))  # create subplots based on number of columns
    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])  # create boxplot for each column
        axes[i].set_title(f'Boxplot of {col}')  # set title for each boxplot
    # adjust the layout (automatically adjust the subplot parameters to give specified padding)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Main logic for detecting and handling outlier detection
#
# The given steps show how we have performed detecting and handling outlier on our dataset:
#
# 1. Identify numerical columns (except the target 'price' column)
# 2. Plot boxplots before outlier handling to visualize the distribution and potential outliers
# 3. Detect outliers using the Interquartile Range (IQR) method
# 4. Handle outliers using the capping method (setting values to the 1st and 99th percentiles)
# 5. Plot boxplots after outlier handling to see the effect
# 6. Calculate and print the percentage of data retained after outlier handling
# 7. Visualize the distribution of flight prices after outlier handling
#
# These processes have helped us identify and mitigate the impact of extreme values in our dataset, which could potentially create a negative impact to our data.
#


# %%
# Identify numerical columns (excluding date columns)
# get the list of numerical columns but exclude the 'price' column because it's the target
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude target columns because outliers in target columns may be valid data points and give valuable information
numerical_cols = [col for col in numerical_cols if 'price' not in col]

# Plot boxplots before outlier handling
print("Plotting boxplots before outlier handling...")
plot_boxplots(df, numerical_cols)

# Detect outliers
outliers = detect_outliers(df, numerical_cols)
print("Number of outliers in each column: ", outliers)
total_rows = len(df)  # total of rows before outlier handling
print(f"Total number of rows before outlier handling: {total_rows}")

# Handle outliers (here we'll use capping method). By setting the lower and upper bounds for each numerical column, if the value is below the 1st percentile or above the 99th percentile respectively, replace it with the corresponding bound. (lower bound for values below 1st percentile, upper bound for values above 99th percentile)
for col in numerical_cols:
    lower_bound = df[col].quantile(0.01)  # 1st percentile
    upper_bound = df[col].quantile(0.99)  # 99th percentile
    df[col] = df[col].clip(lower_bound, upper_bound)

# Plot boxplots after outlier handling
plot_boxplots(df, numerical_cols)


# Calculate and print the percentage of data retained after outlier handling
rows_after_outlier_handling = len(df)  # total of rows after outlier handling
print(
    f"Total number of rows after outlier handling: {rows_after_outlier_handling}")
# calculate the percentage of data retained
percentage_retained = (rows_after_outlier_handling / total_rows) * 100

print(
    f"Percentage of data retained after outlier handling: {percentage_retained:.2f}%")

# the distribution of the 'price' column
plt.figure(figsize=(10, 6))
# create a histogram of the 'price' column (with kernel density estimation)
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Flight Prices After Outlier Handling')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Show the information of the dataset after handling the outliers

# %%
df.info()

# %% [markdown]
# ### Analyze and Visualize data

# %% [markdown]
# #### Display the distribution of flight duration using **box plot**

# %%
# The distribution of flight duration
plt.figure(figsize=(12, 8))
sns.boxplot(x=df['flight_duration_in_minutes'])
plt.title('Box Plot of Flight Duration')
plt.xlabel('Flight Duration (in minutes)')
plt.show()

# %% [markdown]
# ### Linear correlation between numerical attributes using **heat map**

# %%
# linear correlation between numerical attributes
# [col for col in df.columns if df[col].dtype=="int"]
correlation = df.select_dtypes(include=["int"]).corr()
color_map = sns.diverging_palette(260, -10, s=50, l=80, n=6, as_cmap=True)
plt.subplots(figsize=(17, 17))
sns.heatmap(correlation, cmap=color_map, annot=True, square=True)

# %% [markdown]
# ### Distribution of the flight price using **histogram plot**

# %%
# distribution of the flight price
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Flight Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Flight prices by airline using **box plot**

# %%
# Box plot of prices by airline
plt.figure(figsize=(12, 6))
sns.boxplot(x='airline', y='price', data=df)
plt.title('Flight Prices by Airline')
# rotate the x-axis labels by 90 degrees and set the labels to the airline names
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ###  Relationship between flight price vs. flight duration using **scatter plot**

# %%
# Scatter plot of price vs. flight duration
plt.figure(figsize=(10, 6))
sns.scatterplot(x='flight_duration_in_minutes', y='price', data=df)
plt.title('Price vs. Flight Duration')
plt.xlabel('Flight Duration (minutes)')
plt.ylabel('Price')
plt.show()

# %% [markdown]
# ### Flight Price based on Airline using **bar chart**

# %%
# Analyze Airline vs Price
df.groupby('airline')['price'].mean().sort_values(
    ascending=False).plot(kind='bar', figsize=(12, 6))

# %% [markdown]
# ### Average price by year using **bar chart**

# %%
# average price by year
df.groupby('year')['price'].mean().plot(kind='bar', figsize=(12, 6))


# %% [markdown]
# ### Price Trends Over Time using **line chart**

# %%
# Price Trends Over Time
monthly_avg_price = df.groupby('month')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='month', y='price', data=monthly_avg_price)
plt.title('Average Price Trend by Month')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.show()

# %% [markdown]
# ### Analyze how many flights are there for each day of the week using **bar chart**

# %%
# How many flights are there for each day of the week?
plt.figure(figsize=(10, 6))
sns.countplot(x='day_of_week', data=df)
plt.title('Number of Flights by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Flights')
plt.show()


# %% [markdown]
# ### Analyze the relationship between the day of the week vs price using **bar chart** (monday is 0, sunday is 6)

# %%
# day of the week vs price (monday is 0, sunday is 6)
day_of_the_week_avg_price = df.groupby(
    'day_of_week')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='day_of_week', y='price', data=day_of_the_week_avg_price)
plt.title('Average Price by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Price')
plt.show()


# %% [markdown]
# ### Analyze the average price by departure city and arrival city using **bar chart**

# %%
# Analyze the average price by departure city and arrival city
cities: list[str] = ['departure_city', 'arrival_city']
fig, axes = plt.subplots(2, 1, figsize=(20, 10))
for i, city in enumerate(cities):
    avg_prices = df.groupby(city)['price'].mean().sort_values(ascending=False)
    avg_prices.plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Average Price by {city}')
    axes[i].set_xlabel(city)
    axes[i].set_ylabel('Average Price')
    axes[i].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Analzye the price distribution by number of stops using **violin plot**

# %%
# Price Distribution by Number of Stops
plt.figure(figsize=(10, 6))
sns.violinplot(x='stops', y='price', data=df)
plt.title('Price Distribution by Number of Stops')
plt.xlabel('Number of Stops')
plt.ylabel('Price')
plt.show()

# %% [markdown]
# ### Price vs Days Before Flight using **scatter plot**

# %%
# Price vs Days Before Flight
plt.figure(figsize=(12, 6))
sns.scatterplot(x='departure_date_distance', y='price', data=df)
plt.title('Price vs Departure Date Distance')
plt.xlabel('Departure Date Distance')
plt.ylabel('Price')
plt.show()

# %% [markdown]
# ### Analyze the price based on the `departure_city` using **box plot**

# %%
# the departure_city vs price
plt.figure(figsize=(15, 10))
plt.xticks(rotation=90)
sns.boxplot(x='departure_city', y='price',
            data=df.sort_values('price', ascending=False))

# %% [markdown]
# ### Analyze the price based on the `arrival_city` using **box plot**

# %%
# arrival_city vs price
plt.figure(figsize=(15, 10))
plt.xticks(rotation=90)
sns.boxplot(x='arrival_city', y='price',
            data=df.sort_values('price', ascending=False))

# %% [markdown]
# ### Display **Pair Plot** for All Key Features

# %%
# Pair Plot for Key Features
key_features = ['price', 'flight_duration_in_minutes', 'stops', 'departure_time_in_minutes_from_midnight',
                'day_of_month', 'month', 'day_of_week', 'year', 'departure_date_distance']
sns.pairplot(df[key_features], height=2.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### **Bar plot** for displaying the distribution of price categories

# %%
# Bar plot of price categories
plt.figure(figsize=(8, 6))
sns.countplot(x='price_category', data=df)
plt.title('Distribution of Price Categories')
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ### Flight Duration Distribution by Price Category using **box plot**

# %%
# Flight Duration Distribution by Price Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='price_category', y='flight_duration_in_minutes', data=df)
plt.title('Flight Duration Distribution by Price Category')
plt.xlabel('Price Category')
plt.ylabel('Flight Duration (in minutes)')
plt.show()

# %% [markdown]
# ### Display the summarized statistics using `describe()` function

# %%
# Statistical Summary
summary = df.describe()
print("Statistical Summary of Numerical Features:")
print(summary)

# %% [markdown]
# ## Data Spliting, Scaling and Encoding

# %% [markdown]
# ### Data Splitting
# In this section, data is being prepared for modelling and training by:
# 1. Separating features and target variables:
#     - Features (X): All columns except `price` and `price_category`
#     - Regression target (y): `price`
#     - Classification target (z): `price_category`
# 2. Splitting the data into training and testing sets:
#     - Using `train_test_split()` to split data with a 80% for training and 20% for testing.
#     - Creating separate splits for regression and classification tasks.
#
# By executing the below cell, we have created separate datasets for:
#  - Regression: Predicting the exact price
#  - Classification: Predicting the price category
#

# %%
# Separate features and target
# enforce exclusive the target variable
X = df.drop(columns=['price', 'price_category'], axis=1)
y = df['price']  # target variable for regression
z = df['price_category']  # target variable for classification

# split the data for regression model
X_train_rg, X_test_rg, y_train_rg, y_test_rg = train_test_split(
    X, y, test_size=0.20, random_state=9214)
# split the data for classification model
X_train_cl, X_test_cl, z_train_cl, z_test_cl = train_test_split(
    X, z, test_size=0.20, random_state=9214)

# %% [markdown]
# ### Visualize the distribution of numerical column to find out the best scaling method using **bar chart**

# %%
# visualize the distribution of numerical column to choose the best scaling method
categorical_cols: list[str] = ['departure_city', 'arrival_city', 'airline']
# get the list of numerical columns
numerical_cols: list[str] = [
    col for col in X.columns if col not in categorical_cols]
for col in numerical_cols:
    plt.figure(figsize=(10, 12))
    # count the unique values in each column and visualize it in bar chart
    df[col].value_counts().plot(kind='hist')
    plt.title(f'Histogram of {col}')
    # rotate 45 deg the label in xaxis for better reading
    plt.xticks(rotation=45)
    plt.show()  # show the plot

# %% [markdown]
# ### Feature Scaling and Encoding
#
# After analyzing the distribution of the numerical columns, we observed that the values are not normally distributed. Therefore, `MinMaxScaler` is the best algorithm to used for normalizing these columns.
#
# For categorical variables, `OneHotEncoder` is used since these variables are nominal (not ordinal), thus one-hot encoding the most suitable method.
#
# The process involves:
# 1. Scaling numerical variables using `MinMaxScaler`
# 2. Encoding categorical variables using `OneHotEncoder`
# 3. Combining the scaled numerical and encoded categorical variables
#
# This preprocessing is done separately for both regression and classification models to ensure no data leakage occurs between the training and testing sets.
#
#

# %%
# Scale numerical variables
# # After analyzing the distribution of the selected numerical columns, we can see that the values are not normally distributed. Therefore, we will use the MinMaxScaler to normalize the numerical columns.
scaler = MinMaxScaler()
X_train_scaled_rg = pd.DataFrame(scaler.fit_transform(X_train_rg[numerical_cols]),
                                 columns=numerical_cols,
                                 index=X_train_rg.index)  # Fit the scaler on training data to learn the scaling parameters (min, max). Then, transform the training data using those parameters.
X_test_scaled_rg = pd.DataFrame(scaler.transform(X_test_rg[numerical_cols]),
                                columns=numerical_cols,
                                index=X_test_rg.index)  # only transform the test set to avoid data leakage

X_train_scaled_cl = pd.DataFrame(scaler.fit_transform(X_train_cl[numerical_cols]),
                                 columns=numerical_cols,
                                 index=X_train_cl.index)  # Fit the scaler on training data to learn the scaling parameters (min, max). Then, transform the training data using those parameters.
X_test_scaled_cl = pd.DataFrame(scaler.transform(X_test_cl[numerical_cols]),
                                columns=numerical_cols,
                                index=X_test_cl.index)  # only transform the test set to avoid data leakage

# One-hot encode categorical variables. We will use the OneHotEncoder to encode the categorical variables because the categorical variables are nominal (not ordinal) so one-hot encoding is the most suitable method.
encoder = OneHotEncoder(
    drop='first', sparse_output=False, handle_unknown='ignore')

# for regression model
X_train_encoded_rg = pd.DataFrame(encoder.fit_transform(X_train_rg[categorical_cols]),
                                  columns=encoder.get_feature_names_out(
                                      categorical_cols),
                                  index=X_train_rg.index)   # Fit the encoder on training data to learn the categories. Then, transform the training data using those categories.
X_test_encoded_rg = pd.DataFrame(encoder.transform(X_test_rg[categorical_cols]),
                                 columns=encoder.get_feature_names_out(
                                     categorical_cols),
                                 index=X_test_rg.index)  # only transform the test set to avoid data leakage

# for classification model
X_train_encoded_cl = pd.DataFrame(encoder.fit_transform(X_train_cl[categorical_cols]),
                                  columns=encoder.get_feature_names_out(
                                      categorical_cols),
                                  index=X_train_cl.index)    # Fit the encoder on training data to learn the categories. Then, transform the training data using those categories.
X_test_encoded_cl = pd.DataFrame(encoder.transform(X_test_cl[categorical_cols]),
                                 columns=encoder.get_feature_names_out(
                                     categorical_cols),
                                 index=X_test_cl.index)  # only transform the test set to avoid data leakage


# Combine encoded categorical and scaled numerical variables

# for regression model
X_train_preprocessed_rg = pd.concat(
    [X_train_encoded_rg, X_train_scaled_rg], axis=1)
X_test_preprocessed_rg = pd.concat(
    [X_test_encoded_rg, X_test_scaled_rg], axis=1)

# for classification model
X_train_preprocessed_cl = pd.concat(
    [X_train_encoded_cl, X_train_scaled_cl], axis=1)
X_test_preprocessed_cl = pd.concat(
    [X_test_encoded_cl, X_test_scaled_cl], axis=1)

# %% [markdown]
# ## Feature Selection

# %% [markdown]
# ### Feature Selection for **regression model**

# %% [markdown]
# ### `stepwise_regression()` helper function
# `stepwise_regression()` helper function to perform stepwise regression for feature selection.
#
# **Input**
# - `X` (pd.DataFrame): Training data
# - `y` (pd.Series): Target variable
# - `significance_level_in` (float): Significance level for entering (default: 0.05)
# - `significance_level_out` (float): Significance level for removing (default: 0.05)
#
# **Output**
# - Returns a list of selected features
#
# **Functionalities**
# 1. Performs forward selection: Appends features that has the **lowest p-value** (has the highest impact to the model)
# 2. Performs backward elimination: Removes no longer significant features
# 3. Iteratively appends and pops features until no changes occur
# 4. Prints the features being added or removed along with their p-values
# 5. Returns the final list of selected features
#

# %%
# Function to calculate p-values and perform stepwise regression


def stepwise_regression(X: Annotated[pd.DataFrame, Doc("Training data")],
                        y: Annotated[pd.Series, Doc("Target variable")],
                        significance_level_in: Annotated[float, Doc(
                            "Significance level for entering")] = 0.05,
                        significance_level_out: Annotated[float, Doc(
                            "Significance level for removing")] = 0.05
                        ) -> Annotated[list, Doc("List of selected features")]:
    """
    Perform stepwise regression to select features for the regression model.

    Parameters:
    - X (pd.DataFrame): Training data.
    - y (pd.Series): Target variable.
    - significance_level_in (float): Significance level for entering.
    - significance_level_out (float): Significance level for removing.

    Returns:
    - List of selected features.
    """

    # Start with no features
    included: list[str] = []

    while True:
        changed = False  # flag to check if any feature was added or removed

        # Forward Selection: Add the feature that improves the model the most (lowest p-value)
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            # Fit OLS model with current features plus new feature
            model = sm.OLS(y, sm.add_constant(
                X[included + [new_column]])).fit()
            # Store p-value of new feature
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()

        # If best p-value is below threshold, add the feature
        if best_pval < significance_level_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            print(
                f"Adding feature '{best_feature}' with p-value {best_pval:.4f}")

        # Backward Elimination: Remove features that are no longer significant
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # Exclude constant
        worst_pval = pvalues.max()

        # If worst p-value is above threshold, remove the feature
        if worst_pval > significance_level_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            print(
                f"Removing feature '{worst_feature}' with p-value {worst_pval:.4f}")

        # If no changes were made, exit the loop
        if not changed:
            break

    return included

# %% [markdown]
# ### Shows the final features along with their p-values and prepare final data frame for training for regression


# %%
# Add a constant term to the features (required for statsmodels OLS)
# This is necessary for the intercept term in the regression model
X_train_with_const_rg = sm.add_constant(X_train_preprocessed_rg)

# Perform stepwise regression to select the most significant features
# This process will iteratively add and remove features based on their statistical significance
final_features_rg = stepwise_regression(X_train_preprocessed_rg, y_train_rg)

# Print the final features that were selected after stepwise regression
# This helps us understand which features were deemed most important for the model
print("\nFinal features after stepwise regression:")
print(final_features_rg)

# Prepare final datasets for modeling using only the selected features
# This ensures we're using the most relevant features for our regression model
X_train_final_rg = X_train_preprocessed_rg[final_features_rg]
X_test_final_rg = X_test_preprocessed_rg[final_features_rg]

# Create the final data frame for regression
# This combines the training and test sets, including both features and target variable
# The resulting dataframe will be used for further analysis and model evaluation
df_regression = pd.concat([pd.concat([X_train_final_rg, y_train_rg], axis=1), pd.concat(
    [X_test_final_rg, y_test_rg], axis=1)], axis=0)

# %% [markdown]
# ### Show the first 5 values of regression dataset

# %%
df_regression.head(5)

# %% [markdown]
# ## Feature selection for **classification model**

# %% [markdown]
# ### `feature_selection_ramdomforestclassifier()` helper function
# `feature_selection_ramdomforestclassifier()` helper function used for performing feature selection using Random Forest Classifier and the elbow method.
#
# **Input**
# - `X` (pd.DataFrame): Training data features
# - `y` (pd.Series): Target variable
#
# **Output**
# - Returns a list of selected features
#
# **Functionalities**
# 1. Initializes and fits a `Random Forest Classifier`
# 2. Calculates the importances of a feature
# 3. Sorts features by importances
# 4. Calculates the cumulative of the importances
# 5. Finds the elbow point using the difference in cumulative importances
# 6. Selects features up to the elbow point
# 7. Prints the automatically selected features and their importances
# 8. Plots the cumulative feature importance graph
# 9. Returns the list of selected features
#

# %%
#  We can modify our approach to use a technique called "elbow method" to automatically select the number of features. This method looks at the cumulative importance of features and selects a point where adding more features doesn't significantly increase the total importance.


def feature_selection_ramdomforestclassifier(X: Annotated[pd.DataFrame, "Training data features"],
                                             y: Annotated[pd.Series,
                                                          "Target variable"]
                                             ) -> Annotated[list, "List of selected features"]:
    """
    Perform feature selection using Random Forest Classifier and the elbow method.

    Parameters:
    - X (pd.DataFrame): Training data features.
    - y (pd.Series): Target variable.

    Returns:
    - List of selected features.
    """
    # Initialize and fit the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Sort features by importance
    feature_importances = pd.DataFrame(
        {'feature': X.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values(
        'importance', ascending=False)

    # Calculate cumulative importances
    cumulative_importances = np.cumsum(feature_importances['importance'])

    # Find the elbow point
    diff = np.diff(cumulative_importances)
    # Add 1 because diff reduces the array size by 1
    elbow = np.argmin(diff) + 1

    # Select features up to the elbow point
    selected_features = feature_importances['feature'][:elbow].tolist()

    print(f"\nAutomatically selected {len(selected_features)} features:")
    print(feature_importances.head(elbow))

    # Plot cumulative importances
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_importances) + 1),
             cumulative_importances, 'b-')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.axvline(x=elbow, color='r', linestyle='--', label='Elbow point')
    plt.legend()
    plt.show()

    return selected_features

# %% [markdown]
# ### Prepare final datasets for classification modeling


# %%
# X_train_preprocessed_cl and z_train_cl are preprocessed features and target variable
selected_features = feature_selection_ramdomforestclassifier(
    X_train_preprocessed_cl, z_train_cl)

# Prepare final datasets for classification modeling
X_train_final_cl = X_train_preprocessed_cl[selected_features]
X_test_final_cl = X_test_preprocessed_cl[selected_features]

# Final data frame for regression
df_classification = pd.concat([pd.concat([X_train_final_cl, z_train_cl], axis=1), pd.concat(
    [X_test_final_cl, z_test_cl], axis=1)], axis=0)

# %% [markdown]
# ## Model Selection and Training

# %% [markdown]
# ### Preliminary testing to check the score of the model MSE score for each model

# %%
models = [LinearRegression(), SVR(), RandomForestRegressor(),
          GradientBoostingRegressor()]
for model in models:
    scores = cross_val_score(model, X_train_final_rg,
                             y_train_rg, cv=5, scoring='neg_mean_squared_error')
    print(
        f"Model: {model.__class__.__name__} - Mean MSE: {-scores.mean():.4f} (±{scores.std() * 2:.4f})")

# %% [markdown]
# ### Define Allowed Model

# %%
# Strict typing for model name
RegressionModel = Union[RandomForestRegressor, GradientBoostingRegressor]

ClassificationModel = Union[RandomForestClassifier]

# %% [markdown]
# ### `model_train()` function
# `model_train()` function used for **training regression models** and **evaluating their performance**.
#
# **Input**
# - `model_name` (RegressionModel): Model instance (RandomForestRegressor or GradientBoostingRegressor)
# - `X_train` (pd.DataFrame): Training data features
# - `X_test` (pd.DataFrame): Testing data features
# - `y_train` (pd.Series): Training target variable
# - `y_test` (pd.Series): Testing target variable
#
# **Output**
# - None (prints evaluation metrics)
#
# **Functionalities**
# 1. Fits the model on the training data
# 2. Prints the training score
# 3. Prints the test score
# 4. Makes predictions on the test set
# 5. Calculates and prints the following evaluation metrics:
#    - R^2 Score
#    - Mean Squared Error
#    - Mean Absolute Error
#
# **Note:** This function is designed for regression models and uses appropriate evaluation metrics for regression tasks.
#

# %%
# Function to train and evaluate regression models with numeric evaluation metrics


def model_train(model_name: Annotated[RegressionModel, Doc("Model instance")],
                X_train: Annotated[pd.DataFrame, Doc("Training data")],
                X_test: Annotated[pd.DataFrame, Doc("Testing data")],
                y_train: Annotated[pd.Series, Doc("Training target")],
                y_test: Annotated[pd.Series, Doc("Testing target")]
                ) -> None:
    """
    Train a regression model and print various evaluation metrics.

    Parameters:
    - model_name (RandomForestRegressor | GradientBoostingRegressor): Instance of the regression model to be trained.
    - X_train (pd.DataFrame): Features of the training data.
    - X_test (pd.DataFrame): Features of the testing data.
    - y_train (pd.Series): Target values of the training data.
    - y_test (pd.Series): Target values of the testing data.

    Returns:
    - None (prints evaluation metrics)
    """
    # Print the name of the regression model being used
    print(f" Regression Model: {model_name}")

    # Fit the model on the training data
    model = model_name.fit(X_train, y_train)

    # Calculate and print the training score (R^2 score on training data)
    print(f"Training Score: {model.score(X_train, y_train)}")

    # Calculate and print the test score (R^2 score on test data)
    print(f"Test Score: {model.score(X_test, y_test)}")

    # Make predictions on the test set
    predict_value = model.predict(X_test)

    # Calculate and print various evaluation metrics
    # Coefficient of determination
    print(f"R^2 Score: {r2_score(y_test, predict_value)}")
    # Average squared difference between predicted and actual values
    print(f"Mean Square Error: {mean_squared_error(y_test, predict_value)}")
    # Average absolute difference between predicted and actual values
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predict_value)}")


# %% [markdown]
# ### `model_train_classification()` function
# `model_train_classification()` function used for **training classification models** and **evaluating their performance**.
#
# **Input**
# - `model_name` (ClassificationModel): Model instance (e.g., RandomForestClassifier)
# - `X_train` (pd.DataFrame): Training data features
# - `X_test` (pd.DataFrame): Testing data features
# - `y_train` (pd.Series): Training target variable
# - `y_test` (pd.Series): Testing target variable
#
# **Output**
# - None (prints evaluation metrics)
#
# **Functionalities**
# 1. Fits the model on the training data
# 2. Makes predictions on the test set
# 3. Calculates and prints the following evaluation metrics:
#    - Accuracy
#    - Precision (weighted average)
#    - Recall (weighted average)
#    - F1 Score (weighted average)
#    - Confusion Matrix
#
# **Note:** This function is designed for classification models and uses appropriate evaluation metrics for classification tasks.
#

# %%
# Function to train and evaluate classification models
def model_train_classification(model_name: Annotated[ClassificationModel, Doc("Model instance")],
                               X_train: Annotated[pd.DataFrame, Doc("Training data")],
                               X_test: Annotated[pd.DataFrame, Doc("Testing data")],
                               y_train: Annotated[pd.Series, Doc("Training target")],
                               y_test: Annotated[pd.Series,
                                                 Doc("Testing target")]
                               ) -> None:
    """
    Train a classification model and print various evaluation metrics.

    Parameters:
    - model_name (ClassificationModel): Instance of the classification model to be trained.
    - X_train (pd.DataFrame): Features of the training data.
    - X_test (pd.DataFrame): Features of the testing data.
    - y_train (pd.Series): Target values of the training data.
    - y_test (pd.Series): Target values of the testing data.

    Returns:
    - None: This function doesn't return any value, it prints the evaluation metrics.
    """

    # Fit the model on the training data
    model = model_name.fit(X_train, y_train)

    # Use the trained model to make predictions on the test set
    predict_value = model.predict(X_test)

    # Calculate and print various evaluation metrics

    # Accuracy: The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined
    print(f"Accuracy: {accuracy_score(y_test, predict_value)}")

    # Precision: The ratio of correctly predicted positive observations to the total predicted positive observations
    # Using weighted average for multi-class problems
    print(
        f"Precision: {precision_score(y_test, predict_value, average='weighted')}")

    # Recall: The ratio of correctly predicted positive observations to all observations in actual class
    # Using weighted average for multi-class problems
    print(f"Recall: {recall_score(y_test, predict_value, average='weighted')}")

    # F1 Score: The weighted average of Precision and Recall
    # Using weighted average for multi-class problems
    print(f"F1 Score: {f1_score(y_test, predict_value, average='weighted')}")

    # Confusion Matrix: A table used to describe the performance of a classification model on a set of test data for which the true values are known
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predict_value)}")

# %% [markdown]
# ### Train regression model using **RandomForestRegressor**


# %%
# Random Forest (for predict numerical value)
model1 = RandomForestRegressor(random_state=9214)
# Train the model
model_train(model1, X_train_final_rg, X_test_final_rg, y_train_rg, y_test_rg)

# %% [markdown]
# ### Train regression model using **GradientBoostingRegressor**

# %%
# Gradient Boosting Regressor
model2 = GradientBoostingRegressor(random_state=9214)
# Train the model
model_train(model2, X_train_final_rg, X_test_final_rg, y_train_rg, y_test_rg)

# %% [markdown]
# ### Train regression model using **RandomForestClassifier**

# %%
# Random Forest Classifier
model3 = RandomForestClassifier(random_state=9214)
# Train the model
model_train_classification(model3, X_train_final_cl,
                           X_test_final_cl, z_train_cl, z_test_cl)

# %% [markdown]
# ## Validate **regression model**

# %%
# The data use to train the regression model also use to validation (should be the whole data not the splitted data because will use kfold to valid the model)
X_cross_val_rg = df_regression.drop(columns=['price'], axis=1)
# Get the target variable
y_cross_val_rg = df_regression['price']

# %%
# validation for the regression model
models = [model1, model2]  # list of models
for i, model in enumerate(models, start=1):
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_cross_val_rg, y_cross_val_rg, cv=5)

    print(f"Model {i} Cross-validation scores: {cv_scores}")
    print(f"Model {i} Mean CV score: {cv_scores.mean():.4f}")
    print(f"Model {i} Standard deviation of CV score: {cv_scores.std():.4f}")

# %% [markdown]
# ## Validate **classification model**

# %%
# the data use to train the classification model also use to validation (should be the whole data not the splitted data because will use kfold to valid the model)
X_cross_val_cl = df_classification.drop(columns=['price_category'], axis=1)
y_cross_val_cl = df_classification['price_category']

# %%
# Validation for the classification model
# List of models to evaluate
models = [model3]

# Define metrics for evaluation
# Accuracy: Proportion of correct predictions
# Precision: Ratio of true positives to all positive predictions
# Recall: Ratio of true positives to all actual positives
# F1 Score: Harmonic mean of precision and recall
metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

# Perform 5-fold cross-validation for each model
for i, model in enumerate(models, 1):
    print(f"\nModel {i}:")
    for metric_name, metric_func in metrics.items():
        # Calculate cross-validation scores
        scores = cross_val_score(
            model, X_cross_val_cl, y_cross_val_cl, cv=5, scoring=make_scorer(metric_func))
        # Print results for each metric
        print(f"{metric_name}:")
        print(f"  Scores: {scores}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std. Dev: {scores.std():.4f}")

# %% [markdown]
# ## Hyper-parameter tuning

# %% [markdown]
# ### Hyper-parameter tuning for **Random Forest Regression** model

# %% [markdown]
# We perform hyperparameter tuning for our Random Forest Regression model by following the given procedure:
#
# 1. **Parameter Grid**: Define a grid of hyperparameters to search through, including:
#    - `n_estimators`: Number of trees in the forest (500 or 600)
#    - `max_depth`: Maximum depth of the trees (35 or 40)
#    - `min_samples_split`: Minimum number of samples required to split an internal node (15 or 20)
#    - `min_samples_leaf`: Minimum number of samples required to be at a leaf node (3 or 4)
#    - `max_features`: Number of features to consider when looking for the best split ('sqrt')
#    - `bootstrap`: Whether bootstrap samples are used when building trees (True)
#
# 2. **Custom Scoring Function**: Define a custom scoring function that penalizes overfitting by considering both cross-validation and training scores.
#
# 3. **RandomizedSearchCV**: Use `RandomizedSearchCV` to perform the hyperparameter search, which is more efficient than an exhaustive grid search for large parameter spaces.
#
# 4. **Model Fitting**: Fit the `RandomizedSearchCV` object to our training data.
#
# 5. **Results**: Print the best parameters found and the corresponding best score.
#
#

# %%
# Define the parameter grid
param_grid = {
    'n_estimators': [500, 600],         # Keep high number of trees
    # Slightly shallower trees to reduce overfitting
    'max_depth': [35, 40],
    'min_samples_split': [15, 20],      # Tighten split to control overfitting
    # Increase min_samples_leaf for stability
    'min_samples_leaf': [3, 4],
    'max_features': ['sqrt'],           # Keep 'sqrt' as it worked well
    'bootstrap': [True]
}

# Define a custom scoring function that penalizes overfitting


def custom_score(estimator, X, y):
    # Get cross-validation score
    cv_score = np.mean(cross_val_score(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error'))

    # Fit the estimator and get training score
    estimator.fit(X, y)
    train_score = -mean_squared_error(y, estimator.predict(X))

    # Penalize the difference between train and cv scores
    penalty = abs(train_score - cv_score)

    return cv_score - penalty


# Create a random forest regressor
rf = RandomForestRegressor(random_state=9214)

# Set up the Random search
ran_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=custom_score
)

# Fit the random search
ran_search.fit(X_train_final_rg, y_train_rg)

# Print the best parameters and score
print("Best parameters:", ran_search.best_params_)
print("Best score:", ran_search.best_score_)

# %% [markdown]
# ### Predict and display the **Mean Squared Error**, **Mean Absolute Error** and **R-squared** score for the Hyper-parameter tuned

# %%
# Create a new model with the best parameters from the random search
best_rf = RandomForestRegressor(**ran_search.best_params_, random_state=9214)

# Fit the model to the training data
best_rf.fit(X_train_final_rg, y_train_rg)

# Make predictions on the test set
y_pred = best_rf.predict(X_test_final_rg)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_rg, y_pred)
r2 = r2_score(y_test_rg, y_pred)
mbe = mean_absolute_error(y_test_rg, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mbe}")
print(f"R^2 Score: {r2}")

# %% [markdown]
# ### Display the **importance of features** for the Hyper-parameter tuned

# %%
# Print feature importances
for feature, importance in zip(X_cross_val_rg.columns, best_rf.feature_importances_):
    print(f"{feature}: {importance}")

# %% [markdown]
# ### Print the **training score** for the tuned model

# %%
# print training score
print(f"Training Score: {best_rf.score(X_train_final_rg, y_train_rg)}")

# %% [markdown]
# ### Display the **Cross-validation scores**, **Mean CV Score** and **Standard devitation of CV score** for the Hyper-parameter tuned model

# %%
# Perform cross-validation on the best Random Forest model
cv_val = cross_val_score(best_rf, X_cross_val_rg,
                         y_cross_val_rg, cv=5, scoring='r2')

# Print the cross-validation scores
print(f"Cross-validation scores: {cv_val}")

# Print the mean cross-validation score
print(f" Mean CV score: {cv_val.mean():.4f}")

# Print the standard deviation of the cross-validation scores
print(f"Standard deviation of CV score: {cv_val.std():.4f}")

# %% [markdown]
# ### Hyper-parameter tuning for **Gradient Boosting Regressor** model

# %% [markdown]
# We perform hyperparameter tuning for our Gradient Boosting Regressor model by following this procedure:
#
# 1. **Parameter Grid**: Define a grid of hyperparameters to search through, including:
#    - `n_estimators`: Number of boosting stages (100 or 200)
#    - `learning_rate`: Shrinks the contribution of each tree (0.05 or 0.1)
#    - `max_depth`: Maximum depth of the individual regression estimators (3 or 4)
#    - `min_samples_split`: Minimum number of samples required to split an internal node (5 or 10)
#    - `min_samples_leaf`: Minimum number of samples required to be at a leaf node (2 or 4)
#    - `subsample`: Fraction of samples used for fitting the individual base learners (0.8)
#    - `max_features`: Number of features to consider when looking for the best split ('sqrt' or None)
#
# 2. **Custom Scoring Function**: Define a custom scoring function that penalizes overfitting by considering both cross-validation and training scores.
#
# 3. **GridSearchCV**: Use `GridSearchCV` to perform an exhaustive search over the specified parameter grid.
#
# 4. **Model Fitting**: Fit the `GridSearchCV` object to our training data.
#
# 5. **Results**: Print the best parameters found and the corresponding best score.
#
#

# %%
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'subsample': [0.8],
    'max_features': ['sqrt', None]
}

# Define a custom scoring function that penalizes overfitting


def custom_score(estimator, X, y):
    # Get cross-validation score
    cv_score = np.mean(cross_val_score(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error'))

    # Fit the estimator and get training score
    estimator.fit(X, y)
    train_score = -mean_squared_error(y, estimator.predict(X))

    # Penalize the difference between train and cv scores
    penalty = abs(train_score - cv_score)

    return cv_score - penalty  # We want to maximize this


# Create a random forest regressor
gb = GradientBoostingRegressor(random_state=9214)

# Set up the Grid search
grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=custom_score
)

# Fit the Grid search
grid_search.fit(X_train_final_rg, y_train_rg)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# %% [markdown]
# ### Predict and display the **Mean Squared Error**, **Mean Absolute Error** and **R-squared** score for the Hyper-parameter tuned

# %%
# Create a new Gradient Boosting Regressor model with the best parameters found from grid search
best_gb = GradientBoostingRegressor(
    **grid_search.best_params_, random_state=9214)

# Fit the model to the training data
# This trains the model on our prepared training dataset
best_gb.fit(X_train_final_rg, y_train_rg)

# Make predictions on the test set
# We use the trained model to predict prices for our test data
y_pred = best_gb.predict(X_test_final_rg)

# Calculate various performance metrics
# Mean Squared Error (MSE): Average squared difference between predicted and actual values
mse = mean_squared_error(y_test_rg, y_pred)

# R-squared (R2) Score: Proportion of variance in dependent variable predictable from independent variable(s)
r2 = r2_score(y_test_rg, y_pred)

# Mean Absolute Error (MAE): Average absolute difference between predicted and actual values
mae = mean_absolute_error(y_test_rg, y_pred)

# Print out the performance metrics
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Calculate and print the training score
# This shows how well the model fits the training data, which can be used to check for overfitting
print(f"Training Score: {best_gb.score(X_train_final_rg, y_train_rg)}")

# %% [markdown]
# ### Display the **Cross-validation scores**, **Mean CV Score** and **Standard devitation of CV score** for the Hyper-parameter tuned model

# %%
# Perform 5-fold cross-validation using the best Gradient Boosting model
# This helps assess how well the model generalizes to unseen data
cv_val = cross_val_score(best_gb, X_cross_val_rg,
                         y_cross_val_rg, cv=5, scoring='r2')

# Print the individual cross-validation scores for each fold
print(f"Cross-validation scores: {cv_val}")

# Calculate and print the mean cross-validation score
# This gives us an overall measure of the model's performance across all folds
print(f"Mean CV score: {cv_val.mean():.4f}")

# Calculate and print the standard deviation of the cross-validation scores
# This indicates how consistent the model's performance is across different folds
print(f"Standard deviation of CV score: {cv_val.std():.4f}")
