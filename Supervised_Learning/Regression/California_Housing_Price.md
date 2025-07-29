# California Housing Price Prediction

## Overview
This project aims to predict the median house values in California using the California Housing Dataset. The dataset contains various features related to housing, such as the average number of rooms, the median income of the area, and more. Using these features, we build and evaluate multiple machine learning models to predict house prices.

## Dataset
The dataset used in this project is the California Housing Dataset, which is available in the sklearn library. It contains information on:
•	MedInc (Median Income in the block)
•	HouseAge (Age of the house)
•	AveRooms (Average number of rooms)
•	AveBedrms (Average number of bedrooms)
•	Population (Population in the block)
•	AveOccup (Average occupancy)
•	Latitude & Longitude (Geographical information)
•	MedHouseVal (Median house value in USD)

## Steps Involved
### 1.	Loading the Data:
 -	We use the fetch_california_housing() function from scikit-learn to load the dataset.
 -  The dataset is converted into a pandas DataFrame for easier handling.
### 2.	Data Preprocessing:
 - Missing Values: The dataset doesn't contain missing values, but if it did, we would handle them by either filling them  with the mean/median or dropping the rows/columns.
 - Outliers: We identify and cap outliers (values that are significantly different from the rest of the data) using percentiles (1st and 99th percentile).
 - Skewness: We check the skewness of numerical features. If a feature is highly skewed, we apply log transformation to make it more normally distributed.
 - Feature Scaling: We standardize numerical features to have a mean of 0 and a standard deviation of 1. This helps improve the performance of certain models.
### 3.	Model Building: 
  We implement the following regression algorithms:
 - Linear Regression: A simple algorithm that assumes a linear relationship between the features and the target variable.
 - Decision Tree Regressor: A non-linear model that splits the data into decision nodes to make predictions.
 - Random Forest Regressor: An ensemble method that combines multiple decision trees to improve prediction accuracy.
 - Gradient Boosting Regressor: Another ensemble method that builds trees sequentially, each learning from the mistakes of the previous one.
 - Support Vector Regressor (SVR): A model that finds a hyperplane to maximize the margin between data points, ideal for higher-dimensional data.

### 4.	Model Evaluation:
 We evaluate the performance of each model using the following metrics:
 - Mean Squared Error (MSE)
 - Mean Absolute Error (MAE)
 - R-squared (R²)
These metrics help us understand how well the model is predicting the house prices and how close the predictions are to the actual values.

### 5.	Results:
 - After evaluating each model, we compare their performance and identify the best-performing algorithm.
 - We also identify the worst-performing model and provide reasoning for its performance.

## Conclusion
The goal of this project is to demonstrate how various machine learning models can be used to predict housing prices. By comparing multiple models, we can choose the one that works best for our dataset and achieve better prediction accuracy.


