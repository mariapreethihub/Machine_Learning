# Firewall Action Prediction using Machine Learning

## Project Overview
This project aims to predict the “Action” taken by the firewall based on traffic data using machine learning techniques. Here the ‘Action’ column serves as class label (target variable), has four possible values:
1.	Allow: Traffic permitted.
2.	Deny: Traffic blocked.
3.	Drop: Traffic dropped.
4.	Reset-Both: Reset connection.
Since the dependant variable is of categorical in nature with multiple classes, the problem is a multi-class classification task.

## Dataset Overview
The dataset for this analysis is taken from UCI Machine Learning Repository. It contains 65,532 instances and 12 features including the target variable. The target variable has four possible values indicating the Action taken by firewall. 
The class label ‘Action’ is categorical in nature.

Dataset Link: https://archive.ics.uci.edu/dataset/542/internet+firewall+data
## Library imports
The following python libraries were used for analysis:
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns
- import warnings
  warnings.filterwarnings('ignore')
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import LabelEncoder
- from imblearn.over_sampling import SMOTE 
- from sklearn.preprocessing import StandardScaler
- from sklearn.linear_model import LogisticRegression
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.svm import SVC
- from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,classification_report

## Dataset Loading 
The dataset is in CSV format, and it was loaded into a pandas DataFrame using the pd.read_csv() function.

## Exploratory Data Analysis(EDA)
In this phase, we explore and analyze the basic structure and characteristics of the dataset. The following aspects are examined:

- Dataset shape (number of rows and columns)
- Data types of each feature
- Unique values and distributions
- Statistical summary (mean, median, standard deviation, etc.)
- Missing values
This initial analysis helps for a better understanding of the dataset and for the preprocessing steps.

## Data preprocessing
Various dataset preprocessing steps were performed to prepare data for modelling. Same is briefly explained in this part:

a. Removing class
- The class 'reset-both' has only 54 rows compared to other classes. Model will not learn enough from this 54 samples. It might completely ignore or predict something else. Hence, it is better to drop the class as it is irrelevant.

b. Removing irrelevant fields 
- The given dataset has separate fields ‘Bytes Sent’ and ‘Bytes Received’, which already sum up to total traffic. Therefore, the column ‘Bytes’ (representing total bytes) is dropped to avoid redundancy.
  
c. Handling Null Values
- The dataset does not contain any null values.
     
d. Finding and removing duplicate value
- 8362 duplicate rows were identified and removed using the function drop_duplicates().
     
e. Outlier detection and handling
- Outliers in the numerical fields (excluding Target) are detected using the boxplot.
- The Inter Quartile Method (IQR) method was used to handled outliers by clipping the extreme values to lower and upper bound.
    
f. Handling Skewness
- Features NAT Destination Port, Bytes Sent, Bytes Received, Packets, pkts_sent and pkts_received are highly skewed.
- As these columns include negative and zero values, a cube root transformation method is used to reduce skewness.

After treating skewness, the dataset was revalidated to ensure that there are no null values and duplicated values and the outliers are handled properly.

## Feature Engineering

 To improve model performance, features are changed or created. In this part the following steps were performed:
 
a. Renaming Columns

- For better readability and consistency, the column ‘Action’ is renamed as 'TARGET'
  
b. Creating new feature

-	Features pkts_sent and pkts_received are combined to get new feature ‘Total_Packets’.

## Encoding categorical variables
  
  - In the dataset, the target column represents categorical values.
  - To help model predict these easily Label Encoding technique is used as the numbers are just class ID’s and not values in order.
  

## Feature Selection
In this step, the most important features were selected to improve the model's performance and reduce complexity.

  - The relationship between each feature and the target variable is found using correlation matrix.
  - Features that showed low / no correlation and high relation with the target were removed.
    
## Splitting Features and Target (X and y)

   - All the features except target is assigned to X.
   - Target value is assigned to y.
     
## Train-Test Split

   - Dataset was split into equal proportion for testing and training purpose.
   - 75% of the data was used for training, and 25% was used for testing.
     
## Handling Imbalanced Data

   - Class distribution of target variable is found using value_counts() and it is observed that data is imbalanced.
   - Using SMOTE technique, the issue is addressed by increasing the minority class by generating synthetic samples.
     
## Feature Scaling
   Scaling is performed to bring all numerical features into the same scale.
   - In this project, Standard Scaling is applied using the StandardScaler.
   - Standard Scaler is chosen because the dataset contains outliers.

## Model selection 
   For prediction, the initially the model the trained using the following classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Classifier

## Check for overfitting
- Train and test accuracy score is found for the model using accuracy_score.
-	Difference between train accuracy score and test accuracy score is more than 20% for all the ML algorithms which suggests overfitting.  
-	As Random Forest is good in handling overfitting and SVM may bias because of overfitting Random Forest Classifier Algorithm is chosen for further evaluation.
  
 ## Model Evaluation
  - The performance of the model is evaluated using key matrics like accuracy, recall score, precision score and F1 score
  - Based on evaluation results:
  - Random Forest Classifier has evaluation score near to 1 in terms of accuracy, precision score and F1 score.
    
## Final Predictions

  - Final Predictions were made using the test dataset by passing selected input parameters.
  - The features used for prediction are 'Source Port', 'Destination Port', 'NAT Source Port','NAT Destination Port', 'Bytes Sent', 'Bytes Received','Elapsed Time (sec).

## Conclusion and Key Findings
 - This project demonstrates how machine learning techniques can be used to predict firewall actions based on network traffic.
 - Random Forest Classifier showed the best overall performance, achieving high accuracy,
   precision, recall and F1 score.
 - The dataset was highly imbalanced and this was handled using the SMOTE technique to oversample minority classes in the training set.   

  



