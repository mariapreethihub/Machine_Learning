# Credit Card Default Prediction using Machine Learning

## Project Overview
This project aims to predict credit card payment defaulters using machine learning techniques.Since the dependant variable 
is of categorical in nature indicating whether a customer will default or not,a classification algorithm is used for prediction.

## Dataset Overview
The dataset for this analysis is taken from UCI Machine Learning Repository. It contains 30000 instances and 25 features, 
including the target variable. The target variables indicates the default status of credit card payment where 1=Default 
and 0=No Default.Some features like sex,marriage and education are stored as integer but represents categorical variable.

Dataset Link:(https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## Library imports
The following python libraries were used for analysis:
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns
- import warnings
  warnings.filterwarnings('ignore')
- from sklearn.model_selection import train_test_split,GridSearchCV
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
In this phase, we explore and analyze the basic structure and characteristics of the dataset. The following aspects are 
examined:

- Dataset shape (number of rows and columns)
- Data types of each feature
- Unique values and distributions
- Statistical summary (mean, median, standard deviation, etc.)
- Missing values
This initial analysis helps for a better understanding of the dataset and  for the preprocessing steps.

## Datapreprocessing
Various dataset preprocessing steps were performed to prepare data for modelling.Same is brifely explained in this part:

a. Removing irrelavent fields 
   - The numerical column 'ID' does not contribute to prediction and hence dropped.
     
b. Handling Null Values
   - The dataset does not contain any null values.
     
c. Finding and removing duplicate value
   - 35 duplicate rows were identified and removed using the function drop_duplicates().
     
d. Outlier detection and handling
   - Outliers in the numerical fields(excluding Target) are detected using the boxplot.
   - The Inter Quartile Method(IQR) method was used to handled outliers by clipping the extreme values to lower and upper
     bound.
     
e. Handling Skewness
   - Features BILL_AMT 1 to 6 and PAY_AMT 1 to 6 are highly skewed.
   - As these columns include negative and zero values,a cube root transformation method is used to reduce skewness.

   After treating skewness, the dataset was revalidated  to ensure that there are no null values and duplicated values and the
   outliers are handled properly.

## Feature Engineering

 To improve model performance, features are changed or created.In this part the following steps were performed:
 
a. Renaming Columns

   For better readability and consistency columns were renamed as under:
   
   - Target column 'default payment next month' as 'TARGET'
   - PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6 as REPAYMENT-SEPT,REPAYMENT-AUG,REPAYMENT-JULY,REPAYMENT-JUNE,REPAYMENT-MAY &
     REPAYMENT-APRIL
   - BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6 as BILL_AMT-SEPT,BILL_AMT-AUG,BILL_AMT-JULY,BILL_AMT-JUNE,
     BILL_AMT-MAY& BILL_AMT-APRIL
   - PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6 as AMT_PAID-SEPT,AMT_PAID-AUG,AMT_PAID-JULY,AMT_PAID-JUNE,AMT_PAID-MAY&
     AMT_PAID-APRIL 
  
b. Handling undefined values

   Some categorical features like SEX, EDUCATION and MARRIAGE has undefined values .The undefined were identified and removed by isin().

## Encoding categorical variables
  
  - In the dataset, the columns EDUCATION and MARRIAGE are of datatype integer, however they represent categorical values.
  - To help model predict these easily OneHotEncoding technique is used.
  - This technique converts categorical values into 0 and 1 to treat them as numeric.

## Feature Selection
In this step, the most important features were selected to improve the model's performance and reduce complexity.

  - The relationship between each feature and the target variable is found using correlation matrix.
  - Features that showed low or no correlation with the target were considered less important and were removed.
    
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
   - StandardScaler is chosen because the dataset contains outliers.

## Model selection 
   For prediction, the initialy the model the trained using the following classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Classifier

## Model Evaluation
  - The performance of each model is evaluated using key matrics like accuracy,recall score, precision score and F1 score
  - Based on evaluation results :
     - Random Forest Classifer has better evaluation score in terms of accuracy,precision score and F1 score.
     - Logisitic regression model was able to identify most of the defaulter.
  -Random Forest Classification model for prediction as the model shows a balanced performance across all metrics.

    
## Hyperparameter Tuning
  In this step, the models were tuned with best parameters to improve the evaluation metrics which inturn improve efficiency.

  - The method used for hypertuning is GridSearchCV. A parameter grid is passed to find the best parameter.
  - The model is retrained using best parameter obtained.
  - Random Forest Classifier model performance has increased in terms of all paramters.
 ## Final Predictions

  - Final Predictions were made using the test dataset by passing selected input paramters.
  - The features used for prediction are LIMIT_BAL', 'SEX', 'AGE', 'REPAYMENT-SEPT', 'REPAYMENT-AUG','REPAYMENT-JULY', 'REPAYMENT-JUNE', 'REPAYMENT-MAY', 'REPAYMENT-APRIL',
    'BILL_AMT-SEPT', 'BILL_AMT-AUG', 'BILL_AMT-JULY', 'AMT_PAID-SEPT','AMT_PAID-AUG', 'AMT_PAID-JULY', 'AMT_PAID-JUNE', 'AMT_PAID-MAY',
    'AMT_PAID-APRIL', 'MARRIAGE_2', 'MARRIAGE_3', 'EDUCATION_2','EDUCATION_3', 'EDUCATION_4'.
  - Predictions were also performed using custom input cases.
  - Before prediction ,the input values are scaled using the same scaling technique that was applied during training to maintain consistency.

## Conclusion and Key Findings

 - This project demonstrates how machine learning technique can be used for credit card payment defaults.
 - Among the tested models, Random Forest Classifer has the best overall performance with good score in accuracy,
   precision,recall and F1 score.
 - The dataset was highly imbalance and same is handling using SMOTE technique.
 - Early identification of defaulter helps Financial institutions to identify likely defaulter
   there by reducing the credit risk.
 - The model gives banks sufficient lead time to follow up with potential defaulters and take preventive actions, thereby reducing financial loss.
   
  ## Opportunities for Improvement
  
 - As a point for improvement customer behavioural features can be included.
 - Also,advanced models may be used for much more better performance.
 - The current model uses historical repayment behavior and billing information to predict defaults. However, it does not check if the minimum payment made by the customer meets a specific threshold.
 - In real-world banking, minimum due amounts are often a fixed percentage of the credit limit or outstanding balance, and not paying this minimum can be an early indicator that the customer is facing financial difficulty.
 - Including a feature that checks whether customer has consistently paid at least the minimum due could enhance model's ability to identfy high risk customer.
 - Also,the current model tends to misclassify customers who have not utilized their credit limit at all as potential defaulters. This is due to a lack of sufficient training samples.
 - To address this, the training data shoud be included with more examples of such “no usage” cases, or same has to be  handled separately before model prediction.



