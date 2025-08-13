# Breast Cancer Classification

## Objective

The objective of this assessment is to evaluate the understanding and ability to apply supervised learning techniques to a real-world dataset.
## Dataset
We use the Breast Cancer dataset available in the sklearn.datasets library.
This dataset contains features computed from digitized images of breast mass fine-needle aspirate (FNA) and the target variable indicating whether the tumor is malignant or benign.

## Steps:
### 1. Loading and Preprocessing 
  a.Loaded dataset from sklearn.datasets.load_breast_cancer.
  b.Explored the dataset,found correlation of features with target.Highly correlated and low related features are removed.
  c.Checked for missing values and duplicate values.

### 2. Implementation of Classification Algorithm
  Implemented five classification algorithms:

  a.Logistic Regression
  b.Decision Tree Classifier
  c.Random Forest Classifier
  d.Support Vector Machine (SVM)
  e.k-Nearest Neighbors (k-NN)

### 3. Model Comparison
Evaluated models using accuracy score on test data.

Best performing algorithm:  Logistic Regression
Worst performing algorithm: Decision Tree
