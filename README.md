# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients
### Developed by: Malligesh
### RegisterNumber:  21222330119
## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   - Import pandas for data manipulation, scikit-learn for modeling and evaluation, and seaborn and matplotlib for visualization.  

2. **Dataset Loading**:  
   - Load the dataset from the specified URL.  

3. **Dataset Inspection**:  
   - Examine the dataset by displaying basic information and previewing the data to understand its structure.  

4. **Target Variable Encoding**:  
   - Convert the `class` column into a binary variable indicating whether a patient is diabetic.  

5. **Feature and Target Definition**:  
   - Separate the dataset into the feature matrix (X) and target variable (y).  

6. **Data Splitting**:  
   - Divide the data into training and testing subsets with an 80-20 split.  

7. **Model Training**:  
   - Train a Logistic Regression model using the training data.  

8. **Prediction Generation**:  
   - Use the trained model to predict labels for the test dataset.  

9. **Model Evaluation**:  
   - Assess model performance by calculating accuracy, precision, recall, F1-score, and by generating a classification report.  

10. **Confusion Matrix Visualization**:  
   - Plot a heatmap of the confusion matrix to visualize the model’s evaluation results.  

## Program:
```
# Program to implement Logistic Regression for classifying food choices based on nutritional information.

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
df = pd.read_csv(url)

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Encode the target column ('class') into binary labels
# Assuming 'Diabetic' means classifying 'In Moderation' as 1 (diabetic-friendly) and others as 0
df['class'] = df['class'].str.strip("'")  # Remove surrounding quotes
df['Diabetic'] = df['class'].apply(lambda x: 1 if x == 'In Moderation' else 0)

# Define features and target
X = df.drop(columns=['class', 'Diabetic'])  # Features
y = df['Diabetic']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Output:
<img width="697" alt="Screenshot 2024-11-17 at 8 05 07 PM" src="https://github.com/user-attachments/assets/abde9e60-445e-46ab-98c0-a1a16969ed01">


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
