import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")



# Use LabelEncoder for 'gender' column
label_encoders_gender = LabelEncoder()
df['gender'] = label_encoders_gender.fit_transform(df['gender'])

# Use LabelEncoder for 'smoking_history' column
label_encoders_smoking= LabelEncoder()
df['smoking_history'] = label_encoders_smoking.fit_transform(df['smoking_history'])

# Split the dataset into features and target
X = df.drop(columns=['diabetes'])  # Features
y = df['diabetes']  # Target variable

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)
# Calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
print("Logistic Regression Accuracy:", lr_accuracy)
# Save Logistic Regression model to a pickle file
with open('lr_model.pkl', 'wb') as file:
    pickle.dump((lr_model, lr_accuracy), file)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print("Random Forest Accuracy:", rf_accuracy)
# Save Random Forest model to a pickle file
with open('rf_model.pkl', 'wb') as file:
    pickle.dump((rf_model, rf_accuracy), file)

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
print("SVM Accuracy:", svm_accuracy)
# Save SVM model to a pickle file
with open('svm_model.pkl', 'wb') as file:
    pickle.dump((svm_model, svm_accuracy), file)

# Save LabelEncoders to a pickle file
with open('label_encoders_gender.pkl', 'wb') as file:
    pickle.dump(label_encoders_gender, file)

with open('label_encoders_smoking.pkl', 'wb') as file:
    pickle.dump(label_encoders_smoking, file)


