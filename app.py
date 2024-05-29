from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved models and accuracies
with open('lr_model.pkl', 'rb') as file:
    lr_model, lr_accuracy = pickle.load(file)

with open('rf_model.pkl', 'rb') as file:
    rf_model, rf_accuracy = pickle.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_model, svm_accuracy = pickle.load(file)

# Load LabelEncoders
with open('label_encoders_gender.pkl', 'rb') as file:
    label_encoders1 = pickle.load(file)

with open('label_encoders_smoking.pkl', 'rb') as file:
    label_encoders2 = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    gender = request.form['gender']
    smoking_history = request.form['smoking_history']
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = int(request.form['blood_glucose_level'])
    print("Working")
    # Convert categorical variables using LabelEncoder
    gender_encoded = label_encoders1.transform([gender])[0]
    smoking_history_encoded = label_encoders2.transform([smoking_history])[0]
    # Transform gender using label encoder
    
    # Make predictions using the models
    lr_prediction = lr_model.predict([[gender_encoded, age, hypertension, heart_disease, smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]])
    rf_prediction = rf_model.predict([[gender_encoded, age, hypertension, heart_disease, smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]])
    svm_prediction = svm_model.predict([[gender_encoded, age, hypertension, heart_disease, smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]])
    print("predicting")
    # Calculate ensemble prediction
    ensemble_prediction = np.mean([lr_prediction, rf_prediction, svm_prediction])
    if ensemble_prediction==1:
        result="Patient is diabetic"
    else:
        result="Patient is not diabetic"
    
    return render_template('index.html', 
                           lr_prediction=lr_prediction, 
                           lr_accuracy=lr_accuracy,
                           rf_prediction=rf_prediction, 
                           rf_accuracy=rf_accuracy,
                           svm_prediction=svm_prediction, 
                           svm_accuracy=svm_accuracy,
                           ensemble_prediction=ensemble_prediction,
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)

