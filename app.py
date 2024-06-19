from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and scaler
with open('model/knn_credit_score_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load Encoders
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_marital_status = LabelEncoder()
le_home_ownership = LabelEncoder()
le_credit_score = LabelEncoder()

# Define classes for each encoder
le_gender.classes_ = np.array(['Female', 'Male'])
le_education.classes_ = np.array(['Associate\'s Degree', 'Bachelor\'s Degree', 'Doctorate', 'High School Diploma', 'Master\'s Degree'])
le_marital_status.classes_ = np.array(['Married', 'Single'])
le_home_ownership.classes_ = np.array(['Owned', 'Rented'])
le_credit_score.classes_ = np.array(['Average', 'High', 'Low'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = request.form.get('age', '')
        gender = request.form.get('gender', '')
        income = request.form.get('income', '')
        education = request.form.get('education', '')
        marital_status = request.form.get('marital_status', '')
        number_of_children = request.form.get('number_of_children', '')
        home_ownership = request.form.get('home_ownership', '')

        if 'predict' in request.form:
            # Encode Input
            gender_encoded = le_gender.transform([gender])[0]
            education_encoded = le_education.transform([education])[0]
            marital_status_encoded = le_marital_status.transform([marital_status])[0]
            home_ownership_encoded = le_home_ownership.transform([home_ownership])[0]
            
            input_data = np.array([[int(age), gender_encoded, float(income), education_encoded, marital_status_encoded, int(number_of_children), home_ownership_encoded]])
            
            # Scale Input
            input_data_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = knn_model.predict(input_data_scaled)
            credit_score = le_credit_score.inverse_transform(prediction)[0]

            color_class = 'text-green-500' if credit_score == 'High' else 'text-yellow-500' if credit_score == 'Average' else 'text-red-500'
            
            return render_template('index.html', age=age, gender=gender, income=income, education=education, marital_status=marital_status, number_of_children=number_of_children, home_ownership=home_ownership, prediction_text=credit_score, color_class=color_class)
        
        elif 'clear' in request.form:
            return render_template('index.html')

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
