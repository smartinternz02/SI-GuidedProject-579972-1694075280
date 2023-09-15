from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model and initialize the Flask app
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    if request.method == 'POST':
        # Define the list of symptoms
        col = [
            'itching', 'continuous_sneezing', 'shivering', 'joint_pain',
            'stomach_pain', 'vomiting', 'fatigue', 'weight_loss', 'restlessness',
            'lethargy', 'high_fever', 'headache', 'dark_urine', 'nausea',
            'pain_behind_the_eyes', 'constipation', 'abdominal_pain', 'diarrhoea',
            'mild_fever', 'yellowing_of_eyes', 'malaise', 'phlegm', 'congestion',
            'chest_pain', 'fast_heart_rate', 'neck_pain', 'dizziness',
            'puffy_face_and_eyes', 'knee_pain', 'muscle_weakness',
            'passage_of_gases', 'irritability', 'muscle_pain', 'belly_pain',
            'abnormal_menstruation', 'increased_appetite', 'lack_of_concentration',
            'visual_disturbances', 'receiving_blood_transfusion', 'coma',
            'history_of_alcohol_consumption', 'blood_in_sputum', 'palpitations',
            'inflammatory_nails', 'yellow_crust_ooze'
        ]

        # Get the selected symptoms from the form
        selected_symptoms = [col[x] for x in range(0, 45) if col[x] in request.form]

        # Create a binary array for symptoms
        b = [1 if col[i] in selected_symptoms else 0 for i in range(0, 45)]
        b = np.array(b).reshape(1, -1)

        # Make the prediction using the model
        prediction = model.predict(b)
        predicted_disease = prediction[0]
        print(f"Predicted Disease: {predicted_disease}")
        return render_template('results.html', prediction_text='{}'.format(predicted_disease))

if __name__ == "__main__":
    app.debug = True
    app.run()
