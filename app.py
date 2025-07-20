import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(_name_)

# Load model and scaler
model = pickle.load(open('G:/AIML/ML_projects/Traffic_volume/model.pkl', 'rb'))
scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/scale.pkl', 'rb'))

@app.route('/')  # Home route
def home():
    return render_template('index.html')  # Make sure you have this HTML file

@app.route('/predict', methods=['POST', 'GET'])  # Prediction route
def predict():
    # Read input values from form and convert to float
    input_feature = [float(x) for x in request.form.values()]
    features_values = np.array([input_feature])  # Reshape for model

    # Column names used for the model input
    names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
              'hour', 'minutes', 'seconds']]  # Ensure it matches model training

    # Create DataFrame
    data = pd.DataFrame(features_values, columns=names[0])

    # Apply scaler correctly (use transform, not fit_transform)
    data_scaled = scale.transform(data)

    # Predict traffic volume
    prediction = model.predict(data_scaled)
    print(f"Prediction: {prediction}")

    # Show result in index.html
    result_text = f"Estimated Traffic Volume: {prediction[0]}"
    return render_template('index.html', prediction_text=result_text)

# Run app
if _name_ == '_main_':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
