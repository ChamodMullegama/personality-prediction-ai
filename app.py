import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

# Load the trained model and scaler
model = joblib.load('model/best_CatBoost_model.pkl')  
scaler = joblib.load('model/scaler.pkl')  

# Define feature names
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                     'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']
one_hot_features = ['Stage_fear_No', 'Stage_fear_Yes', 
                   'Drained_after_socializing_No', 'Drained_after_socializing_Yes']
all_features = numerical_features + one_hot_features

def preprocess_input(data):
    """Preprocess input data for prediction"""
    input_df = pd.DataFrame([data])
    
   
    for cat in categorical_features:
        if cat == 'Stage_fear':
            input_df['Stage_fear_Yes'] = 1 if data.get('Stage_fear') == 'Yes' else 0
            input_df['Stage_fear_No'] = 1 if data.get('Stage_fear') == 'No' else 0
        elif cat == 'Drained_after_socializing':
            input_df['Drained_after_socializing_Yes'] = 1 if data.get('Drained_after_socializing') == 'Yes' else 0
            input_df['Drained_after_socializing_No'] = 1 if data.get('Drained_after_socializing') == 'No' else 0
    

    if input_df['Stage_fear_Yes'].iloc[0] + input_df['Stage_fear_No'].iloc[0] == 0:
        input_df['Stage_fear_No'] = 1
    if input_df['Drained_after_socializing_Yes'].iloc[0] + input_df['Drained_after_socializing_No'].iloc[0] == 0:
        input_df['Drained_after_socializing_No'] = 1
    
    input_df = input_df.reindex(columns=all_features, fill_value=0)
   
    input_processed = scaler.transform(input_df)
    
    return input_processed

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    try:
        data = request.get_json()
        required_fields = numerical_features + categorical_features
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Convert numerical inputs to float
        for field in numerical_features:
            try:
                data[field] = float(data[field])
            except ValueError:
                return jsonify({'error': f'Invalid value for {field}'}), 400
        
        # Validate categorical inputs
        for field in categorical_features:
            if data[field] not in ['Yes', 'No']:
                return jsonify({'error': f'Invalid value for {field}'}), 400
        
        processed_input = preprocess_input(data)
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)
        
        personality = 'Extrovert' if prediction[0] == 0 else 'Introvert'
        confidence = max(probability[0]) * 100
        
        return jsonify({
            'personality': personality,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)