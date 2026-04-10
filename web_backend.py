from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import json
import pandas as pd
from pathlib import Path

app = Flask(__name__, static_folder='static', template_folder='templates')
PROJECT_ROOT = Path(__file__).resolve().parent

# Load models and encoders once
model = joblib.load(PROJECT_ROOT / 'random_forest_model.pkl')
feature_encoders = joblib.load(PROJECT_ROOT / 'feature_encoders.pkl')
target_encoder = joblib.load(PROJECT_ROOT / 'target_encoder.pkl')
with open(PROJECT_ROOT / 'feature_names.json', 'r') as f:
    feature_names = json.load(f)
with open(PROJECT_ROOT / 'model_metadata.json', 'r') as f:
    metadata = json.load(f)

ALLOWED_PLOTS = {
    'confusion_matrix.png',
    'roc_curve.png',
    'feature_importance.png'
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        'status': 'ok',
        'metadata': metadata
    })


@app.route('/plots/<path:filename>', methods=['GET'])
def get_plot(filename):
    if filename not in ALLOWED_PLOTS:
        return jsonify({'status': 'error', 'error': 'Plot not found'}), 404
    return send_from_directory(PROJECT_ROOT, filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        if not data:
            return jsonify({'status': 'error', 'error': 'No input data received'}), 400

        if 'Age' in data:
            data['Age'] = int(data['Age'])

        missing_features = [feature for feature in feature_names if feature not in data]
        if missing_features:
            return jsonify({'status': 'error', 'error': f"Missing features: {', '.join(missing_features)}"}), 400

        # Encode categorical features
        for feature, encoder in feature_encoders.items():
            if feature in data:
                data[feature] = encoder.transform([data[feature]])[0]

        # Prepare DataFrame
        input_df = pd.DataFrame([{k: data[k] for k in feature_names}])

        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        label = target_encoder.inverse_transform([pred])[0]

        class_probabilities = {}
        for i, probability in enumerate(proba):
            class_label = target_encoder.inverse_transform([i])[0]
            class_probabilities[class_label] = round(float(probability) * 100, 2)

        confidence = round(float(proba[pred]) * 100, 2)

        return jsonify({
            'status': 'ok',
            'prediction': label,
            'confidence': confidence,
            'probabilities': class_probabilities
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
