# Diabetes Prediction System

A machine learning-based web application for predicting diabetes risk based on patient symptoms and characteristics.

## 🎯 Features

- **Interactive Web Interface**: User-friendly HTML/CSS/JavaScript frontend served by Flask
- **Machine Learning Models**: Random Forest classifier with hyperparameter tuning
- **Real-time Predictions**: Instant diabetes risk assessment
- **Model Performance Metrics**: Detailed accuracy and ROC-AUC scores
- **Visual Analytics**: Confusion matrix, ROC curves, and feature importance plots

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Run the training script to create the ML models:

```bash
python train_model.py
```

This will:
- Load and preprocess the diabetes dataset
- Train Logistic Regression and Random Forest models
- Perform hyperparameter tuning
- Generate performance visualizations
- Save trained models and encoders

Expected output files:
- `random_forest_model.pkl`
- `logistic_model.pkl`
- `scaler.pkl`
- `feature_encoders.pkl`
- `target_encoder.pkl`
- `feature_names.json`
- `model_metadata.json`
- Various visualization PNG files

### Step 3: Launch the Web Application

```bash
python web_backend.py
```

Open your browser at `http://localhost:5000`

## ☁️ Deploy On Vercel

This project is configured for Vercel Python serverless deployment.

- Flask entrypoint: `app.py`
- Main app module: `web_backend.py`
- Vercel config: `vercel.json`

If you previously saw `No flask entrypoint found`, redeploy after these files are present.

Health check endpoint: `/health`

## 📊 Dataset

The system uses `diabetes_data_upload.csv` containing:
- **520 patient records**
- **16 features** including age, gender, and various diabetes symptoms
- **Binary classification**: Positive or Negative for diabetes

## 🔬 How to Use

1. **Open the web application** in your browser
2. **Enter patient information**:
   - Age and gender
   - Various symptoms (Polyuria, Polydipsia, weight loss, etc.)
3. **Click "Predict Diabetes Risk"**
4. **View results**:
   - Prediction (Positive/Negative)
   - Confidence score
   - Probability breakdown

## 📁 Project Structure

```
Diabetes/
│
├── web_backend.py                            # Flask backend for the HTML UI
├── templates/index.html                      # Main HTML page
├── static/style.css                          # Stylesheet
├── static/app.js                             # Frontend JavaScript
├── train_model.py                            # Model training script
├── copy_of_diabetes_disease_prediction_system.py  # Original notebook
├── diabetes_data_upload.csv                  # Dataset
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
│
└── Generated files (after training):
    ├── random_forest_model.pkl
    ├── logistic_model.pkl
    ├── scaler.pkl
    ├── feature_encoders.pkl
    ├── target_encoder.pkl
    ├── feature_names.json
    ├── model_metadata.json
    └── Various .png visualization files
```

## 🛠️ Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

## 📈 Model Performance

The Random Forest model achieves:
- High accuracy on test data
- Excellent ROC-AUC score
- Balanced precision and recall

Detailed metrics are displayed in the web application's "Model Info" tab.

## ⚠️ Disclaimer

**This tool is for educational and screening purposes only.**

- It is NOT a substitute for professional medical diagnosis
- Always consult with qualified healthcare professionals
- Do not make medical decisions based solely on this prediction

## 🐛 Troubleshooting

### Models not found error
- Run `python train_model.py` first to generate the model files

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Port already in use
- Change the port in `web_backend.py` (app.run) to another value, for example 8502

## 📝 License

This project is for educational purposes.

## 👨‍💻 Contributing

Feel free to fork, improve, and submit pull requests!

## 📧 Contact

For questions or issues, please open an issue in the project repository.
