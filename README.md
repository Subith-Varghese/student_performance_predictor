🎓 Student Performance Predictor

Predict students’ Math scores based on demographic and academic features using Machine Learning. This project includes data preprocessing, model training, evaluation, and deployment via a web interface.

Project Overview

Many educators and institutions want to understand factors influencing student performance. This project predicts Math scores using features like:

Gender

Race/Ethnicity

Parental Level of Education

Lunch Type

Test Preparation Course

Reading Score

Writing Score

We train multiple regression models and select the best-performing one for deployment.

Project Structure

student_performance_predictor/
│
├── data/                                 # Dataset folder
│   └── students-performance-in-exams/   # Downloaded dataset
│       └── StudentsPerformance.csv
│
├── notebooks/                            # Jupyter notebooks
│   └── student_performance_EDA.ipynb     # EDA and visualization
│
├── src/                                  # Source code
│   ├── __init__.py
│   ├── logger.py                         # Logging utility
│   ├── components/                       # Core components
│   │   ├── __init__.py
│   │   ├── data_downloader.py            # Download dataset from Kaggle
│   │   ├── data_transformation.py        # Preprocessing (encoding, scaling)
│   │   ├── model_trainer.py              # Train, evaluate, save ML models
│   │   ├── predictor.py                  # Prediction module
│   │
│   ├── pipelines/                        # Pipelines
│   │   ├── __init__.py
│   │   ├── training_pipeline.py          # End-to-end training pipeline
│   │   ├── predict_pipeline.py           # User prediction pipeline
│
├── artifacts/                            # Saved models and preprocessors
│   ├── preprocessor.pkl                  # Scaler & encoder
│   ├── best_model.pkl                     # Trained ML model
│
├── templates/                            # Flask HTML templates
│   └── home.html                          # Prediction form
│
├── app.py                                # Flask web app
├── requirements.txt                      # Python dependencies
├── setup.py                              
└── README.md                             
