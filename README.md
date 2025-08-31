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

```
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
│   └── home.html                         # Prediction form
│   └── result.html                          
│
├── app.py                                # Flask web app
├── requirements.txt                      # Python dependencies
└── README.md                             
```

---
Project Workflow
1️⃣ Data Downloading
- Dataset is downloaded from Kaggle using opendatasets.
- Path: data/students-performance-in-exams/StudentsPerformance.csv.

2️⃣ Data Preprocessing
- Features are separated into numerical (reading score, writing score) and categorical (gender, race/ethnicity, lunch, test preparation course, parental level of education).
- Categorical features are one-hot encoded.
- Numerical features are standard scaled.
- The preprocessing pipeline is saved as artifacts/preprocessor.pkl.

3️⃣ Train-Test Split
- Dataset is split 80% train / 20% test.
- Split datasets are saved as CSV for reproducibility.

4️⃣ Model Training
- Regression models trained:
  - Linear Regression
  - Ridge
  - Lasso

- Models are evaluated using: R2 score, RMSE, MAE.
- Best model is saved as artifacts/best_model.pkl.

5️⃣ Prediction

- Users can input new student data through a Flask web app.
- The pipeline preprocesses inputs and predicts the Math score.

6️⃣ Deployment

- Flask app runs locally:
```
python app.py
```
- Accessible via browser: http://127.0.0.1:5000/.
---
| Feature                     | Input Type | Options                                                                                             |
| --------------------------- | ---------- | --------------------------------------------------------------------------------------------------- |
| Gender                      | Dropdown   | Male, Female                                                                                        |
| Race/Ethnicity              | Dropdown   | Group A, B, C, D, E                                                                                 |
| Parental Level of Education | Dropdown   | Some High School, High School, Some College, Associate's Degree, Bachelor's Degree, Master's Degree |
| Lunch Type                  | Dropdown   | Standard, Free/Reduced                                                                              |
| Test Preparation Course     | Dropdown   | None, Completed                                                                                     |
| Reading Score               | Number     | 0–100                                                                                               |
| Writing Score               | Number     | 0–100                                                                                               |
---
