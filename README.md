# 🎓 Student Performance Predictor

Predict students’ Math scores based on demographic and academic features using Machine Learning. This project includes data preprocessing, model training, evaluation, and deployment via a web interface.

## Project Overview

Many educators and institutions want to understand factors influencing student performance. This project predicts Math scores using features like:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

We train multiple regression models and select the best-performing one for deployment.

---
## Project Structure

```
student_performance_predictor/
│
├── notebooks/                            # Jupyter notebooks
│   ├── EDA STUDENT PERFORMANCE.ipynb  # Exploratory Data Analysis
│   └── MODEL TRAINING.ipynb           # Model training & evaluation
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
## Project Workflow
### 1️⃣ Data Downloading
- Dataset is downloaded from Kaggle using opendatasets.
- Path: data/students-performance-in-exams/StudentsPerformance.csv.

### 2️⃣ Data Preprocessing
- Features are separated into numerical (reading score, writing score) and categorical (gender, race/ethnicity, lunch, test preparation course, parental level of education).
- Categorical features are one-hot encoded.
- Numerical features are standard scaled.
- The preprocessing pipeline is saved as artifacts/preprocessor.pkl.

### 3️⃣ Train-Test Split
- Dataset is split 80% train / 20% test.
- Split datasets are saved as CSV for reproducibility.

### 4️⃣ Model Training
- Regression models trained:
  - Linear Regression
  - Ridge
  - Lasso

- Models are evaluated using: R2 score, RMSE, MAE.
- Best model is saved as artifacts/best_model.pkl.

### 5️⃣ Prediction

- Users can input new student data through a Flask web app.
- The pipeline preprocesses inputs and predicts the Math score.

### 6️⃣ Deployment

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
```
# Clone the repository
git clone https://github.com/Subith-Varghese/student_performance_predictor.git
cd student_performance_predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (optional)
python src/components/data_downloader.py
```
---
Running the Training Pipeline
```
python src/pipelines/training_pipeline.py
```
- This trains the model and saves the artifacts (preprocessor.pkl & best_model.pkl).
- It also saves train/test datasets as CSV.
---

Running the Prediction Pipeline
```
python src/pipelines/predict_pipeline.py
```
- Accepts user inputs for all required features.
- Returns predicted Math score.
---
Running the Web App
```
python app.py
```
- Open browser: http://127.0.0.1:5000/
- Fill student details and get predicted Math score.

---
## Notebooks

### 1. **EDA_STUDENT_PERFORMANCE.ipynb**
**Exploratory Data Analysis (EDA)**  
- Initial data exploration: missing values, duplicates, data types, statistics  
- Feature engineering: `total_score`, `average`  
- Visualization:
  - Distribution plots (histograms, KDE)  
  - Pie charts for categorical features  
  - Pairplots for multivariate analysis  
  - Boxplots to detect outliers  
- Insights:
  - Female students generally perform better overall  
  - Male students score higher in Math  
  - Standard lunch improves overall performance  
  - Parental education positively correlates with student scores  

### 2. **MODEL_TRAINING.ipynb**
**Preprocessing & Model Training**  
- Features encoding: One-hot encoding for categorical variables, scaling numerical columns  
- Train-test split: 80/20  
- Models trained:
  - Linear Regression
  - Ridge & Lasso Regression
  - K-Nearest Neighbors
  - Decision Tree & Random Forest
  - XGBoost, CatBoost, AdaBoost
- Model evaluation using **R², MAE, RMSE**
- Hyperparameter tuning using GridSearchCV for Ridge and Lasso
- Best performing models: Linear, Ridge, and Lasso Regression

---
## KeyInsights & Observations
- **Female students** generally perform better overall.
- **Males** tend to have higher Math scores than females.
- **Standard lunch** improves overall performance.
- **Parental education** is positively correlated with performance.
- **Test Preparation**: Completing test prep courses improves overall scores  
- **Modeling**: Linear, Ridge, and Lasso regression models perform similarly with ~88% R² on test set


