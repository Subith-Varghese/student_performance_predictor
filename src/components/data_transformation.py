import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from src.logger import logger
import os

class DataTransformation:
    def __init__(self):
        self.preprocessor = None

    def fit_transformer(self, X):
        num_features = ['reading score', 'writing score']
        cat_features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course','parental level of education']

        oh_transformer = OneHotEncoder(drop='first',handle_unknown="ignore")
        scaler = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("onehot", oh_transformer, cat_features),
                ("scaler", scaler, num_features)
            ]
        )
        X_transformed = self.preprocessor.fit_transform(X)
        logger.info("Data transformation completed successfully.")
        return X_transformed

    def transform_new_data(self, X):
        X_transformed = self.preprocessor.transform(X)
        return X_transformed

    def save_preprocessor(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        logger.info(f"Preprocessor saved at {file_path}")
