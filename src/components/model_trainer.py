import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import logger
import numpy as np


class ModelTrainer:
    def __init__(self, model_type="linear"):
        if model_type.lower() == "ridge":
            self.model = Ridge(alpha=0.1)
        elif model_type.lower() == "lasso":
            self.model = Lasso(alpha=0.01)
        else:
            self.model = LinearRegression()
        self.model_type = model_type

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_type} model trained successfully.")
        return self.model

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse  = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        logger.info(f"Evaluation Metrics - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def save_model(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved at {file_path}")
