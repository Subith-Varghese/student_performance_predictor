import pickle
import pandas as pd
from src.logger import logger

class Predictor:
    def __init__(self, preprocessor_path="artifacts/preprocessor.pkl", model_path="artifacts/best_model.pkl"):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        self.preprocessor = None
        self.model = None

    def load_artifacts(self):
        """Load the preprocessor and trained model"""
        try:
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("✅ Preprocessor & Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {e}")
            raise e

    def predict(self, input_data: dict):
        """Predict math score for given user input"""
        try:
            if self.preprocessor is None or self.model is None:
                self.load_artifacts()

            df = pd.DataFrame([input_data])
            transformed_data = self.preprocessor.transform(df)
            prediction = self.model.predict(transformed_data)
            return prediction[0]
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise e
