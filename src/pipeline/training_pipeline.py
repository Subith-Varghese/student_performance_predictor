import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger

if __name__ == "__main__":
    try:
        # ---------------------------
        # Step 1: Load Dataset
        # ---------------------------
        data_path = r"data\students-performance-in-exams\StudentsPerformance.csv"  # Change path if needed
        logger.info("📥 Loading dataset...")
        df = pd.read_csv(data_path)
        logger.info(df.head())
        logger.info(df.info())
        logger.info(f"✅ Dataset loaded successfully with shape {df.shape}")

        # ---------------------------
        # Step 2: Separate Features & Target
        # ---------------------------
        X = df.drop(columns=['math score'], axis=1)
        y = df['math score']

        # ---------------------------
        # Step 3: Data Transformation
        # ---------------------------
        transformer = DataTransformation()
        X_transformed = transformer.fit_transformer(X)
        transformer.save_preprocessor("artifacts/preprocessor.pkl")
        logger.info("✅ Data transformation completed.")

        # ---------------------------
        # Step 4: Train-Test Split
        # ---------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )
        logger.info(f"🔹 Train/Test split completed: {X_train.shape}, {X_test.shape}")

        # ---------------------------
        # Step 5: Save Train & Test CSVs
        # ---------------------------
        try:
            os.makedirs("data", exist_ok=True)
            train_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
            test_df = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

            train_path = os.path.join("data", "train.csv")
            test_path = os.path.join("data", "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info(f"💾 Train dataset saved at: {train_path}")
            logger.info(f"💾 Test dataset saved at: {test_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save train/test CSVs: {e}")

        # ---------------------------
        # Step 6: Train Model
        # ---------------------------
        trainer = ModelTrainer(model_type="linear")
        trainer.train(X_train, y_train)

        # ---------------------------
        # Step 7: Evaluate Model
        # ---------------------------
        train_metrics = trainer.evaluate(X_train, y_train)
        test_metrics = trainer.evaluate(X_test, y_test)

        logger.info(f"📊 Training Metrics: {train_metrics}")
        logger.info(f"📊 Testing Metrics: {test_metrics}")

        # ---------------------------
        # Step 8: Save Model
        # ---------------------------
        trainer.save_model("artifacts/best_model.pkl")
        logger.info("✅ Best model saved successfully.")

    except Exception as e:
        logger.error(f"❌ Pipeline Execution Failed: {e}")
