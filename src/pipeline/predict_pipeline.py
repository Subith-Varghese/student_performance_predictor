from src.components.predictor import Predictor


if __name__ == "__main__":
    print("\n Please provide student details for prediction\n")

    gender = input("Enter Gender (male/female): ").strip().lower()
    race = input("Enter Race/Ethnicity (group A/B/C/D/E): ").strip()
    parental_edu = input("Enter Parental Level of Education: ").strip()
    lunch = input("Enter Lunch Type (standard/free/reduced): ").strip()
    test_prep = input("Enter Test Preparation Course (none/completed): ").strip()
    reading_score = float(input("Enter Reading Score (0-100): ").strip())
    writing_score = float(input("Enter Writing Score (0-100): ").strip())

    user_input = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parental_edu,
        "lunch": lunch,
        "test preparation course": test_prep,
        "reading score": reading_score,
        "writing score": writing_score
    }

    pipeline = Predictor()
    predicted_score = pipeline.predict(user_input)

    print("\n Prediction Successful!")
    print(f" Predicted Math Score: {predicted_score:.2f}")
