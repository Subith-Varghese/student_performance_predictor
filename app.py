from flask import Flask, render_template, request
from src.components.predictor import Predictor
from src.logger import logger

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        gender = request.form['gender']
        race = request.form['race']
        parental_edu = request.form['parental_edu']
        lunch = request.form['lunch']
        test_prep = request.form['test_prep']
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])

        # Create input dict for prediction
        user_input = {
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parental_edu,
            "lunch": lunch,
            "test preparation course": test_prep,
            "reading score": reading_score,
            "writing score": writing_score
        }

        # Make prediction
        predictor = Predictor()
        predicted_score = predictor.predict(user_input)

        return render_template("result.html", predicted_score=predicted_score)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return render_template("result.html", predicted_score=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
