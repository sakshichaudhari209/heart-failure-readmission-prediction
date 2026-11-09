from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Please train and save your model first."
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------
# Home Page
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------
        # Get values from HTML form
        # -------------------------------------------

        age = float(request.form["Age"])
        gender = request.form["Gender"]

        bmi = float(request.form["BMI"])

        smoking_status = request.form["Smoking_Status"]
        alcohol_consumption = request.form["Alcohol_Consumption"]

        systolic_bp = float(request.form["Systolic_BP"])
        diastolic_bp = float(request.form["Diastolic_BP"])

        heart_rate = float(request.form["Heart_Rate"])

        cholesterol = float(request.form["Cholesterol"])

        diabetes = int(request.form["Diabetes"])
        hypertension = int(request.form["Hypertension"])

        kidney_disease = int(request.form["Kidney_Disease"])

        length_of_stay = float(
            request.form["Length_of_Stay"]
        )

        icu_admission = int(
            request.form["ICU_Admission"]
        )

        emergency_admission = int(
            request.form["Emergency_Admission"]
        )


        # -------------------------------------------
        # Create input DataFrame
        # -------------------------------------------

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender],

            "BMI": [bmi],

            "Smoking_Status": [smoking_status],

            "Alcohol_Consumption": [
                alcohol_consumption
            ],

            "Systolic_BP": [systolic_bp],

            "Diastolic_BP": [diastolic_bp],

            "Heart_Rate": [heart_rate],

            "Cholesterol": [cholesterol],

            "Diabetes": [diabetes],

            "Hypertension": [hypertension],

            "Kidney_Disease": [kidney_disease],

            "Length_of_Stay": [
                length_of_stay
            ],

            "ICU_Admission": [
                icu_admission
            ],

            "Emergency_Admission": [
                emergency_admission
            ]
        })


        # -------------------------------------------
        # Make prediction
        # -------------------------------------------

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]


        # -------------------------------------------
        # Prediction result
        # -------------------------------------------

        probability_percentage = round(
            probability * 100,
            2
        )


        if prediction == 1:

            result = (
                "High Risk of 30-Day Readmission"
            )

            message = (
                "The model predicts that the "
                "patient may be readmitted "
                "within 30 days."
            )

        else:

            result = (
                "Low Risk of 30-Day Readmission"
            )

            message = (
                "The model predicts that the "
                "patient is unlikely to be "
                "readmitted within 30 days."
            )


        return render_template(
            "index.html",

            prediction=result,

            message=message,

            probability=probability_percentage
        )


    except Exception as e:

        return render_template(
            "index.html",

            error=str(e)
        )


# ---------------------------------------------------
# Run Flask Application
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )