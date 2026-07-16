from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


# Load saved model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get uploaded file
    file = request.files["file"]


    # Read CSV
    data = pd.read_csv(file)


    # Remove labels column
    if "labels" in data.columns:
        data = data.drop("labels", axis=1)



    # Encode categorical columns
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category").cat.codes



    # Scale data
    data_scaled = scaler.transform(data)



    # Prediction
    predictions = model.predict(data_scaled)



    # Convert encoded prediction to original labels
    predictions = label_encoder.inverse_transform(predictions)



    # Count predictions
    summary = pd.Series(predictions).value_counts()



    # Total records
    total_records = len(predictions)



    # Attack counts
    normal = summary.get("Normal", 0)
    dos = summary.get("DoS", 0)
    probe = summary.get("Probe", 0)
    r2l = summary.get("R2L", 0)
    u2r = summary.get("U2R", 0)



    # Threat calculation

    attacks = total_records - normal


    if total_records > 0:
        attack_percentage = (attacks / total_records) * 100
    else:
        attack_percentage = 0



    if attack_percentage < 20:
        threat = "Low"

    elif attack_percentage < 50:
        threat = "Medium"

    else:
        threat = "High"



    return render_template(
        "result.html",

        total_records=total_records,

        normal=normal,
        dos=dos,
        probe=probe,
        r2l=r2l,
        u2r=u2r,

        threat=threat
    )



if __name__ == "__main__":
    app.run()