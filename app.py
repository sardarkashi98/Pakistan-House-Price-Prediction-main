from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form inputs
        property_type = int(request.form["type"])
        location = int(request.form["location"])
        city = int(request.form["city"])
        purpose = int(request.form["purpose"])
        baths = float(request.form["baths"])
        beds = float(request.form["beds"])
        area = float(request.form["area"])

        # Convert into array
        features = np.array([[property_type, location, city, purpose, baths, beds, area]])
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction_text=f"Predicted Price: {prediction:,.0f} PKR")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
