from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("airbnb.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = {
            "City": request.form.get("City"),
            "RoomType": request.form.get("Roomtype"),  #  Fix: Match HTML name="Roomtype"
            "Bedrooms": int(request.form.get("Bedrooms")),
            "Bathrooms": int(request.form.get("Bathrooms")),
            "GuestsCapacity": int(request.form.get("GuestsCapacity")),
            "HasWifi": int(request.form.get("HasWifi")),
            "HasAC": int(request.form.get("HasAC")),
            "DistanceFromCityCenter": float(request.form.get("DistanceFromCityCenter"))
        }

        # Convert input data to DataFrame and predict
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        return render_template("index.html", predictt=f"Predicted Price: ₹{prediction:.2f}")
    
    return render_template("index.html")


if __name__ == "__main__":  #  Fix: remove extra parentheses
    app.run(host="0.0.0.0", port=5000, debug=True)
