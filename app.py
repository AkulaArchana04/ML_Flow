from flask import Flask,render_template,request
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load("airbnb.pkl")

@app.route("/")

def index():
    return render_template("index.html")

#it is to transfer the data
@app.route("/predict",methods=["POST"])
def home():
    if request.method=="POST":
        data={
            "City":request.form.get("City"),
            "RoomType":request.form.get("RoomType"),
            "Bedrooms":int(request.form.get("Bedrooms")),
            "Bathrooms":int(request.form.get("Bathrooms")),
            "GuestsCapacity":int(request.form.get("GuestsCapacity")),
            "HasWifi":int(request.form.get("HasWifi")),
            "HasAC":int(request.form.get("HasAC")),
            "DistanceFromCityCenter":float(request.form.get("DistanceFromCityCenter"))
        }

        #prediction
        df=pd.DataFrame([data])
        prediction=model.predict(df)[0]
        return render_template("index.html",predictt=prediction)
    else:
        return render_template("index.html")


if __name__==("__main__"):
    app.run(host="0.0.0.0",port=5000,debug=True)