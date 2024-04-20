from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import keras
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML Model Code --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@app.route('/')
@app.route('/about')
def about():

    return render_template("about.html")
    


@app.route('/CHDPredictor')
def cafeOccupancyPredictor():

    return render_template("CHDPredictor.html")

def preprocessDataAndPredict(PovertyRate, MedianFamilyIncome, lakidshalfshare, laseniorshalfshare, TractSeniors, TractWhite, TractAsian, TractSNAP):
    # keep all inputs in array
    data = [PovertyRate, MedianFamilyIncome, lakidshalfshare, laseniorshalfshare, TractSeniors, TractWhite, TractAsian, TractSNAP]

    # Create Data Frame
    data = pd.DataFrame({'PovertyRate': [PovertyRate], 'MedianFamilyIncome': [MedianFamilyIncome],
         'lakidshalfshare': [lakidshalfshare], 'laseniorshalfshare': [laseniorshalfshare], 'TractSeniors': [TractSeniors],
         'TractWhite': [TractWhite], 'TractAsian': [TractAsian], 'TractSNAP': [TractSNAP]})

    # open file
    file = open("finalModel.pkl", "rb")

    # Load the scaler
    scaler = joblib.load("scaler.pkl")

    
    data_scaled = scaler.transform(data)


    # load trained model
    trained_model = joblib.load(file)

    # predict
    prediction = trained_model.predict(data_scaled)

    return prediction[0]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # get form data
        PovertyRate = request.form.get('PovertyRate')
        MedianFamilyIncome = request.form.get('MedianFamilyIncome')
        lakidshalfshare = request.form.get('lakidshalfshare')
        laseniorshalfshare = request.form.get('laseniorshalfshare')
        TractSeniors = request.form.get('TractSeniors')
        TractWhite = request.form.get('TractWhite')
        TractAsian = request.form.get('TractAsian')
        TractSNAP = request.form.get('TractSNAP')

    

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(PovertyRate, MedianFamilyIncome, lakidshalfshare, laseniorshalfshare, TractSeniors, TractWhite, TractAsian, TractSNAP)
            # pass prediction to template
            return render_template('predict.html', prediction=prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass

@app.route('/resume')
def resume():

    return render_template("resume.html")
    

@app.route('/other_proj')
def other_proj():

    return render_template("other_proj.html")
    


if __name__ == '__main__':
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # app.debug = True
    app.run(host="localhost", port=5000)