from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = load_model('modelCB')
cols =['incident_severity', 'insured_hobbies', 'capital_loss', 'collision_type', 'incident_state', 'policy_annual_premium', 'loss_by_claims', 'property_claim']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen)
    prediction = prediction.Label[0]   
    return render_template('index.html', prediction_text='This is a fraud (1) or not (0) -----> {} '.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)