# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:03:52 2020

@author: Nandhu
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    req=request.form
    diabete=float(req.get("diabetes"))
    blood_pressure=float(req.get("BP"))
    heart_disease=float(req.get("heart"))
    pregnants=float(req.get("pregnant"))
    
    int_features=(diabete,blood_pressure,heart_disease,pregnants)
    final_features = np.array(int_features, dtype=np.float32)
    final_features = [np.array(int_features)]
    
    
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='priority score :  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)