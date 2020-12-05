# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:40:55 2020

@author: shhiv
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model = pickle.load(open('classification_model.pkl','rb'))
transformer = pickle.load(open('cv_transformer.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():  
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if(request.method == 'POST'):
        message = request.form['message']
        data = [message]
        vect = transformer.transform(data).toarray()
        prediction = model.predict(vect)
    return render_template('results.html',prediction = prediction)

if __name__ == '__main__':
	app.run(debug=True)