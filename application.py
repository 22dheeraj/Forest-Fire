from flask import Flask, Request, jsonify,render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model=pickle.load(open('models/scaler.pkl', 'rb'))

#@app.route("/", methods=['GET', 'POST'])
#3def index():
 #  if request.method=='POST':
  #     pass
   #else:
    #    return render_template('home.html')
   #return render_template('home.html')

@app.route('/', methods=['GET', 'Post'])
def predict_datapoint():
    if request.method=='POST':
       Temperature = float(request.form.get('Temperature'))
       RH = float(request.form.get('RH'))
       Ws = float(request.form.get('Ws'))
       Rain = float(request.form.get('Rain'))
       FFMC = float(request.form.get('FFMC'))
       DMC = float(request.form.get('DMC'))
       ISI = float(request.form.get('ISI'))
       Classes = float(request.form.get('Classes'))
       Region = float(request.form.get('Region'))

       new_data= scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
       result = ridge_model.predict(new_data)

       return render_template('home.html', results=result[0])


    else:
       return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
