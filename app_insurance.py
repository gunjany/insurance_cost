# Creating a front-end web app using flask

from flask import url_for, request, Flask, redirect, render_template, jsonify
from pycaret.regression import predict_model, load_model
import pandas as pd
import numpy as np
import pickle

app_insurance = Flask(__name__, template_folder = 'template')

model = load_model('deployment_30052020')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app_insurance.route('/')
def home():
    return render_template("home.html")


@app_insurance.route('/predict', methods = ['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html', pred = 'Expected Bill will be {}'.format((prediction)))

# @app_insurance.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

if __name__ == '__main__':
    app_insurance.run(debug = True)