# Making and training the Linear Regression model on the dataset = 'insurance'
from pycaret.datasets import get_data

dataset = get_data('insurance')

from pycaret.regression import *
# Experiment 1 using simple model creation without any feature scaling
# s1 = setup(dataset, target = 'charges', session_id = 123)

# lr = create_model('lr')
# plot_model(lr)

# Experiment 2 adding some additional parameters
s2 = setup(dataset, target = 'charges', session_id = 123,
          normalize = True,
          polynomial_features = True,
          trigonometry_features = True,
          feature_interaction = True,
          bin_numeric_features = ['age', 'bmi'])

lr = create_model('lr')

plot_model(lr)
save_model(lr, 'deployment_30052020')

# import requests
# url = 'https://pycaret-insurance.herokuapp.com/predict_api'
# pred = requests.post(url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})
# print(pred.json())