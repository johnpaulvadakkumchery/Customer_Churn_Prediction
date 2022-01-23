import flask
from flask import request
app = flask.Flask(__name__)

import numpy as np

from flask_cors import CORS
CORS(app)


__model = None
__scaler = None

@app.route('/')
def hello():
    return '<h1> Hello World!</h1>'


@app.route('/predict', methods=['GET','POST'])
def predict():
    #return '<h1>Prediction!</h1>'
        
    print("loading saved models...start")

    global __model
    global __scaler
    
  
    from tensorflow.keras.models import load_model
    from pickle import load

    # Loading the model
    __model = load_model('models/churn_prediction_model.h5')
    

    # load the scaler
    __scaler = load(open('models/scaler.pkl', 'rb'))
            
    print("loading saved models...complete")
    
    #print(__model.summary())
    #####################################################################################################
    age = float(request.args['age'])
    
    if request.args['gender'] == "Male":
        gender = 1
    else:
        gender = 0
        
    credit_score = float(request.args['cscore'])
    
    if request.args['geography'] == 'Germany':
        Germany = 1
        Spain = 0
    elif request.args['geography'] == 'Spain':
        Spain = 1
        Germany = 0
    else:
        Germany,Spain = 0,0

    tenure = float(request.args['tenure'])
    balance =  float(request.args['balance'])
    num_of_products = float(request.args['nprdcts'])
    
    if request.args['hascreditcard'] == "Yes":
        has_credit_card = 1
    else:
        has_credit_card = 0
        
    if request.args['isactivemember'] == "Yes":
        is_active_member = 1
    else:
        is_active_member = 0
        
    estimated_salary = float(request.args['salary'])
    
    #####################################################################################################
    
    a = np.array([[credit_score,gender,age,tenure,balance,num_of_products,has_credit_card,is_active_member,estimated_salary,Germany,Spain]])
    a_scaled = __scaler.transform(a)
    
    predicted_prob =  __model.predict(a_scaled)[0][0]
    print(str(predicted_prob))
    return str(predicted_prob)

app.run(debug=True)

