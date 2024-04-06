from flask import Flask,render_template, render_template,request
import pickle
import numpy as np
from sklearn import *
from sklearn.linear_model._base import _preprocess_data
#from sklearn.ensemble._iforest import *
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
 
@app.route('/')
def hello_world():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 6)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict',methods = ['POST'])
def get_result():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    result = ValuePredictor(to_predict_list)        
    if int(result)== 1:
        prediction ='Income more than 50K'
    else:
        prediction ='Income less that 50K'           
    return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)

