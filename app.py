from flask import Flask, render_template, request
import numpy as np
import pickle

KNN_model = pickle.load(open('KNN_model.pkl','rb'))
DT_model = pickle.load(open('DT_model.pkl','rb'))

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('new_home.html')

@app.route('/predict',methods=['POST'])
def predict_class():
    sepal_l = request.form['sepal_l']
    sepal_w = request.form['sepal_w']
    petal_l = request.form['petal_l']
    petal_w = request.form['petal_w']

    arr= np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    #pred_KNN = str(KNN_model.predict(arr))
    pred_DT = str(DT_model.predict(arr)[0])

    if pred_DT=='1':
        return '1__Iris-setosa'
    if pred_DT=='2':
        return '2__Iris-versicolor'
    if pred_DT=='2':
        return '3__Iris-virginica'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070, debug=True)