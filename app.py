from flask import Flask, render_template, request
import pickle

app=Flask(__name__, static_url_path='/static')
GB=pickle.load(open('savedmodelGB.sav','rb'))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

@app.route('/predict',methods=['POST','GET'])
def predict():
    decisions = float(request.form['decisions'])
    meaningful = float(request.form['meaningful'])
    personal_identity = float(request.form['personal_identity'])
    Feeling_good = float(request.form['Feeling_good'])
    knowledge = float(request.form['knowledge'])
    not_available = float(request.form['not_available'])
    uncertainty = float(request.form['uncertainty'])
    preference = float(request.form['preference'])
    purchased = float(request.form['purchased'])
    result = GB.predict([[decisions,meaningful,personal_identity,Feeling_good,knowledge,not_available,
                        uncertainty,preference,purchased]])[0]
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)
