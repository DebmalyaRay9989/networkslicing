from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('telecom.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST'])
def predict():
    output = 0
    prediction = 0
    if request.method == 'POST':
        lte_5g_category = int(request.form['lte_5g_category'])
        time = int(request.form['time'])
        packet_loss_rate = float(request.form['packet_loss_rate'])
        packet_delay = int(request.form['packet_delay'])
        io_t = int(request.form['io_t'])
        lte_5g = int(request.form['lte_5g'])
        gbr = int(request.form['gbr'])
        non_gbr = int(request.form['non_gbr'])
        ar_vr_gaming = int(request.form['ar_vr_gaming'])
        healthcare = int(request.form['healthcare'])
        industry_4_0 = int(request.form['industry_4_0'])
        io_t_devices = int(request.form['io_t_devices'])
        public_safety = int(request.form['public_safety'])
        smart_city_and_home = int(request.form['smart_city_and_home'])
        smart_transportation = int(request.form['smart_transportation'])
        smartphone = int(request.form['smartphone'])

        prediction = model.predict([[lte_5g_category, time, packet_loss_rate, packet_delay, io_t, lte_5g, gbr, non_gbr, ar_vr_gaming,
                                   healthcare, industry_4_0, io_t_devices, public_safety, smart_city_and_home, smart_transportation, smartphone]])
        output = int(prediction)
        if prediction == 1:
            return render_template('main.html', prediction_text="It is predicted as category : 1")
        elif prediction == 2:
            return render_template('main.html', prediction_text="It is predicted as category : 2")
        elif prediction == 3:
            return render_template('main.html', prediction_text="It is predicted as category : 3")
    else:
        return render_template('main.html')


if __name__ == "__main__":
    app.run(debug=True)


