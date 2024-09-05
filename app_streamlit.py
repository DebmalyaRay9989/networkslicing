
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('telecom.pkl', 'rb'))

# Initialize the scaler
standard_to = StandardScaler()


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting inputs from the form
            lte_5g_category = int(request.form['lte_5g_category'])
            time_str = request.form['time']
            
            # Parse time in different formats (e.g., 'HH:MM' or 'HH:MM:SS')
            try:
                time_obj = datetime.strptime(time_str, '%H:%M:%S')
            except ValueError:
                time_obj = datetime.strptime(time_str, '%H:%M')
            time = time_obj.hour * 60 + time_obj.minute  # Convert time to total minutes
            
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

            # Feature scaling
            features = [[lte_5g_category, time, packet_loss_rate, packet_delay, io_t, lte_5g, gbr, non_gbr, ar_vr_gaming,
                         healthcare, industry_4_0, io_t_devices, public_safety, smart_city_and_home, smart_transportation, smartphone]]
            scaled_features = standard_to.transform(features)

            # Prediction
            prediction = model.predict(scaled_features)[0]

            # Return the predicted category
            if prediction == 1:
                return render_template('main.html', prediction_text="It is predicted as category: 1")
            elif prediction == 2:
                return render_template('main.html', prediction_text="It is predicted as category: 2")
            elif prediction == 3:
                return render_template('main.html', prediction_text="It is predicted as category: 3")
        except Exception as e:
            return render_template('main.html', prediction_text=f"Error in prediction: {str(e)}")
    else:
        return render_template('main.html')


if __name__ == "__main__":
    app.run(debug=True)

