
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('telecom.pkl', 'rb'))

# Initialize the scaler
standard_to = StandardScaler()

# Streamlit app
st.title("Telecom Network Category Prediction")

# Input fields
lte_5g_category = st.number_input('LTE/5G Category', min_value=0, max_value=5, step=1)
time = st.text_input('Time (HH:MM or HH:MM:SS)', '00:00')
packet_loss_rate = st.number_input('Packet Loss Rate', min_value=0.0, max_value=100.0, step=0.01)
packet_delay = st.number_input('Packet Delay (ms)', min_value=0, max_value=1000, step=1)
io_t = st.selectbox('IoT Involvement (0 for No, 1 for Yes)', [0, 1])
lte_5g = st.selectbox('LTE/5G (0 for LTE, 1 for 5G)', [0, 1])
gbr = st.number_input('GBR', min_value=0, max_value=10, step=1)
non_gbr = st.number_input('Non-GBR', min_value=0, max_value=10, step=1)
ar_vr_gaming = st.selectbox('AR/VR Gaming (0 for No, 1 for Yes)', [0, 1])
healthcare = st.selectbox('Healthcare (0 for No, 1 for Yes)', [0, 1])
industry_4_0 = st.selectbox('Industry 4.0 (0 for No, 1 for Yes)', [0, 1])
io_t_devices = st.number_input('IoT Devices', min_value=0, max_value=1000, step=1)
public_safety = st.selectbox('Public Safety (0 for No, 1 for Yes)', [0, 1])
smart_city_and_home = st.selectbox('Smart City and Home (0 for No, 1 for Yes)', [0, 1])
smart_transportation = st.selectbox('Smart Transportation (0 for No, 1 for Yes)', [0, 1])
smartphone = st.selectbox('Smartphone (0 for No, 1 for Yes)', [0, 1])

# Convert time input to total minutes
if ':' in time:
    from datetime import datetime
    try:
        time_obj = datetime.strptime(time, '%H:%M:%S')
    except ValueError:
        time_obj = datetime.strptime(time, '%H:%M')
    total_minutes = time_obj.hour * 60 + time_obj.minute
else:
    total_minutes = int(time)  # If input is directly in minutes

# When the "Predict" button is clicked
if st.button('Predict'):
    # Feature scaling
    features = np.array([[lte_5g_category, total_minutes, packet_loss_rate, packet_delay, io_t, lte_5g, gbr, non_gbr, 
                          ar_vr_gaming, healthcare, industry_4_0, io_t_devices, public_safety, 
                          smart_city_and_home, smart_transportation, smartphone]])
    scaled_features = standard_to.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]

    # Display the prediction
    if prediction == 1:
        st.success("It is predicted as category: 1")
    elif prediction == 2:
        st.success("It is predicted as category: 2")
    elif prediction == 3:
        st.success("It is predicted as category: 3")
