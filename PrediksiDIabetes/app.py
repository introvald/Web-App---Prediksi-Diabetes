import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
def load_model():
    return pickle.load(open("diabetes_model.sav", "rb"))

def load_scaler():
    return pickle.load(open("scaler.sav", "rb"))

def predict_diabetes(input_data, model, scaler):
    # Preprocess input data
    std_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(std_data)

    return prediction

def main():
    st.title('Prediksi Diabetes')
    st.write('Masukkan data pasien untuk melakukan prediksi')

    # Define input fields for user to input patient data
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=17, step=1)
    glucose = st.number_input("Glukosa (mg/dL)", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0, max_value=122, step=1)
    skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=99, step=1)
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, step=1)
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=67.1, step=0.1)
    diabetes_pedigree = st.number_input("Riwayat Diabetes Keluarga", min_value=0.078, max_value=2.42, step=0.001)
    age = st.number_input("Usia (tahun)", min_value=21, max_value=81, step=1)

    # Create a numpy array from the user input data
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)

    # Load the model and scaler
    model = load_model()
    scaler = load_scaler()

    if st.button('Jalankan Prediksi'):
        # Perform prediction
        prediction = predict_diabetes(input_data, model, scaler)

        # Display prediction result
        if prediction[0] == 0:
            st.write("Pasien tidak terkena diabetes :D")
        else:
            st.write("Pasien terkena diabetes T_T")

if __name__ == '__main__':
    main()



