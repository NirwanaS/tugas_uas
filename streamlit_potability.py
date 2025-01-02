import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

# Title and description
st.title("Prediksi Potabilitas Air")
st.write("Aplikasi ini memprediksi apakah air layak minum berdasarkan parameter yang diinputkan pengguna menggunakan model Random Forest.")

# Input form
def user_input_features():
    st.subheader("Masukkan Data untuk Prediksi")
    ph = st.text_input('pH', '7.0')
    Hardness = st.text_input('Hardness', '200.0')
    Solids = st.text_input('Solids (mg/L)', '30000.0')
    Chloramines = st.text_input('Chloramines', '7.0')
    Sulfate = st.text_input('Sulfate', '250.0')
    Conductivity = st.text_input('Conductivity', '400.0')
    Organic_carbon = st.text_input('Organic Carbon', '15.0')
    Trihalomethanes = st.text_input('Trihalomethanes', '60.0')
    Turbidity = st.text_input('Turbidity', '5.0')
    
    if st.button("Prediksi"):
        try:
            data = {
                'ph': float(ph),
                'Hardness': float(Hardness),
                'Solids': float(Solids),
                'Chloramines': float(Chloramines),
                'Sulfate': float(Sulfate),
                'Conductivity': float(Conductivity),
                'Organic_carbon': float(Organic_carbon),
                'Trihalomethanes': float(Trihalomethanes),
                'Turbidity': float(Turbidity)
            }
            return pd.DataFrame(data, index=[0])
        except ValueError:
            st.error("Harap masukkan nilai numerik yang valid di semua kolom.")
            return None
    return None

input_df = user_input_features()

if input_df is not None:
    # Load dataset and preprocess
    @st.cache_data
    def load_and_preprocess_data():
        # Load data
        water_data = pd.read_csv('water_potability.csv')

        # Handle missing values
        water_data = water_data.fillna(water_data.mean())

        # Features and target
        features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        target = 'Potability'

        X = water_data[features]
        y = water_data[target]

        # Resampling to balance the classes
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        # Scaling
        scaler = MinMaxScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)

        return X_resampled_scaled, y_resampled, scaler

    X_resampled_scaled, y_resampled, scaler = load_and_preprocess_data()

    # Train model
    @st.cache_resource
    def train_model(X, y):
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model

    model = train_model(X_resampled_scaled, y_resampled)

    # Prediction
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    # Display results
    st.subheader("Hasil Prediksi")
    result = "Layak Minum" if prediction[0] == 1 else "Tidak Layak Minum"
    st.write(f"Berdasarkan input data, air ini diprediksi *{result}*.")

    # Display model accuracy
    st.subheader("Akurasi Model")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Split the data for accuracy calculation
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)
    model_accuracy = RandomForestClassifier(random_state=42)
    model_accuracy.fit(X_train, y_train)
    y_pred = model_accuracy.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi model Random Forest adalah: {accuracy*100:.2f}%.")