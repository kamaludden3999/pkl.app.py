import pickle

# Assuming `logistic_model` is your trained logistic regression model
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(logistic_model, file)
  import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title("Logistic Regression Model Deployment")
st.write("This app allows users to input data and get predictions from the Logistic Regression model.")

# Sidebar for user inputs
st.sidebar.header("Input Features")
# Replace these feature names with the actual ones used in your model
feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0)

# Button to make predictions
if st.sidebar.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Make predictions
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Display the prediction
    st.write(f"Prediction: {'Class 1' if prediction[0] == 1 else 'Class 0'}")
    st.write(f"Probability of Class 1: {probability[0][1]:.2f}")

st.write("Developed with Streamlit")
streamlit run app.py

