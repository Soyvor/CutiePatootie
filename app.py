import streamlit as st
import pandas as pd
import joblib

def load_data():
    # Load your dataset
    try:
        df = pd.read_csv('/content/heart.csv')  # Replace with your dataset path
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'heart_disease_data.csv' is in the directory.")
        return None

def load_model():
    # Load your trained SVM model
    try:
        svm_model = joblib.load('/content/svm_model.pkl')  # Replace with your model file path
        return svm_model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'svm_model.pkl' is in the directory.")
        return None

def preprocess_input(user_inputs, df):
    """
    Preprocess user inputs to match the feature set used during training.
    This includes one-hot encoding for categorical features.
    """
    # Create a DataFrame for user input
    user_data = pd.DataFrame([user_inputs])
    
    # One-hot encode user input to match the original training dataset
    encoded_data = pd.get_dummies(user_data)

    # Ensure that the columns match the model's training data
    for col in df.columns:
        if col not in encoded_data.columns:
            encoded_data[col] = 0  # Add missing columns with default value 0
    
    # Align columns with the original dataset's feature order
    encoded_data = encoded_data[df.columns]

    return encoded_data

def main():
    # Set app title
    st.title('Heart Disease Prediction')

    # Load dataset and model
    df = load_data()
    svm_model = load_model()

    if df is None or svm_model is None:
        return  # Exit if dataset or model is missing

    # Display the dataset overview
    st.write('**Dataset Overview:**')
    st.write(df.head())

    # Collect user inputs
    st.sidebar.title('User Input')
    user_inputs = {}

    for column in df.columns[:-1]:  # Assuming the last column is the target
        if df[column].dtype == 'object' or df[column].nunique() <= 50:
            user_inputs[column] = st.sidebar.selectbox(f'Select {column}', df[column].unique())
        else:
            user_inputs[column] = st.sidebar.number_input(f'Enter {column}', value=float(df[column].mean()))

    # Preprocess user inputs
    try:
        processed_input = preprocess_input(user_inputs, df.drop(columns=[df.columns[-1]]))
    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        return

    # Make predictions
    if st.sidebar.button('Predict'):
        try:
            prediction = svm_model.predict(processed_input)
            if prediction[0] == 1:
                st.write('**Prediction:** You are predicted to have heart disease.')
            else:
                st.write('**Prediction:** You are predicted to be healthy.')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
