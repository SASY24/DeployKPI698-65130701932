import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Load model and encoders
@st.cache_resource
def load_model():
    model_path = 'model_kpi.65130701932'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it's in the correct location.")
        return None

model_data = load_model()
if model_data:
    model, department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder = model_data

# Load your DataFrame
@st.cache_data
def load_data():
    file_path = 'Uncleaned_employees_final_dataset.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.drop('employee_id', axis=1)
    else:
        st.error(f"Data file '{file_path}' not found. Please ensure it's in the correct location.")
        return None

df = load_data()

# Function to prepare input data
def prepare_input_data(input_df):
    # Ensure all necessary columns are present
    required_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel', 
                        'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
                        'awards_won', 'avg_training_score']
    
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # or some default value
    
    # Encode categorical variables
    input_df['department'] = department_encoder.transform(input_df['department'])
    input_df['region'] = region_encoder.transform(input_df['region'])
    input_df['education'] = education_encoder.transform(input_df['education'])
    input_df['gender'] = gender_encoder.transform(input_df['gender'])
    input_df['recruitment_channel'] = recruitment_channel_encoder.transform(input_df['recruitment_channel'])
    
    # Ensure correct order of columns
    return input_df[required_columns]

# Streamlit App
st.title('Employee KPIs App')

# ... (rest of the code remains the same until the prediction part)

# Tab 1: Predict KPIs
if st.session_state.tab_selected == 0 and model_data and df is not None:
    st.header('Predict KPIs')

    # User Input Form
    department = st.selectbox('Department', department_encoder.classes_)
    region = st.selectbox('Region', region_encoder.classes_)
    education = st.selectbox('Education', education_encoder.classes_)
    gender = st.radio('Gender', gender_encoder.classes_)
    recruitment_channel = st.selectbox('Recruitment Channel', recruitment_channel_encoder.classes_)
    no_of_trainings = st.slider('Number of Trainings', 1, 10, 1)
    age = st.slider('Age', 18, 60, 30)
    previous_year_rating = st.slider('Previous Year Rating', 1.0, 5.0, 3.0)
    length_of_service = st.slider('Length of Service', 1, 20, 5)
    awards_won = st.checkbox('Awards Won')
    avg_training_score = st.slider('Average Training Score', 40, 100, 70)

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'department': [department],
        'region': [region],
        'education': [education],
        'gender': [gender],
        'recruitment_channel': [recruitment_channel],
        'no_of_trainings': [no_of_trainings],
        'age': [age],
        'previous_year_rating': [previous_year_rating],
        'length_of_service': [length_of_service],
        'awards_won': [1 if awards_won else 0],
        'avg_training_score': [avg_training_score]
    })

    # Prepare input data
    user_input_prepared = prepare_input_data(user_input)

    # Predicting
    try:
        prediction = model.predict(user_input_prepared)
        # Display Result
        st.subheader('Prediction Result:')
        st.write('KPIs_met_more_than_80:', prediction[0])
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Debug information:")
        st.write(f"Input data shape: {user_input_prepared.shape}")
        st.write(f"Input data columns: {user_input_prepared.columns}")

# ... (rest of the code remains the same)

# Tab 3: Predict from CSV
elif st.session_state.tab_selected == 2 and model_data:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read CSV file
        csv_df_org = pd.read_csv(uploaded_file)
        csv_df_org = csv_df_org.dropna()
        
        csv_df = csv_df_org.copy()
        if 'employee_id' in csv_df.columns:
            csv_df = csv_df.drop('employee_id', axis=1)
        
        # Prepare input data
        csv_df_prepared = prepare_input_data(csv_df)

        try:
            # Predicting
            predictions = model.predict(csv_df_prepared)

            # Add predictions to the DataFrame
            csv_df_org['KPIs_met_more_than_80'] = predictions

            # Display the DataFrame with predictions
            st.subheader('Predicted Results:')
            st.write(csv_df_org)

            # Visualize predictions based on a selected feature
            st.subheader('Visualize Predictions')

            # Select feature for visualization
            feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df_org.columns)

            # Plot the number of employees based on KPIs for the selected feature
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.countplot(x=feature_for_visualization, hue='KPIs_met_more_than_80', data=csv_df_org, palette='viridis')
            plt.title(f'Number of Employees based on KPIs - {feature_for_visualization}')
            plt.xlabel(feature_for_visualization)
            plt.ylabel('Number of Employees')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Debug information:")
            st.write(f"Input data shape: {csv_df_prepared.shape}")
            st.write(f"Input data columns: {csv_df_prepared.columns}")

else:
    st.error("Please ensure all required files are present and the model is loaded correctly.")
