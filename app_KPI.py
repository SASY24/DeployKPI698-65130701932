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

# Streamlit App
st.title('Employee KPIs App')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Predict KPIs', 'Visualize Data', 'Predict from CSV']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

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

    # Categorical Data Encoding
    user_input['department'] = department_encoder.transform(user_input['department'])
    user_input['region'] = region_encoder.transform(user_input['region'])
    user_input['education'] = education_encoder.transform(user_input['education'])
    user_input['gender'] = gender_encoder.transform(user_input['gender'])
    user_input['recruitment_channel'] = recruitment_channel_encoder.transform(user_input['recruitment_channel'])

    # Predicting
    prediction = model.predict(user_input)

    # Display Result
    st.subheader('Prediction Result:')
    st.write('KPIs_met_more_than_80:', prediction[0])

# Tab 2: Visualize Data
elif st.session_state.tab_selected == 1 and df is not None:
    st.header('Visualize Data')

    # Select condition feature
    condition_feature = st.selectbox('Select Condition Feature:', df.columns)

    # Set default condition values
    default_condition_values = ['Select All'] + df[condition_feature].unique().tolist()

    # Select condition values
    condition_values = st.multiselect('Select Condition Values:', default_condition_values)

    # Handle 'Select All' choice
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    if len(condition_values) > 0:
        # Filter DataFrame based on selected condition
        filtered_df = df[df[condition_feature].isin(condition_values)]

        # Plot the number of employees based on KPIs
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=condition_feature, hue='KPIs_met_more_than_80', data=filtered_df, palette='viridis')
        plt.title('Number of Employees based on KPIs')
        plt.xlabel(condition_feature)
        plt.ylabel('Number of Employees')
        st.pyplot(fig)

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
        
        # Check if all required columns are present
        required_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']
        missing_columns = set(required_columns) - set(csv_df.columns)
        
        if missing_columns:
            st.error(f"The following required columns are missing from the CSV: {', '.join(missing_columns)}")
        else:
            # Categorical Data Encoding
            csv_df['department'] = department_encoder.transform(csv_df['department'])
            csv_df['region'] = region_encoder.transform(csv_df['region'])
            csv_df['education'] = education_encoder.transform(csv_df['education'])
            csv_df['gender'] = gender_encoder.transform(csv_df['gender'])
            csv_df['recruitment_channel'] = recruitment_channel_encoder.transform(csv_df['recruitment_channel'])

            # Predicting
            predictions = model.predict(csv_df)

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
else:
    st.error("Please ensure all required files are present and the model is loaded correctly.")
