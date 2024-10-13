import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ---------------------------
# Caching Functions for Efficiency
# ---------------------------

@st.cache_resource
def load_model():
    """Load the machine learning model and encoders."""
    model_path = os.path.join(os.path.dirname(__file__), 'model_kpi.65130701932')
    try:
        with open(model_path, 'rb') as file:
            model, department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder = pickle.load(file)
        return model, department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Load the employee dataset."""
    data_path = os.path.join(os.path.dirname(__file__), 'Uncleaned_employees_final_dataset.csv')
    try:
        df = pd.read_csv(data_path)
        df = df.drop('employee_id', axis=1)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}. Please ensure the data file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# ---------------------------
# Load Model and Data
# ---------------------------

model, department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder = load_model()
df = load_data()

# ---------------------------
# Streamlit App Layout
# ---------------------------

st.title('Employee KPIs App')

# Define tabs using Streamlit's tabs feature for better user experience
tab1, tab2, tab3 = st.tabs(['Predict KPIs', 'Visualize Data', 'Predict from CSV'])

# ---------------------------
# Tab 1: Predict KPIs
# ---------------------------

with tab1:
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

    # ---------------------------
    # One-Hot Encoding for Categorical Variables
    # ---------------------------

    categorical_features = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    user_input_encoded = pd.get_dummies(user_input, columns=categorical_features)

    # ---------------------------
    # Ensure All Expected Features Are Present
    # ---------------------------

    # Define all possible categories for each categorical feature
    departments = department_encoder.classes_
    regions = region_encoder.classes_
    educations = education_encoder.classes_
    genders = gender_encoder.classes_
    recruitment_channels = recruitment_channel_encoder.classes_

    # Create dummy variable columns for all categories
    for dep in departments:
        col_name = f'department_{dep}'
        if col_name not in user_input_encoded.columns:
            user_input_encoded[col_name] = 0

    for reg in regions:
        col_name = f'region_{reg}'
        if col_name not in user_input_encoded.columns:
            user_input_encoded[col_name] = 0

    for edu in educations:
        col_name = f'education_{edu}'
        if col_name not in user_input_encoded.columns:
            user_input_encoded[col_name] = 0

    for gen in genders:
        col_name = f'gender_{gen}'
        if col_name not in user_input_encoded.columns:
            user_input_encoded[col_name] = 0

    for rc in recruitment_channels:
        col_name = f'recruitment_channel_{rc}'
        if col_name not in user_input_encoded.columns:
            user_input_encoded[col_name] = 0

    # ---------------------------
    # Reorder Columns to Match Training Data
    # ---------------------------

    expected_columns = [
        'no_of_trainings',
        'age',
        'previous_year_rating',
        'length_of_service',
        'awards_won',
        'avg_training_score',
    ]

    # Add all dummy variables in the expected order
    expected_columns += [f'department_{dep}' for dep in departments]
    expected_columns += [f'region_{reg}' for reg in regions]
    expected_columns += [f'education_{edu}' for edu in educations]
    expected_columns += [f'gender_{gen}' for gen in genders]
    expected_columns += [f'recruitment_channel_{rc}' for rc in recruitment_channels]

    # Reorder the DataFrame columns and ensure all expected columns are present
    user_input_encoded = user_input_encoded.reindex(columns=expected_columns, fill_value=0)

    # ---------------------------
    # Debugging: Check Feature Alignment
    # ---------------------------

    st.write("Input Features:", user_input_encoded.columns.tolist())
    st.write("Number of Input Features:", user_input_encoded.shape[1])

    # ---------------------------
    # Prediction
    # ---------------------------

    try:
        prediction = model.predict(user_input_encoded)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Display Result
    st.subheader('Prediction Result:')
    st.write('KPIs_met_more_than_80:', prediction[0])

# ---------------------------
# Tab 2: Visualize Data
# ---------------------------

with tab2:
    st.header('Visualize Data')

    # Select condition feature
    condition_feature = st.selectbox('Select Condition Feature:', df.columns)

    # Set default condition values
    default_condition_values = ['Select All'] + df[condition_feature].unique().tolist()

    # Select condition values
    condition_values = st.multiselect('Select Condition Values:', default_condition_values, default=['Select All'])

    # Handle 'Select All' choice
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    if condition_values:
        # Filter DataFrame based on selected condition
        filtered_df = df[df[condition_feature].isin(condition_values)]

        # Plot the number of employees based on KPIs
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=condition_feature, hue='KPIs_met_more_than_80', data=filtered_df, palette='viridis')
        plt.title('Number of Employees based on KPIs')
        plt.xlabel(condition_feature)
        plt.ylabel('Number of Employees')
        st.pyplot(fig)
    else:
        st.warning("Please select at least one condition value to visualize the data.")

# ---------------------------
# Tab 3: Predict from CSV
# ---------------------------

with tab3:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV file
            csv_df_org = pd.read_csv(uploaded_file)
            csv_df_org = csv_df_org.dropna()
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        # Check if 'employee_id' exists and drop it
        if 'employee_id' in csv_df_org.columns:
            csv_df = csv_df_org.drop('employee_id', axis=1)
        else:
            csv_df = csv_df_org.copy()

        # ---------------------------
        # One-Hot Encoding for Categorical Variables
        # ---------------------------

        csv_df_encoded = pd.get_dummies(csv_df, columns=categorical_features)

        # Create dummy variable columns for all categories
        for dep in departments:
            col_name = f'department_{dep}'
            if col_name not in csv_df_encoded.columns:
                csv_df_encoded[col_name] = 0

        for reg in regions:
            col_name = f'region_{reg}'
            if col_name not in csv_df_encoded.columns:
                csv_df_encoded[col_name] = 0

        for edu in educations:
            col_name = f'education_{edu}'
            if col_name not
