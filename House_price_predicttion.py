import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Title
st.title("🏡 House Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # ============================
    # 1. Feature Engineering
    # ============================
    # Extract BHK from 'size'
    if 'size' in data.columns:
        data['BHK'] = data['size'].str.extract(r'(\d+)').astype(float)
        data.drop('size', axis=1, inplace=True)
        st.success("Extracted BHK from size.")
    else:
        st.warning("Column 'size' not found.")

    # Convert total_sqft
    if 'total_sqft' in data.columns:
        def convert_sqft_number(x):
            try:
                token = str(x).split("-")
                if len(token) == 2:
                    return (float(token[0]) + float(token[1])) / 2
                return float(x)
            except:
                return None
        data['total_sqft'] = data['total_sqft'].apply(convert_sqft_number)
    else:
        st.warning("Column 'total_sqft' not found.")

    # Simplify availability
    if 'availability' in data.columns:
        def simplify_availability(value):
            try:
                value = str(value).strip()
                return "Ready to Move" if "Ready To Move" in value else "18-Dec"
            except:
                return "Unknown"
        data['availability'] = data['availability'].apply(simplify_availability)

    # Drop 'society' if exists
    if 'society' in data.columns:
        data.drop('society', axis=1, inplace=True)

    # Handle rare locations
    if 'location' in data.columns:
        threshold = st.number_input("Group locations with count less than:", min_value=1, value=50)
        location_stats = data['location'].value_counts()
        locations_less_than_threshold = location_stats[location_stats < threshold].index
        data['location'] = data['location'].apply(lambda x: 'other' if x in locations_less_than_threshold else x)

    # Handle bath vs BHK rule
    if 'bath' in data.columns and 'BHK' in data.columns:
        data = data[~(data['bath'] > data['BHK'] + 1)]

    # Fill missing values
    if 'bath' in data.columns:
        data['bath'].fillna(data['bath'].median(), inplace=True)
    if 'balcony' in data.columns:
        data['balcony'].fillna(2, inplace=True)
    if 'BHK' in data.columns:
        data = data.dropna(subset=['BHK'])

    # Label Encoding for categorical columns
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    if categorical_cols:
        lb = LabelEncoder()
        for col in categorical_cols:
            data[col] = lb.fit_transform(data[col].astype(str))

    st.subheader("Cleaned Data")
    st.write(data.head())

    # ============================
    # 2. Feature Selection & Model Training
    # ============================
    st.subheader("Model Training")

    features = st.multiselect("Select Features", data.columns)
    target = st.selectbox("Select Target", data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Train-Test Split
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        st.success("Model Trained Successfully!")

        # ============================
        # 3. Prediction
        # ============================
        st.subheader("Make a Prediction")
        user_input = []
        for feature in features:
            user_input.append(st.number_input(f"Enter value for {feature}", value=0.0))

        if st.button("Predict Price"):
            prediction = model.predict([user_input])
            st.write(f"### Predicted Price: {prediction[0]:,.2f}")
