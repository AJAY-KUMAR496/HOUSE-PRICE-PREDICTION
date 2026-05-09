import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('column_names.pkl')
transformer = joblib.load('transformer.pkl')

# Remove target column safely
if 'TARGET(PRICE_IN_LACS)' in columns:
    feature_columns = [col for col in columns if col != 'TARGET(PRICE_IN_LACS)']
else:
    feature_columns = columns

# Options
POSTED_BY_OPTIONS = ['Owner', 'Dealer', 'Builder']
BHK_RK_OPTIONS = ['BHK', 'RK']

# Page setup
st.set_page_config(page_title='House Price Predictor')

st.title(' House Price Prediction App')
st.markdown('Enter the property details to estimate the price.')

# Layout
col1, col2 = st.columns(2)

with col1:
    posted_by = st.selectbox('Posted By', POSTED_BY_OPTIONS)
    under_construction = st.selectbox('Under Construction', [0, 1])
    rera = st.selectbox('RERA Approved', [0, 1])
    bhk_no = st.number_input('BHK Number', min_value=1, max_value=20, value=2)
    bhk_or_rk = st.selectbox('BHK or RK', BHK_RK_OPTIONS)

with col2:
    square_ft = st.number_input('Square Footage', min_value=100.0, value=1000.0)
    ready_to_move = st.selectbox('Ready to Move', [0, 1])
    resale = st.selectbox('Resale Property', [0, 1])
    longitude = st.number_input('Longitude', value=12.97)
    latitude = st.number_input('Latitude', value=77.59)

# Prediction button
if st.button('Predict Price'):
    # Create input dataframe
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Numerical values
    input_data['UNDER_CONSTRUCTION'] = under_construction
    input_data['RERA'] = rera
    input_data['BHK_NO.'] = bhk_no
    input_data['SQUARE_FT'] = square_ft
    input_data['READY_TO_MOVE'] = ready_to_move
    input_data['RESALE'] = resale
    input_data['LONGITUDE'] = longitude
    input_data['LATITUDE'] = latitude

    # One-hot encoding mapping
    posted_col = f'POSTED_BY_{posted_by}'
    bhk_col = f'BHK_OR_RK_{bhk_or_rk}'

    if posted_col in input_data.columns:
        input_data[posted_col] = 1

    if bhk_col in input_data.columns:
        input_data[bhk_col] = 1

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction_transformed = model.predict(input_scaled)

    # Inverse transform logic
    dummy_input = np.zeros((1, 2))
    dummy_input[0, 1] = prediction_transformed[0]
    
    # Inverse transform
    actual_results = transformer.inverse_transform(dummy_input)
    final_price = float(actual_results[0, 1])

    # Show result in decimals
    st.success(f'### Estimated Price: ₹ {final_price:.2f}')
    st.balloons()
