# # streamlit_farm_advisor.py
# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder

# # === Load Models ===
# rf_crop = joblib.load(r'Models\model_crop_suitability.pkl')
# rf_yield = joblib.load(r'Models\model_yield_forecast.pkl')
# rf_price = joblib.load(r'Models\model_price_forecast.pkl')
# rf_profit = joblib.load(r'Models\model_profit_estimation.pkl')
# rf_risk = joblib.load(r'Models\model_risk_assessment.pkl')

# #lode encoder
# le = joblib.load(r'Models/encoder_district.pkl') #universal encoder as used only once

# # === Load Encoders (saved during training) ===
# # le_district = joblib.load('Models\encoder_district.pkl')
# # le_soil = joblib.load('Models\encoder_soil_type.pkl')
# # le_water = joblib.load('Models\encoder_water_requirement.pkl')
# # le_season = joblib.load('Models\encoder_season.pkl')

# # === Load Sample Data (preprocessed base table for market prices) ===
# get_mandi_prices = pd.read_csv(r"C:\Users\athud\OneDrive\Desktop\MSc\2nd Sem\Mini Project\Sample_data.csv")

# # === Helper to encode user input ===
# def encode_inputs(location, soil_type, water_availability, season='Kharif', arrival_qty=1000, modal_price=4000):
#     district_enc = le.transform([location])[0]
#     soil_enc = le.transform([soil_type])[0]
#     water_enc = le.transform([water_availability])[0]
#     season_enc = le.transform([season])[0]

#     return [district_enc, soil_enc, water_enc, season_enc, arrival_qty, modal_price]

# # === Prediction functions ===
# def predict_crop_suitability(encoded_features):
#     prediction = rf_crop.predict([encoded_features])
#     return prediction

# def predict_yield(encoded_features):
#     prediction = rf_yield.predict([encoded_features])
#     return prediction[0]

# def predict_market_price(encoded_features):
#     prediction = rf_price.predict([encoded_features])
#     return prediction[0]

# def predict_profitability(encoded_features):
#     prediction = rf_profit.predict([encoded_features])
#     return prediction[0]

# def predict_risk(encoded_features):
#     prediction = rf_risk.predict([encoded_features])
#     return prediction[0]

# # === Streamlit App ===
# def main():
#     st.set_page_config(page_title="Farmer Crop Advisor", layout="wide")
#     st.title("🌾 Smart Crop Advisor for Farmers")

#     st.sidebar.header("📝 Enter Your Farm Details")

#     location = st.sidebar.selectbox("Select your location (District)", le.classes_)
#     farm_size = st.sidebar.number_input("Farm Size (Hectares)", min_value=0.1, max_value=100.0, value=1.0)
#     soil_type = st.sidebar.selectbox("Soil Type", le.classes_)
#     water_availability = st.sidebar.selectbox("Water Availability", le.classes_)
#     budget = st.sidebar.number_input("Investment Budget (₹)", min_value=1000, max_value=500000, value=50000)

#     if st.sidebar.button("Get Crop Recommendations"):
#         # Encode inputs
#         encoded_features = encode_inputs(location, soil_type, water_availability)

#         # Crop suitability prediction
#         suitable_crop = predict_crop_suitability(encoded_features)

#         # Decode predicted crop name (reverse LabelEncoder for commodity)
#         crop_name = le.inverse_transform(suitable_crop)[0] if hasattr(le, 'inverse_transform') else suitable_crop[0]

#         st.subheader("🌱 Recommended Crop for Your Field")
#         st.success(crop_name)

#         # Current Market Prices
#         st.subheader("💹 Current Market Prices (MSAMB)")
#         mandi_df = get_mandi_prices
#         st.dataframe(mandi_df)

#         # Detailed Report
#         st.subheader("📊 Estimated Yield, Price & Profitability")
#         predicted_yield = predict_yield(encoded_features) * farm_size
#         predicted_price = predict_market_price(encoded_features)
#         estimated_profit = predict_profitability(encoded_features)
#         estimated_risk = predict_risk(encoded_features)

#         report_data = pd.DataFrame({
#             'Crop': [crop_name],
#             'Predicted Yield (tons)': [round(predicted_yield, 2)],
#             'Predicted Market Price (₹/Qtl)': [round(predicted_price, 2)],
#             'Estimated Profit (₹)': [round(estimated_profit, 2)],
#             'Estimated Risk (Price Volatility)': [round(estimated_risk, 2)]
#         })

#         st.dataframe(report_data)

# if __name__ == '__main__':
#     main()

### Part 2 of the code


# streamlit_farm_advisor.py
import streamlit as st
import pandas as pd
import joblib
import os

# === Load Models ===
rf_crop = joblib.load(r'Models\model_crop_suitability.pkl')
rf_yield = joblib.load(r'Models\model_yield_forecast.pkl')
rf_price = joblib.load(r'Models\model_price_forecast.pkl')
rf_profit = joblib.load(r'Models\model_profit_estimation.pkl')
rf_risk = joblib.load(r'Models\model_risk_assessment.pkl')

# === Load Label Encoders ===
# Load individual encoders for each categorical feature
le_district = joblib.load(r'Models\encoder_district.pkl')
le_soil = joblib.load(r'Models\encoder_soil_type.pkl')
le_water = joblib.load(r'Models\encoder_water_requirement.pkl')
le_season = joblib.load(r'Models\encoder_season.pkl')
le_commodity = joblib.load(r'Models\encoder_commodity.pkl')  # For decoding crop predictions

# === Load Sample Data (preprocessed base table for market prices) ===
try:
    # Update the path to where your sample data is stored
    get_mandi_prices = pd.read_csv(r"C:\Users\athud\OneDrive\Desktop\MSc\2nd Sem\Mini Project\Sample_data.csv")
except FileNotFoundError:
    # Create dummy data if file not found
    st.warning("Sample price data file not found. Using dummy data.")
    get_mandi_prices = pd.DataFrame({
        'Crop': ['Rice', 'Wheat', 'Cotton'],
        'Price': [2000, 1800, 5500],
        'Market': ['Mandi A', 'Mandi B', 'Mandi C']
    })

# === Helper to encode user input ===
def encode_inputs(location, soil_type, water_availability, season='Kharif', arrival_qty=1000, modal_price=4000):
    # Use the correct encoder for each feature
    district_enc = le_district.transform([location])[0]
    soil_enc = le_soil.transform([soil_type])[0]
    water_enc = le_water.transform([water_availability])[0]
    season_enc = le_season.transform([season])[0]

    return [district_enc, soil_enc, water_enc, season_enc, arrival_qty, modal_price]

# === Prediction functions ===
def predict_crop_suitability(encoded_features):
    prediction = rf_crop.predict([encoded_features])
    return prediction

def predict_yield(encoded_features):
    prediction = rf_yield.predict([encoded_features])
    return prediction[0]

def predict_market_price(encoded_features):
    prediction = rf_price.predict([encoded_features])
    return prediction[0]

def predict_profitability(encoded_features):
    prediction = rf_profit.predict([encoded_features])
    return prediction[0]

def predict_risk(encoded_features):
    prediction = rf_risk.predict([encoded_features])
    return prediction[0]

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Farmer Crop Advisor", layout="wide")
    st.title("🌾 Smart Crop Advisor for Farmers")

    st.sidebar.header("📝 Enter Your Farm Details")

    # Use the appropriate encoder classes for each dropdown
    location = st.sidebar.selectbox("Select your location (District)", le_district.classes_)
    farm_size = st.sidebar.number_input("Farm Size (Hectares)", min_value=0.1, max_value=100.0, value=1.0)
    soil_type = st.sidebar.selectbox("Soil Type", le_soil.classes_)
    water_availability = st.sidebar.selectbox("Water Availability", le_water.classes_)
    season = st.sidebar.selectbox("Growing Season", le_season.classes_)
    budget = st.sidebar.number_input("Investment Budget (₹)", min_value=1000, max_value=500000, value=50000)

    if st.sidebar.button("Get Crop Recommendations"):
        # Encode inputs using the correct encoders
        encoded_features = encode_inputs(location, soil_type, water_availability, season)

        # Crop suitability prediction
        suitable_crop_encoded = predict_crop_suitability(encoded_features)
        
        # Decode the crop prediction using the commodity encoder
        crop_name = le_commodity.inverse_transform(suitable_crop_encoded)[0]

        st.subheader("🌱 Recommended Crop for Your Field")
        st.success(crop_name)

        # Current Market Prices
        st.subheader("💹 Current Market Prices (MSAMB)")
        mandi_df = get_mandi_prices.head()
        st.dataframe(mandi_df)

        # Detailed Report
        st.subheader("📊 Estimated Yield, Price & Profitability")
        predicted_yield = predict_yield(encoded_features) * farm_size
        predicted_price = predict_market_price(encoded_features)
        estimated_profit = predict_profitability(encoded_features)
        estimated_risk = predict_risk(encoded_features)

        report_data = pd.DataFrame({
            'Crop': [crop_name],
            'Predicted Yield (tons)': [round(predicted_yield, 2)],
            'Predicted Market Price (₹/Qtl)': [round(predicted_price, 2)],
            'Estimated Profit (₹)': [round(estimated_profit, 2)],
            'Estimated Risk (Price Volatility)': [round(estimated_risk, 2)]
        })

        st.dataframe(report_data)

if __name__ == '__main__':
    main()