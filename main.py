# streamlit_farm_advisor.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import calendar
import sklearn

# Set page configuration
st.set_page_config(
    page_title="Smart Farm Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388e3c;
        margin-top: 2rem;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffa000;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        color: #333; /* A dark gray for good contrast */
        /* Or simply don't specify a color to use the browser default */
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='main-header'>🌾 Smart Crop Advisor for Farmers</h1>", unsafe_allow_html=True)

# Create tabs for organization
tabs = st.tabs(["🏠 Home", "🌱 Crop Recommendation", "💰 Economic Analysis", "🌦️ Weather & Calendar"])

# === Load Models ===
@st.cache_resource
def load_models():
    models = {
        'crop': joblib.load(r'Models\model_crop_suitability.pkl'),
        'yield': joblib.load(r'Models\model_yield_forecast.pkl'),
        'price': joblib.load(r'Models\model_price_forecast.pkl'),
        'profit': joblib.load(r'Models\model_profit_estimation.pkl'),
        'risk': joblib.load(r'Models\model_risk_assessment.pkl')
    }
    return models

# === Load Label Encoders ===
@st.cache_resource
def load_encoders():
    encoders = {
        'district': joblib.load(r'Models\encoder_district.pkl'),
        'soil': joblib.load(r'Models\encoder_soil_type.pkl'),
        'water': joblib.load(r'Models\encoder_water_requirement.pkl'),
        'season': joblib.load(r'Models\encoder_season.pkl'),
        'commodity': joblib.load(r'Models\encoder_commodity.pkl')
    }
    return encoders

# === Load Sample Data ===
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv(r"C:\Users\athud\OneDrive\Desktop\MSc\2nd Sem\Mini Project\Sample_data.csv")
        return df
    except FileNotFoundError:
        st.warning("Sample data file not found. Using dummy data.")
        # Create dummy data
        crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']
        markets = ['Mandi A', 'Mandi B', 'Mandi C']
        
        dummy_data = []
        for crop in crops:
            for market in markets:
                dummy_data.append({
                    'Crop': crop,
                    'Market': market,
                    'Price': np.random.randint(1500, 6000),
                    'Arrival_Qty': np.random.randint(100, 5000),
                    'Date': datetime.date.today() - datetime.timedelta(days=np.random.randint(0, 30))
                })
        
        return pd.DataFrame(dummy_data)

# === Load Weather Data ===
@st.cache_data
def load_weather_data():
    # In a real app, you would fetch this from a weather API
    # For now, we'll create some dummy weather data
    districts = ['District A', 'District B', 'District C', 'District D']
    
    today = datetime.date.today()
    weather_data = []
    
    for district in districts:
        for day in range(7):
            date = today + datetime.timedelta(days=day)
            weather_data.append({
                'District': district,
                'Date': date,
                'Min_Temp': np.random.randint(18, 24),
                'Max_Temp': np.random.randint(28, 38),
                'Humidity': np.random.randint(50, 95),
                'Rainfall_mm': max(0, np.random.normal(5, 10)),
                'Condition': np.random.choice(['Sunny', 'Partly Cloudy', 'Cloudy', 'Light Rain', 'Heavy Rain'], p=[0.5, 0.2, 0.1, 0.15, 0.05])
            })
    
    return pd.DataFrame(weather_data)

# === Helper Functions ===
def encode_inputs(location, soil_type, water_availability, season='Kharif', arrival_qty=1000, modal_price=4000):
    encoders = load_encoders()
    
    district_enc = encoders['district'].transform([location])[0]
    soil_enc = encoders['soil'].transform([soil_type])[0]
    water_enc = encoders['water'].transform([water_availability])[0]
    season_enc = encoders['season'].transform([season])[0]

    return [district_enc, soil_enc, water_enc, season_enc, arrival_qty, modal_price]

def predict_crop_suitability(encoded_features):
    models = load_models()
    prediction = models['crop'].predict([encoded_features])
    return prediction

def predict_yield(encoded_features):
    models = load_models()
    prediction = models['yield'].predict([encoded_features])
    return prediction[0]

def predict_market_price(encoded_features):
    models = load_models()
    prediction = models['price'].predict([encoded_features])
    return prediction[0]

def predict_profitability(encoded_features):
    models = load_models()
    prediction = models['profit'].predict([encoded_features])
    return prediction[0]

def predict_risk(encoded_features):
    models = load_models()
    prediction = models['risk'].predict([encoded_features])
    return prediction[0]

def get_crop_recommendations(location, soil_type, water_availability, season, additional_filters=None):
    # Get top n crop recommendations instead of just one
    encoders = load_encoders()
    models = load_models()
    
    encoded_features = encode_inputs(location, soil_type, water_availability, season)
    
    # For a real app, you would use model.predict_proba to get top n crops
    # Here we'll simulate by returning a few options
    crop_indices = [0, 1, 2]  # In a real app, these would be the indices of top crops
    
    recommendations = []
    for idx in crop_indices:
        # Decode crop name
        crop_name = encoders['commodity'].inverse_transform([idx])[0]
        
        # Get predictions for this crop
        encoded_features_copy = encoded_features.copy()
        encoded_features_copy[3] = idx  # Replace crop index
        
        predicted_yield = predict_yield(encoded_features_copy)
        predicted_price = predict_market_price(encoded_features_copy)
        estimated_profit = predict_profitability(encoded_features_copy)
        estimated_risk = predict_risk(encoded_features_copy)
        
        recommendations.append({
            'Crop': crop_name,
            'Suitability_Score': np.random.randint(70, 100),  # In a real app, get this from the model
            'Predicted_Yield': predicted_yield, 
            'Predicted_Price': predicted_price,
            'Estimated_Profit': estimated_profit,
            'Risk_Score': estimated_risk
        })
    
    return pd.DataFrame(recommendations).sort_values('Suitability_Score', ascending=False)

def generate_farming_calendar(crop, season):
    # This would be based on crop-specific data in a real app
    today = datetime.date.today()
    current_month = today.month
    
    # Define activities for the crop
    activities = [
        "Land Preparation", 
        "Sowing/Planting", 
        "Fertilizer Application", 
        "Irrigation", 
        "Pest Management", 
        "Weed Management", 
        "Harvesting", 
        "Post-harvest Processing"
    ]
    
    # Create a calendar with start and end dates for each activity
    calendar_data = []
    month_offset = 0
    
    for i, activity in enumerate(activities):
        start_date = today + datetime.timedelta(days=i*15 + np.random.randint(0, 5))
        end_date = start_date + datetime.timedelta(days=np.random.randint(5, 15))
        
        calendar_data.append({
            'Activity': activity,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Duration_Days': (end_date - start_date).days,
            'Status': 'Upcoming' if start_date > today else ('In Progress' if today <= end_date else 'Completed')
        })
    
    return pd.DataFrame(calendar_data)

def calculate_roi(investment, profit):
    return (profit / investment) * 100

def format_currency(amount):
    return f"₹{amount:,.2f}"

# === Sidebar Form ===
with st.sidebar:
    st.header("📝 Farm Details")

    encoders = load_encoders()
    # Farm details form
    # st.write(encoders['district'].classes_)
    location = st.selectbox("Select location (District)", encoders['district'].classes_)
    farm_size = st.number_input("Farm Size (Hectares)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        soil_type = st.selectbox("Soil Type", encoders['soil'].classes_)
    with col2:
        water_availability = st.selectbox("Water Availability", encoders['water'].classes_)
    
    season = st.selectbox("Growing Season", encoders['season'].classes_)
    budget = st.number_input("Investment Budget (₹)", min_value=1000, max_value=1000000, value=50000, step=1000)
    
    # Advanced filters (expandable)
    with st.expander("Advanced Filters"):
        max_risk = st.slider("Maximum Risk Tolerance", 0, 100, 50)
        min_profit = st.slider("Minimum Profit Expectation (₹)", 0, 100000, 10000, step=1000)
        preferences = st.multiselect("Preferences", 
                                      ["Organic Farming", "Low Water Usage", "Disease Resistant", "Export Quality", "Quick Harvest"])
    
    analyze_button = st.button("Analyze & Recommend Crops")

# === Home Tab ===
with tabs[0]:
    st.markdown("""
    <div class='info-box'>
        <h3>Welcome to Smart Crop Advisor!</h3>
        <p>This application helps farmers make data-driven decisions by recommending suitable crops,
        providing yield and price forecasts, and estimating profitability.</p>
        <p>Use the sidebar to enter your farm details and get personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Current Date", value=datetime.date.today().strftime("%d %b %Y"))
    
    with col2:
        # Example metric that would be dynamic in a real app
        weather_df = load_weather_data()

        location_map = {
            'District A': 0,
            'District B': 1,
            'District C': 2
        }
        
        # Reverse mapping (number → text)
        reverse_location_map = {v: k for k, v in location_map.items()}

        # Decode the location from encoded value to text
        location_text = reverse_location_map[location]   # Now location_text = 'District A'

        # st.dataframe(weather_df) ############ tring this out to see if empty
        weather_df_today = weather_df[weather_df['Date'] == datetime.date.today()]
        if not weather_df_today.empty and location_text in weather_df_today['District'].values:
            weather_today = weather_df_today[weather_df_today['District'] == location_text].iloc[0]
            st.metric(label=f"Temperature ({location_text})", 
                     value=f"{weather_today['Max_Temp']}°C", 
                     delta=f"{weather_today['Max_Temp'] - weather_today['Min_Temp']}°C range")
        else:
            st.metric(label="Temperature", value="30°C", delta="10°C range")
    
    with col3:
        # Example rainfall metric
        if not weather_df_today.empty and location_text in weather_df_today['District'].values:
            weather_today = weather_df_today[weather_df_today['District'] == location_text].iloc[0]
            st.metric(label="Rainfall", 
                     value=f"{weather_today['Rainfall_mm']:.1f} mm", 
                     delta=f"{weather_today['Humidity']}% humidity")
        else:
            st.metric(label="Rainfall", value="2.5 mm", delta="65% humidity")
    
    # Quick access cards
    st.subheader("Quick Access")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        st.markdown("""
        <div style="background-color:#e8f5e9; padding:15px; border-radius:10px; height:auto; color:#000; min-height: 80px;">
            <h3 style="color:#2e7d32; margin-top: 0; margin-bottom: 0.5rem;">🌱 Crop Recommendations</h3>
            <p style="margin-top: 0; margin-bottom: 0; line-height: 1.4;">Get personalized crop recommendations based on your farm's conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with quick_col2:
        st.markdown("""
        <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; height:auto; color:#000; min-height: 80px;">
            <h3 style="color:#1565c0; margin-top: 0; margin-bottom: 0.5rem;">💰 Economic Analysis</h3>
            <p style="margin-top: 0; margin-bottom: 0; line-height: 1.4;">Get personalized crop recommendations based on your farm's conditions.</p>
        </div>
        """, unsafe_allow_html=True)

    with quick_col3:
        st.markdown("""
        <div style="background-color:#fff8e1; padding:15px; border-radius:10px; height:auto; color:#000; min-height: 80px;">
            <h3 style="color:#ff8f00; margin-top: 0; margin-bottom: 0.5rem;">🌦️ Weather Forecast</h3>
            <p style="margin-top: 0; margin-bottom: 0; line-height: 1.4;">Get personalized crop recommendations based on your farm's conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent trends section
    st.markdown("<h2 class='sub-header'>Market Trends</h2>", unsafe_allow_html=True)
    mandi_df = load_sample_data()
    
    # Grouping data for visualization
    if 'Date' in mandi_df.columns and 'Price' in mandi_df.columns and 'Crop' in mandi_df.columns:
        try:
            mandi_df['Date'] = pd.to_datetime(mandi_df['Date'])
            price_trends = mandi_df.groupby(['Crop', 'Date'])['Price'].mean().reset_index()
            
            # Create a line chart of price trends
            fig = px.line(price_trends, x='Date', y='Price', color='Crop', 
                          title='Price Trends Over Time',
                          labels={'Price': 'Price (₹/Qtl)', 'Date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Could not generate price trends chart with the available data.")
    else:
        st.warning("Market data does not contain required columns for visualization.")

# === Crop Recommendation Tab ===
with tabs[1]:
    if analyze_button or st.session_state.get('analyzed', False):
        st.session_state['analyzed'] = True
        
        st.markdown("<h2 class='sub-header'>Crop Recommendations for Your Farm</h2>", unsafe_allow_html=True)
        
        # Get recommendations (multiple crops instead of just one)
        recommendations = get_crop_recommendations(
            location, soil_type, water_availability, season,
            additional_filters={'max_risk': max_risk, 'min_profit': min_profit, 'preferences': preferences}
        )
        
        # Display recommendations in cards
        for i, (index, row) in enumerate(recommendations.iterrows()):
            with st.container():
                st.markdown(f"""
                <div style="background-color:#f1f8e9; padding:15px; border-radius:10px; margin-bottom:20px; color:#000 ;
                            border-left: 5px solid {'#4caf50' if i==0 else '#8bc34a'}">
                    <h3 style="color:#2e7d32">{"🥇 " if i==0 else "🌱 "}{row['Crop']}</h3>
                    <p><b>Suitability Score:</b> {row['Suitability_Score']}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Predicted Yield", value=f"{row['Predicted_Yield']:.2f} ton/ha")
                with col2:
                    st.metric(label="Market Price", value=f"₹{row['Predicted_Price']:.2f}/qtl")
                with col3:
                    st.metric(label="Est. Profit", value=f"₹{row['Estimated_Profit']:,.2f}")
                with col4:
                    # Create a risk indicator
                    risk_level = "Low" if row['Risk_Score'] < 30 else ("Medium" if row['Risk_Score'] < 70 else "High")
                    risk_color = "green" if risk_level == "Low" else ("orange" if risk_level == "Medium" else "red")
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <p style="font-size:0.8em;margin-bottom:0;">Risk Level</p>
                        <p style="color:{risk_color};font-weight:bold;font-size:1.2em;margin-top:0;">{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add crop details expandable section
        for i, (index, row) in enumerate(recommendations.iterrows()):
            with st.expander(f"🔍 More details about {row['Crop']}"):
                st.write(f"### {row['Crop']} Cultivation Details")
                
                # Create tabs for different information categories
                crop_tabs = st.tabs(["General Info", "Cultivation Practices", "Disease Management", "Market Info"])
                
                with crop_tabs[0]:
                    st.write(f"**Scientific Name:** [Example Scientific Name]")
                    st.write(f"**Growing Season:** {season}")
                    st.write(f"**Crop Duration:** [Example: 90-120 days]")
                    st.write(f"**Optimal Temperature:** [Example: 25-30°C]")
                    st.write(f"**Water Requirement:** {water_availability}")
                
                with crop_tabs[1]:
                    st.write("**Seed Rate:** [Example: 5-10 kg/ha]")
                    st.write("**Spacing:** [Example: 30 cm × 10 cm]")
                    st.write("**Fertilizer Requirement:**")
                    st.write("- Nitrogen: [Example: 100 kg/ha]")
                    st.write("- Phosphorus: [Example: 50 kg/ha]")
                    st.write("- Potassium: [Example: 50 kg/ha]")
                
                with crop_tabs[2]:
                    st.write("**Common Diseases:**")
                    st.write("1. [Example Disease 1]")
                    st.write("2. [Example Disease 2]")
                    st.write("**Prevention Measures:**")
                    st.write("- [Example Prevention Measure 1]")
                    st.write("- [Example Prevention Measure 2]")
                
                with crop_tabs[3]:
                    st.write(f"**Current Market Price:** ₹{row['Predicted_Price']:.2f}/qtl")
                    st.write(f"**Price Trend:** [Increasing/Stable/Decreasing]")
                    st.write(f"**Major Markets:** [Example: Market 1, Market 2]")
                    
                    # Sample price trend chart
                    dates = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=180), periods=6, freq='M')
                    prices = np.random.normal(row['Predicted_Price'], row['Predicted_Price'] * 0.1, size=6)
                    price_df = pd.DataFrame({'Date': dates, 'Price': prices})
                    
                    fig = px.line(price_df, x='Date', y='Price', title=f'{row["Crop"]} Price Trend (6 months)')
                    fig.update_layout(yaxis_title='Price (₹/qtl)')
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class='info-box'>
            <h3>Get Your Personalized Crop Recommendations</h3>
            <p>Please fill in your farm details in the sidebar and click "Analyze & Recommend Crops" to get started.</p>
            <p>Our ML models will analyze your inputs and suggest the most suitable crops for your farm conditions.</p>
        </div>
        """, unsafe_allow_html=True)

# === Economic Analysis Tab ===
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Economic Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get('analyzed', False):
        st.info("Please enter your farm details and click 'Analyze & Recommend Crops' in the sidebar first.")
    else:
        # Get the recommended crops
        recommendations = get_crop_recommendations(
            location, soil_type, water_availability, season,
            additional_filters={'max_risk': max_risk, 'min_profit': min_profit, 'preferences': preferences}
        )
        
        # Cost and profit analysis
        st.subheader("Cost & Profit Analysis")
        
        # Allow user to adjust cost parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            seed_cost_factor = st.slider("Seed Cost Factor", 0.5, 1.5, 1.0, 0.1)
        with col2:
            fertilizer_cost_factor = st.slider("Fertilizer Cost Factor", 0.5, 1.5, 1.0, 0.1)
        with col3:
            labor_cost_factor = st.slider("Labor Cost Factor", 0.5, 1.5, 1.0, 0.1)
        
        # Create cost analysis table
        cost_breakdown = []
        
        for i, (index, row) in enumerate(recommendations.iterrows()):
            # Generate realistic cost breakdown
            base_cost = row['Estimated_Profit'] * 0.5  # Assume profit is about 50% of revenue
            seed_cost = base_cost * 0.15 * seed_cost_factor
            fertilizer_cost = base_cost * 0.25 * fertilizer_cost_factor
            pesticide_cost = base_cost * 0.10
            irrigation_cost = base_cost * 0.15
            labor_cost = base_cost * 0.25 * labor_cost_factor
            other_cost = base_cost * 0.10
            
            total_cost = seed_cost + fertilizer_cost + pesticide_cost + irrigation_cost + labor_cost + other_cost
            
            # Calculate revenue
            revenue = row['Predicted_Yield'] * farm_size * row['Predicted_Price'] * 10  # Convert qtl to kg
            
            # Adjusted profit
            adjusted_profit = revenue - total_cost
            
            # ROI
            roi = calculate_roi(total_cost, adjusted_profit)
            
            cost_breakdown.append({
                'Crop': row['Crop'],
                'Seed Cost': seed_cost,
                'Fertilizer Cost': fertilizer_cost,
                'Pesticide Cost': pesticide_cost,
                'Irrigation Cost': irrigation_cost,
                'Labor Cost': labor_cost,
                'Other Costs': other_cost,
                'Total Cost': total_cost,
                'Revenue': revenue,
                'Profit': adjusted_profit,
                'ROI (%)': roi
            })
        
        cost_df = pd.DataFrame(cost_breakdown)
        
        # Display cost breakdown
        st.subheader("Cost Breakdown")
        formatted_cost_df = cost_df.copy()
        for col in cost_df.columns:
            if col not in ['Crop', 'ROI (%)']:
                formatted_cost_df[col] = formatted_cost_df[col].apply(format_currency)
        
        st.dataframe(formatted_cost_df, use_container_width=True)
        
        # Create visualizations
        st.subheader("Cost & Profit Visualization")
        
        # 1. Cost breakdown pie chart
        selected_crop = st.selectbox("Select Crop for Cost Analysis", cost_df['Crop'].unique())
        selected_crop_costs = cost_df[cost_df['Crop'] == selected_crop].iloc[0]
        
        cost_categories = ['Seed Cost', 'Fertilizer Cost', 'Pesticide Cost', 'Irrigation Cost', 'Labor Cost', 'Other Costs']
        cost_values = [selected_crop_costs[cat] for cat in cost_categories]
        
        fig1 = px.pie(
            values=cost_values,
            names=cost_categories,
            title=f"Cost Breakdown for {selected_crop}",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Profit comparison bar chart
        profit_comparison = cost_df[['Crop', 'Total Cost', 'Revenue', 'Profit']]
        profit_comparison_melted = pd.melt(
            profit_comparison, 
            id_vars=['Crop'],
            value_vars=['Total Cost', 'Revenue', 'Profit'],
            var_name='Category',
            value_name='Amount'
        )
        
        fig2 = px.bar(
            profit_comparison_melted,
            x='Crop',
            y='Amount',
            color='Category',
            title="Cost, Revenue & Profit Comparison",
            labels={'Amount': 'Amount (₹)', 'Crop': 'Crop'},
            barmode='group',
            color_discrete_sequence=['#f44336', '#4caf50', '#2196f3']
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # ROI comparison
        roi_comparison = cost_df[['Crop', 'ROI (%)']]
        fig3 = px.bar(
            roi_comparison,
            x='Crop',
            y='ROI (%)',
            title="Return on Investment (ROI) Comparison",
            labels={'ROI (%)': 'ROI (%)', 'Crop': 'Crop'},
            color='ROI (%)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Breakeven analysis
        st.subheader("Breakeven Analysis")
        
        # Allow user to select a crop for breakeven analysis
        crop_for_breakeven = st.selectbox("Select Crop for Breakeven Analysis", cost_df['Crop'].unique(), key="breakeven_crop")
        crop_data = cost_df[cost_df['Crop'] == crop_for_breakeven].iloc[0]
        
        # Calculate breakeven yield
        crop_price_per_kg = recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicted_Price'].iloc[0] / 100  # Convert qtl to kg
        breakeven_yield = crop_data['Total Cost'] / (crop_price_per_kg * farm_size)
        
        # Display breakeven metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Breakeven Yield", f"{breakeven_yield:.2f} kg/ha")
        with col2:
            st.metric("Current Yield Estimate", f"{recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicted_Yield'].iloc[0] * 1000:.2f} kg/ha")
        with col3:
            buffer_percentage = ((recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicted_Yield'].iloc[0] * 1000 - breakeven_yield) / breakeven_yield) * 100
            st.metric("Yield Buffer", f"{buffer_percentage:.2f}%")
        
        # Create breakeven chart
        price_variations = np.linspace(crop_price_per_kg * 0.5, crop_price_per_kg * 1.5, 100)
        breakeven_yields = []
        
        for price in price_variations:
            breakeven_yields.append(crop_data['Total Cost'] / (price * farm_size))
        
        breakeven_df = pd.DataFrame({
            'Price (₹/kg)': price_variations,
            'Breakeven Yield (kg/ha)': breakeven_yields
        })
        
        fig4 = px.line(
            breakeven_df,
            x='Price (₹/kg)',
            y='Breakeven Yield (kg/ha)',
            title=f"Breakeven Analysis for {crop_for_breakeven}",
            labels={'Price (₹/kg)': 'Price (₹/kg)', 'Breakeven Yield (kg/ha)': 'Breakeven Yield (kg/ha)'}
        )

# ======================== Buffer code ================================     
        # # Add a point for current price and yield
        # current_price = crop_price_per_kg
        # current_yield = recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicte
                                                                                       
        
        ## Add a point for current price and yield
        # current_price = crop_price_per_kg
        # current_breakeven_yield = crop_data['Total Cost'] / (current_price * farm_size)

        # fig4.add_scatter(
        #     x=[current_price],
        #     y=[current_breakeven_yield],
        #     mode='markers',
        #     marker=dict(size=10, color='red', symbol='x'),
        #     name='Current Price Point'
        # )

        # st.plotly_chart(fig4, use_container_width=True)

# ======================== Buffer code ================================
        # Add a point for current price and yield
        current_price = crop_price_per_kg
        current_breakeven_yield = crop_data['Total Cost'] / (current_price * farm_size)

        fig4.add_scatter(
            x=[current_price],
            y=[current_breakeven_yield],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Current Price Point'
        )

        # Add a horizontal line for the current yield estimate
        current_yield_estimate = recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicted_Yield'].iloc[0] * 1000

        fig4.add_shape(
            type="line",
            x0=price_variations.min(),
            y0=current_yield_estimate,
            x1=price_variations.max(),
            y1=current_yield_estimate,
            line=dict(color="green", width=2, dash="dash"),
        )

        fig4.add_annotation(
            x=price_variations.min(),
            y=current_yield_estimate,
            text="Current Yield Estimate",
            showarrow=False,
            xshift=-100,
            yshift=10,
            font=dict(color="green")
        )

        st.plotly_chart(fig4, use_container_width=True)

        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")

        # Create sensitivity analysis for price and yield changes
        sensitivity_price_changes = [-30, -20, -10, 0, 10, 20, 30]
        sensitivity_yield_changes = [-30, -20, -10, 0, 10, 20, 30]

        base_price = crop_price_per_kg
        base_yield = recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicted_Yield'].iloc[0] * 1000

        sensitivity_data = []

        for price_change in sensitivity_price_changes:
            for yield_change in sensitivity_yield_changes:
                adjusted_price = base_price * (1 + price_change/100)
                adjusted_yield = base_yield * (1 + yield_change/100)
                
                revenue = adjusted_yield * adjusted_price * farm_size
                profit = revenue - crop_data['Total Cost']
                roi = (profit / crop_data['Total Cost']) * 100
                
                sensitivity_data.append({
                    'Price_Change': price_change,
                    'Yield_Change': yield_change,
                    'Profit': profit,
                    'ROI': roi
                })

        sensitivity_df = pd.DataFrame(sensitivity_data)

        # Create heatmap
        profit_pivot = sensitivity_df.pivot(
            index='Yield_Change', 
            columns='Price_Change', 
            values='Profit'
        )

        fig5 = px.imshow(
            profit_pivot,
            labels=dict(x="Price Change (%)", y="Yield Change (%)", color="Profit (₹)"),
            x=sensitivity_price_changes,
            y=sensitivity_yield_changes,
            title=f"Profit Sensitivity Analysis for {crop_for_breakeven}",
            color_continuous_scale="RdYlGn"
        )

        st.plotly_chart(fig5, use_container_width=True)

        # Add risk analysis
        st.subheader("Risk Analysis")

        # Display risk metrics
        risk_score = recommendations[recommendations['Crop'] == crop_for_breakeven]['Risk_Score'].iloc[0]
        risk_level = "Low" if risk_score < 30 else ("Medium" if risk_score < 70 else "High")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Risk Score", f"{risk_score:.1f}/100")
        with col2:
            st.metric("Risk Level", risk_level)
        with col3:
            st.metric("Profit Variability", f"±{np.random.randint(10, 30)}%")

        # Risk breakdown
        risk_factors = [
            "Weather Risk", 
            "Pest & Disease Risk", 
            "Market Price Risk", 
            "Input Cost Risk", 
            "Policy Change Risk"
        ]
        risk_values = np.random.randint(10, 100, size=len(risk_factors))

        risk_df = pd.DataFrame({
            'Risk_Factor': risk_factors,
            'Risk_Score': risk_values
        })

        fig6 = px.bar(
            risk_df,
            x='Risk_Factor',
            y='Risk_Score',
            title=f"Risk Breakdown for {crop_for_breakeven}",
            labels={'Risk_Score': 'Risk Score (0-100)', 'Risk_Factor': 'Risk Factor'},
            color='Risk_Score',
            color_continuous_scale='Reds'
        )

        st.plotly_chart(fig6, use_container_width=True)

# === Weather & Calendar Tab ===
with tabs[3]:
    st.markdown("<h2 class='sub-header'>Weather Forecast & Farming Calendar</h2>", unsafe_allow_html=True)
    
    # Create two columns for weather and calendar
    weather_col, calendar_col = st.columns([1, 1])
    
    with weather_col:
        st.subheader("🌦️ Weather Forecast")
        
        # Load weather data
        weather_df = load_weather_data()
        # st.dataframe(weather_df) # to see if empty
        today_weather = None

        #Decoding the location data manualy 
        location_map = {
            'District A': 0,
            'District B': 1,
            'District C': 2
        }
        
        # Reverse mapping (number → text)
        reverse_location_map = {v: k for k, v in location_map.items()}

        # Decode the location from encoded value to text
        location_text = reverse_location_map[location]   # Now location_text = 'District A'


        # Filter weather data for selected location
        if location_text in weather_df['District'].values:
            location_weather = weather_df[weather_df['District'] == location_text]
            
            # Current weather
            today_weather = location_weather[location_weather['Date'] == datetime.date.today()]
            
            if not today_weather.empty:
                today_weather = today_weather.iloc[0]
                
                # Display current weather in a nice box
                st.markdown(f"""
                <div style="background-color:#e3f2fd; padding:15px; color:#000; border-radius:10px; margin-bottom:20px;">
                    <h3>Current Weather in {location_text}</h3>
                    <p><strong>Date:</strong> {today_weather['Date'].strftime('%d %b %Y')}</p>
                    <p><strong>Temperature:</strong> {today_weather['Min_Temp']}°C - {today_weather['Max_Temp']}°C</p>
                    <p><strong>Humidity:</strong> {today_weather['Humidity']}%</p>
                    <p><strong>Rainfall:</strong> {today_weather['Rainfall_mm']:.1f} mm</p>
                    <p><strong>Condition:</strong> {today_weather['Condition']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 7-day forecast
            st.subheader("7-Day Forecast")
            
            # Create a forecast table
            forecast_df = location_weather.sort_values('Date') ####### might come empty
            
            # Add weather icons based on condition
            weather_icons = {
                'Sunny': '☀️',
                'Partly Cloudy': '⛅',
                'Cloudy': '☁️',
                'Light Rain': '🌦️',
                'Heavy Rain': '🌧️'
            }
            
            forecast_df['Icon'] = forecast_df['Condition'].map(lambda x: weather_icons.get(x, '❓'))
            forecast_df['Day'] = forecast_df['Date'].map(lambda x: x.strftime('%a'))
            
            # Display compact forecast
            cols = st.columns(min(7, len(forecast_df)))
            for i, (_, row) in enumerate(forecast_df.iterrows()):
                if i < len(cols):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="text-align:center; padding:10px;">
                            <p style="font-size:1.2em; margin-bottom:5px;">{row['Day']}</p>
                            <p style="font-size:2em; margin:0;">{row['Icon']}</p>
                            <p style="margin-top:5px;">{row['Min_Temp']}° | {row['Max_Temp']}°</p>
                            <p style="font-size:0.8em; margin:0;">{row['Rainfall_mm']:.1f} mm</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Weather trends
            st.subheader("Temperature & Rainfall Trends")
            
            # Create temperature and rainfall trends
            fig = go.Figure()
            
            # Add temperature lines
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Max_Temp'],
                name='Max Temp',
                line=dict(color='#f44336', width=2),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Min_Temp'],
                name='Min Temp',
                line=dict(color='#2196f3', width=2),
                mode='lines+markers'
            ))
            
            # Add rainfall bars
            fig.add_trace(go.Bar(
                x=forecast_df['Date'],
                y=forecast_df['Rainfall_mm'],
                name='Rainfall (mm)',
                marker_color='#4caf50',
                opacity=0.7,
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title='Temperature & Rainfall Forecast',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                yaxis2=dict(
                    title='Rainfall (mm)',
                    overlaying='y',
                    side='right',
                    range=[0, max(forecast_df['Rainfall_mm']) * 2]
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather impact on crops
            with st.expander("Weather Impact on Crops"):
                st.write("### Weather Impact on Selected Crops")
                
                if st.session_state.get('analyzed', False):
                    recommendations = get_crop_recommendations(
                        location, soil_type, water_availability, season
                    )
                    
                    for _, row in recommendations.iterrows():
                        crop = row['Crop']
                        # Simulate weather impact analysis
                        weather_impact = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
                        impact_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}[weather_impact]
                        
                        st.markdown(f"""
                        <div style="margin-bottom:10px;color:#000;">
                            <p><strong>{crop}:</strong> <span style="color:{impact_color};">{weather_impact} impact</span></p>
                            <ul style="margin-top:5px;">
                                <li>Temperature conditions: {'Favorable' if today_weather['Max_Temp'] < 35 else 'Unfavorable'}</li>
                                <li>Humidity conditions: {'Favorable' if today_weather['Humidity'] < 80 else 'Risk of fungal diseases'}</li>
                                <li>Rainfall adequacy: {'Adequate' if today_weather['Rainfall_mm'] > 5 else 'Insufficient'}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Please analyze your farm details in the Crop Recommendation tab first.")
        else:
            st.warning(f"Weather data not available for {location}. Please select another location.")

    with calendar_col:
        st.subheader("🗓️ Farming Calendar")
        
        # Allow user to select a crop for calendar
        if st.session_state.get('analyzed', False):
            recommendations = get_crop_recommendations(
                location, soil_type, water_availability, season
            )
            
            crop_for_calendar = st.selectbox("Select crop for farming calendar", 
                                            recommendations['Crop'].tolist())
            
            # Generate farming calendar for selected crop
            calendar_df = generate_farming_calendar(crop_for_calendar, season)
            
            # Display calendar data
            st.markdown("""
            <div style="background-color:#f1f8e9; color:#000; padding:15px; border-radius:10px; margin-bottom:20px;">
                <h3>Farming Calendar</h3>
                <p>Plan your agricultural activities based on the recommended timeline below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a Gantt chart for the farming calendar
            fig = px.timeline(
                calendar_df, 
                x_start="Start_Date", 
                x_end="End_Date", 
                y="Activity",
                color="Status",
                color_discrete_map={
                    'Completed': '#4caf50',
                    'In Progress': '#ff9800',
                    'Upcoming': '#2196f3'
                },
                title=f"Farming Calendar for {crop_for_calendar}"
            )
            
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Activity",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add calendar details in a table
            st.subheader("Activity Details")
            
            # Format dates for display
            calendar_df['Start_Date'] = pd.to_datetime(calendar_df['Start_Date'], errors='coerce')
            calendar_df['End_Date'] = pd.to_datetime(calendar_df['End_Date'], errors='coerce')

            calendar_df['Start_Date'] = calendar_df['Start_Date'].dt.strftime('%d %b %Y')
            calendar_df['End_Date'] = calendar_df['End_Date'].dt.strftime('%d %b %Y')

            
            st.dataframe(
                calendar_df[['Activity', 'Start_Date', 'End_Date', 'Duration_Days', 'Status']],
                use_container_width=True
            )
            
            # Add activity recommendations based on weather
            st.subheader("Activity Recommendations")
            
            # Get today's date
            today = datetime.date.today()
            
            # Find activities that are current or upcoming
            current_activities = calendar_df[calendar_df['Status'] != 'Completed']
            
            if not current_activities.empty:
                for _, activity in current_activities.iterrows():
                    activity_name = activity['Activity']
                    status = activity['Status']
                    
                    # Generate weather-based recommendations
                    if activity_name == "Land Preparation":
                        if status == "Upcoming":
                            st.markdown(f"""
                            <div style="background-color:#e8f5e9; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <p><strong>Land Preparation:</strong> Good time to prepare your land as weather conditions are favorable.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        # st.dataframe(today_weather) ######### to see if empty or not
                    elif activity_name == "Sowing/Planting":
                        if today_weather['Rainfall_mm'] > 10:
                            st.markdown(f"""
                            <div style="background-color:#fff8e1; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <p><strong>Sowing/Planting:</strong> Consider delaying sowing by 1-2 days due to heavy rainfall.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color:#e8f5e9; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <p><strong>Sowing/Planting:</strong> Weather conditions are favorable for sowing.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    elif activity_name == "Irrigation":
                        if today_weather['Rainfall_mm'] > 5:
                            st.markdown(f"""
                            <div style="background-color:#e8f5e9; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <p><strong>Irrigation:</strong> Skip irrigation as rainfall is sufficient.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color:#fff8e1; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <p><strong>Irrigation:</strong> Low rainfall expected. Plan irrigation accordingly.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    elif activity_name == "Pest Management":
                        if today_weather['Humidity'] > 80:
                            st.markdown(f"""
                            <div style="background-color:#ffebee; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <p><strong>Pest Management:</strong> High humidity increases disease risk. Consider preventive spraying.</p>
                            </div>
                            """, unsafe_allow_html=True)

        else:
            st.info("Please analyze your farm details in the Crop Recommendation tab first to generate a farming calendar.")
        
        # Weather alerts
        st.subheader("⚠️ Weather Alerts")
        
        # Simulate weather alerts
        alerts = []

        if today_weather is not None:
            if today_weather['Rainfall_mm'] > 15:
                alerts.append({
                    'type': 'Heavy Rainfall',
                    'message': f"Heavy rainfall expected ({today_weather['Rainfall_mm']:.1f} mm). Take precautions to prevent waterlogging.",
                    'severity': 'High'
                })
            if today_weather['Max_Temp'] > 38:
                alerts.append({
                    'type': 'High Temperature',
                    'message': f"Temperature expected to reach {today_weather['Max_Temp']}°C. Ensure crops are adequately irrigated.",
                    'severity': 'Medium'
                })
            if today_weather['Humidity'] > 85:
                alerts.append({
                    'type': 'High Humidity',
                    'message': f"High humidity ({today_weather['Humidity']}%) increases risk of fungal diseases. Monitor crops closely.",
                    'severity': 'Medium'
                })

        if alerts:
            for alert in alerts:
                alert_color = {'High': '#ffebee', 'Medium': '#fff8e1', 'Low': '#e8f5e9'}[alert['severity']]
                border_color = {'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}[alert['severity']]
                
                st.markdown(f"""
                <div style="background-color:{alert_color};color:#000; padding:10px; border-radius:5px; margin-bottom:10px; border-left:4px solid {border_color};">
                    <p><strong>{alert['type']}:</strong> {alert['message']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#e8f5e9; color:#000; padding:10px; border-radius:5px; margin-bottom:10px;">
                <p>No severe weather alerts for your location today.</p>
            </div>
            """, unsafe_allow_html=True)


# with tabs[3]:
#     import folium
#     import random
#     from streamlit_folium import st_folium  # ✅ ADD THIS IMPORT

#     # Sample locations in Maharashtra
#     locations = {
#         "Mumbai": (19.0760, 72.8777),
#         "Pune": (18.5204, 73.8567),
#         "Nagpur": (21.1458, 79.0882),
#         "Nashik": (19.9975, 73.7898),
#         "Aurangabad": (19.8762, 75.3433),
#         "Loni Kalbhor": (18.5167, 73.9333),
#     }

#     def create_random_weather_data():
#         weather_data = {}
#         for city in locations:
#             weather_data[city] = {
#                 "main": {
#                     "temp": round(random.uniform(20, 40), 1),
#                     "humidity": random.randint(50, 90),
#                 },
#                 "weather": [
#                     {
#                         "description": random.choice(
#                             ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
#                         ),
#                         "icon": random.choice(["01d", "02d", "09d", "03d"]),
#                     }
#                 ],
#                 "wind": {"speed": round(random.uniform(5, 20), 1)},
#             }
#         return weather_data

#     def create_weather_map(weather_data):
#         if not weather_data:
#             return None

#         latitudes = [coord[0] for coord in locations.values()]
#         longitudes = [coord[1] for coord in locations.values()]
#         center_lat = sum(latitudes) / len(latitudes)
#         center_lon = sum(longitudes) / len(longitudes)

#         m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

#         for city, data in weather_data.items():
#             if 'main' in data and 'weather' in data:
#                 temp = data['main']['temp']
#                 humidity = data['main']['humidity']
#                 description = data['weather'][0]['description'].title()
#                 icon = data['weather'][0]['icon']
#                 icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png"
#                 lat, lon = locations[city]

#                 html = f"""
#                     <h3>{city}</h3>
#                     <img src="{icon_url}" alt="{description}" width="50">
#                     <p>Temperature: {temp}°C</p>
#                     <p>Humidity: {humidity}%</p>
#                     <p>Description: {description}</p>
#                 """
#                 iframe = folium.IFrame(html=html, width=150, height=150)
#                 popup = folium.Popup(iframe, max_width=200)
#                 folium.Marker([lat, lon], popup=popup).add_to(m)

#         return m

#     def create_weather_dataframe(weather_data):
#         if not weather_data:
#             return None

#         data_list = []
#         for city, data in weather_data.items():
#             if 'main' in data and 'weather' in data:
#                 temp = data['main']['temp']
#                 humidity = data['main']['humidity']
#                 description = data['weather'][0]['description'].title()
#                 wind_speed = data.get('wind', {}).get('speed', None)
#                 data_list.append(
#                     {
#                         "City": city,
#                         "Temperature (°C)": temp,
#                         "Humidity (%)": humidity,
#                         "Description": description,
#                         "Wind Speed (m/s)": wind_speed,
#                     }
#                 )
#         df = pd.DataFrame(data_list)
#         return df

#     # ✅ DIRECTLY RUN YOUR MAIN CONTENT (NO NEED FOR A FUNCTION)
#     st.markdown("<h2 class='sub-header'>Weather & Calendar</h2>", unsafe_allow_html=True)
#     st.info("Displays current weather information for selected locations in Maharashtra.")

#     # Generate random weather data
#     if 'weather_data' not in st.session_state:
#         st.session_state['weather_data'] = create_random_weather_data()

#     weather_data = st.session_state['weather_data']

#     # Create and display the map
#     weather_map = create_weather_map(weather_data)
#     if weather_map:
#         st_folium(weather_map, width=700, height=500)  # ✅ FIXED
#     else:
#         st.error("Failed to generate the weather map.")

#     # Create and display the DataFrame
#     weather_df = create_weather_dataframe(weather_data)
#     if weather_df is not None:
#         st.subheader("Weather Data Table")
#         st.dataframe(weather_df)
#     else:
#         st.error("Failed to generate the weather data table.")
