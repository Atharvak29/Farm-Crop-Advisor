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
tabs = st.tabs(["🏠 Home", "🌱 Crop Recommendation", "💰 Economic Analysis", "🌦️ Weather & Calendar", "📊 Data Explorer", "ℹ️ About"])

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
        weather_df_today = weather_df[weather_df['Date'] == datetime.date.today()]
        if not weather_df_today.empty and location in weather_df_today['District'].values:
            weather_today = weather_df_today[weather_df_today['District'] == location].iloc[0]
            st.metric(label=f"Temperature ({location})", 
                     value=f"{weather_today['Max_Temp']}°C", 
                     delta=f"{weather_today['Max_Temp'] - weather_today['Min_Temp']}°C range")
        else:
            st.metric(label="Temperature", value="30°C", delta="10°C range")
    
    with col3:
        # Example rainfall metric
        if not weather_df_today.empty and location in weather_df_today['District'].values:
            weather_today = weather_df_today[weather_df_today['District'] == location].iloc[0]
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
        <div style="background-color:#e8f5e9; padding:15px; border-radius:10px; height:150px;">
            <h3 style="color:#2e7d32">🌱 Crop Recommendations</h3>
            <p>Get personalized crop recommendations based on your farm's conditions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with quick_col2:
        st.markdown("""
        <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; height:150px;">
            <h3 style="color:#1565c0">💰 Economic Analysis</h3>
            <p>Calculate potential profits, ROI, and market prices for different crops</p>
        </div>
        """, unsafe_allow_html=True)
    
    with quick_col3:
        st.markdown("""
        <div style="background-color:#fff8e1; padding:15px; border-radius:10px; height:150px;">
            <h3 style="color:#ff8f00">🌦️ Weather Forecast</h3>
            <p>View weather predictions and plan your farming activities accordingly</p>
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
                <div style="background-color:#f1f8e9; padding:15px; border-radius:10px; margin-bottom:20px; 
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
        
        # # Add a point for current price and yield
        # current_price = crop_price_per_kg
        # current_yield = recommendations[recommendations['Crop'] == crop_for_breakeven]['Predicte
                                                                                       
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

        st.plotly_chart(fig4, use_container_width=True)