# 🌾 Smart Farm Advisor

An intelligent decision-support system that recommends the most suitable crop based on **soil**, **district**, **season**, and **water requirement** — empowering farmers and agri-businesses to make informed, data-driven crop selection decisions.

---

> "Sow data. Reap insights. Cultivate prosperity." 🌾

## 🚀 Project Overview

Agriculture faces challenges like fluctuating weather, soil degradation, and uncertain market demand. Our system leverages **machine learning** on real agri-market and soil data to:

* Recommend the best crop to cultivate in given conditions
* Optimize productivity, reduce risk, and maximize profitability
* Empower farmers with smart agri-decision tools

**Tech Stack:**

* Python (Pandas, Scikit-learn)
* Streamlit (for user-friendly web app)
* Machine Learning (Decision Trees)

---

## 📊 Dataset

The dataset used contains real-world agricultural market and soil parameters:

| Feature                       | Description                               |
| ----------------------------- | ----------------------------------------- |
| District                      | Name of district (e.g., Pune, Nashik)     |
| APMC\_Name                    | Agriculture Produce Market Committee Name |
| Commodity                     | Crop name (target variable)               |
| Arrival\_Quantity\_Qtl        | Quantity of crop arrived in market (Qtl)  |
| Modal\_Price\_Rs\_Qtl         | Modal price (Rs/Qtl)                      |
| Soil\_Type                    | Loam, Sandy etc.                          |
| Water\_Requirement            | Low, Medium etc.                          |
| Season                        | Kharif, Rabi                              |
| Average\_Yield\_Tons\_per\_Ha | Yield                                     |
| Typical\_Costs\_Rs\_per\_Ha   | Cost of production                        |
| Export\_Potential             | High, Medium, Low                         |

---

## 🛠️ Features

✅ Crop recommendation based on soil, water, and district  ✅ Simple, interactive Streamlit web interface  ✅ Easy-to-interpret Decision Tree model  ✅ Export potential insights for market-driven decisions  ✅ Scalable framework to integrate larger datasets

---

## 📈 Model Architecture

We trained a **Decision Tree Classifier** on encoded categorical features (District, Soil, Season, Water). The model predicts the **Commodity** most suitable under the given conditions.

### Key Steps:

* Data preprocessing and label encoding
* Feature selection: District, Soil\_Type, Season, Water\_Requirement
* Model training and evaluation
* Deployment via Streamlit app

---

## 🧠 Research Impact & Capabilities

This project demonstrates:

* **Applied Machine Learning in Agriculture**: Bridging the gap between agri-expertise and data science
* **Feature Importance Analysis**: Understanding key factors influencing crop suitability
* **Decision Support System Design**: Translating research into actionable tools for end-users
* **Scalability Potential**: Framework adaptable to integrate weather, market prices, and geospatial data in future iterations

---

## 🌐 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/agri-crop-recommender.git
cd agri-crop-recommender

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 📌 Future Work

* Integrate real-time weather and market data
* Deploy advanced models (Random Forests, XGBoost)
* Enhance UI/UX and multilingual support
* Collaborate with agri-research bodies for larger datasets
* Implement mobile app

---


