import pickle
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
LOW_TEMP, HIGH_TEMP = 50, 100        # the temperatures (F) the model learned from
MAX_RAIN = 3.0                       # the model saw very few days wetter than this (inches)
LOW_SALES, HIGH_SALES = 200, 300     # below LOW_SALES is a slow day, above HIGH_SALES is a busy one
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------

# Load the trained model (we trained it once, in Colab, and saved it to a file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date. The app gets the weather forecast (high temperature and rain), "
           "checks whether it is a weekend, and predicts how many cones you will sell.")

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=16),    # the forecast reaches about 16 days ahead
)

if st.button("Predict sales", type="primary"):
    # 1. Ask the weather API for the forecast: high temperature and total rain
    latitude, longitude = CITIES[city]
    response = requests.get(
        URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "auto",
            "start_date": str(selected_date),
            "end_date": str(selected_date),
        },
        timeout=30,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    temperature = daily["temperature_2m_max"][0]
    rain = daily["precipitation_sum"][0] or 0.0     # the API can return None; treat it as dry

    # 2. Weekend or not comes from the date itself (Monday = 0 ... Saturday = 5, Sunday = 6)
    weekend = 1 if selected_date.weekday() >= 5 else 0

    # 3. Give the model the same inputs it was trained on, with the same names, then predict
    features = pd.DataFrame({"temperature": [temperature], "rain": [rain], "weekend": [weekend]})
    prediction = max(0, model.predict(features[model.feature_names_in_])[0])

    # 4. Show the result
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Forecast high", f"{temperature:.0f} °F")
    col2.metric("Forecast rain", f"{rain:.2f} in")
    col3.metric("Day", selected_date.strftime("%A"), "Weekend" if weekend else "Weekday",
                delta_color="off")
    col4.metric("Predicted sales", f"{prediction:.0f} cones")

    if temperature < LOW_TEMP or temperature > HIGH_TEMP:
        st.warning(f"This temperature is outside the range the model learned from ({LOW_TEMP} to {HIGH_TEMP} °F). "
                   "Treat the number with caution.")
    if rain > MAX_RAIN:
        st.warning(f"This much rain (over {MAX_RAIN} in) is beyond what the model mostly learned from. "
                   "Treat the number with caution.")

    if prediction > HIGH_SALES:
        st.info("Busy day: make sure you are fully stocked.")
    elif prediction > LOW_SALES:
        st.info("A normal day is expected.")
    else:
        st.info("A slow day: consider a promotion, or order less.")
