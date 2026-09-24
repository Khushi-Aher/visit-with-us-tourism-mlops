
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "model.joblib"
META_PATH = BASE / "model_metadata.json"


st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


with st.spinner("Loading prediction model..."):
    model = load_model()


with open(META_PATH, "r") as file:
    metadata = json.load(file)


st.title(
    "✈️ Visit with Us — Wellness Tourism Package Predictor"
)

st.write(
    "Enter customer details to estimate the probability "
    "of purchasing the Wellness Tourism Package."
)


with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        contact = st.selectbox(
            "Type of Contact",
            ["Company Invited", "Self Inquiry"]
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )


    with col2:

        persons = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=10,
            value=2
        )

        hotel = st.selectbox(
            "Preferred Property Star",
            [3, 4, 5]
        )

        marital = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Single",
                "Divorced"
            ]
        )

        trips = st.number_input(
            "Number of Trips",
            min_value=1.0,
            max_value=30.0,
            value=3.0
        )

        passport = st.selectbox(
            "Passport",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        own_car = st.selectbox(
            "Own Car",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


    with col3:

        children = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=5,
            value=0
        )

        income = st.number_input(
            "Monthly Income",
            min_value=1000.0,
            max_value=200000.0,
            value=20000.0
        )

        pitch_score = st.slider(
            "Pitch Satisfaction Score",
            1,
            5,
            3
        )

        product = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King"
            ]
        )

        followups = st.number_input(
            "Number of Followups",
            min_value=0,
            max_value=10,
            value=3
        )

        duration = st.number_input(
            "Duration of Pitch",
            min_value=1.0,
            max_value=60.0,
            value=15.0
        )


    submitted = st.form_submit_button(
        "Predict Purchase Probability"
    )


if submitted:

    customer = pd.DataFrame([{

        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "PreferredPropertyStar": hotel,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
        "PitchSatisfactionScore": pitch_score,
        "ProductPitched": product,
        "NumberOfFollowups": followups,
        "DurationOfPitch": duration

    }])


    probability = float(
        model.predict_proba(customer)[:, 1][0]
    )

    prediction = int(
        probability >= metadata["threshold"]
    )


    st.metric(
        "Purchase Probability",
        f"{probability:.1%}"
    )


    if prediction == 1:

        st.success(
            "Predicted outcome: Potential buyer"
        )

    else:

        st.info(
            "Predicted outcome: Lower purchase likelihood"
        )


    st.caption(
        f"Operating threshold: "
        f"{metadata['threshold']:.2f}"
    )
