
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Predict button */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.65rem 1rem;
    transition: all 0.2s ease;
}

/* Button hover effect */
.stButton > button:hover {
    border-color: #60a5fa;
    transform: translateY(-1px);
}

/* Rounded bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

/* Predicted price styling */
.price-result {
    color: #34d399;
    font-size: 2.6rem;
    font-weight: 750;
    letter-spacing: -1px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED MODEL ----------------
model_data = joblib.load("boston_house_price_model_final.joblib")

model = model_data["model"]
saved_scaler = model_data["scaler"]
selected_features = model_data["selected_features"]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🏠 Boston Housing")
    st.caption("Machine Learning Dashboard")

    st.markdown("---")

    st.subheader("🤖 Model Information")
    st.info("Final Tuned Random Forest")

    st.markdown("### 📊 Model Performance")
    st.metric("R² Score", "87.69%")
    st.metric("RMSE", "3.00")
    st.metric("MAE", "2.02")

    st.caption(
        "RMSE and MAE are measured in thousands of dollars."
    )

    st.markdown("---")

    st.subheader("🔬 Model Details")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write(f"**Selected Features:** {len(selected_features)}")
    st.write("**Feature Selection:** RFE")
    st.write("**Scaling:** StandardScaler")

    st.markdown("---")
    st.caption("Boston House Price Prediction Project")

# ---------------- CENTERED MAIN PAGE ----------------
left_space, center, right_space = st.columns([1, 3, 1])

with center:

    st.title("🏠 Boston House Price Predictor")
    st.caption("Machine Learning • House Price Estimation")

    st.markdown("---")

    # ---------------- PROPERTY INPUTS ----------------
    st.subheader("🏡 Enter Property Details")
    st.caption("Enter the values for your property.")

    with st.container(border=True):

        col1, col2 = st.columns(2)

        with col1:
            CRIM = st.number_input(
                "CRIM — Crime Rate",
                min_value=0.0,
                value=0.1,
                step=0.01
            )

            RM = st.number_input(
                "RM — Average Rooms",
                min_value=0.0,
                value=6.2,
                step=0.1
            )

            RAD = st.number_input(
                "RAD — Highway Accessibility",
                min_value=0,
                value=4,
                step=1
            )

            PTRATIO = st.number_input(
                "PTRATIO — Pupil-Teacher Ratio",
                min_value=0.0,
                value=18.0,
                step=0.1
            )

        with col2:
            NOX = st.number_input(
                "NOX — Nitric Oxide",
                min_value=0.0,
                value=0.5,
                step=0.01
            )

            DIS = st.number_input(
                "DIS — Distance",
                min_value=0.0,
                value=4.5,
                step=0.1
            )

            TAX = st.number_input(
                "TAX — Property Tax",
                min_value=0,
                value=300,
                step=1
            )

            LSTAT = st.number_input(
                "LSTAT — Lower-Status Population (%)",
                min_value=0.0,
                value=10.0,
                step=0.1
            )

    # ---------------- PREDICTION BUTTON ----------------
    st.write("")

    predict_button = st.button(
        "🔮 Predict House Price",
        type="primary",
        use_container_width=True
    )

    # ---------------- PREDICTION ----------------
    if predict_button:

        new_house = pd.DataFrame({
            "CRIM": [CRIM],
            "NOX": [NOX],
            "RM": [RM],
            "DIS": [DIS],
            "RAD": [RAD],
            "TAX": [TAX],
            "PTRATIO": [PTRATIO],
            "LSTAT": [LSTAT]
        })

    
        new_house = new_house[selected_features]
        new_house_scaled = saved_scaler.transform(new_house)
        prediction = model.predict(new_house_scaled)[0]

        # MEDV is expressed in thousands of dollars
        price = prediction * 1000

        st.markdown("---")
        st.markdown("### 🏠 Prediction Result")

        with st.container(border=True):

            st.caption("ESTIMATED HOUSE PRICE")

            st.markdown(
                f'<div class="price-result">${price:,.2f}</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"Approximately **${price / 1000:,.2f}K**"
            )

            st.caption(
                "This is a model estimate, not a guaranteed market price."
            )
