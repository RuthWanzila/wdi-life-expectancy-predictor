
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="WDI Life Expectancy Predictor",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("Life Expectancy Predictor")
st.subheader("World Bank World Development Indicators")

st.write(
    "This application uses development, health, education, and economic "
    "indicators to predict life expectancy at birth."
)

st.info(
    "The prediction is generated using an XGBoost regression model trained "
    "on World Bank World Development Indicators (WDI) data."
)

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("📊 Development Indicators")

st.sidebar.write(
    "Adjust the indicators below to generate a predicted life expectancy."
)

electricity = st.sidebar.slider(
    "Electricity Access (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.1
)

clean_water = st.sidebar.slider(
    "Clean Water Access (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=0.1
)

primary_edu = st.sidebar.slider(
    "Primary Completion Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=0.1
)

maternal_mortality = st.sidebar.number_input(
    "Maternal Mortality (per 100,000 live births)",
    min_value=5.0,
    max_value=1000.0,
    value=150.0,
    step=5.0
)

health_exp_raw = st.sidebar.number_input(
    "Health Expenditure Per Capita (USD)",
    min_value=5.0,
    max_value=10000.0,
    value=250.0,
    step=10.0
)

gdp_raw = st.sidebar.number_input(
    "GDP Per Capita (USD)",
    min_value=100.0,
    max_value=120000.0,
    value=3500.0,
    step=100.0
)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
input_df = pd.DataFrame([{
    "Health_Exp_Log": np.log1p(health_exp_raw),
    "Electricity_Access_Pct": electricity,
    "Primary_Completion_Rate": primary_edu,
    "GDP_Per_Capita_Log": np.log1p(gdp_raw),
    "Maternal_Mortality_Rate": maternal_mortality,
    "Clean_Water_Access_Pct": clean_water
}])

# Make sure feature order matches training
feature_order = [
    "Health_Exp_Log",
    "Electricity_Access_Pct",
    "Primary_Completion_Rate",
    "GDP_Per_Capita_Log",
    "Maternal_Mortality_Rate",
    "Clean_Water_Access_Pct"
]

input_df = input_df[feature_order]

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("best_wdi_xgboost.json")
    return model


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.divider()

try:
    model = load_model()

    prediction = model.predict(input_df)[0]

    # Keep prediction within a realistic range
    prediction = np.clip(prediction, 0, 100)

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------
    st.header("🔮 Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Life Expectancy",
            f"{prediction:.1f} years"
        )

    with col2:
        st.metric(
            "GDP Per Capita",
            f"${gdp_raw:,.0f}"
        )

    with col3:
        st.metric(
            "Health Expenditure",
            f"${health_exp_raw:,.0f}"
        )

    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------
    st.subheader("📌 Interpretation")

    if prediction < 50:
        interpretation = (
            "The predicted life expectancy is relatively low. "
            "This may reflect weaker socioeconomic, healthcare, "
            "education, or infrastructure conditions."
        )

    elif prediction < 65:
        interpretation = (
            "The predicted life expectancy falls within a moderate range. "
            "There may still be opportunities to improve healthcare, "
            "education, infrastructure, and economic conditions."
        )

    elif prediction < 75:
        interpretation = (
            "The predicted life expectancy is relatively high, suggesting "
            "generally favorable development and health conditions."
        )

    else:
        interpretation = (
            "The predicted life expectancy is high, indicating relatively "
            "strong socioeconomic, healthcare, education, and infrastructure "
            "conditions."
        )

    st.write(interpretation)

    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------
    st.subheader("📊 Input Summary")

    display_df = pd.DataFrame({
        "Indicator": [
            "Electricity Access",
            "Clean Water Access",
            "Primary Completion Rate",
            "Maternal Mortality",
            "Health Expenditure Per Capita",
            "GDP Per Capita"
        ],
        "Value": [
            f"{electricity:.1f}%",
            f"{clean_water:.1f}%",
            f"{primary_edu:.1f}%",
            f"{maternal_mortality:.0f} per 100k",
            f"${health_exp_raw:,.0f}",
            f"${gdp_raw:,.0f}"
        ]
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------
    st.subheader("🤖 Model Information")

    st.write(
        "Model: XGBoost Regressor"
    )

    st.write(
        "Target: Life Expectancy at Birth (Years)"
    )

    st.write(
        "Data Source: World Bank World Development Indicators (WDI)"
    )

except FileNotFoundError:
    st.error(
        "The trained model file 'best_wdi_xgboost.json' was not found. "
        "Make sure the model file is in the same folder as app.py."
    )

except Exception as e:
    st.error(
        f"The model could not generate a prediction. Error: {e}"
    )
