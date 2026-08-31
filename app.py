import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from xgboost import XGBRegressor


# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="WDI Health Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# 2. CUSTOM THEME
# ==============================================================================

st.markdown(
    """
    <style>

    /* ============================================================
       GLOBAL
       ============================================================ */

    .stApp {
        background-color: #F8FAFC;
        font-family: "Inter", system-ui, -apple-system, sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ============================================================
       HEADER
       ============================================================ */

    .header-container {
        background: linear-gradient(
            135deg,
            #0F172A 0%,
            #1E293B 100%
        );
        padding: 32px 36px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10);
    }

    .header-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
        color: #F8FAFC;
    }

    .header-subtitle {
        font-size: 1rem;
        color: #CBD5E1;
        margin-bottom: 12px;
    }

    .header-description {
        font-size: 0.92rem;
        color: #94A3B8;
        max-width: 850px;
        line-height: 1.6;
    }

    /* ============================================================
       SECTION HEADINGS
       ============================================================ */

    .section-title {
        color: #0F172A;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 0.9rem;
        margin-bottom: 18px;
    }

    /* ============================================================
       PREDICTION CARD
       ============================================================ */

    .prediction-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 30px;
        min-height: 215px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
    }

    .prediction-label {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .prediction-value {
        color: #0F172A;
        font-size: 3.4rem;
        line-height: 1;
        font-weight: 850;
        margin-top: 16px;
    }

    .prediction-unit {
        color: #0EA5E9;
        font-size: 1.25rem;
        font-weight: 700;
    }

    .prediction-note {
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 12px;
    }

    /* ============================================================
       STATUS
       ============================================================ */

    .status-badge {
        padding: 11px 14px;
        border-radius: 10px;
        font-weight: 650;
        font-size: 0.86rem;
        margin-top: 16px;
    }

    .status-high {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    .status-medium {
        background: #FEFCE8;
        color: #854D0E;
        border: 1px solid #FEF08A;
    }

    .status-low {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    /* ============================================================
       PROFILE CARD
       ============================================================ */

    .profile-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 26px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
    }

    .profile-row {
        margin-bottom: 16px;
    }

    .profile-label-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .profile-label {
        color: #475569;
        font-size: 0.86rem;
        font-weight: 650;
    }

    .profile-value {
        color: #0F172A;
        font-size: 0.86rem;
        font-weight: 750;
    }

    .progress-track {
        width: 100%;
        height: 8px;
        background: #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: #0EA5E9;
        border-radius: 10px;
    }

    /* ============================================================
       SCENARIO CARDS
       ============================================================ */

    .scenario-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        min-height: 145px;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.04);
    }

    .scenario-name {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .scenario-value {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .scenario-change {
        color: #0EA5E9;
        font-size: 0.85rem;
        font-weight: 650;
        margin-top: 5px;
    }

    /* ============================================================
       SIDEBAR
       ============================================================ */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* ============================================================
       SMALL TEXT
       ============================================================ */

    .disclaimer {
        color: #64748B;
        font-size: 0.78rem;
        line-height: 1.55;
        padding-top: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 3. MODEL LOADING
# ==============================================================================

MODEL_PATH = "best_wdi_xgboost.json"

FEATURE_NAMES = [
    "Health_Exp_Log",
    "Electricity_Access_Pct",
    "Primary_Completion_Rate",
    "GDP_Per_Capita_Log",
    "Maternal_Mortality_Rate",
    "Clean_Water_Access_Pct"
]


@st.cache_resource
def load_model():
    """Load the trained XGBoost regression model."""

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    return model


try:
    model = load_model()

except Exception:
    st.error(
        "The trained prediction model could not be loaded. "
        f"Please verify that '{MODEL_PATH}' is available in the application directory."
    )
    st.stop()


# ==============================================================================
# 4. SIDEBAR — POLICY CONTROLS
# ==============================================================================

with st.sidebar:

    st.markdown("## ⚙️ Policy Controls")

    st.caption(
        "Adjust development indicators to explore how alternative "
        "national conditions influence the model's predicted life expectancy."
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Infrastructure
    # --------------------------------------------------------------------------

    st.markdown("### ⚡ Infrastructure & Utilities")

    electricity = st.slider(
        "Electricity Access (% Population)",
        min_value=0.0,
        max_value=100.0,
        value=68.0,
        step=0.5
    )

    clean_water = st.slider(
        "Clean Water Access (% Population)",
        min_value=0.0,
        max_value=100.0,
        value=74.0,
        step=0.5
    )

    # --------------------------------------------------------------------------
    # Healthcare
    # --------------------------------------------------------------------------

    st.markdown("### 🏥 Healthcare Delivery")

    maternal_mortality = st.slider(
        "Maternal Mortality (per 100,000 births)",
        min_value=5.0,
        max_value=1000.0,
        value=180.0,
        step=5.0
    )

    health_exp_raw = st.number_input(
        "Health Expenditure per Capita ($)",
        min_value=5.0,
        max_value=10000.0,
        value=220.0,
        step=25.0
    )

    # --------------------------------------------------------------------------
    # Education & Economy
    # --------------------------------------------------------------------------

    st.markdown("### 📈 Education & Economy")

    primary_edu = st.slider(
        "Primary Completion Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=0.5
    )

    gdp_raw = st.number_input(
        "GDP per Capita ($)",
        min_value=100.0,
        max_value=120000.0,
        value=2800.0,
        step=250.0
    )


# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================

input_data = pd.DataFrame([{
    "Health_Exp_Log": np.log1p(health_exp_raw),
    "Electricity_Access_Pct": electricity,
    "Primary_Completion_Rate": primary_edu,
    "GDP_Per_Capita_Log": np.log1p(gdp_raw),
    "Maternal_Mortality_Rate": maternal_mortality,
    "Clean_Water_Access_Pct": clean_water
}])


# ==============================================================================
# 6. CURRENT PREDICTION
# ==============================================================================

predicted_life_exp = float(model.predict(input_data)[0])


# ==============================================================================
# 7. HEADER
# ==============================================================================

st.markdown(
    """
    <div class="header-container">

        <div class="header-title">
            National Life Expectancy Intelligence System
        </div>

        <div class="header-subtitle">
            World Bank WDI • Predictive Analytics & Scenario Planning
        </div>

        <div class="header-description">
            Explore how national development indicators relate to predicted
            life expectancy using a machine-learning model trained on
            World Bank development data.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 8. PREDICTION + DEVELOPMENT PROFILE
# ==============================================================================

col_prediction, col_profile = st.columns([1, 1.35], gap="large")


# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

with col_prediction:

    st.markdown(
        '<div class="section-title">Model Output</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Predicted Life Expectancy
            </div>

            <div class="prediction-value">
                {predicted_life_exp:.1f}
                <span class="prediction-unit">Years</span>
            </div>

            <div class="prediction-note">
                Current scenario • Model-generated prediction
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # Prediction outlook
    # --------------------------------------------------------------------------

    if predicted_life_exp >= 75:

        st.markdown(
            """
            <div class="status-badge status-high">
                Higher predicted outlook
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predicted_life_exp >= 65:

        st.markdown(
            """
            <div class="status-badge status-medium">
                Moderate predicted outlook
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-badge status-low">
                Lower predicted outlook
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="disclaimer">
            This classification describes the model's predicted outcome.
            It is not an official development classification.
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------------------
# Development Profile
# ------------------------------------------------------------------------------

with col_profile:

    st.markdown(
        '<div class="section-title">Development Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Current values supplied to the prediction model.'
        '</div>',
        unsafe_allow_html=True
    )

    profile_html = '<div class="profile-card">'

    profile_items = [
        ("Electricity Access", electricity),
        ("Clean Water Access", clean_water),
        ("Primary Completion Rate", primary_edu)
    ]

    for label, value in profile_items:

        profile_html += f"""
        <div class="profile-row">

            <div class="profile-label-row">

                <span class="profile-label">
                    {label}
                </span>

                <span class="profile-value">
                    {value:.1f}%
                </span>

            </div>

            <div class="progress-track">

                <div
                    class="progress-fill"
                    style="width:{min(value, 100):.1f}%;">
                </div>

            </div>

        </div>
        """

    profile_html += "</div>"

    st.markdown(profile_html, unsafe_allow_html=True)

    st.markdown("")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Health Expenditure",
            f"${health_exp_raw:,.0f}"
        )

    with metric2:
        st.metric(
            "GDP per Capita",
            f"${gdp_raw:,.0f}"
        )

    with metric3:
        st.metric(
            "Maternal Mortality",
            f"{maternal_mortality:,.0f}"
        )


st.markdown("---")


# ==============================================================================
# 9. MODEL DRIVERS
# ==============================================================================

st.markdown(
    '<div class="section-title">Model Drivers</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    "Relative importance of the indicators used by the XGBoost model."
    '</div>',
    unsafe_allow_html=True
)


clean_labels = {
    "Maternal_Mortality_Rate": "Maternal Mortality Rate",
    "Electricity_Access_Pct": "Electricity Access (%)",
    "Clean_Water_Access_Pct": "Clean Water Access (%)",
    "Health_Exp_Log": "Log Health Expenditure",
    "Primary_Completion_Rate": "Primary Completion Rate",
    "GDP_Per_Capita_Log": "Log GDP per Capita"
}


importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": importances
})

feat_df["Feature"] = feat_df["Feature"].map(clean_labels)

feat_df = feat_df.sort_values(
    "Importance",
    ascending=True
)


col_chart, col_insights = st.columns([1.6, 1], gap="large")


# ------------------------------------------------------------------------------
# Feature Importance Chart
# ------------------------------------------------------------------------------

with col_chart:

    fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=55, t=15, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Relative Feature Importance",
        yaxis_title="",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------------------------
# Dynamic Insights
# ------------------------------------------------------------------------------

with col_insights:

    st.markdown("### 💡 Model Insights")

    top_features = feat_df.sort_values(
        "Importance",
        ascending=False
    ).head(3)

    for _, row in top_features.iterrows():

        st.markdown(
            f"• **{row['Feature']}** is among the model's strongest "
            "predictive features."
        )

    st.markdown(
        """
        <div class="disclaimer">
            Feature importance describes how strongly variables contribute
            to the model's predictions. It does not establish causation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# 10. SCENARIO ANALYSIS
# ==============================================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Scenario Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    "Explore how alternative development assumptions change the model's "
    "predicted outcome."
    '</div>',
    unsafe_allow_html=True
)


# ==============================================================================
# Scenario Definitions
# ==============================================================================

scenario_data = {

    "Current Baseline": {
        "electricity": electricity,
        "clean_water": clean_water,
        "maternal_mortality": maternal_mortality,
        "health_exp": health_exp_raw,
        "primary_edu": primary_edu,
        "gdp": gdp_raw
    },

    "Infrastructure Improvement": {
        "electricity": min(electricity + 15, 100),
        "clean_water": min(clean_water + 15, 100),
        "maternal_mortality": max(maternal_mortality - 40, 5),
        "health_exp": health_exp_raw * 1.20,
        "primary_edu": min(primary_edu + 8, 100),
        "gdp": gdp_raw * 1.15
    },

    "High Development": {
        "electricity": min(electricity + 30, 100),
        "clean_water": min(clean_water + 25, 100),
        "maternal_mortality": max(maternal_mortality - 80, 5),
        "health_exp": health_exp_raw * 1.50,
        "primary_edu": min(primary_edu + 15, 100),
        "gdp": gdp_raw * 1.35
    }
}


# ==============================================================================
# Scenario Predictions
# ==============================================================================

scenario_predictions = []


for scenario_name, values in scenario_data.items():

    scenario_input = pd.DataFrame([{
        "Health_Exp_Log": np.log1p(values["health_exp"]),
        "Electricity_Access_Pct": values["electricity"],
        "Primary_Completion_Rate": values["primary_edu"],
        "GDP_Per_Capita_Log": np.log1p(values["gdp"]),
        "Maternal_Mortality_Rate": values["maternal_mortality"],
        "Clean_Water_Access_Pct": values["clean_water"]
    }])

    prediction = float(
        model.predict(scenario_input)[0]
    )

    scenario_predictions.append({
        "Scenario": scenario_name,
        "Predicted Life Expectancy": prediction
    })


scenario_df = pd.DataFrame(scenario_predictions)

baseline_prediction = scenario_df.loc[
    scenario_df["Scenario"] == "Current Baseline",
    "Predicted Life Expectancy"
].iloc[0]


# ==============================================================================
# 11. SCENARIO CARDS
# ==============================================================================

scenario_cols = st.columns(3, gap="medium")


for column, (_, row) in zip(
    scenario_cols,
    scenario_df.iterrows()
):

    scenario_name = row["Scenario"]
    prediction = row["Predicted Life Expectancy"]
    change = prediction - baseline_prediction

    with column:

        if scenario_name == "Current Baseline":
            change_text = "Current selected conditions"
        else:
            change_text = (
                f"{change:+.1f} years vs baseline"
            )

        st.markdown(
            f"""
            <div class="scenario-card">

                <div class="scenario-name">
                    {scenario_name}
                </div>

                <div class="scenario-value">
                    {prediction:.1f} years
                </div>

                <div class="scenario-change">
                    {change_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("")


# ==============================================================================
# 12. SCENARIO COMPARISON CHART
# ==============================================================================

fig_scenario = px.bar(
    scenario_df,
    x="Scenario",
    y="Predicted Life Expectancy",
    text="Predicted Life Expectancy"
)

fig_scenario.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig_scenario.update_layout(
    height=360,
    margin=dict(l=20, r=20, t=25, b=30),
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis_title="Predicted Life Expectancy (Years)",
    xaxis_title="",
    showlegend=False
)

st.plotly_chart(
    fig_scenario,
    use_container_width=True
)


st.caption(
    "Illustrative model scenarios — not causal estimates or guaranteed "
    "policy outcomes."
)


# ==============================================================================
# 13. ABOUT THE MODEL
# ==============================================================================

st.markdown("---")

with st.expander("ℹ️ About the Model"):

    about_col1, about_col2 = st.columns(2)

    with about_col1:

        st.markdown("### Data & Target")

        st.markdown(
            """
            **Data Source**

            World Bank World Development Indicators (WDI)

            **Prediction Target**

            Life expectancy at birth, total (years)

            **Model**

            XGBoost Regression
            """
        )

    with about_col2:

        st.markdown("### Input Indicators")

        st.markdown(
            """
            - Health expenditure per capita
            - Electricity access
            - Primary completion rate
            - GDP per capita
            - Maternal mortality
            - Clean water access
            """
        )

    st.markdown("---")

    st.markdown(
        """
        ### Important Interpretation Note

        The model uses historical development indicators to generate
        predictions based on patterns learned during training.

        Selected monetary variables are log-transformed before being
        passed to the model.

        Scenario results represent the model's response to hypothetical
        changes in input values. They should not be interpreted as causal
        estimates, guaranteed forecasts, or evidence that changing one
        indicator will directly produce a specific change in life expectancy.
        """
    )


# ==============================================================================
# 14. FOOTER
# ==============================================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94A3B8;
        font-size:0.75rem;
        padding-top:30px;
    ">
        World Bank WDI Predictive Analytics • XGBoost Life Expectancy Model
    </div>
    """,
    unsafe_allow_html=True
)
