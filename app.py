import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os

# ============================================================
# LOAD BEST MODEL
# ============================================================

final_model = joblib.load("random_forest_model.joblib")

MODEL_R2 = 0.9422
MODEL_RMSE = 2.02
MODEL_MAE = 1.14


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_life_expectancy(
    health_expenditure,
    electricity_access,
    primary_completion,
    gdp_per_capita,
    maternal_mortality,
    clean_water_access
):

    # Apply the same transformations used during model training
    gdp_log = np.log1p(gdp_per_capita)
    health_exp_log = np.log1p(health_expenditure)

    # Create input dataframe with exact model feature names
    input_data = pd.DataFrame([{
        "Health_Exp_Log": health_exp_log,
        "Electricity_Access_Pct": electricity_access,
        "Primary_Completion_Rate": primary_completion,
        "GDP_Per_Capita_Log": gdp_log,
        "Maternal_Mortality_Rate": maternal_mortality,
        "Clean_Water_Access_Pct": clean_water_access
    }])

    # Generate prediction
    prediction = final_model.predict(input_data)[0]
    prediction = round(float(prediction), 2)

    # Categorize prediction
    if prediction >= 75:
        category = "High Predicted Life Expectancy"
        color = "#22c55e"

    elif prediction >= 65:
        category = "Moderate Predicted Life Expectancy"
        color = "#f59e0b"

    else:
        category = "Lower Predicted Life Expectancy"
        color = "#ef4444"

    # ========================================================
    # VISIBLE RESULT CARD
    # ========================================================

    result_card = f"""
    <div style="
        width: 100%;
        box-sizing: border-box;
        background: linear-gradient(135deg, #172554 0%, #0f766e 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 35px 25px;
        margin: 10px 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.35);
    ">

        <div style="
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: #bfdbfe;
            margin-bottom: 12px;
        ">
            ESTIMATED LIFE EXPECTANCY
        </div>

        <div style="
            font-size: 58px;
            line-height: 1;
            font-weight: 900;
            color: #ffffff;
            margin: 5px 0;
        ">
            {prediction:.2f}
        </div>

        <div style="
            font-size: 18px;
            font-weight: 500;
            color: #dbeafe;
            margin-top: 8px;
        ">
            Years
        </div>

        <div style="
            display: inline-block;
            margin-top: 18px;
            padding: 9px 18px;
            border-radius: 30px;
            background: rgba(15,23,42,0.65);
            border: 1px solid {color};
            color: {color};
            font-size: 15px;
            font-weight: 700;
        ">
            {category}
        </div>

    </div>
    """

    # ========================================================
    # INTERPRETATION
    # ========================================================

    interpretation = f"""
### Prediction Summary

**Estimated Life Expectancy:** {prediction:.2f} years

**Prediction Category:** {category}

The estimate is generated using six development indicators from the
World Bank World Development Indicators dataset.

**Model:** Random Forest Regressor  
**R²:** 0.942  
**Mean Absolute Error:** 1.14 years
"""

    return result_card, interpretation


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

/* ==========================================================
   GLOBAL
   ========================================================== */

.gradio-container {
    background: #eaf0f6 !important;
    color: #172033 !important;
    font-family: Inter, Arial, sans-serif !important;
}

body {
    background: #eaf0f6 !important;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    background:
        linear-gradient(
            135deg,
            #172554 0%,
            #1e3a8a 45%,
            #0f766e 100%
        );

    padding: 38px;
    border-radius: 22px;
    margin-bottom: 24px;

    border: 1px solid #334155;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.30);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
}

.hero-subtitle {
    color: #bfdbfe;
    font-size: 18px;
    margin-top: 10px;
}

.hero-text {
    color: #dbeafe;
    margin-top: 15px;
    font-size: 15px;
    line-height: 1.6;
}


/* ==========================================================
   CARDS
   ========================================================== */

.card {
    background: #ffffff !important;
    border: 1px solid #d6dee8 !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
}


/* ==========================================================
   CARD HEADINGS
   ========================================================== */

.card h2,
.card h3,
.card h4,
.gradio-container h2,
.gradio-container h3 {
    color: #172033 !important;
    font-weight: 800 !important;
}


/* ==========================================================
   CARD TEXT
   ========================================================== */

.card p,
.card li,
.card label {
    color: #334155 !important;
}


/* ==========================================================
   MODEL METRICS
   ========================================================== */

.metric-card {
    background: #111827;

    border: 1px solid #334155;

    border-radius: 16px;

    text-align: center;

    padding: 17px;

    min-height: 75px;
}

.metric-title {
    color: #60a5fa;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 5px;
}

.metric-value {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 800;
}


/* ==========================================================
   BUTTON
   ========================================================== */

#predict-button {
    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        ) !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 12px !important;

    font-weight: 700 !important;

    height: 52px;

    margin-top: 10px;

    box-shadow:
        0 5px 18px rgba(37,99,235,0.25);
}

#predict-button:hover {
    filter: brightness(1.08);
}


/* ==========================================================
   INPUT LABELS
   ========================================================== */

label {
    color: #334155 !important;
}


/* ==========================================================
   SLIDERS
   ========================================================== */

input[type="range"] {
    accent-color: #38bdf8;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 25px;
    padding: 15px;
    font-size: 12px;
}

"""


# ============================================================
# APP
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="Life Expectancy Prediction Dashboard"
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-title">
                🌍 Life Expectancy Prediction Dashboard
            </div>

            <div class="hero-subtitle">
                World Bank World Development Indicators
            </div>

            <div class="hero-text">
                Explore how health, economic, education, and
                infrastructure indicators relate to predicted
                life expectancy.
            </div>

        </div>
        """
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    with gr.Row():

        gr.HTML(
            """
            <div class="metric-card">
                <div class="metric-title">MODEL</div>
                <div class="metric-value">Random Forest</div>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class="metric-card">
                <div class="metric-title">R² SCORE</div>
                <div class="metric-value">{MODEL_R2:.3f}</div>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class="metric-card">
                <div class="metric-title">RMSE</div>
                <div class="metric-value">{MODEL_RMSE:.2f}</div>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class="metric-card">
                <div class="metric-title">MAE</div>
                <div class="metric-value">{MODEL_MAE:.2f} yrs</div>
            </div>
            """
        )


    # ========================================================
    # MAIN CONTENT
    # ========================================================

    with gr.Row():

        # ----------------------------------------------------
        # INPUT SECTION
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown("## Development Indicators")

            health_expenditure = gr.Number(
                label="🏥 Health Expenditure per Capita (USD)",
                value=350,
                minimum=0
            )

            electricity_access = gr.Slider(
                minimum=0,
                maximum=100,
                value=75,
                step=1,
                label="⚡ Electricity Access (%)"
            )

            primary_completion = gr.Slider(
                minimum=0,
                maximum=100,
                value=85,
                step=1,
                label="🎓 Primary Completion Rate (%)"
            )

            gdp_per_capita = gr.Number(
                label="💰 GDP per Capita (USD)",
                value=5000,
                minimum=0
            )

            maternal_mortality = gr.Number(
                label="👩 Maternal Mortality Rate (per 100k)",
                value=150,
                minimum=0
            )

            clean_water_access = gr.Slider(
                minimum=0,
                maximum=100,
                value=80,
                step=1,
                label="💧 Clean Water Access (%)"
            )

            predict_button = gr.Button(
                "Predict Life Expectancy",
                variant="primary",
                size="lg",
                elem_id="predict-button"
            )


        # ----------------------------------------------------
        # RESULT SECTION
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown("## Prediction Result")

            prediction_card = gr.HTML(
                value="""
                <div style="
                    background:#ffffff;
                    border:1px solid #334155;
                    border-radius:20px;
                    padding:45px 20px;
                    text-align:center;
                    color:#94a3b8;
                ">

                    <div style="
                        font-size:13px;
                        letter-spacing:1px;
                        font-weight:700;
                        margin-bottom:12px;
                    ">
                        PREDICTION RESULT
                    </div>

                    <div style="
                        font-size:16px;
                        line-height:1.5;
                    ">
                        Enter the development indicators
                        and click <b>Predict Life Expectancy</b>.
                    </div>

                </div>
                """
            )

            interpretation = gr.Markdown(
                """
### Prediction Summary

Your prediction will appear here after running the model.
"""
            )


    # ========================================================
    # EXAMPLE SCENARIOS
    # ========================================================

    gr.Markdown("## Example Development Profiles")

    gr.Examples(
        examples=[
            [100, 30, 60, 1200, 550, 45],
            [450, 85, 88, 6500, 180, 82],
            [4500, 99, 98, 45000, 12, 99]
        ],

        inputs=[
            health_expenditure,
            electricity_access,
            primary_completion,
            gdp_per_capita,
            maternal_mortality,
            clean_water_access
        ]
    )


    # ========================================================
    # ABOUT THE MODEL
    # ========================================================

    with gr.Accordion(
        "About This Model",
        open=False
    ):

        gr.Markdown(
            """
### Model Inputs

- Health Expenditure Per Capita
- Electricity Access
- Primary Completion Rate
- GDP Per Capita
- Maternal Mortality Rate
- Clean Water Access

### Target

**Life Expectancy at Birth (Years)**

### Dataset

**World Bank World Development Indicators (WDI)**

### Model

**Random Forest Regressor**

The model was trained using development indicators from
2000–2022. GDP per capita and health expenditure were
log-transformed before modeling.
"""
        )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">
            Life Expectancy Prediction Model • World Bank WDI
        </div>
        """
    )


    # ========================================================
    # EVENT
    # ========================================================

    predict_button.click(
        fn=predict_life_expectancy,

        inputs=[
            health_expenditure,
            electricity_access,
            primary_completion,
            gdp_per_capita,
            maternal_mortality,
            clean_water_access
        ],

        outputs=[
            prediction_card,
            interpretation
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
