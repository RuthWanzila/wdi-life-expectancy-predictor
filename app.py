import gradio as gr
import numpy as np
import pandas as pd
import joblib

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

    gdp_log = np.log1p(gdp_per_capita)
    health_exp_log = np.log1p(health_expenditure)

    input_data = pd.DataFrame([{
        "Health_Exp_Log": health_exp_log,
        "Electricity_Access_Pct": electricity_access,
        "Primary_Completion_Rate": primary_completion,
        "GDP_Per_Capita_Log": gdp_log,
        "Maternal_Mortality_Rate": maternal_mortality,
        "Clean_Water_Access_Pct": clean_water_access
    }])

    prediction = final_model.predict(input_data)[0]
    prediction = round(float(prediction), 2)

    if prediction >= 75:
        category = "High Predicted Life Expectancy"
        color = "#22c55e"

    elif prediction >= 65:
        category = "Moderate Predicted Life Expectancy"
        color = "#f59e0b"

    else:
        category = "Lower Predicted Life Expectancy"
        color = "#ef4444"

   result_card = f"""
<div style="
    background: #111827;
    border: 1px solid #374151;
    border-radius: 16px;
    padding: 30px 24px;
    text-align: center;
    width: 100%;
    box-sizing: border-box;
    color: #ffffff;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
">

    <div style="
        color: #9ca3af;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    ">
        ESTIMATED LIFE EXPECTANCY
    </div>

    <div style="
        color: #38bdf8;
        font-size: 52px;
        font-weight: 800;
        line-height: 1.1;
    ">
        {prediction:.2f}
    </div>

    <div style="
        color: #d1d5db;
        font-size: 16px;
        margin-top: 4px;
    ">
        Years
    </div>

    <div style="
        color: {color};
        font-weight: 700;
        font-size: 18px;
        margin-top: 18px;
    ">
        {category}
    </div>

</div>
"""
    interpretation = f"""
### Model Assessment

**Estimated Life Expectancy:** {prediction:.2f} years

**Category:** {category}

The prediction is generated using health, education, economic,
infrastructure, and development indicators from the World Bank
World Development Indicators dataset.
"""

    return result_card, interpretation


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

.gradio-container{
    background:#0f172a;
    color:white;
    font-family:Inter, Arial, sans-serif;
}

/* Hero */

.hero{
    background:
    linear-gradient(
    135deg,
    #1e3a8a,
    #0ea5e9
    );

    padding:40px;

    border-radius:24px;

    margin-bottom:25px;

    box-shadow:
    0 10px 30px rgba(0,0,0,.25);
}

.hero-title{
    font-size:42px;
    font-weight:800;
    color:white;
}

.hero-subtitle{
    color:#dbeafe;
    font-size:18px;
    margin-top:8px;
}

.hero-text{
    color:#e0f2fe;
    margin-top:15px;
}

/* Cards */

.card{
    background:#111827;
    border:1px solid #334155;
    border-radius:20px;
    padding:22px;
}

/* Buttons */

#predict-button{
    background:
    linear-gradient(
    135deg,
    #2563eb,
    #06b6d4
    );

    color:white !important;

    border:none !important;

    border-radius:14px !important;

    font-weight:700 !important;

    height:52px;
}

#predict-button:hover{
    transform:translateY(-2px);
}

/* Metric Cards */

.metric-card{
    background:#111827;
    border:1px solid #334155;
    border-radius:18px;

    text-align:center;

    padding:16px;
}

.metric-title{
    color:#60a5fa;
    font-size:14px;
    font-weight:600;
}

.metric-value{
    color:white;
    font-size:24px;
    font-weight:800;
}

/* Result Card */

.result-card{

    background:
    linear-gradient(
    135deg,
    #172554,
    #0f766e
    );

    border-radius:24px;

    padding:35px;

    text-align:center;

    box-shadow:
    0 6px 20px rgba(0,0,0,0.25);
}

.result-value{
    font-size:64px;
    font-weight:900;
    color:white;
}

.result-unit{
    font-size:20px;
    color:#dbeafe;
}

/* Footer */

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:20px;
    font-size:13px;
}

"""


# ============================================================
# APP
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
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
                World Bank World Development Indicators (WDI)
            </div>

            <div class="hero-text">
                Explore the relationship between development indicators
                and predicted life expectancy using a machine learning model.
            </div>

        </div>
        """
    )

    # ========================================================
    # MODEL METRICS
    # ========================================================

    with gr.Row():

        gr.HTML(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Model</div>
                <div class='metric-value'>Random Forest</div>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>R²</div>
                <div class='metric-value'>{MODEL_R2:.3f}</div>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>RMSE</div>
                <div class='metric-value'>{MODEL_RMSE:.2f}</div>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>MAE</div>
                <div class='metric-value'>{MODEL_MAE:.2f}</div>
            </div>
            """
        )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    with gr.Row():

        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        with gr.Column(scale=1, elem_classes="card"):

            gr.Markdown("## Development Indicators")

            health_expenditure = gr.Number(
                label="🏥 Health Expenditure per Capita (USD)",
                value=350
            )

            electricity_access = gr.Slider(
                0,
                100,
                value=75,
                step=1,
                label="⚡ Electricity Access (%)"
            )

            primary_completion = gr.Slider(
                0,
                100,
                value=85,
                step=1,
                label="🎓 Primary Completion Rate (%)"
            )

            gdp_per_capita = gr.Number(
                label="💰 GDP per Capita (USD)",
                value=5000
            )

            maternal_mortality = gr.Number(
                label="👩 Maternal Mortality Rate",
                value=150
            )

            clean_water_access = gr.Slider(
                0,
                100,
                value=80,
                step=1,
                label="💧 Clean Water Access (%)"
            )

            predict_button = gr.Button(
                "Predict Life Expectancy",
                elem_id="predict-button"
            )

        # ----------------------------------------------------
        # OUTPUTS
        # ----------------------------------------------------

        with gr.Column(scale=1, elem_classes="card"):

            gr.Markdown("## Prediction Result")

            prediction_card = gr.HTML()

            interpretation = gr.Markdown()

    # ========================================================
    # EXAMPLES
    # ========================================================

    gr.Markdown("## Example Development Profiles")

    gr.Examples(
        examples=[
            [100,30,60,1200,550,45],
            [450,85,88,6500,180,82],
            [4500,99,98,45000,12,99]
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
    # ABOUT MODEL
    # ========================================================

    with gr.Accordion(
        "About This Model",
        open=False
    ):

        gr.Markdown(
            """
### Inputs

- Health Expenditure Per Capita
- Electricity Access
- Primary Completion Rate
- GDP Per Capita
- Maternal Mortality Rate
- Clean Water Access

### Target Variable

Life Expectancy at Birth (Years)

### Dataset

World Bank World Development Indicators (WDI)

### Model

Random Forest Regressor
"""
        )

    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class='footer'>
            AnalystLab Africa Data Science Capstone Project
        </div>
        """
    )

    # ========================================================
    # EVENTS
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
import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
