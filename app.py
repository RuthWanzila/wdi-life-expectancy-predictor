import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os


# ============================================================
# LOAD MODEL
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

    # Apply transformations used during model training
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

    # Prediction
    prediction = float(final_model.predict(input_data)[0])
    prediction = round(prediction, 2)

    # Classification
    if prediction >= 75:
        category = "High Predicted Life Expectancy"
        category_color = "#4ade80"

    elif prediction >= 65:
        category = "Moderate Predicted Life Expectancy"
        category_color = "#fbbf24"

    else:
        category = "Lower Predicted Life Expectancy"
        category_color = "#f87171"

    # ========================================================
    # RESULT CARD
    # ========================================================

    result_card = f"""
    <div class="result-card">

        <div class="result-label">
            ESTIMATED LIFE EXPECTANCY
        </div>

        <div class="result-number">
            {prediction:.2f}
        </div>

        <div class="result-unit">
            YEARS
        </div>

        <div class="result-category"
             style="color:{category_color};
                    border-color:{category_color};">
            {category}
        </div>

    </div>
    """

    # ========================================================
    # INTERPRETATION
    # ========================================================

    interpretation = f"""
    <div class="interpretation">

        <div class="interpretation-title">
            Prediction Summary
        </div>

        <p>
            The model estimates a life expectancy of
            <strong>{prediction:.2f} years</strong>
            based on the six development indicators provided.
        </p>

        <div class="summary-grid">

            <div>
                <span>Model</span>
                <strong>Random Forest</strong>
            </div>

            <div>
                <span>R² Score</span>
                <strong>0.942</strong>
            </div>

            <div>
                <span>MAE</span>
                <strong>1.14 years</strong>
            </div>

            <div>
                <span>RMSE</span>
                <strong>2.02 years</strong>
            </div>

        </div>

    </div>
    """

    return result_card, interpretation


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

/* ==========================================================
   GLOBAL
   ========================================================== */

body {
    background: #020617 !important;
}

.gradio-container {
    background: #020617 !important;
    color: #f8fafc !important;
    font-family: Arial, Helvetica, sans-serif !important;
    max-width: 1200px !important;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    background: linear-gradient(
        135deg,
        #172554 0%,
        #1e3a8a 50%,
        #0f766e 100%
    );

    padding: 32px;
    border-radius: 20px;
    margin-bottom: 22px;

    border: 1px solid #334155;

    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.hero-title {
    color: #ffffff !important;
    font-size: 36px;
    font-weight: 800;
    line-height: 1.2;
}

.hero-subtitle {
    color: #bfdbfe !important;
    font-size: 17px;
    margin-top: 8px;
}

.hero-description {
    color: #dbeafe !important;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 14px;
}


/* ==========================================================
   CARDS
   ========================================================== */

.card {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 18px !important;
    padding: 22px !important;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-title {
    color: #ffffff !important;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 20px;
}

.section-subtitle {
    color: #94a3b8 !important;
    font-size: 13px;
    margin-bottom: 18px;
}


/* ==========================================================
   MODEL METRICS
   ========================================================== */

.metric-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    min-height: 72px;
}

.metric-title {
    color: #60a5fa !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.7px;
}

.metric-value {
    color: #ffffff !important;
    font-size: 21px;
    font-weight: 800;
    margin-top: 5px;
}


/* ==========================================================
   INPUT LABELS
   ========================================================== */

.gradio-container label {
    color: #e2e8f0 !important;
}

.gradio-container label span {
    color: #e2e8f0 !important;
}


/* ==========================================================
   INPUT BOXES
   ========================================================== */

.gradio-container input {
    background: #111827 !important;
    color: #ffffff !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

.gradio-container input:focus {
    border-color: #38bdf8 !important;
}


/* ==========================================================
   NUMBER INPUT TEXT
   ========================================================== */

.gradio-container input[type="number"] {
    color: #ffffff !important;
}


/* ==========================================================
   BUTTON
   ========================================================== */

#predict-button {
    background: linear-gradient(
        135deg,
        #2563eb,
        #0891b2
    ) !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 11px !important;

    font-weight: 800 !important;

    height: 52px;

    margin-top: 12px;

    box-shadow: 0 5px 18px rgba(37,99,235,0.25);
}

#predict-button:hover {
    filter: brightness(1.1);
}


/* ==========================================================
   RESULT CARD
   ========================================================== */

.result-card {
    width: 100%;
    box-sizing: border-box;

    background: linear-gradient(
        135deg,
        #172554 0%,
        #0f766e 100%
    );

    border: 1px solid #475569;
    border-radius: 18px;

    padding: 35px 20px;

    text-align: center;

    box-shadow: 0 8px 25px rgba(0,0,0,0.35);

    margin-bottom: 15px;
}

.result-label {
    color: #bfdbfe !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-number {
    color: #ffffff !important;
    font-size: 56px;
    font-weight: 900;
    line-height: 1;
    margin-top: 12px;
}

.result-unit {
    color: #dbeafe !important;
    font-size: 15px;
    font-weight: 600;
    margin-top: 8px;
}

.result-category {
    display: inline-block;

    margin-top: 18px;

    padding: 8px 16px;

    border-radius: 30px;

    background: rgba(2, 6, 23, 0.55);

    border: 1px solid;

    font-size: 13px;
    font-weight: 800;
}


/* ==========================================================
   INTERPRETATION
   ========================================================== */

.interpretation {
    background: #111827;

    border: 1px solid #334155;

    border-radius: 14px;

    padding: 18px;

    color: #cbd5e1 !important;

    line-height: 1.6;
}

.interpretation-title {
    color: #ffffff !important;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 8px;
}

.interpretation strong {
    color: #ffffff !important;
}

.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;

    gap: 10px;

    margin-top: 15px;
}

.summary-grid div {
    background: #0f172a;

    border: 1px solid #334155;

    border-radius: 9px;

    padding: 10px;

    text-align: center;
}

.summary-grid span {
    display: block;

    color: #94a3b8 !important;

    font-size: 11px;
}

.summary-grid strong {
    display: block;

    color: #f8fafc !important;

    font-size: 14px;

    margin-top: 3px;
}


/* ==========================================================
   EXAMPLES
   ========================================================== */

.examples-title {
    color: #ffffff !important;
    font-size: 20px;
    font-weight: 800;
    margin: 25px 0 12px;
}


/* ==========================================================
   ACCORDION
   ========================================================== */

.gradio-container .accordion {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
}

.gradio-container .accordion button {
    color: #ffffff !important;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    color: #64748b !important;
    margin-top: 25px;
    padding: 15px;
    font-size: 12px;
}

"""


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="Life Expectancy Prediction Dashboard"
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML("""
    <div class="hero">

        <div class="hero-title">
            🌍 Life Expectancy Prediction Dashboard
        </div>

        <div class="hero-subtitle">
            World Bank World Development Indicators
        </div>

        <div class="hero-description">
            Predict life expectancy using health, economic,
            education, and infrastructure indicators.
        </div>

    </div>
    """)


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        Model Performance
    </div>
    """)

    with gr.Row():

        gr.HTML("""
        <div class="metric-card">
            <div class="metric-title">MODEL</div>
            <div class="metric-value">Random Forest</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="metric-card">
            <div class="metric-title">R² SCORE</div>
            <div class="metric-value">{MODEL_R2:.3f}</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">{MODEL_RMSE:.2f}</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="metric-card">
            <div class="metric-title">MAE</div>
            <div class="metric-value">{MODEL_MAE:.2f} yrs</div>
        </div>
        """)


    # ========================================================
    # MAIN DASHBOARD
    # ========================================================

    with gr.Row():

        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.HTML("""
            <div class="section-title">
                Development Indicators
            </div>

            <div class="section-subtitle">
                Enter values for the six indicators used by the model.
            </div>
            """)

            health_expenditure = gr.Number(
                label="🏥 Health Expenditure per Capita (USD)",
                value=350,
                minimum=0
            )

            electricity_access = gr.Number(
                label="⚡ Electricity Access (% of Population)",
                value=75,
                minimum=0,
                maximum=100
            )

            primary_completion = gr.Number(
                label="🎓 Primary Completion Rate (%)",
                value=85,
                minimum=0,
                maximum=100
            )

            gdp_per_capita = gr.Number(
                label="💰 GDP per Capita (USD)",
                value=5000,
                minimum=0
            )

            maternal_mortality = gr.Number(
                label="👩 Maternal Mortality Rate (per 100,000)",
                value=150,
                minimum=0
            )

            clean_water_access = gr.Number(
                label="💧 Clean Water Access (% of Population)",
                value=80,
                minimum=0,
                maximum=100
            )

            predict_button = gr.Button(
                "Predict Life Expectancy",
                variant="primary",
                size="lg",
                elem_id="predict-button"
            )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.HTML("""
            <div class="section-title">
                Prediction Result
            </div>
            """)

            prediction_card = gr.HTML("""
            <div class="result-card">

                <div class="result-label">
                    PREDICTION RESULT
                </div>

                <div style="
                    color:#cbd5e1;
                    font-size:15px;
                    margin-top:15px;
                    line-height:1.6;
                ">
                    Enter the development indicators
                    and click <strong>Predict Life Expectancy</strong>.
                </div>

            </div>
            """)

            interpretation = gr.HTML("""
            <div class="interpretation">

                <div class="interpretation-title">
                    Prediction Summary
                </div>

                <p>
                    Your prediction will appear here after
                    running the model.
                </p>

            </div>
            """)


    # ========================================================
    # EXAMPLE SCENARIOS
    # ========================================================

    gr.HTML("""
    <div class="examples-title">
        Example Development Profiles
    </div>
    """)

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
    # ABOUT
    # ========================================================

    with gr.Accordion(
        "About This Model",
        open=False
    ):

        gr.Markdown("""
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

GDP per capita and health expenditure were log-transformed
before modeling.

The model was evaluated using an 80/20 train-test split.
""")


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div class="footer">
        Life Expectancy Prediction • World Bank WDI
    </div>
    """)


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
