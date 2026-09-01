# Life Expectancy Prediction Using World Bank WDI

## Project Overview

This project investigates factors influencing life expectancy using selected indicators from the World Bank World Development Indicators (WDI) dataset.

The project follows an end-to-end data science workflow covering data preparation, exploratory data analysis, machine learning, model evaluation, interpretation, and deployment.

## Problem Statement

Life expectancy varies considerably across countries and over time. Understanding the development indicators associated with these differences can provide useful insights for public health and development planning.

This project asks:

> **How well can life expectancy be predicted using selected health, economic, education, and infrastructure indicators from the World Bank WDI dataset?**

## Dataset

The project uses the **World Bank World Development Indicators (WDI)** dataset.

**Source:** World Bank World Development Indicators

The analysis focuses on the period **2000–2022** and uses the following indicators:

* Life Expectancy
* Health Expenditure per Capita
* Electricity Access
* Primary Completion Rate
* GDP per Capita
* Maternal Mortality Rate
* Clean Water Access

## Methodology

The project follows these steps:

1. Data collection and understanding
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Machine learning model development
6. Model evaluation
7. Feature importance analysis
8. Insights and recommendations
9. Model deployment

### Data Preparation

The WDI dataset was filtered to the selected indicators and transformed into a country-year format.

Missing values were handled using forward/backward filling within countries followed by median imputation for remaining missing predictor values.

Extreme predictor values were capped at the 1st and 99th percentiles.

GDP per capita and health expenditure per capita were log-transformed to reduce skewness.

## Exploratory Analysis

The analysis identified strong relationships between life expectancy and several development indicators.

Key relationships included:

* Electricity access showed a strong positive relationship with life expectancy.
* Maternal mortality showed a strong negative relationship with life expectancy.
* Clean water access and primary completion rates showed positive relationships with life expectancy.
* GDP per capita also showed a positive relationship, with diminishing gains at higher income levels.

These relationships represent statistical associations and should not be interpreted as proof of causation.

## Machine Learning

Three regression models were developed:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

### Model Performance

| Model             |  MAE |   MSE | RMSE |         R² |
| ----------------- | ---: | ----: | ---: | ---------: |
| Random Forest     | 1.14 |  4.09 | 2.02 | **0.9422** |
| XGBoost           | 1.24 |  4.27 | 2.07 |     0.9396 |
| Linear Regression | 2.46 | 11.61 | 3.41 |     0.8357 |

The **Random Forest Regressor** achieved the best overall performance.

Its R² score of **0.9422** indicates strong predictive performance on the test dataset, while its MAE of approximately **1.14 years** means that predictions differed from actual values by about 1.14 years on average.

## Key Insights

The analysis suggests that health, infrastructure, education, and economic development indicators contain substantial predictive information about life expectancy.

Maternal mortality, economic conditions, electricity access, clean water access, health expenditure, and education were among the important variables considered by the model.

## Recommendations

* Strengthen maternal and healthcare services, particularly in areas with high maternal mortality.
* Improve access to clean water and reliable electricity as part of broader public health and infrastructure development.
* Support education and economic development initiatives that contribute to improved living conditions.
* Use predictive models as analytical tools alongside local socioeconomic and health information when making policy decisions.

## Deployment

The trained Random Forest model was deployed using **Gradio** to provide an interactive life expectancy prediction interface.

Users can enter values for the selected development indicators and receive an estimated life expectancy.

**Deployment:** Add the live Gradio/hosting link here after deployment.

## Project Files

```text
life-expectancy-wdi/
│
├── notebook/
│   └── Life_Expectancy_WDI_Capstone.ipynb
│
├── deployment/
│   ├── app.py
│   ├── requirements.txt
│   └── random_forest_model.joblib
│
├── README.md
├── .gitignore
└── LICENSE
```

## Limitations

* The analysis uses a selected subset of WDI indicators rather than all available development indicators.
* Missing observations required imputation.
* The model identifies predictive relationships rather than causal effects.
* A random train-test split was used and may not fully account for the temporal structure of the data.

## Future Work

Future improvements could include:

* Time-based model validation
* Hyperparameter tuning
* Additional WDI indicators
* SHAP-based model explainability
* Country-level comparison and scenario analysis

## Data Source

World Bank World Development Indicators:
https://datatopics.worldbank.org/world-development-indicators/
