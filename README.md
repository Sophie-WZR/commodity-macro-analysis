# Global Commodity Macro Analysis

A compact macroeconomic research project using real-world data from FRED and the World Bank to study:

- oil prices and inflation
- commodity prices and GDP growth

The project is designed as a reproducible notebook-style analysis with a focus on economic interpretation, clear visuals, and policy-relevant framing.

## Project Goal

This project asks two core macro questions:

1. How strongly are oil prices associated with inflation?
2. Do broader commodity price cycles move with real GDP growth?

The analysis emphasizes:

- time series alignment across daily, monthly, and quarterly data
- inflation and growth feature engineering
- correlation analysis
- simple regression
- lag effects
- rolling relationships over time

## Data Sources

The project uses publicly available macroeconomic datasets:

- FRED `DCOILWTICO`: WTI crude oil prices
- FRED `CPIAUCSL`: headline CPI
- FRED `CPILFESL`: core CPI
- FRED `FEDFUNDS`: effective federal funds rate
- FRED `GDPC1`: real GDP
- World Bank Pink Sheet: monthly commodity price indices

## Main Analysis

### 1. Oil Prices vs Inflation

The first section studies the relationship between oil prices and inflation using monthly data.

It includes:

- monthly oil price aggregation
- headline CPI year-over-year inflation
- core CPI year-over-year inflation
- oil vs headline inflation comparison
- oil vs core inflation comparison
- lag analysis at 1, 3, and 6 months
- 12-month rolling correlation
- simple OLS regression
- oil plus monetary-policy control regression

### 2. Commodity Prices vs GDP Growth

The second section studies whether broad commodity price conditions move with real economic activity.

It includes:

- extraction of the World Bank Total Commodity Index
- monthly-to-quarterly aggregation
- quarterly commodity growth
- quarterly real GDP growth
- correlation analysis
- simple regression of GDP growth on commodity growth

## Key Findings

In the current sample:

- oil prices are more strongly associated with headline inflation than with core inflation
- the oil-inflation relationship is positive and economically intuitive in simple regressions
- the relationship is time-varying rather than constant over time
- the oil coefficient remains positive after controlling for the federal funds rate
- commodity growth and GDP growth are positively associated in the quarterly sample

These findings should be interpreted as reduced-form empirical relationships rather than causal estimates.

## Why This Project Matters

This project is relevant for macroeconomic and commodity-market research roles because it demonstrates the ability to:

- work with real-world macro and commodity datasets
- clean and align multi-frequency time series
- build interpretable economic features
- present empirical results through tables and charts
- connect quantitative results to macroeconomic intuition

## Repository Structure

- [outputs/notebooks/macro_analysis_notebook_style.py](/Users/wangzhuoran/Desktop/global-commodity-macro-analysis/outputs/notebooks/macro_analysis_notebook_style.py): main notebook-style analysis script
- [outputs/figures](/Users/wangzhuoran/Desktop/global-commodity-macro-analysis/outputs/figures): exported charts
- [data](/Users/wangzhuoran/Desktop/global-commodity-macro-analysis/data): local source datasets

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib statsmodels openpyxl
```

Run the notebook-style script:

```bash
python outputs/notebooks/macro_analysis_notebook_style.py
```

Run the project dashboard:

```bash
streamlit run app.py
```

Figures will be saved to:

```text
outputs/figures/
```

The dashboard presents the same project as a polished research briefing with:

- a high-level workflow view
- key empirical findings
- figure-driven storytelling
- a portfolio-friendly presentation layer

## Packaging Angle

If you are using this project in a job application, the strongest framing is:

- commodity-market analysis linked to macro outcomes
- inflation pass-through from oil prices
- comparison of headline and core inflation
- time-varying macro relationships
- clear, reproducible research workflow

For ready-to-use resume bullets and interview framing, see:

- [project_packaging.md](/Users/wangzhuoran/Desktop/global-commodity-macro-analysis/outputs/project_packaging.md)
