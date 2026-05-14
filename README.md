# Global Commodity Macro Analysis

This project analyzes relationships between commodity prices and macroeconomic indicators using public datasets from FRED and the World Bank.

The analysis focuses on two questions:

- how oil prices relate to inflation
- whether broader commodity-price cycles move with real GDP growth

The repository includes a reproducible notebook workflow, exported figures, and a lightweight Streamlit dashboard for interactive exploration.

---

## Project Overview

The project combines multiple macroeconomic time series with different frequencies and reporting structures, including daily oil prices, monthly inflation data, and quarterly GDP data.

Main components include:

- oil price and inflation analysis
- headline vs core inflation comparison
- lag and rolling-correlation analysis
- simple regression models with macro controls
- commodity growth vs GDP growth analysis
- dashboard-based visualization and reporting

---

## Research Questions

1. How strongly are oil prices associated with inflation?
2. Do commodity-price cycles move with real GDP growth?

---

## Preview

### Oil Prices vs Inflation

![Oil Prices vs Inflation](outputs/figures/oil_vs_inflation_dual_axis.png)

### Headline vs Core Inflation

![Headline vs Core Inflation](outputs/figures/headline_vs_core_inflation.png)

### Rolling Correlation

![Rolling Correlation](outputs/figures/rolling_correlation_oil_inflation.png)

### Commodity Prices vs GDP Growth

![Commodity Prices vs GDP Growth](outputs/figures/commodity_vs_gdp_growth_timeseries.png)

---

## Main Findings

- Oil prices are more strongly associated with headline inflation than core inflation
- The oil-inflation relationship changes over time rather than remaining constant
- The strongest oil-headline relationship appears at shorter lag horizons
- Oil prices remain positively associated with inflation after controlling for the federal funds rate
- Commodity-price growth and GDP growth remain positively associated in the quarterly sample

These results should be interpreted as reduced-form empirical relationships rather than causal estimates.

---

## Data Sources

The project uses:

- FRED `DCOILWTICO`: WTI crude oil prices
- FRED `CPIAUCSL`: headline CPI
- FRED `CPILFESL`: core CPI
- FRED `FEDFUNDS`: effective federal funds rate
- FRED `GDPC1`: real GDP
- World Bank Pink Sheet commodity indices

---

## Methods

The workflow includes:

- multi-frequency time-series alignment
- inflation and growth-rate construction
- lag correlation analysis
- rolling-window correlation analysis
- OLS regression analysis
- monthly-to-quarterly aggregation
- dashboard visualization using Streamlit

---

## Repository Structure

```text
commodity-macro-analysis/
├── app.py
├── data/
├── outputs/
│   ├── figures/
│   └── notebooks/
│       └── macro_analysis.ipynb
├── requirements.txt
└── README.md