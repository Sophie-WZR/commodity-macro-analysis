from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Global Commodity Macro Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_base_dir() -> Path:
    base = Path.cwd().resolve()
    for candidate in [base, *base.parents]:
        if (candidate / "data").exists() and (candidate / "outputs").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root with data/ and outputs/.")


BASE_DIR = resolve_base_dir()
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


@st.cache_data
def load_fred_series(series_id: str, value_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{series_id}.csv"
    df = pd.read_csv(path)
    value_col = [col for col in df.columns if col != "observation_date"][0]
    df = df.rename(columns={"observation_date": "DATE", value_col: value_name})
    df["DATE"] = pd.to_datetime(df["DATE"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df[["DATE", value_name]].dropna().sort_values("DATE").reset_index(drop=True)


@st.cache_data
def load_commodity_index() -> pd.DataFrame:
    path = DATA_DIR / "CMO-Historical-Data-Monthly.xlsx"
    raw = pd.read_excel(path, sheet_name="Monthly Indices", header=None)
    header_row = raw.iloc[5].copy()
    header_row.iloc[0] = "DATE"
    commodity = raw.iloc[9:].copy()
    commodity.columns = header_row
    commodity = commodity.loc[:, ["DATE", "Total Index"]]
    commodity = commodity.rename(columns={"Total Index": "commodity_index"})
    commodity["DATE"] = pd.to_datetime(
        commodity["DATE"].astype(str), format="%YM%m", errors="coerce"
    )
    commodity["commodity_index"] = pd.to_numeric(
        commodity["commodity_index"], errors="coerce"
    )
    return commodity.dropna().sort_values("DATE").reset_index(drop=True)


@st.cache_data
def build_analysis_frames() -> dict[str, pd.DataFrame | float | int]:
    oil = load_fred_series("DCOILWTICO", "oil_price")
    headline = load_fred_series("CPIAUCSL", "headline_cpi")
    core = load_fred_series("CPILFESL", "core_cpi")
    fed = load_fred_series("FEDFUNDS", "fed_funds")
    gdp = load_fred_series("GDPC1", "real_gdp")
    commodity = load_commodity_index()

    oil_monthly = oil.set_index("DATE").resample("MS").mean().reset_index()
    headline["headline_inflation_yoy"] = headline["headline_cpi"].pct_change(
        12, fill_method=None
    )
    core["core_inflation_yoy"] = core["core_cpi"].pct_change(12, fill_method=None)

    oil_inflation = (
        oil_monthly.merge(
            headline[["DATE", "headline_inflation_yoy"]], on="DATE", how="inner"
        )
        .merge(core[["DATE", "core_inflation_yoy"]], on="DATE", how="inner")
        .merge(fed, on="DATE", how="inner")
        .dropna()
        .reset_index(drop=True)
    )
    for lag in [1, 3, 6]:
        oil_inflation[f"oil_price_lag{lag}"] = oil_inflation["oil_price"].shift(lag)

    oil_inflation["rolling_corr_12m_headline"] = (
        oil_inflation["oil_price"].rolling(12).corr(oil_inflation["headline_inflation_yoy"])
    )
    oil_inflation["rolling_corr_12m_core"] = (
        oil_inflation["oil_price"].rolling(12).corr(oil_inflation["core_inflation_yoy"])
    )

    commodity_quarterly = (
        commodity.set_index("DATE").resample("QS").mean().reset_index().sort_values("DATE")
    )
    commodity_quarterly["commodity_growth"] = commodity_quarterly["commodity_index"].pct_change(
        fill_method=None
    )
    gdp["gdp_growth"] = gdp["real_gdp"].pct_change(fill_method=None)

    commodity_gdp = (
        commodity_quarterly.merge(gdp[["DATE", "gdp_growth"]], on="DATE", how="inner")
        .dropna()
        .reset_index(drop=True)
    )

    headline_corr = oil_inflation[["oil_price", "headline_inflation_yoy"]].corr().iloc[0, 1]
    core_corr = oil_inflation[["oil_price", "core_inflation_yoy"]].corr().iloc[0, 1]
    commodity_corr = commodity_gdp[["commodity_growth", "gdp_growth"]].corr().iloc[0, 1]

    lag_table = []
    for lag in [1, 3, 6]:
        lag_corr = (
            oil_inflation[[f"oil_price_lag{lag}", "headline_inflation_yoy"]]
            .dropna()
            .corr()
            .iloc[0, 1]
        )
        lag_table.append({"Lag (months)": lag, "Correlation": lag_corr})
    lag_df = pd.DataFrame(lag_table)

    return {
        "oil_inflation": oil_inflation,
        "commodity_gdp": commodity_gdp,
        "headline_corr": headline_corr,
        "core_corr": core_corr,
        "commodity_corr": commodity_corr,
        "lag_df": lag_df,
    }


def figure_path(name: str) -> Path:
    return FIGURES_DIR / name


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f1e8;
            --panel: rgba(255,255,255,0.76);
            --ink: #1f2a2a;
            --muted: #55615b;
            --line: rgba(31, 42, 42, 0.09);
            --accent: #b65a3a;
            --accent-2: #3f6b66;
            --accent-3: #d6a449;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214,164,73,0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(63,107,102,0.16), transparent 24%),
                linear-gradient(180deg, #f7f3ea 0%, #ece5d8 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem 2.1rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,248,238,0.72));
            backdrop-filter: blur(10px);
            box-shadow: 0 16px 50px rgba(45, 39, 31, 0.08);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: var(--accent-2);
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .hero-title {
            font-size: 3rem;
            line-height: 1.0;
            font-weight: 700;
            margin: 0 0 0.8rem 0;
            color: var(--ink);
        }

        .hero-subtitle {
            font-size: 1.05rem;
            line-height: 1.7;
            color: var(--muted);
            max-width: 850px;
            margin-bottom: 0.4rem;
        }

        .glass-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            background: var(--panel);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 28px rgba(36, 32, 26, 0.06);
            height: 100%;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(247,240,230,0.76));
            min-height: 132px;
            box-shadow: 0 10px 24px rgba(36, 32, 26, 0.05);
        }

        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }

        .metric-value {
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.55rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.98rem;
            margin-bottom: 0.9rem;
        }

        div[data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid rgba(31, 42, 42, 0.08);
            box-shadow: 0 14px 40px rgba(36, 32, 26, 0.08);
        }

        .stDataFrame, div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
frames = build_analysis_frames()
oil_inflation = frames["oil_inflation"]
lag_df = frames["lag_df"]

latest_date = oil_inflation["DATE"].max().strftime("%b %Y")
best_lag = int(lag_df.loc[lag_df["Correlation"].abs().idxmax(), "Lag (months)"])

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Macroeconomic Research Showcase</div>
        <div class="hero-title">Global Commodity Macro Analysis</div>
        <div class="hero-subtitle">
            A portfolio-style dashboard built from FRED and World Bank data to present the full
            workflow, visual evidence, and key findings behind a commodity-focused macro project.
            The emphasis is on oil-inflation pass-through, time-varying macro relationships, and
            the link between commodity cycles and output growth.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Project Lens")
    st.write("This dashboard presents the project as a compact research briefing rather than a BI dashboard.")
    st.markdown("### Data Sources")
    st.write("- FRED: WTI oil, headline CPI, core CPI, fed funds, real GDP")
    st.write("- World Bank Pink Sheet: Total Commodity Index")
    st.markdown("### Workflow")
    st.write("1. Load multi-frequency data")
    st.write("2. Standardize dates and numeric fields")
    st.write("3. Engineer inflation and growth rates")
    st.write("4. Compare headline vs core inflation")
    st.write("5. Test lag and rolling relationships")
    st.write("6. Summarize regression and macro narrative")
    st.markdown("### Current Sample")
    st.write(f"Oil-inflation sample through `{latest_date}`")


cols = st.columns(4)
with cols[0]:
    metric_card(
        "Oil vs Headline",
        f"{frames['headline_corr']:.3f}",
        "Stronger co-movement with headline CPI than with core CPI.",
    )
with cols[1]:
    metric_card(
        "Oil vs Core",
        f"{frames['core_corr']:.3f}",
        "Lower than headline, consistent with more direct energy pass-through.",
    )
with cols[2]:
    metric_card(
        "Best Lag",
        f"{best_lag}M",
        "The strongest oil-headline link appears at a short horizon.",
    )
with cols[3]:
    metric_card(
        "Commodity vs GDP",
        f"{frames['commodity_corr']:.3f}",
        "Broad commodity growth stays positively associated with output growth.",
    )

st.markdown("")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Research Question</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The project asks whether oil prices act as a visible transmission channel into inflation and whether broader commodity cycles move with real economic activity. The design intentionally stays interpretable: a small number of macro series, clean transformations, and reduced-form evidence that can be explained clearly in an interview or research setting.</div>',
        unsafe_allow_html=True,
    )
    st.image(str(figure_path("oil_vs_inflation_dual_axis.png")), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why This Is Job-Relevant</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-copy">
        This dashboard highlights the parts of the project that map cleanly to a macro research analyst workflow:
        loading public data, aligning multiple frequencies, creating interpretable features, presenting strong visuals,
        and turning reduced-form results into a concise macro narrative.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
        The strongest headline from the project is that oil is more closely linked to headline inflation than to core inflation,
        while the rolling and lag sections show that the relationship is time-varying rather than mechanically constant.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["Oil and Inflation", "Commodity and GDP", "Project Workflow"])

with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.image(str(figure_path("headline_vs_core_inflation.png")), use_container_width=True)
        st.image(str(figure_path("oil_headline_core_scatter.png")), use_container_width=True)
    with c2:
        st.image(str(figure_path("rolling_correlation_oil_inflation.png")), use_container_width=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Lag Structure</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Short-horizon lag comparisons help show whether pass-through is immediate or delayed. In this sample, the strongest oil-headline relationship shows up at the 1-month horizon, then fades at 3 and 6 months.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            lag_df.style.format({"Correlation": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.image(str(figure_path("commodity_vs_gdp_growth_timeseries.png")), use_container_width=True)
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Macro Readout</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">The commodity-growth and GDP-growth block keeps the second part of the project intentionally simple. It broadens the story from inflation pass-through into macro cycle co-movement and shows that the project is not narrowly about one commodity series alone.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="section-copy"><strong>Current correlation:</strong> {frames["commodity_corr"]:.3f}. This is directionally positive and consistent with the idea that stronger commodity cycles often coincide with stronger real activity.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.image(str(figure_path("commodity_vs_gdp_growth_scatter.png")), use_container_width=True)

with tab3:
    c1, c2 = st.columns([0.9, 1.1], gap="large")
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Workflow Design</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-copy">
            <strong>1. Source data:</strong> FRED and World Bank series stored locally.
            </div>
            <div class="section-copy">
            <strong>2. Frequency alignment:</strong> daily oil aggregated to monthly; monthly commodity index aggregated to quarterly.
            </div>
            <div class="section-copy">
            <strong>3. Feature engineering:</strong> headline and core inflation, commodity growth, GDP growth, lags, rolling correlations.
            </div>
            <div class="section-copy">
            <strong>4. Interpretation:</strong> reduced-form correlations and regressions translated into macro narrative.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Portfolio Positioning</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-copy">
            This project is strongest when framed as a compact research briefing:
            commodity-market signals, inflation pass-through, time-varying macro relationships,
            and a reproducible workflow built from public data.
            </div>
            <div class="section-copy">
            It is not presented as a causal macro model. Instead, it demonstrates the practical
            skills most relevant to macro research support: data handling, charting, empirical
            structure, and concise interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("")
st.caption("Built with Streamlit to present the workflow and findings of the macro analysis project in a research-briefing format.")
