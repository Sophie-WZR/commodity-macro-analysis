from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Global Commodity Macro Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def resolve_base_dir() -> Path:
    base = Path.cwd().resolve()
    for candidate in [base, *base.parents]:
        if (candidate / "data").exists() and (candidate / "outputs").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root with data/ and outputs/.")


BASE_DIR = resolve_base_dir()
DATA_DIR = BASE_DIR / "data"

PALETTE = {
    "ink": "#1e252f",
    "muted": "#626f7d",
    "line": "rgba(30, 37, 47, 0.12)",
    "accent": "#0057a8",
    "accent_soft": "#2f6da7",
    "sage": "#5d738d",
    "gold": "#7d8ea5",
    "cream": "#ffffff",
    "bg": "#f6f8fb",
    "plot_bg": "#ffffff",
}


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


def run_ols(
    y: pd.Series, X: pd.DataFrame, cov_type: str = "HAC", cov_kwds: dict | None = None
):
    X = sm.add_constant(X)
    return sm.OLS(y, X, missing="drop").fit(cov_type=cov_type, cov_kwds=cov_kwds)


@st.cache_data
def build_analysis_frames() -> dict[str, object]:
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
        commodity_quarterly.merge(
            gdp[["DATE", "real_gdp", "gdp_growth"]], on="DATE", how="inner"
        )
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

    headline_model = run_ols(
        oil_inflation["headline_inflation_yoy"],
        oil_inflation[["oil_price"]],
        cov_kwds={"maxlags": 3},
    )
    core_model = run_ols(
        oil_inflation["core_inflation_yoy"],
        oil_inflation[["oil_price"]],
        cov_kwds={"maxlags": 3},
    )
    controlled_model = run_ols(
        oil_inflation["headline_inflation_yoy"],
        oil_inflation[["oil_price", "fed_funds"]],
        cov_kwds={"maxlags": 3},
    )
    commodity_model = run_ols(
        commodity_gdp["gdp_growth"], commodity_gdp[["commodity_growth"]], cov_type="nonrobust"
    )

    return {
        "oil_inflation": oil_inflation,
        "commodity_gdp": commodity_gdp,
        "headline_corr": headline_corr,
        "core_corr": core_corr,
        "commodity_corr": commodity_corr,
        "lag_df": lag_df,
        "headline_model": headline_model,
        "core_model": core_model,
        "controlled_model": controlled_model,
        "commodity_model": commodity_model,
    }


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: #f6f8fb;
            --panel: rgba(255, 255, 255, 0.92);
            --panel-strong: rgba(255, 255, 255, 1);
            --ink: {PALETTE["ink"]};
            --muted: {PALETTE["muted"]};
            --line: {PALETTE["line"]};
            --accent: {PALETTE["accent"]};
            --accent-2: {PALETTE["sage"]};
            --accent-3: {PALETTE["gold"]};
        }}

        .stApp {{
            background: #f6f8fb;
            color: var(--ink);
        }}

        header[data-testid="stHeader"], .stAppToolbar {{
            display: none;
        }}

        [data-testid="collapsedControl"] {{
            display: none;
        }}

        .block-container {{
            max-width: 1120px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stSidebar"] {{
            display: none;
        }}

        .hero {{
            padding: 0.05rem 0 0.15rem 0;
            margin-bottom: 0.15rem;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.9fr);
            gap: 1.1rem;
            align-items: end;
        }}

        .hero-kicker {{
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.7rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.28rem;
        }}

        .hero-title {{
            font-size: 3.18rem;
            line-height: 0.98;
            font-weight: 740;
            letter-spacing: -0.03em;
            margin: 0 0 0.3rem 0;
            color: var(--ink);
            max-width: 640px;
        }}

        .hero-subtitle {{
            font-size: 1.04rem;
            line-height: 1.42;
            color: var(--muted);
            max-width: 600px;
            margin-bottom: 0.4rem;
        }}

        .hero-core {{
            color: var(--ink);
            font-size: 1.5rem;
            line-height: 1.22;
            margin-top: 0.18rem;
            max-width: 440px;
            font-weight: 660;
            letter-spacing: -0.02em;
        }}

        .hero-sidecar {{
            border-radius: 4px;
            background: rgba(0, 87, 168, 0.03);
            padding: 0.45rem 0 0.15rem 0.95rem;
            border-left: 2px solid rgba(0, 87, 168, 0.16);
        }}

        .hero-sidecar-title {{
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .hero-sidecar-value {{
            font-size: 2.95rem;
            line-height: 1;
            font-weight: 760;
            color: var(--accent);
            margin-bottom: 0.12rem;
        }}

        .hero-sidecar-copy {{
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.36;
            max-width: 220px;
        }}

        .takeaway-strip {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(30, 37, 47, 0.12);
        }}

        .takeaway-item {{
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 0.45rem;
            align-items: start;
        }}

        .takeaway-index {{
            color: rgba(0, 87, 168, 0.2);
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            line-height: 1;
            padding-top: 0.02rem;
        }}

        .takeaway-copy {{
            color: var(--ink);
            font-size: 0.92rem;
            line-height: 1.34;
            max-width: 290px;
        }}

        .findings-list {{
            margin: 0;
            padding-left: 1rem;
            color: var(--ink);
            font-size: 0.96rem;
            line-height: 1.45;
        }}

        .findings-list li {{
            margin-bottom: 0.4rem;
        }}

        .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.3rem;
            margin-bottom: 0.35rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.42rem;
            border: 1px solid var(--line);
            border-radius: 3px;
            background: #ffffff;
            color: var(--muted);
            font-size: 0.8rem;
        }}

        .selection-context, .inline-context {{
            margin-top: 0.2rem;
            color: var(--muted);
            font-size: 0.84rem;
            letter-spacing: 0.01em;
        }}

        .section-wrap {{
            margin-top: 1.35rem;
            margin-bottom: 0.15rem;
        }}

        .module {{
            margin-top: 1.15rem;
            padding-top: 0.15rem;
            border-top: 1px solid rgba(30, 37, 47, 0.08);
            animation: moduleFadeIn 0.55s ease both;
        }}

        .module:first-of-type {{
            border-top: none;
            margin-top: 1rem;
        }}

        .module-head {{
            margin-bottom: 0.5rem;
        }}

        .module-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            align-items: end;
            margin: 0.45rem 0 0.25rem 0;
        }}

        .module-context {{
            margin: 0.1rem 0 0.55rem 0;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.4;
        }}

        @keyframes moduleFadeIn {{
            from {{
                opacity: 0;
                transform: translateY(8px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .section-kicker {{
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.68rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .section-title {{
            font-size: 1.52rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.18rem;
            letter-spacing: -0.02em;
        }}

        .section-subtitle {{
            color: var(--muted);
            line-height: 1.36;
            font-size: 0.84rem;
            max-width: 850px;
            margin-bottom: 0.2rem;
        }}

        .metric-card {{
            border: none;
            border-radius: 4px;
            padding: 0.2rem 0 0.05rem 0;
            height: 132px;
            background: transparent;
            box-shadow: none;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .metric-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 0.45rem;
            min-height: 2.1rem;
        }}

        .metric-value {{
            font-size: 2.3rem;
            line-height: 1;
            font-weight: 780;
            color: var(--ink);
            margin-bottom: 0.2rem;
        }}

        .metric-note {{
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.34;
            min-height: 2.2rem;
            padding-top: 0.45rem;
            border-top: 1px solid rgba(30, 37, 47, 0.08);
        }}

        .comparison-card {{
            padding: 0.1rem 0 0.18rem 0;
            min-height: 132px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .comparison-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
            align-items: end;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid rgba(30, 37, 47, 0.08);
        }}

        .comparison-item-title {{
            font-size: 0.74rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.18rem;
        }}

        .comparison-item-value {{
            font-size: 2.15rem;
            line-height: 1;
            font-weight: 760;
            color: var(--ink);
        }}

        .comparison-footnote {{
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.55rem;
        }}

        .note-grid-card {{
            min-height: 88px;
            display: flex;
            align-items: flex-start;
        }}

        .compact-note {{
            border-top: 1px solid rgba(30, 37, 47, 0.08);
            padding-top: 0.45rem;
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.4;
        }}

        .key-inline {{
            border-top: 1px solid rgba(30, 37, 47, 0.14);
            padding-top: 0.7rem;
            margin-top: 0.5rem;
            max-width: 920px;
        }}

        .key-inline ul {{
            margin: 0;
            padding-left: 1.05rem;
        }}

        .key-inline li {{
            margin-bottom: 0.38rem;
            color: var(--ink);
            font-size: 0.94rem;
            line-height: 1.42;
        }}

        .insight-list {{
            margin: 0;
            padding-left: 1.1rem;
            color: var(--muted);
            line-height: 1.5;
        }}

        .insight-list li {{
            margin-bottom: 0.5rem;
        }}

        .chart-card {{
            border-top: 1px solid rgba(30, 37, 47, 0.12);
            border-right: none;
            border-left: none;
            border-bottom: none;
            border-radius: 0;
            padding: 0.5rem 0 0 0;
            background: transparent;
            box-shadow: none;
        }}

        .chart-head {{
            padding: 0.08rem 0 0.32rem 0;
        }}

        .chart-title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.14rem;
        }}

        .chart-subtitle {{
            font-size: 0.8rem;
            line-height: 1.36;
            color: var(--muted);
        }}

        div[data-baseweb="tab-list"] {{
            gap: 1.15rem;
            align-items: flex-end;
            border-bottom: 1px solid rgba(30, 37, 47, 0.10);
            padding-bottom: 0.15rem;
            margin-bottom: 0.6rem;
        }}

        div[data-baseweb="tab-border"] {{
            display: none;
        }}

        div[data-baseweb="tab-highlight"] {{
            display: none;
        }}

        button[data-baseweb="tab"] {{
            background: transparent;
            border: none;
            border-bottom: 1.5px solid transparent;
            border-radius: 0;
            color: rgba(30, 37, 47, 0.58);
            padding: 0 0 0.55rem 0;
            min-height: auto;
            font-size: 0.96rem;
            font-weight: 560;
            letter-spacing: -0.01em;
            box-shadow: none;
            transition:
                color 0.18s ease,
                border-color 0.18s ease,
                transform 0.18s ease;
        }}

        button[data-baseweb="tab"]:hover {{
            background: transparent;
            color: rgba(30, 37, 47, 0.82);
        }}

        button[data-baseweb="tab"]:focus {{
            background: transparent;
            box-shadow: none;
            outline: none;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background: transparent;
            color: var(--ink);
            font-weight: 680;
            border-bottom-color: rgba(0, 87, 168, 0.88);
        }}

        html {{
            scroll-behavior: smooth;
        }}

        .source-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .source-table th, .source-table td {{
            border-bottom: 1px solid var(--line);
            text-align: left;
            padding: 0.7rem 0.4rem;
            color: var(--muted);
            font-size: 0.92rem;
        }}

        .source-table th {{
            color: var(--ink);
            font-weight: 700;
        }}

        .divider {{
            height: 1px;
            background: rgba(17, 17, 17, 0.1);
            margin: 0.85rem 0 0.35rem 0;
        }}

        .stCaption {{
            color: var(--muted);
            font-size: 0.88rem;
        }}

        .source-list {{
            margin: 0.05rem 0 0 0;
            padding-left: 1rem;
            color: rgba(98, 111, 125, 0.9);
            font-size: 0.84rem;
            line-height: 1.48;
        }}

        .source-list li {{
            margin-bottom: 0.34rem;
        }}

        .appendix-block {{
            padding-top: 0.2rem;
        }}

        .appendix-title {{
            color: var(--ink);
            font-size: 1rem;
            font-weight: 660;
            margin-bottom: 0.25rem;
        }}

        .appendix-copy {{
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
            max-width: 520px;
        }}

        @media (max-width: 900px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
            }}

            .takeaway-strip {{
                grid-template-columns: 1fr;
            }}

            .comparison-grid {{
                grid-template-columns: 1fr;
                gap: 0.55rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-wrap module-head">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def module_open() -> None:
    st.markdown('<section class="module">', unsafe_allow_html=True)


def module_close() -> None:
    st.markdown("</section>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def comparison_card(
    label: str,
    left_title: str,
    left_value: str,
    mid_title: str,
    mid_value: str,
    right_title: str,
    right_value: str,
    footnote: str,
) -> None:
    st.markdown(
        f"""
        <div class="comparison-card">
            <div>
                <div class="comparison-label">{label}</div>
                <div class="comparison-grid">
                    <div>
                        <div class="comparison-item-title">{left_title}</div>
                        <div class="comparison-item-value">{left_value}</div>
                    </div>
                    <div>
                        <div class="comparison-item-title">{mid_title}</div>
                        <div class="comparison-item-value">{mid_value}</div>
                    </div>
                    <div>
                        <div class="comparison-item-title">{right_title}</div>
                        <div class="comparison-item-value">{right_value}</div>
                    </div>
                </div>
            </div>
            <div class="comparison-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def narrative_card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title" style="font-size:1.35rem;">{title}</div>
            <div class="section-subtitle" style="max-width:none; margin-bottom:0;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_frame(title: str, subtitle: str, fig: go.Figure, key: str) -> None:
    st.markdown(
        f"""
        <div class="chart-card">
            <div class="chart-head">
                <div class="chart-title">{title}</div>
                <div class="chart-subtitle">{subtitle}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
    st.markdown("</div>", unsafe_allow_html=True)


def badge_row(items: list[str]) -> None:
    html = "".join([f'<span class="badge">{item}</span>' for item in items])
    st.markdown(f'<div class="badge-row">{html}</div>', unsafe_allow_html=True)


def apply_plot_style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=22, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["plot_bg"],
        font=dict(color=PALETTE["ink"], size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.0)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)", bordercolor=PALETTE["line"], font_color=PALETTE["ink"]
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=PALETTE["line"],
        tickfont=dict(color=PALETTE["muted"]),
        title_font=dict(color=PALETTE["muted"]),
    )
    fig.update_yaxes(
        gridcolor="rgba(31,48,45,0.04)",
        zeroline=False,
        linecolor=PALETTE["line"],
        tickfont=dict(color=PALETTE["muted"]),
        title_font=dict(color=PALETTE["muted"]),
    )
    return fig


def build_dual_axis_chart(df: pd.DataFrame, inflation_choice: str) -> go.Figure:
    metric_col = (
        "headline_inflation_yoy" if inflation_choice == "Headline CPI" else "core_inflation_yoy"
    )
    metric_label = "Headline Inflation" if inflation_choice == "Headline CPI" else "Core Inflation"
    metric_color = PALETTE["accent"] if inflation_choice == "Headline CPI" else PALETTE["sage"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["DATE"],
            y=df["oil_price"],
            mode="lines",
            name="WTI Oil Price",
            line=dict(color=PALETTE["ink"], width=3.6),
            hovertemplate="%{x|%b %Y}<br>Oil price: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["DATE"],
            y=df[metric_col] * 100,
            mode="lines",
            name=metric_label,
            line=dict(color=metric_color, width=2.4, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>Inflation: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_annotation(
        x=df["DATE"].iloc[-1],
        y=df["oil_price"].iloc[-1],
        text="Oil",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font=dict(color=PALETTE["ink"], size=12),
    )
    fig.add_annotation(
        x=df["DATE"].iloc[-1],
        y=(df[metric_col] * 100).iloc[-1],
        text="Headline CPI" if inflation_choice == "Headline CPI" else "Core CPI",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font=dict(color=metric_color, size=12),
    )
    fig.update_yaxes(title_text="WTI oil price (USD/bbl)", secondary_y=False)
    fig.update_yaxes(title_text=f"{metric_label} (%)", secondary_y=True)
    fig.update_layout(showlegend=False)
    return apply_plot_style(fig, height=450)


def build_scatter_chart(df: pd.DataFrame, inflation_choice: str, lag_months: int) -> go.Figure:
    metric_col = (
        "headline_inflation_yoy" if inflation_choice == "Headline CPI" else "core_inflation_yoy"
    )
    metric_label = "Headline Inflation" if inflation_choice == "Headline CPI" else "Core Inflation"
    x_col = "oil_price" if lag_months == 0 else f"oil_price_lag{lag_months}"
    x_label = "WTI oil price" if lag_months == 0 else f"WTI oil price ({lag_months}-month lag)"
    plot_df = df[[x_col, metric_col, "DATE"]].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[metric_col] * 100,
            mode="markers",
            marker=dict(
                size=9,
                color=PALETTE["accent"] if inflation_choice == "Headline CPI" else PALETTE["sage"],
                opacity=0.62,
                line=dict(color="rgba(255,255,255,0.7)", width=0.8),
            ),
            hovertemplate="%{customdata|%b %Y}<br>Oil: %{x:.1f}<br>Inflation: %{y:.2f}%<extra></extra>",
            customdata=plot_df["DATE"],
            name=metric_label,
        )
    )
    if len(plot_df) > 1:
        trend_x = plot_df[x_col]
        trend_y = plot_df[metric_col] * 100
        slope, intercept = pd.Series(trend_y).cov(pd.Series(trend_x)) / pd.Series(trend_x).var(), trend_y.mean() - (pd.Series(trend_y).cov(pd.Series(trend_x)) / pd.Series(trend_x).var()) * trend_x.mean()
        x_min, x_max = trend_x.min(), trend_x.max()
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[intercept + slope * x_min, intercept + slope * x_max],
                mode="lines",
                line=dict(color="rgba(0, 87, 168, 0.35)" if inflation_choice == "Headline CPI" else "rgba(93, 115, 141, 0.45)", width=1.6),
                hoverinfo="skip",
                name="Trend",
            )
        )
    fig.update_xaxes(title_text=f"{x_label} (USD/bbl)")
    fig.update_yaxes(title_text=f"{metric_label} (%)")
    fig.update_layout(showlegend=False)
    return apply_plot_style(fig, height=420)


def build_rolling_chart(df: pd.DataFrame, rolling_view: str) -> go.Figure:
    fig = go.Figure()
    if rolling_view in ("Both", "Headline"):
        fig.add_trace(
            go.Scatter(
                x=df["DATE"],
                y=df["rolling_corr_12m_headline"],
                mode="lines",
                name="Oil vs headline inflation",
                line=dict(color=PALETTE["accent"], width=3),
            )
        )
    if rolling_view in ("Both", "Core"):
        fig.add_trace(
            go.Scatter(
                x=df["DATE"],
                y=df["rolling_corr_12m_core"],
                mode="lines",
                name="Oil vs core inflation",
                line=dict(color="rgba(93, 115, 141, 0.6)", width=2.2, dash="dash"),
            )
        )
    fig.add_hline(y=0, line_color="rgba(31,48,45,0.25)", line_dash="dot")
    fig.update_yaxes(title_text="12-month rolling correlation")
    return apply_plot_style(fig, height=400)


def build_lag_bar(lag_df: pd.DataFrame) -> go.Figure:
    colors = [
        PALETTE["accent"] if row == lag_df["Correlation"].abs().idxmax() else "rgba(93, 115, 141, 0.5)"
        for row in lag_df.index
    ]
    fig = go.Figure(
        go.Bar(
            x=lag_df["Lag (months)"].astype(str),
            y=lag_df["Correlation"],
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.65)", width=0.8)),
            hovertemplate="Lag %{x} months<br>Correlation: %{y:.3f}<extra></extra>",
        )
    )
    peak_row = lag_df.loc[lag_df["Correlation"].abs().idxmax()]
    fig.add_annotation(
        x=str(int(peak_row["Lag (months)"])),
        y=float(peak_row["Correlation"]),
        text="strongest",
        showarrow=False,
        yshift=16,
        font=dict(color=PALETTE["accent"], size=11),
    )
    fig.update_xaxes(title_text="Lag window")
    fig.update_yaxes(title_text="Correlation with headline inflation")
    return apply_plot_style(fig, height=330)


def build_commodity_cycle_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["DATE"],
            y=df["commodity_growth"] * 100,
            mode="lines",
            name="Commodity growth",
            line=dict(color=PALETTE["gold"], width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["DATE"],
            y=df["gdp_growth"] * 100,
            mode="lines",
            name="GDP growth",
            line=dict(color=PALETTE["sage"], width=3),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Commodity growth (%)", secondary_y=False)
    fig.update_yaxes(title_text="GDP growth (%)", secondary_y=True)
    return apply_plot_style(fig, height=430)


def build_commodity_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["quarter_label"] = plot_df["DATE"].dt.to_period("Q").astype(str)
    fig = go.Figure(
        go.Scatter(
            x=plot_df["commodity_growth"] * 100,
            y=plot_df["gdp_growth"] * 100,
            mode="markers",
            marker=dict(size=8.5, color="rgba(125, 142, 165, 0.58)", opacity=0.68),
            customdata=plot_df["quarter_label"],
            hovertemplate="%{customdata}<br>Commodity growth: %{x:.2f}%<br>GDP growth: %{y:.2f}%<extra></extra>",
            name="Quarterly observations",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Commodity growth (%)")
    fig.update_yaxes(title_text="GDP growth (%)")
    return apply_plot_style(fig, height=360)


inject_styles()
frames = build_analysis_frames()
oil_inflation = frames["oil_inflation"]
commodity_gdp = frames["commodity_gdp"]
lag_df = frames["lag_df"]
controlled_model = frames["controlled_model"]
headline_model = frames["headline_model"]
core_model = frames["core_model"]
commodity_model = frames["commodity_model"]

latest_date = oil_inflation["DATE"].max().strftime("%b %Y")
best_lag = int(lag_df.loc[lag_df["Correlation"].abs().idxmax(), "Lag (months)"])
controlled_oil_coef = controlled_model.params["oil_price"]


st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Macroeconomic Research Dashboard</div>
                <div class="hero-title">Global Commodity Macro Analysis</div>
                <div class="hero-subtitle">
                    FRED and World Bank data are used to study oil-inflation pass-through and the link between commodity cycles and output growth.
                </div>
                <div class="hero-core">
                    Oil tracks headline inflation more closely than core, with the strongest pass-through appearing within one month.
                </div>
            </div>
            <div class="hero-sidecar">
                <div class="hero-sidecar-title">Primary read</div>
                <div class="hero-sidecar-value">+0.030</div>
                <div class="hero-sidecar-copy">
                    Headline correlation exceeds core correlation by 0.030 in the current reduced-form specification.
                </div>
            </div>
        </div>
        <div class="takeaway-strip">
            <div class="takeaway-item">
                <div class="takeaway-index">01</div>
                <div class="takeaway-copy">Oil is more strongly linked to headline inflation than to core inflation.</div>
            </div>
            <div class="takeaway-item">
                <div class="takeaway-index">02</div>
                <div class="takeaway-copy">The strongest simple association appears at a 1-month lag rather than at longer horizons.</div>
            </div>
            <div class="takeaway-item">
                <div class="takeaway-index">03</div>
                <div class="takeaway-copy">The relationship weakens over time, but remains directionally positive after adding fed funds.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header(
    "Key Findings",
    "Top-Line Readout",
    "Top-line metrics computed directly from the project data and reduced-form models.",
)
module_open()
metric_cols = st.columns([1.75, 1.05], gap="large")
with metric_cols[0]:
    comparison_card(
        "Oil and Inflation Correlation",
        "Headline",
        f"{frames['headline_corr']:.3f}",
        "Core",
        f"{frames['core_corr']:.3f}",
        "Difference",
        f"{(frames['headline_corr'] - frames['core_corr']):+.3f}",
        "The headline link is modestly stronger and carries the main pass-through story.",
    )
with metric_cols[1]:
    st.markdown(
        f'<div class="appendix-block"><div class="appendix-title">Meta insight</div><div class="appendix-copy">Peak pass-through appears at <strong>{best_lag} month</strong>; the oil coefficient remains positive after policy control at <strong>{controlled_oil_coef:.4f}</strong>.</div></div>',
        unsafe_allow_html=True,
    )
st.markdown(
    f'<div class="module-context">FRED and World Bank data | Correlation, OLS, HAC | Sample through {latest_date}</div>',
    unsafe_allow_html=True,
)
module_close()

module_open()
section_header(
    "Oil and Inflation",
    "Oil and Inflation Over Time",
    "The time-series view anchors the project: oil and inflation rise together most clearly in the headline specification.",
)
inflation_choice = "Headline CPI"
lag_months = 0
rolling_view = "Both"

inflation_choice = st.segmented_control(
    "Inflation lens",
    ["Headline CPI", "Core CPI"],
    default="Headline CPI",
    key="inflation_choice",
)
st.markdown(
    f'<div class="module-context">Time-series view: {inflation_choice}</div>',
    unsafe_allow_html=True,
)
chart_frame(
    "Oil and Inflation",
    "WTI oil is plotted against the selected inflation series in a dual-axis view.",
    build_dual_axis_chart(oil_inflation, inflation_choice),
    key="dual_axis",
)
oil_notes = st.columns(2, gap="large")
with oil_notes[0]:
    st.markdown(
        f'<div class="compact-note note-grid-card"><strong>Interpretation.</strong>&nbsp;Headline correlation ({frames["headline_corr"]:.3f}) remains above core ({frames["core_corr"]:.3f}), consistent with more direct energy pass-through.</div>',
        unsafe_allow_html=True,
    )
with oil_notes[1]:
    st.markdown(
        f'<div class="compact-note note-grid-card"><strong>Controlled estimate.</strong>&nbsp;The oil coefficient stays positive at {controlled_oil_coef:.4f} after adding the federal funds rate.</div>',
        unsafe_allow_html=True,
    )
module_close()

module_open()
section_header(
    "Headline vs Core Inflation",
    "Oil-Inflation Cross-Section",
    "The scatter view shows how the oil-inflation link changes when inflation is measured as headline versus core and when oil enters with lags.",
)
lag_months = st.segmented_control(
    "Scatter lag",
    [0, 1, 3, 6],
    default=0,
    format_func=lambda x: "Current" if x == 0 else f"{x}M",
    key="lag_months",
)
st.markdown(
    f'<div class="module-context">Scatter view: {inflation_choice} | {"Current oil price" if lag_months == 0 else f"{lag_months}-month lagged oil price"}</div>',
    unsafe_allow_html=True,
)
chart_frame(
    "Oil and CPI: Cross-Section Evidence",
    "Each point is a monthly observation under the selected inflation lens and lag specification.",
    build_scatter_chart(oil_inflation, inflation_choice, lag_months),
    key="scatter",
)
st.markdown(
    '<div class="compact-note"><strong>Model note.</strong> Oil coefficients are positive across specifications.</div>',
    unsafe_allow_html=True,
)
module_close()

module_open()
section_header(
    "Lag Response",
    "How quickly the oil signal fades",
    "Shorter lags carry more of the oil-headline inflation relationship than longer ones.",
)
chart_frame(
    "Lag Structure",
    "The 1-, 3-, and 6-month lag comparison shows how the oil signal decays with time.",
    build_lag_bar(lag_df),
    key="lagbar",
)
st.markdown(
    '<div class="compact-note"><strong>Reading.</strong> The strongest simple link appears at 1 month, then weakens as the lag extends.</div>',
    unsafe_allow_html=True,
)
module_close()

module_open()
section_header(
    "Rolling Correlation",
    "Positive, but unstable over time",
    "A 12-month rolling view shows that the oil-inflation link changes materially across the sample.",
)
chart_frame(
    "Rolling Correlation",
    "Headline and core rolling correlations are shown together to emphasize instability rather than a fixed coefficient.",
    build_rolling_chart(oil_inflation, "Both"),
    key="rolling",
)
st.markdown(
    '<div class="compact-note"><strong>Reading.</strong> Rolling correlations remain mostly positive, but the strength of the relationship is clearly time-varying rather than constant.</div>',
    unsafe_allow_html=True,
)
module_close()

module_open()
section_header(
    "Commodity Prices and GDP Growth",
    "Broader Commodity Cycle Link",
    "The secondary block asks whether broader commodity-price cycles move with real activity as well as with inflation.",
)

chart_frame(
    "Commodity Growth and GDP Growth",
    "Quarterly commodity growth and GDP growth are plotted together to show broad macro-cycle co-movement.",
    build_commodity_cycle_chart(commodity_gdp),
    key="commodity_cycle",
)
st.markdown(
    f'<div class="compact-note"><strong>Reading.</strong> The relationship is positive but modest: correlation {frames["commodity_corr"]:.3f}, regression coefficient {commodity_model.params["commodity_growth"]:.4f}.</div>',
    unsafe_allow_html=True,
)
with st.expander("Show quarterly scatter", expanded=False):
    chart_frame(
        "Quarterly Scatter",
        "The quarterly relationship is modest but directionally positive.",
        build_commodity_scatter(commodity_gdp),
        key="commodity_scatter",
    )
module_close()

module_open()
section_header(
    "Method and Limits",
    "Notes on construction and interpretation",
    "These results are descriptive rather than causal and are intentionally kept reduced-form.",
)

note_cols = st.columns([1.2, 0.9], gap="large")
with note_cols[0]:
    st.markdown(
        """
        <div class="appendix-block">
            <div class="appendix-title">Construction</div>
            <ul class="source-list">
                <li>Daily WTI oil prices are aggregated to monthly means before merging with CPI and fed funds data.</li>
                <li>Headline and core inflation are computed as year-over-year changes in CPI.</li>
                <li>Commodity growth and GDP growth are computed at quarterly frequency.</li>
                <li>Lag comparisons focus on 1-, 3-, and 6-month oil lags, with HAC-robust errors in the controlled model.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with note_cols[1]:
    st.markdown(
        """
        <div class="appendix-block">
            <div class="appendix-title">Interpretation limits</div>
            <ul class="source-list">
                <li>These results are associational rather than causal.</li>
                <li>The regressions are intentionally reduced-form and omit broader macro controls.</li>
                <li>Rolling correlations are descriptive and should not be read as structural estimates.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
module_close()

module_open()
section_header(
    "Data Sources",
    "Series used in the dashboard",
    "All figures are computed from local project data pulled from FRED and the World Bank Pink Sheet.",
)

st.markdown(
    """
    <ul class="source-list">
        <li><strong>DCOILWTICO</strong> (FRED): daily WTI oil aggregated to monthly frequency.</li>
        <li><strong>CPIAUCSL</strong> and <strong>CPILFESL</strong> (FRED): headline and core CPI used for year-over-year inflation.</li>
        <li><strong>FEDFUNDS</strong> (FRED): federal funds rate used as a policy control.</li>
        <li><strong>GDPC1</strong> (FRED): quarterly real GDP used for GDP growth.</li>
        <li><strong>Total Commodity Index</strong> (World Bank Pink Sheet): commodity index aggregated to quarterly frequency.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)
module_close()

st.caption(
    "Reduced-form macro dashboard built with Streamlit, Plotly, and statsmodels."
)
