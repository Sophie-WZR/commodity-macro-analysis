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
    "ink": "#1f302d",
    "muted": "#66726c",
    "line": "rgba(31, 48, 45, 0.10)",
    "accent": "#9f5a3f",
    "accent_soft": "#cf8c6f",
    "sage": "#5f7a73",
    "gold": "#bf9a59",
    "cream": "#fbf7f1",
    "bg": "#f5efe6",
    "plot_bg": "rgba(255,255,255,0.55)",
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
            --bg: #f7f4ee;
            --panel: rgba(255, 255, 255, 0.80);
            --panel-strong: rgba(255, 255, 255, 0.96);
            --ink: {PALETTE["ink"]};
            --muted: {PALETTE["muted"]};
            --line: {PALETTE["line"]};
            --accent: {PALETTE["accent"]};
            --accent-2: {PALETTE["sage"]};
            --accent-3: {PALETTE["gold"]};
        }}

        .stApp {{
            background: linear-gradient(180deg, #faf8f4 0%, #f3eee6 100%);
            color: var(--ink);
        }}

        header[data-testid="stHeader"], .stAppToolbar {{
            display: none;
        }}

        .block-container {{
            max-width: 1260px;
            padding-top: 1rem;
            padding-bottom: 2.5rem;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(249, 246, 240, 0.72);
            border-right: 1px solid var(--line);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--ink);
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
            padding-left: 1.05rem;
            padding-right: 1rem;
        }}

        .hero {{
            padding: 1rem 1.15rem 0.95rem 1.15rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255,255,255,0.80);
            box-shadow: 0 3px 12px rgba(45, 42, 36, 0.03);
            margin-bottom: 0.55rem;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: 1.35fr 0.8fr;
            gap: 0.8rem;
            align-items: start;
        }}

        .hero-kicker {{
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.7rem;
            color: var(--accent-2);
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}

        .hero-title {{
            font-size: 2.22rem;
            line-height: 1.0;
            font-weight: 780;
            letter-spacing: -0.03em;
            margin: 0 0 0.35rem 0;
            color: var(--ink);
        }}

        .hero-subtitle {{
            font-size: 0.88rem;
            line-height: 1.45;
            color: var(--muted);
            max-width: 700px;
            margin-bottom: 0.45rem;
        }}

        .hero-why {{
            border-left: 2px solid rgba(159, 90, 63, 0.30);
            padding-left: 0.7rem;
            color: var(--ink);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-top: 0.2rem;
        }}

        .findings-box {{
            border: 1px solid rgba(31, 48, 45, 0.14);
            border-radius: 10px;
            background: rgba(252, 251, 248, 0.98);
            padding: 0.8rem 0.85rem 0.72rem 0.85rem;
        }}

        .findings-title {{
            font-size: 0.76rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .findings-list {{
            margin: 0;
            padding-left: 1rem;
            color: var(--ink);
            font-size: 0.84rem;
            line-height: 1.45;
        }}

        .findings-list li {{
            margin-bottom: 0.4rem;
        }}

        .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.05rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.34rem 0.62rem;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: rgba(255,255,255,0.62);
            color: var(--muted);
            font-size: 0.78rem;
        }}

        .selection-context {{
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }}

        .section-wrap {{
            margin-top: 1.55rem;
            margin-bottom: 0.2rem;
        }}

        .section-kicker {{
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.73rem;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 0.32rem;
        }}

        .section-title {{
            font-size: 1.42rem;
            font-weight: 760;
            color: var(--ink);
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }}

        .section-subtitle {{
            color: var(--muted);
            line-height: 1.42;
            font-size: 0.84rem;
            max-width: 850px;
            margin-bottom: 0.28rem;
        }}

        .metric-card {{
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.8rem 0.85rem;
            min-height: 148px;
            background: rgba(255,255,255,0.78);
            box-shadow: 0 4px 14px rgba(48, 39, 28, 0.03);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(48, 39, 28, 0.05);
            border-color: rgba(31, 48, 45, 0.14);
        }}

        .metric-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 0.65rem;
        }}

        .metric-value {{
            font-size: 2.15rem;
            line-height: 1;
            font-weight: 760;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }}

        .metric-note {{
            color: var(--muted);
            font-size: 0.79rem;
            line-height: 1.38;
        }}

        .glass-card {{
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.85rem 0.9rem;
            background: rgba(255,255,255,0.72);
            box-shadow: 0 4px 14px rgba(48, 39, 28, 0.03);
            height: 100%;
        }}

        .insight-list {{
            margin: 0;
            padding-left: 1.1rem;
            color: var(--muted);
            line-height: 1.75;
        }}

        .insight-list li {{
            margin-bottom: 0.5rem;
        }}

        .chart-card {{
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.62rem 0.62rem 0.12rem 0.62rem;
            background: rgba(255,255,255,0.76);
            box-shadow: 0 4px 14px rgba(48, 39, 28, 0.03);
        }}

        .chart-head {{
            padding: 0.2rem 0.35rem 0.55rem 0.35rem;
        }}

        .chart-title {{
            font-size: 0.98rem;
            font-weight: 720;
            color: var(--ink);
            margin-bottom: 0.2rem;
        }}

        .chart-subtitle {{
            font-size: 0.84rem;
            line-height: 1.5;
            color: var(--muted);
        }}

        div[data-baseweb="tab-list"] {{
            gap: 1.5rem;
            align-items: flex-end;
            border-bottom: 1px solid rgba(31, 48, 45, 0.10);
            padding-bottom: 0.15rem;
            margin-bottom: 1rem;
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
            border-bottom: 2px solid transparent;
            border-radius: 0;
            color: rgba(31, 48, 45, 0.58);
            padding: 0 0 0.72rem 0;
            min-height: auto;
            font-size: 0.98rem;
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
            color: rgba(31, 48, 45, 0.82);
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
            border-bottom-color: rgba(159, 90, 63, 0.88);
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
        }}

        .source-table th {{
            color: var(--ink);
            font-weight: 700;
        }}

        .divider {{
            height: 1px;
            background: linear-gradient(90deg, rgba(159,90,63,0), rgba(159,90,63,0.26), rgba(159,90,63,0));
            margin: 1.25rem 0 0.4rem 0;
        }}

        .stCaption {{
            color: var(--muted);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        gridcolor="rgba(31,48,45,0.08)",
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
            line=dict(color=PALETTE["ink"], width=3),
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
            line=dict(color=metric_color, width=3),
            hovertemplate="%{x|%b %Y}<br>Inflation: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_annotation(
        x=df["DATE"].iloc[-1],
        y=(df[metric_col] * 100).iloc[-1],
        text="Recent sample end",
        showarrow=False,
        xanchor="right",
        yshift=18,
        font=dict(color=PALETTE["muted"], size=11),
    )
    fig.update_yaxes(title_text="WTI oil price (USD/bbl)", secondary_y=False)
    fig.update_yaxes(title_text=f"{metric_label} (%)", secondary_y=True)
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
                size=10,
                color=PALETTE["accent"] if inflation_choice == "Headline CPI" else PALETTE["sage"],
                opacity=0.72,
                line=dict(color="rgba(255,255,255,0.7)", width=0.8),
            ),
            hovertemplate="%{customdata|%b %Y}<br>Oil: %{x:.1f}<br>Inflation: %{y:.2f}%<extra></extra>",
            customdata=plot_df["DATE"],
            name=metric_label,
        )
    )
    fig.update_xaxes(title_text=f"{x_label} (USD/bbl)")
    fig.update_yaxes(title_text=f"{metric_label} (%)")
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
                line=dict(color=PALETTE["sage"], width=3, dash="dash"),
            )
        )
    fig.add_hline(y=0, line_color="rgba(31,48,45,0.25)", line_dash="dot")
    fig.update_yaxes(title_text="12-month rolling correlation")
    return apply_plot_style(fig, height=400)


def build_lag_bar(lag_df: pd.DataFrame) -> go.Figure:
    colors = [PALETTE["accent"] if row == lag_df["Correlation"].abs().idxmax() else PALETTE["sage"] for row in lag_df.index]
    fig = go.Figure(
        go.Bar(
            x=lag_df["Lag (months)"].astype(str),
            y=lag_df["Correlation"],
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.65)", width=0.8)),
            hovertemplate="Lag %{x} months<br>Correlation: %{y:.3f}<extra></extra>",
        )
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
            marker=dict(size=10, color=PALETTE["gold"], opacity=0.72),
            customdata=plot_df["quarter_label"],
            hovertemplate="%{customdata}<br>Commodity growth: %{x:.2f}%<br>GDP growth: %{y:.2f}%<extra></extra>",
            name="Quarterly observations",
        )
    )
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

with st.sidebar:
    st.markdown("##### Notes")
    st.caption(
        "Reduced-form research dashboard built from public macro data. Results are associational rather than causal."
    )
    st.markdown("##### Sources")
    st.caption("FRED: DCOILWTICO, CPIAUCSL, CPILFESL, FEDFUNDS, GDPC1")
    st.caption("World Bank Pink Sheet: Total Commodity Index")
    st.markdown("##### Sample")
    st.caption(f"Oil-inflation monthly sample through {latest_date}")


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
                <div class="hero-why">
                    Why this matters: commodity shocks remain one of the clearest transmission channels between markets, inflation dynamics, and macro activity.
                </div>
            </div>
            <div>
                <div class="findings-box">
                    <div class="findings-title">Key Findings</div>
                    <ul class="findings-list">
                        <li>Oil is more strongly linked to headline inflation than to core inflation.</li>
                        <li>The oil-inflation relationship is time-varying rather than stable.</li>
                        <li>Oil remains positive after controlling for the federal funds rate.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header(
    "Key Findings",
    "Top-Line Readout",
    "Top-line metrics computed directly from the underlying project data and models.",
)

metric_cols = st.columns(4, gap="large")
with metric_cols[0]:
    metric_card(
        "Oil vs Headline Inflation",
        f"{frames['headline_corr']:.3f}",
        "The oil-headline relationship is the strongest simple contemporaneous correlation in the inflation block.",
    )
with metric_cols[1]:
    metric_card(
        "Oil vs Core Inflation",
        f"{frames['core_corr']:.3f}",
        "Core remains positively related to oil, but less strongly than headline, which fits direct energy pass-through.",
    )
with metric_cols[2]:
    metric_card(
        "Strongest Lag",
        f"{best_lag}M",
        "The lag table suggests the strongest oil-headline link appears at a short horizon in this sample.",
    )
with metric_cols[3]:
    metric_card(
        "Oil Coef. + Fed Funds",
        f"{controlled_oil_coef:.4f}",
        "The oil coefficient remains positive after controlling for the federal funds rate in the HAC-robust specification.",
    )

badge_row(
    [
        "Sources: FRED + World Bank",
        "Methods: Correlation, OLS, HAC",
        "Monthly + Quarterly Frequencies",
        "Reduced-form, not causal",
    ]
)

section_header(
    "Oil and Inflation",
    "Macro Transmission View",
    "Oil-inflation pass-through with direct controls for inflation lens, lag window, and rolling view.",
)
inflation_choice = "Headline CPI"
lag_months = 0
rolling_view = "Both"

oil_left, oil_right = st.columns([1.2, 0.8], gap="large")
with oil_left:
    inflation_choice = st.segmented_control(
        "Inflation lens",
        ["Headline CPI", "Core CPI"],
        default="Headline CPI",
        key="inflation_choice",
    )
    st.caption(
        f"Time-series view: {inflation_choice}"
    )
    chart_frame(
        "Oil Prices and Inflation Over Time",
        "The selected inflation series is shown against WTI oil to make the co-movement and major shock windows easy to read.",
        build_dual_axis_chart(oil_inflation, inflation_choice),
        key="dual_axis",
    )
with oil_right:
    narrative_card(
        "Interpretation",
        f"""
        Oil correlates more strongly with <strong>headline inflation</strong> ({frames['headline_corr']:.3f}) than with
        <strong>core inflation</strong> ({frames['core_corr']:.3f}), consistent with more direct energy pass-through.
        """,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    narrative_card(
        "Methods Snapshot",
        f"""
        Monthly oil prices are aligned with year-over-year CPI inflation. The main controlled model is
        <strong>headline inflation ~ oil price + fed funds</strong>; the oil coefficient is <strong>{controlled_oil_coef:.4f}</strong>.
        """,
    )

section_header(
    "Headline vs Core Inflation",
    "Relationship Quality",
    "Cross-sectional and time-varying views of the oil-inflation relationship.",
)
scatter_left, scatter_right = st.columns([0.9, 1.1], gap="large")
with scatter_left:
    lag_months = st.segmented_control(
        "Scatter lag",
        [0, 1, 3, 6],
        default=0,
        format_func=lambda x: "Current" if x == 0 else f"{x}M",
        key="lag_months",
    )
    st.caption(
        f"Scatter view: {inflation_choice} | {'Current oil price' if lag_months == 0 else f'{lag_months}-month lagged oil price'}"
    )
    chart_frame(
        "Oil and Inflation Scatter",
        "Scatter view under the selected inflation lens and lag specification.",
        build_scatter_chart(oil_inflation, inflation_choice, lag_months),
        key="scatter",
    )
with scatter_right:
    tabs = st.tabs(["Key Findings", "What to Notice"])
    with tabs[0]:
        narrative_card(
            "Headline vs Core Readout",
            f"""
            The key question is whether oil appears more connected to <strong>headline CPI</strong> or <strong>core CPI</strong>.
            In this sample, the evidence points more strongly to headline inflation.
            <ul class="insight-list">
                <li>Headline OLS oil coefficient: <strong>{headline_model.params['oil_price']:.4f}</strong></li>
                <li>Core OLS oil coefficient: <strong>{core_model.params['oil_price']:.4f}</strong></li>
                <li>Controlled oil coefficient: <strong>{controlled_oil_coef:.4f}</strong></li>
            </ul>
            """,
        )
    with tabs[1]:
        narrative_card(
            "Reading the Section",
            """
            The scatter plot shows the relationship under the selected lens.
            The charts below then show whether the effect survives lags and whether it changes across time.
            """,
        )

lag_roll_cols = st.columns(2, gap="large")
with lag_roll_cols[0]:
    chart_frame(
        "Lag Structure",
        "The 1-, 3-, and 6-month lag comparison shows how quickly the oil signal fades.",
        build_lag_bar(lag_df),
        key="lagbar",
    )
with lag_roll_cols[1]:
    rolling_view = st.segmented_control(
        "Rolling view",
        ["Both", "Headline", "Core"],
        default="Both",
        key="rolling_view",
    )
    st.caption(f"Rolling chart view: {rolling_view}")
    chart_frame(
        "Rolling Correlation",
        "The 12-month rolling view shows that the oil-inflation relationship is time-varying.",
        build_rolling_chart(oil_inflation, rolling_view),
        key="rolling",
    )

section_header(
    "Commodity Prices and GDP Growth",
    "Broader Macro Cycle Link",
    "Quarterly commodity growth and GDP growth are used to summarize broader macro cycle co-movement.",
)

commodity_left, commodity_right = st.columns([1.1, 0.9], gap="large")
with commodity_left:
    chart_frame(
        "Commodity Growth vs GDP Growth",
        "Quarterly commodity growth and GDP growth are plotted together to show broad co-movement.",
        build_commodity_cycle_chart(commodity_gdp),
        key="commodity_cycle",
    )
with commodity_right:
    chart_frame(
        "Quarterly Scatter",
        "The relationship is modest but directionally positive.",
        build_commodity_scatter(commodity_gdp),
        key="commodity_scatter",
    )

section_header(
    "Method Notes and Limitations",
    "Research Credibility",
    "Secondary notes on construction and interpretation.",
)

note_cols = st.columns(2, gap="large")
with note_cols[0]:
    with st.expander("Method Notes", expanded=True):
        st.markdown(
            """
            - Daily WTI oil prices are aggregated to monthly means.
            - Headline and core inflation are computed as year-over-year percentage changes in CPI.
            - Commodity growth and GDP growth are computed at quarterly frequency.
            - Lag analysis compares 1-, 3-, and 6-month oil lags.
            - The controlled inflation model uses HAC-robust standard errors with 3 lags.
            """
        )
with note_cols[1]:
    with st.expander("Limitations", expanded=True):
        st.markdown(
            """
            - These results are **associational**, not causal.
            - The regressions are intentionally reduced-form and omit many macro controls.
            - Rolling correlations are descriptive and should not be treated as structural estimates.
            - The project is designed to emphasize clarity and reproducibility rather than model complexity.
            """
        )

section_header(
    "Data Sources",
    "Source Citations",
    "All numbers shown above are computed from the project data stored in the repository.",
)

st.markdown(
    """
    <div class="glass-card">
        <table class="source-table">
            <thead>
                <tr>
                    <th>Series</th>
                    <th>Provider</th>
                    <th>Frequency</th>
                    <th>Role in Dashboard</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>DCOILWTICO</td>
                    <td>FRED</td>
                    <td>Daily → Monthly</td>
                    <td>WTI oil price series</td>
                </tr>
                <tr>
                    <td>CPIAUCSL</td>
                    <td>FRED</td>
                    <td>Monthly</td>
                    <td>Headline CPI inflation</td>
                </tr>
                <tr>
                    <td>CPILFESL</td>
                    <td>FRED</td>
                    <td>Monthly</td>
                    <td>Core CPI inflation</td>
                </tr>
                <tr>
                    <td>FEDFUNDS</td>
                    <td>FRED</td>
                    <td>Monthly</td>
                    <td>Monetary policy control</td>
                </tr>
                <tr>
                    <td>GDPC1</td>
                    <td>FRED</td>
                    <td>Quarterly</td>
                    <td>Real GDP growth</td>
                </tr>
                <tr>
                    <td>Total Commodity Index</td>
                    <td>World Bank Pink Sheet</td>
                    <td>Monthly → Quarterly</td>
                    <td>Broad commodity-cycle indicator</td>
                </tr>
            </tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Built with Streamlit, Plotly, and statsmodels to present the macro analysis project as a polished research-oriented dashboard."
)
