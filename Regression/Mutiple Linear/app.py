from dataclasses import dataclass

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

DEMO_URL = "https://multiple-linear-regression-model-advertising-sales.streamlit.app/"
DATASET_URL = "https://www.kaggle.com/datasets/ashydv/advertising-dataset"

# Exported from the trained multiple linear regression model.
MODEL_COEFFICIENTS = np.array(
    [0.05450927083721978, 0.10094536239295579, 0.0043366468220340446],
    dtype=float,
)
MODEL_INTERCEPT = 4.714126402214127

TV_MAX = 296.4
RADIO_MAX = 49.6
NEWSPAPER_MAX = 114.0

CHANNEL_ORDER = ["TV", "Radio", "Newspaper"]
CHANNEL_COLORS = {
    "TV": "#d16447",
    "Radio": "#2e8b7b",
    "Newspaper": "#4677a8",
}
CHANNEL_MAX = {
    "TV": TV_MAX,
    "Radio": RADIO_MAX,
    "Newspaper": NEWSPAPER_MAX,
}
ACCENT_CLASS = {
    "TV": "warm",
    "Radio": "teal",
    "Newspaper": "blue",
}


@dataclass(frozen=True)
class LinearModelParams:
    """Lightweight prediction model for deployment without sklearn."""

    coef_: np.ndarray
    intercept_: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        return features @ self.coef_.ravel() + self.intercept_


def inject_styles():
    """Apply a custom visual system to the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #17324d;
            --ink-soft: #5e6c7b;
            --paper: #fff9f1;
            --paper-strong: #fffdf8;
            --border: rgba(23, 50, 77, 0.10);
            --shadow: 0 24px 60px rgba(23, 50, 77, 0.10);
            --warm: #d16447;
            --teal: #2e8b7b;
            --blue: #4677a8;
            --sand: #f2e6d8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(209, 100, 71, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(70, 119, 168, 0.16), transparent 24%),
                linear-gradient(180deg, #fffdf8 0%, #f7efe4 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fffaf3 0%, #f3eadb 100%);
            border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        p, li, div {
            color: var(--ink);
        }

        .hero-shell,
        .glass-card,
        .budget-card,
        .metric-card,
        .explanation-panel,
        .sidebar-panel {
            background: rgba(255, 253, 248, 0.90);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }

        .hero-shell {
            border-radius: 30px;
            padding: 2rem;
        }

        .hero-kicker,
        .section-kicker,
        .card-label {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--ink-soft);
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
            font-size: clamp(2.2rem, 3vw, 3.4rem);
            line-height: 1.02;
            margin: 0;
            color: var(--ink);
        }

        .hero-copy {
            font-size: 1.02rem;
            line-height: 1.75;
            color: var(--ink-soft);
            margin-top: 1rem;
            margin-bottom: 0;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.2rem;
        }

        .pill {
            border-radius: 999px;
            border: 1px solid rgba(23, 50, 77, 0.10);
            background: rgba(255, 255, 255, 0.72);
            padding: 0.45rem 0.85rem;
            font-size: 0.88rem;
            color: var(--ink);
        }

        .hero-stat {
            border-radius: 26px;
            padding: 1.55rem;
        }

        .hero-stat-value {
            font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
            font-size: 2.35rem;
            line-height: 1;
            margin: 0.25rem 0 0.5rem;
            color: var(--ink);
        }

        .hero-stat-copy {
            color: var(--ink-soft);
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .mini-grid div {
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--border);
            padding: 0.85rem 0.95rem;
        }

        .mini-grid span {
            display: block;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--ink-soft);
            margin-bottom: 0.35rem;
        }

        .mini-grid strong {
            font-size: 1rem;
            color: var(--ink);
        }

        .section-heading {
            margin-top: 1.5rem;
            margin-bottom: 0.85rem;
        }

        .section-title {
            font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
            font-size: 1.8rem;
            margin: 0;
        }

        .section-copy {
            margin-top: 0.35rem;
            color: var(--ink-soft);
            line-height: 1.7;
        }

        .metric-card,
        .budget-card,
        .glass-card,
        .explanation-panel,
        .sidebar-panel {
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
        }

        .metric-card {
            min-height: 165px;
        }

        .metric-value {
            font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
            font-size: 2rem;
            line-height: 1.05;
            margin: 0.25rem 0 0.6rem;
        }

        .metric-copy,
        .budget-copy {
            color: var(--ink-soft);
            line-height: 1.6;
            margin: 0;
        }

        .budget-channel {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .budget-percent {
            font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
            font-size: 2rem;
            margin: 0.2rem 0 0.35rem;
        }

        .budget-bar {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(23, 50, 77, 0.08);
            overflow: hidden;
            margin-bottom: 0.9rem;
        }

        .budget-bar span {
            display: block;
            height: 100%;
            border-radius: 999px;
        }

        .glass-card h3,
        .explanation-panel h3,
        .sidebar-panel h3 {
            margin-top: 0;
            margin-bottom: 0.7rem;
        }

        .formula {
            font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
            font-size: 1.15rem;
            line-height: 1.7;
            color: var(--ink);
        }

        .formula-note {
            color: var(--ink-soft);
            line-height: 1.7;
            margin-bottom: 0;
        }

        .explanation-panel ul {
            margin: 0.75rem 0 0;
            padding-left: 1.1rem;
        }

        .explanation-panel li {
            margin-bottom: 0.55rem;
            color: var(--ink-soft);
        }

        .takeaway {
            margin-top: 1rem;
            padding-top: 0.95rem;
            border-top: 1px solid var(--border);
            color: var(--ink);
            line-height: 1.7;
        }

        .warm {
            border-top: 4px solid var(--warm);
        }

        .teal {
            border-top: 4px solid var(--teal);
        }

        .blue {
            border-top: 4px solid var(--blue);
        }

        .neutral {
            border-top: 4px solid #b38b59;
        }

        .stLinkButton a {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #17324d 0%, #264f78 100%);
            color: #ffffff;
            font-weight: 700;
            padding: 0.7rem 1.15rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.6rem;
            margin-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.6rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: #17324d;
            color: #ffffff;
        }

        .footer-note {
            margin-top: 2rem;
            padding: 1rem 0 0;
            border-top: 1px solid var(--border);
            color: var(--ink-soft);
            line-height: 1.7;
        }

        .footer-note a {
            color: #1d5a83;
            text-decoration: none;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .hero-shell {
                padding: 1.4rem;
            }

            .mini-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model() -> LinearModelParams:
    """Load exported linear regression coefficients for inference."""
    return LinearModelParams(
        coef_=MODEL_COEFFICIENTS.copy(),
        intercept_=MODEL_INTERCEPT,
    )


def predict_sales(model, tv: float, radio: float, newspaper: float) -> float:
    """Predict sales from the model using a single row of feature values."""
    features = np.array([[tv, radio, newspaper]], dtype=float)
    prediction = model.predict(features)
    return float(prediction[0])


def compute_contributions(model, tv: float, radio: float, newspaper: float):
    """Compute contribution of each feature to the raw linear prediction."""
    coefs = np.array(model.coef_, dtype=float).ravel()
    inputs = np.array([tv, radio, newspaper], dtype=float)
    contributions = coefs * inputs
    return {
        "TV": float(contributions[0]),
        "Radio": float(contributions[1]),
        "Newspaper": float(contributions[2]),
    }


def percent_for(channel: str, value: float) -> float:
    """Convert a channel budget value into a percentage of the dataset maximum."""
    return (value / CHANNEL_MAX[channel]) * 100


def build_metric_card(title: str, value: str, copy: str, accent: str) -> str:
    """Render a compact metric card."""
    return f"""
    <div class="metric-card {accent}">
        <div class="card-label">{title}</div>
        <div class="metric-value">{value}</div>
        <p class="metric-copy">{copy}</p>
    </div>
    """


def build_budget_card(channel: str, budget: float) -> str:
    """Render a current-budget status card."""
    percent = percent_for(channel, budget)
    bar_color = CHANNEL_COLORS[channel]
    max_budget = CHANNEL_MAX[channel]
    return f"""
    <div class="budget-card {ACCENT_CLASS[channel]}">
        <div class="card-label">Current spend</div>
        <div class="budget-channel">{channel}</div>
        <div class="budget-percent">{percent:.0f}%</div>
        <div class="budget-bar"><span style="width:{percent:.0f}%; background:{bar_color};"></span></div>
        <p class="budget-copy">${budget:.1f} of ${max_budget:.1f} observed max budget.</p>
    </div>
    """


def build_panel(title: str, bullets: list[str], takeaway: str, accent: str) -> str:
    """Render an interpretation card."""
    bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
    return f"""
    <div class="explanation-panel {accent}">
        <h3>{title}</h3>
        <ul>{bullet_html}</ul>
        <div class="takeaway">{takeaway}</div>
    </div>
    """


def build_section_heading(kicker: str, title: str, copy: str) -> str:
    """Render a reusable section heading."""
    return f"""
    <div class="section-heading">
        <div class="section-kicker">{kicker}</div>
        <h2 class="section-title">{title}</h2>
        <p class="section-copy">{copy}</p>
    </div>
    """


def plot_contribution_bar(contributions):
    """Create a styled bar chart showing each channel contribution."""
    labels = list(contributions.keys())
    values = list(contributions.values())
    colors = [CHANNEL_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor="#fff9f1")
    ax.set_facecolor("#fff9f1")
    bars = ax.bar(labels, values, color=colors, width=0.56, alpha=0.92)

    span = max(max(values) - min(values), 1.0)
    offset = span * 0.05
    for bar, value in zip(bars, values):
        height = bar.get_height()
        text_y = height + offset if value >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
            color="#17324d",
        )

    ax.set_title(
        "Contribution to predicted sales",
        fontsize=15,
        fontweight="bold",
        color="#17324d",
        loc="left",
    )
    ax.set_ylabel("Sales contribution", fontsize=11, color="#17324d")
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.axhline(y=0, color="#7b8794", linewidth=1.1, alpha=0.8)
    ax.tick_params(axis="x", colors="#17324d", labelsize=11)
    ax.tick_params(axis="y", colors="#17324d", labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    y_min = min(0, min(values) * 1.15)
    y_max = max(values) * 1.18 if max(values) > 0 else 1
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    return fig


def plot_slice(model, fixed_values, variable_name, variable_range):
    """Plot predicted sales across a range for one variable while holding others fixed."""
    tv_values = fixed_values["TV"]
    radio_values = fixed_values["Radio"]
    newspaper_values = fixed_values["Newspaper"]

    y_values = []
    for value in variable_range:
        if variable_name == "TV":
            features = [value, radio_values, newspaper_values]
        elif variable_name == "Radio":
            features = [tv_values, value, newspaper_values]
        else:
            features = [tv_values, radio_values, value]

        prediction = model.predict(np.array([features], dtype=float))[0]
        y_values.append(prediction)

    x_display = variable_range / (CHANNEL_MAX[variable_name] / 100)
    current_percent = percent_for(variable_name, fixed_values[variable_name])
    current_prediction = predict_sales(
        model,
        fixed_values["TV"],
        fixed_values["Radio"],
        fixed_values["Newspaper"],
    )
    color = CHANNEL_COLORS[variable_name]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="#fff9f1")
    ax.set_facecolor("#fff9f1")
    ax.plot(x_display, y_values, color=color, linewidth=3)
    ax.fill_between(x_display, y_values, color=color, alpha=0.14)
    ax.scatter(
        [current_percent],
        [current_prediction],
        s=115,
        color=color,
        edgecolors="#fffdf8",
        linewidth=1.7,
        zorder=5,
    )
    ax.annotate(
        f"{current_percent:.0f}%",
        (current_percent, current_prediction),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        color="#17324d",
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "#fffdf8",
            "ec": color,
            "alpha": 0.96,
        },
    )

    ax.set_title(
        f"{variable_name} budget vs predicted sales",
        fontsize=15,
        fontweight="bold",
        color="#17324d",
        loc="left",
    )
    ax.set_xlabel(f"{variable_name} budget (%)", fontsize=11, color="#17324d")
    ax.set_ylabel("Predicted sales", fontsize=11, color="#17324d")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.tick_params(axis="x", colors="#17324d", labelsize=10)
    ax.tick_params(axis="y", colors="#17324d", labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


def describe_contribution_chart_html(contributions, intercept: float) -> str:
    """Return a beginner-friendly explanation for the contribution bar chart."""
    strongest_channel = max(contributions, key=lambda name: abs(contributions[name]))
    strongest_value = contributions[strongest_channel]
    bullets = [
        "The horizontal axis lists the three advertising channels.",
        "The vertical axis shows how much each channel adds to the final prediction.",
        "Taller bars mean a bigger impact at the current slider settings.",
        "Anything above zero pushes sales up. A negative bar would pull the prediction down.",
    ]
    takeaway = (
        f"{strongest_channel} is the strongest driver right now with a contribution of "
        f"{strongest_value:.2f}. Baseline sales start from the intercept of {intercept:.2f}."
    )
    return build_panel("How to read this chart", bullets, takeaway, "warm")


def describe_slice_plot_html(model, fixed_values, variable_name: str) -> str:
    """Return a beginner-friendly explanation for a budget-vs-sales chart."""
    coefficient_map = dict(zip(CHANNEL_ORDER, np.array(model.coef_, dtype=float).ravel()))
    percent_map = {
        channel: percent_for(channel, fixed_values[channel]) for channel in CHANNEL_ORDER
    }
    current_prediction = predict_sales(
        model,
        fixed_values["TV"],
        fixed_values["Radio"],
        fixed_values["Newspaper"],
    )
    other_channels = [name for name in CHANNEL_ORDER if name != variable_name]
    slope = coefficient_map[variable_name]

    if slope > 0:
        trend_text = (
            f"The line rises, so increasing the {variable_name} budget lifts the sales prediction."
        )
    elif slope < 0:
        trend_text = (
            f"The line falls, so increasing the {variable_name} budget lowers the sales prediction."
        )
    else:
        trend_text = (
            f"The line is nearly flat, so changing the {variable_name} budget barely moves the prediction."
        )

    bullets = [
        f"The x-axis is the {variable_name} budget from 0% to 100% of its observed maximum.",
        "The y-axis is the predicted sales output from the regression model.",
        f"The highlighted point marks your current {variable_name} setting at {percent_map[variable_name]:.0f}%.",
        (
            f"The other channels stay fixed at {other_channels[0]} = {percent_map[other_channels[0]]:.0f}% "
            f"and {other_channels[1]} = {percent_map[other_channels[1]]:.0f}%."
        ),
    ]
    takeaway = f"{trend_text} At the highlighted point, predicted sales are {current_prediction:.2f}."
    return build_panel("How to read this chart", bullets, takeaway, ACCENT_CLASS[variable_name])


def main():
    st.set_page_config(
        page_title="Advertising Sales Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    model = load_model()

    st.sidebar.markdown(
        """
        <div class="sidebar-panel neutral">
            <div class="card-label">Interactive lab</div>
            <h3>Campaign controls</h3>
            <p class="metric-copy">
                Adjust each channel to see how a multiple linear regression model responds.
                Slider values represent percentages of the largest budget seen in the dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tv_percent = st.sidebar.slider("TV Budget %", min_value=0, max_value=100, value=50)
    radio_percent = st.sidebar.slider("Radio Budget %", min_value=0, max_value=100, value=50)
    newspaper_percent = st.sidebar.slider(
        "Newspaper Budget %",
        min_value=0,
        max_value=100,
        value=18,
    )

    st.sidebar.caption("Observed maximum budgets")
    st.sidebar.caption(f"TV: ${TV_MAX:.1f}")
    st.sidebar.caption(f"Radio: ${RADIO_MAX:.1f}")
    st.sidebar.caption(f"Newspaper: ${NEWSPAPER_MAX:.1f}")
    st.sidebar.markdown(f"[Open hosted demo]({DEMO_URL})")

    tv_budget = tv_percent * (TV_MAX / 100)
    radio_budget = radio_percent * (RADIO_MAX / 100)
    newspaper_budget = newspaper_percent * (NEWSPAPER_MAX / 100)

    predicted_sales = predict_sales(model, tv_budget, radio_budget, newspaper_budget)
    contributions = compute_contributions(model, tv_budget, radio_budget, newspaper_budget)
    total_from_contributions = sum(contributions.values())
    intercept = float(model.intercept_)
    strongest_channel = max(contributions, key=lambda name: abs(contributions[name]))
    strongest_value = contributions[strongest_channel]

    hero_left, hero_right = st.columns([1.45, 1])
    with hero_left:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="hero-kicker">Multiple Linear Regression</div>
                <h1 class="hero-title">Advertising Sales Prediction Studio</h1>
                <p class="hero-copy">
                    Explore how TV, Radio, and Newspaper budgets combine inside a trained
                    regression model. The layout is designed to make the prediction, the
                    formula, and every chart easier to read in one pass.
                </p>
                <div class="pill-row">
                    <span class="pill">Beginner-friendly interpretation</span>
                    <span class="pill">Modern Streamlit dashboard</span>
                    <span class="pill">Live hosted demo available</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open Live Demo", DEMO_URL)

    with hero_right:
        st.markdown(
            f"""
            <div class="hero-shell hero-stat warm">
                <div class="card-label">Prediction snapshot</div>
                <div class="hero-stat-value">{predicted_sales:.2f}</div>
                <p class="hero-stat-copy">
                    Estimated sales output for the current campaign mix based on the
                    trained regression formula.
                </p>
                <div class="mini-grid">
                    <div>
                        <span>Strongest channel</span>
                        <strong>{strongest_channel}</strong>
                    </div>
                    <div>
                        <span>Largest contribution</span>
                        <strong>{strongest_value:.2f}</strong>
                    </div>
                    <div>
                        <span>Baseline sales</span>
                        <strong>{intercept:.2f}</strong>
                    </div>
                    <div>
                        <span>Variable lift</span>
                        <strong>{total_from_contributions:.2f}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        build_section_heading(
            "Current mix",
            "Where the campaign stands right now",
            "These cards translate the sidebar slider percentages into actual budget values seen by the model.",
        ),
        unsafe_allow_html=True,
    )

    budget_columns = st.columns(3)
    current_budgets = {
        "TV": tv_budget,
        "Radio": radio_budget,
        "Newspaper": newspaper_budget,
    }
    for column, channel in zip(budget_columns, CHANNEL_ORDER):
        with column:
            st.markdown(build_budget_card(channel, current_budgets[channel]), unsafe_allow_html=True)

    st.markdown(
        build_section_heading(
            "Key metrics",
            "The prediction at a glance",
            "This top row separates the final estimate from the model's baseline and the contribution coming from the chosen budget mix.",
        ),
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].markdown(
        build_metric_card(
            "Predicted sales",
            f"{predicted_sales:.2f}",
            "Final output returned by the model for the current inputs.",
            "warm",
        ),
        unsafe_allow_html=True,
    )
    metric_columns[1].markdown(
        build_metric_card(
            "Strongest driver",
            strongest_channel,
            "Channel with the largest absolute effect on the prediction.",
            "teal",
        ),
        unsafe_allow_html=True,
    )
    metric_columns[2].markdown(
        build_metric_card(
            "Model intercept",
            f"{intercept:.2f}",
            "Baseline sales before any advertising contribution is added.",
            "blue",
        ),
        unsafe_allow_html=True,
    )
    metric_columns[3].markdown(
        build_metric_card(
            "Contribution sum",
            f"{total_from_contributions:.2f}",
            "Total lift created by the TV, Radio, and Newspaper inputs.",
            "neutral",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        build_section_heading(
            "Model logic",
            "How the regression formula is built",
            "The deployed app keeps the trained coefficients directly in code so the hosted version stays lightweight and dependable.",
        ),
        unsafe_allow_html=True,
    )

    formula_col, note_col = st.columns([1.25, 1])
    with formula_col:
        st.markdown(
            f"""
            <div class="glass-card warm">
                <div class="card-label">Prediction formula</div>
                <h3>Linear regression equation</h3>
                <p class="formula">
                    Sales = {intercept:.2f}
                    + ({MODEL_COEFFICIENTS[0]:.4f} x TV)
                    + ({MODEL_COEFFICIENTS[1]:.4f} x Radio)
                    + ({MODEL_COEFFICIENTS[2]:.4f} x Newspaper)
                </p>
                <p class="formula-note">
                    Because the model is linear, each channel contributes independently and
                    then gets added to the intercept. That makes the effect of each slider
                    easy to isolate and compare.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with note_col:
        st.markdown(
            f"""
            <div class="glass-card neutral">
                <div class="card-label">Reference scale</div>
                <h3>What the percentages mean</h3>
                <p class="formula-note">
                    The sliders do not use raw dollar values directly. They use a 0% to 100%
                    scale based on the largest budgets observed in the dataset:
                </p>
                <p class="formula-note">
                    TV max = ${TV_MAX:.1f}<br>
                    Radio max = ${RADIO_MAX:.1f}<br>
                    Newspaper max = ${NEWSPAPER_MAX:.1f}
                </p>
                <p class="formula-note">
                    This makes it easier to compare how strongly each channel reacts without
                    needing to remember three different budget ranges.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        build_section_heading(
            "Contribution view",
            "Which channel is driving the current prediction",
            "The bar chart below breaks the estimate into the amount contributed by each advertising channel.",
        ),
        unsafe_allow_html=True,
    )

    contribution_chart_col, contribution_info_col = st.columns([1.45, 1])
    with contribution_chart_col:
        fig_bar = plot_contribution_bar(contributions)
        st.pyplot(fig_bar, use_container_width=True)
    with contribution_info_col:
        st.markdown(
            describe_contribution_chart_html(contributions, intercept),
            unsafe_allow_html=True,
        )

    st.markdown(
        build_section_heading(
            "Sensitivity curves",
            "How each individual budget changes the forecast",
            "Each tab keeps two channels fixed while one channel moves across its full range from 0% to 100%.",
        ),
        unsafe_allow_html=True,
    )

    fixed_values = {
        "TV": tv_budget,
        "Radio": radio_budget,
        "Newspaper": newspaper_budget,
    }
    percent_range = np.linspace(0, 100, 101)
    plot_ranges = {
        "TV": percent_range * (TV_MAX / 100),
        "Radio": percent_range * (RADIO_MAX / 100),
        "Newspaper": percent_range * (NEWSPAPER_MAX / 100),
    }

    tabs = st.tabs(CHANNEL_ORDER)
    for tab, channel in zip(tabs, CHANNEL_ORDER):
        with tab:
            curve_col, curve_info_col = st.columns([1.45, 1])
            with curve_col:
                fig = plot_slice(model, fixed_values, channel, plot_ranges[channel])
                st.pyplot(fig, use_container_width=True)
            with curve_info_col:
                st.markdown(
                    describe_slice_plot_html(model, fixed_values, channel),
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"""
        <div class="footer-note">
            Dataset source:
            <a href="{DATASET_URL}" target="_blank">Advertising dataset on Kaggle</a><br>
            Live demo:
            <a href="{DEMO_URL}" target="_blank">{DEMO_URL}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
