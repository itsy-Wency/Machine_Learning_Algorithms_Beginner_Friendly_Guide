from dataclasses import dataclass

import altair as alt
import numpy as np
import streamlit as st

GITHUB_URL = "https://github.com/itsy-Wency/Machine_Learning_Algorithms_Beginner_Friendly_Guide"
DATASET_URL = "https://www.kaggle.com/datasets/ashydv/advertising-dataset"

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
    "TV": "#F97316",
    "Radio": "#10B981",
    "Newspaper": "#3B82F6",
}
CHANNEL_MAX = {
    "TV": TV_MAX,
    "Radio": RADIO_MAX,
    "Newspaper": NEWSPAPER_MAX,
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
    """Apply a minimal Streamlit-aware visual layer."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--st-border-color, var(--border-color, rgba(128,128,128,0.18)));
        }

        .hero-card,
        .soft-card {
            background: var(--st-secondary-background-color, var(--secondary-background-color));
            border: 1px solid var(--st-border-color, var(--border-color, rgba(128,128,128,0.18)));
            border-radius: 1.35rem;
            padding: 1.25rem 1.35rem;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(
                90deg,
                var(--st-primary-color, var(--primary-color, #ff4b4b)),
                transparent
            );
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            font-weight: 700;
            opacity: 0.72;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.03;
            margin: 0 0 0.75rem;
            letter-spacing: -0.03em;
        }

        .hero-copy,
        .soft-copy {
            line-height: 1.7;
            opacity: 0.82;
            margin: 0;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .pill {
            border: 1px solid var(--st-border-color, var(--border-color, rgba(128,128,128,0.18)));
            border-radius: 999px;
            padding: 0.36rem 0.75rem;
            font-size: 0.84rem;
            opacity: 0.86;
        }

        .section-copy {
            margin-top: -0.2rem;
            margin-bottom: 0.9rem;
            opacity: 0.78;
            line-height: 1.7;
        }

        .formula-box {
            font-family: var(--st-code-font, monospace);
            font-size: 0.98rem;
            line-height: 1.8;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .tiny-note {
            font-size: 0.92rem;
            opacity: 0.78;
        }

        .footer-links {
            margin-top: 1rem;
            opacity: 0.8;
            line-height: 1.8;
        }

        @media (max-width: 900px) {
            .hero-card,
            .soft-card {
                padding: 1rem 1.05rem;
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


def compute_contributions(model, tv: float, radio: float, newspaper: float) -> dict[str, float]:
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


def contribution_chart(contributions: dict[str, float]) -> alt.Chart:
    """Build a minimalist contribution bar chart."""
    records = [
        {"Channel": channel, "Contribution": contributions[channel]}
        for channel in CHANNEL_ORDER
    ]
    zero_line = alt.Chart(alt.Data(values=[{"Baseline": 0}])).mark_rule(
        strokeDash=[4, 4],
        opacity=0.35,
    ).encode(y="Baseline:Q")

    bars = alt.Chart(alt.Data(values=records)).mark_bar(
        cornerRadiusTopLeft=8,
        cornerRadiusTopRight=8,
        cornerRadiusBottomLeft=8,
        cornerRadiusBottomRight=8,
    ).encode(
        x=alt.X("Channel:N", sort=CHANNEL_ORDER, axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y("Contribution:Q", title="Contribution to sales"),
        color=alt.Color(
            "Channel:N",
            sort=CHANNEL_ORDER,
            legend=None,
            scale=alt.Scale(
                domain=CHANNEL_ORDER,
                range=[CHANNEL_COLORS[channel] for channel in CHANNEL_ORDER],
            ),
        ),
        tooltip=[
            alt.Tooltip("Channel:N"),
            alt.Tooltip("Contribution:Q", format=".2f"),
        ],
    )

    return (
        zero_line + bars
    ).properties(height=320).configure_view(stroke=None)


def slice_chart(model, fixed_values: dict[str, float], variable_name: str) -> alt.Chart:
    """Build a theme-aware interactive line chart for one variable."""
    records = []
    for percent in range(101):
        budget_value = percent * (CHANNEL_MAX[variable_name] / 100)
        if variable_name == "TV":
            features = [budget_value, fixed_values["Radio"], fixed_values["Newspaper"]]
        elif variable_name == "Radio":
            features = [fixed_values["TV"], budget_value, fixed_values["Newspaper"]]
        else:
            features = [fixed_values["TV"], fixed_values["Radio"], budget_value]

        prediction = float(model.predict(np.array([features], dtype=float))[0])
        records.append(
            {
                "BudgetPercent": percent,
                "PredictedSales": prediction,
            }
        )

    current_percent = percent_for(variable_name, fixed_values[variable_name])
    current_prediction = predict_sales(
        model,
        fixed_values["TV"],
        fixed_values["Radio"],
        fixed_values["Newspaper"],
    )
    color = CHANNEL_COLORS[variable_name]

    base = alt.Chart(alt.Data(values=records)).encode(
        x=alt.X(
            "BudgetPercent:Q",
            title=f"{variable_name} budget (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        y=alt.Y("PredictedSales:Q", title="Predicted sales"),
    )

    hover = alt.selection_point(
        nearest=True,
        on="mouseover",
        fields=["BudgetPercent"],
        empty=False,
    )

    line = base.mark_line(color=color, strokeWidth=3).encode(
        tooltip=[
            alt.Tooltip("BudgetPercent:Q", title=f"{variable_name} budget (%)"),
            alt.Tooltip("PredictedSales:Q", title="Predicted sales", format=".2f"),
        ]
    )

    selectors = base.mark_point(opacity=0).add_params(hover)

    hover_points = base.mark_circle(size=68, color=color).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )

    hover_rule = base.mark_rule(color=color).encode(
        opacity=alt.condition(hover, alt.value(0.25), alt.value(0))
    )

    current_point = alt.Chart(
        alt.Data(
            values=[
                {
                    "BudgetPercent": current_percent,
                    "PredictedSales": current_prediction,
                }
            ]
        )
    ).mark_point(
        size=170,
        filled=True,
        color=color,
        shape="diamond",
    ).encode(
        x="BudgetPercent:Q",
        y="PredictedSales:Q",
        tooltip=[
            alt.Tooltip("BudgetPercent:Q", title="Current budget (%)", format=".0f"),
            alt.Tooltip("PredictedSales:Q", title="Current prediction", format=".2f"),
        ],
    )

    return (
        line + selectors + hover_points + hover_rule + current_point
    ).properties(height=340).configure_view(stroke=None)


def contribution_notes(contributions: dict[str, float], intercept: float) -> str:
    """Describe how to interpret the contribution chart."""
    strongest_channel = max(contributions, key=lambda name: abs(contributions[name]))
    strongest_value = contributions[strongest_channel]
    return (
        "- Each bar shows how much one channel adds to the current prediction.\n"
        "- Taller bars mean stronger influence at the current slider values.\n"
        "- The intercept is the model's starting point before any channel lift is added.\n\n"
        f"**Current takeaway:** {strongest_channel} is leading with a contribution of "
        f"`{strongest_value:.2f}`, while baseline sales start at `{intercept:.2f}`."
    )


def slice_notes(model, fixed_values: dict[str, float], variable_name: str) -> str:
    """Describe how to interpret a single-channel response curve."""
    coefficient_map = dict(zip(CHANNEL_ORDER, np.array(model.coef_, dtype=float).ravel()))
    current_prediction = predict_sales(
        model,
        fixed_values["TV"],
        fixed_values["Radio"],
        fixed_values["Newspaper"],
    )
    current_percent = percent_for(variable_name, fixed_values[variable_name])
    other_channels = [name for name in CHANNEL_ORDER if name != variable_name]
    slope = coefficient_map[variable_name]

    if slope > 0:
        trend_line = f"The line rises, so more {variable_name} spend lifts the forecast."
    elif slope < 0:
        trend_line = f"The line falls, so more {variable_name} spend lowers the forecast."
    else:
        trend_line = f"The line is nearly flat, so {variable_name} has little effect here."

    return (
        f"- The x-axis is the {variable_name} budget from 0% to 100% of its observed maximum.\n"
        "- The y-axis is the predicted sales output.\n"
        f"- The diamond marks your current setting at `{current_percent:.0f}%`.\n"
        f"- The other channels stay fixed at `{other_channels[0]} = {percent_for(other_channels[0], fixed_values[other_channels[0]]):.0f}%` "
        f"and `{other_channels[1]} = {percent_for(other_channels[1], fixed_values[other_channels[1]]):.0f}%`.\n\n"
        f"**Current takeaway:** {trend_line} At the current point, predicted sales are `{current_prediction:.2f}`."
    )


def main():
    st.set_page_config(
        page_title="Advertising Sales Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    model = load_model()

    st.sidebar.title("Campaign controls")
    st.sidebar.caption(
        "Adjust the budget mix to see how the regression model responds in real time."
    )
    tv_percent = st.sidebar.slider("TV Budget %", min_value=0, max_value=100, value=50)
    radio_percent = st.sidebar.slider("Radio Budget %", min_value=0, max_value=100, value=50)
    newspaper_percent = st.sidebar.slider(
        "Newspaper Budget %",
        min_value=0,
        max_value=100,
        value=18,
    )
    st.sidebar.divider()
    st.sidebar.caption("Observed maximum budgets")
    st.sidebar.caption(f"TV: ${TV_MAX:.1f}")
    st.sidebar.caption(f"Radio: ${RADIO_MAX:.1f}")
    st.sidebar.caption(f"Newspaper: ${NEWSPAPER_MAX:.1f}")

    tv_budget = tv_percent * (TV_MAX / 100)
    radio_budget = radio_percent * (RADIO_MAX / 100)
    newspaper_budget = newspaper_percent * (NEWSPAPER_MAX / 100)

    fixed_values = {
        "TV": tv_budget,
        "Radio": radio_budget,
        "Newspaper": newspaper_budget,
    }

    predicted_sales = predict_sales(model, tv_budget, radio_budget, newspaper_budget)
    contributions = compute_contributions(model, tv_budget, radio_budget, newspaper_budget)
    total_from_contributions = sum(contributions.values())
    intercept = float(model.intercept_)
    strongest_channel = max(contributions, key=lambda name: abs(contributions[name]))

    hero_col, snapshot_col = st.columns([1.45, 1], gap="large")
    with hero_col:
        st.markdown(
            """
            <div class="hero-card">
                <div class="eyebrow">Multiple Linear Regression</div>
                <h1 class="hero-title">Advertising Sales Prediction</h1>
                <p class="hero-copy">
                    A cleaner, theme-aware Streamlit dashboard for exploring how TV, Radio,
                    and Newspaper budgets combine inside a trained regression model.
                </p>
                <div class="pill-row">
                    <span class="pill">Minimalist layout</span>
                    <span class="pill">Light and dark mode friendly</span>
                    <span class="pill">Code repository linked below</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snapshot_col:
        with st.container(border=True):
            st.caption("Prediction snapshot")
            st.metric(
                "Predicted sales",
                f"{predicted_sales:.2f}",
                delta=f"{strongest_channel} is strongest",
                border=False,
            )
            st.markdown(
                f"""
                <div class="tiny-note">
                    Current variable lift: <strong>{total_from_contributions:.2f}</strong><br>
                    Model intercept: <strong>{intercept:.2f}</strong><br>
                    Code repository: <a href="{GITHUB_URL}" target="_blank">Machine Learning Algorithms: Beginner Friendly Guide</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Current campaign mix")
    st.markdown(
        '<p class="section-copy">Each card shows the active slider percentage, the translated budget value, and the dataset maximum used for scaling.</p>',
        unsafe_allow_html=True,
    )

    mix_cols = st.columns(3, gap="large")
    for column, channel in zip(mix_cols, CHANNEL_ORDER):
        with column:
            with st.container(border=True):
                channel_budget = fixed_values[channel]
                channel_percent = percent_for(channel, channel_budget)
                column_label = {
                    "TV": ":orange[TV]",
                    "Radio": ":green[Radio]",
                    "Newspaper": ":blue[Newspaper]",
                }[channel]
                st.markdown(column_label)
                st.metric(
                    f"{channel} budget",
                    f"{channel_percent:.0f}%",
                    delta=f"${channel_budget:.1f}",
                    delta_description="Current spend",
                    border=False,
                )
                st.progress(int(channel_percent))
                st.caption(f"Observed maximum: ${CHANNEL_MAX[channel]:.1f}")

    st.subheader("Model summary")
    st.markdown(
        '<p class="section-copy">The dashboard keeps the trained coefficients directly in code so the deployed app stays lightweight and reliable.</p>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Predicted sales", f"{predicted_sales:.2f}", border=True)
    metric_cols[1].metric("Strongest driver", strongest_channel, border=True)
    metric_cols[2].metric("Intercept", f"{intercept:.2f}", border=True)
    metric_cols[3].metric("Contribution sum", f"{total_from_contributions:.2f}", border=True)

    formula_col, scale_col = st.columns([1.15, 0.95], gap="large")
    with formula_col:
        with st.container(border=True):
            st.caption("Regression formula")
            st.markdown(
                f"""
                <div class="formula-box">
                Sales = {intercept:.2f}
                + ({MODEL_COEFFICIENTS[0]:.4f} * TV)
                + ({MODEL_COEFFICIENTS[1]:.4f} * Radio)
                + ({MODEL_COEFFICIENTS[2]:.4f} * Newspaper)
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Because the model is linear, each channel contributes independently and then gets added to the baseline intercept."
            )

    with scale_col:
        with st.container(border=True):
            st.caption("How the percentages work")
            st.markdown(
                """
                - `TV` uses the maximum observed budget of `$296.4`
                - `Radio` uses the maximum observed budget of `$49.6`
                - `Newspaper` uses the maximum observed budget of `$114.0`
                """
            )
            st.caption(
                "This standardizes the sliders so beginners can compare channel impact more easily."
            )

    st.subheader("Contribution view")
    st.markdown(
        '<p class="section-copy">This bar chart breaks the prediction into the amount currently contributed by each advertising channel.</p>',
        unsafe_allow_html=True,
    )

    contribution_chart_col, contribution_note_col = st.columns([1.55, 1], gap="large")
    with contribution_chart_col:
        st.altair_chart(contribution_chart(contributions), use_container_width=True)
    with contribution_note_col:
        with st.container(border=True):
            st.caption("How to interpret it")
            st.markdown(contribution_notes(contributions, intercept))

    st.subheader("Sensitivity curves")
    st.markdown(
        '<p class="section-copy">Each tab keeps two channels fixed while one budget moves from 0% to 100%. Hover the line for exact values.</p>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(CHANNEL_ORDER)
    for tab, channel in zip(tabs, CHANNEL_ORDER):
        with tab:
            chart_col, note_col = st.columns([1.55, 1], gap="large")
            with chart_col:
                st.altair_chart(
                    slice_chart(model, fixed_values, channel),
                    use_container_width=True,
                )
            with note_col:
                with st.container(border=True):
                    st.caption("How to interpret it")
                    st.markdown(slice_notes(model, fixed_values, channel))

    st.markdown(
        f"""
        <div class="footer-links">
            Dataset source:
            <a href="{DATASET_URL}" target="_blank">Advertising dataset on Kaggle</a><br>
            Code repository:
            <a href="{GITHUB_URL}" target="_blank">Machine Learning Algorithms: Beginner Friendly Guide</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
