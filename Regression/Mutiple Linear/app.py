import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

MODEL_PATH = Path("advertising_model.pkl")

# Maximum values from the dataset for scaling
TV_MAX = 296.4
RADIO_MAX = 49.6
NEWSPAPER_MAX = 114.0


def load_model(model_path: Path):
    """Load a trained sklearn regression model from a pickle file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    return model


def predict_sales(model, tv: float, radio: float, newspaper: float) -> float:
    """Predict sales from the model using a single row of feature values."""
    features = np.array([[tv, radio, newspaper]], dtype=float)
    prediction = model.predict(features)
    return float(prediction[0])


def compute_contributions(model, tv: float, radio: float, newspaper: float):
    """Compute contribution of each feature to the raw linear prediction."""
    coefs = np.array(model.coef_, dtype=float).ravel()
    if coefs.shape[0] != 3:
        raise ValueError("Expected model with three coefficients: TV, Radio, Newspaper")

    inputs = np.array([tv, radio, newspaper], dtype=float)
    contributions = coefs * inputs
    return {
        "TV": float(contributions[0]),
        "Radio": float(contributions[1]),
        "Newspaper": float(contributions[2]),
    }


def plot_contribution_bar(contributions):
    """Create a bar chart showing how much each input contributes."""
    labels = list(contributions.keys())
    values = list(contributions.values())

    # Define colors for each channel
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red for TV, Teal for Radio, Blue for Newspaper

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Contribution to Predicted Sales", fontsize=14, fontweight='bold')
    ax.set_ylabel("Contribution (Sales Units)", fontsize=12)
    ax.set_xlabel("Advertising Channel", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)  # Add horizontal line at y=0
    
    # Ensure y-axis starts from 0 or the most negative value
    y_min = min(0, min(values) * 1.1)  # Start from 0 or 10% below minimum
    y_max = max(values) * 1.1  # 10% above maximum
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig


def plot_slice(model, fixed_values, variable_name, variable_range):
    """Plot predicted sales across a range for one variable while holding others fixed."""
    tv_values = fixed_values["TV"]
    radio_values = fixed_values["Radio"]
    newspaper_values = fixed_values["Newspaper"]

    x_values = variable_range
    y_values = []
    for value in x_values:
        if variable_name == "TV":
            features = [value, radio_values, newspaper_values]
        elif variable_name == "Radio":
            features = [tv_values, value, newspaper_values]
        else:
            features = [tv_values, radio_values, value]

        prediction = model.predict(np.array([features], dtype=float))[0]
        y_values.append(prediction)

    # Convert x_values back to percentages for display
    if variable_name == "TV":
        x_display = x_values / (TV_MAX / 100)
    elif variable_name == "Radio":
        x_display = x_values / (RADIO_MAX / 100)
    else:
        x_display = x_values / (NEWSPAPER_MAX / 100)

    fig, ax = plt.subplots()
    ax.plot(x_display, y_values)
    current_percent = fixed_values[variable_name] / (TV_MAX / 100) if variable_name == "TV" else (
        fixed_values[variable_name] / (RADIO_MAX / 100) if variable_name == "Radio" else 
        fixed_values[variable_name] / (NEWSPAPER_MAX / 100)
    )
    ax.scatter([current_percent], [predict_sales(model, fixed_values["TV"], fixed_values["Radio"], fixed_values["Newspaper"])], zorder=5)
    ax.set_title(f"{variable_name} Budget % vs Predicted Sales")
    ax.set_xlabel(f"{variable_name} Budget (%)")
    ax.set_ylabel("Predicted Sales")
    ax.legend(["Prediction curve", "Current input"])
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def describe_contribution_chart(contributions, intercept: float) -> str:
    """Return a beginner-friendly explanation for the contribution bar chart."""
    strongest_channel = max(contributions, key=lambda name: abs(contributions[name]))
    strongest_value = contributions[strongest_channel]

    return f"""
### How to interpret this diagram
- The x-axis lists the advertising channels: TV, Radio, and Newspaper.
- The y-axis shows how much each channel adds to or subtracts from the prediction.
- A taller bar means that channel has a bigger effect at the current budget level.
- A bar above 0 pushes predicted sales up, while a bar below 0 pulls predicted sales down.
- The intercept ({intercept:.2f}) is the model's starting value, so it is not shown as a bar.

At the current settings, **{strongest_channel}** has the strongest effect because its contribution is **{strongest_value:.2f}**.
"""


def describe_slice_plot(model, fixed_values, variable_name: str) -> str:
    """Return a beginner-friendly explanation for a budget-vs-sales line plot."""
    coefficient_map = dict(
        zip(["TV", "Radio", "Newspaper"], np.array(model.coef_, dtype=float).ravel())
    )
    percent_map = {
        "TV": fixed_values["TV"] / (TV_MAX / 100),
        "Radio": fixed_values["Radio"] / (RADIO_MAX / 100),
        "Newspaper": fixed_values["Newspaper"] / (NEWSPAPER_MAX / 100),
    }
    current_prediction = predict_sales(
        model,
        fixed_values["TV"],
        fixed_values["Radio"],
        fixed_values["Newspaper"],
    )
    current_percent = percent_map[variable_name]
    other_channels = [name for name in ["TV", "Radio", "Newspaper"] if name != variable_name]
    slope = coefficient_map[variable_name]

    if slope > 0:
        trend_text = (
            f"The line slopes upward, so increasing the {variable_name} budget increases predicted sales."
        )
    elif slope < 0:
        trend_text = (
            f"The line slopes downward, so increasing the {variable_name} budget decreases predicted sales."
        )
    else:
        trend_text = (
            f"The line is almost flat, so changing the {variable_name} budget has very little effect on predicted sales."
        )

    return f"""
### How to interpret this diagram
- The x-axis shows the **{variable_name} budget percentage** from 0% to 100%.
- The y-axis shows the model's **predicted sales**.
- The line shows what happens when only the **{variable_name}** budget changes.
- The dot marks your current choice at **{current_percent:.0f}%**.
- The other budgets stay fixed at **{other_channels[0]} = {percent_map[other_channels[0]]:.0f}%** and **{other_channels[1]} = {percent_map[other_channels[1]]:.0f}%**.
- Because this is a linear regression model, the graph appears as a straight line.

{trend_text} At the highlighted point, the model predicts **{current_prediction:.2f}** sales.
"""


def main():
    st.title("Advertising Sales Prediction (Multiple Linear Regression)")

    st.markdown(
        """
        This app demonstrates how a trained multiple linear regression model uses advertising budgets
        for TV, Radio, and Newspaper to predict product sales.

        Use the sliders below to change the advertising budgets and see how the prediction updates.
        """
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found. Please place a trained pickle file named 'advertising_model.pkl' in the same folder as app.py."
        )
        return

    model = load_model(MODEL_PATH)

    st.sidebar.header("Advertising Budget Sliders (0-100%)")
    st.sidebar.markdown("*Values represent percentage of maximum observed budget in the dataset*")
    
    tv_percent = st.sidebar.slider("TV Budget %", min_value=0, max_value=100, value=50)
    radio_percent = st.sidebar.slider("Radio Budget %", min_value=0, max_value=100, value=50)
    newspaper_percent = st.sidebar.slider("Newspaper Budget %", min_value=0, max_value=100, value=18)
    
    # Convert percentages to actual budget values
    tv_budget = tv_percent * (TV_MAX / 100)
    radio_budget = radio_percent * (RADIO_MAX / 100)
    newspaper_budget = newspaper_percent * (NEWSPAPER_MAX / 100)

    st.subheader("Current input values")
    st.write(
        f"TV: {tv_budget:.1f} ({tv_percent}% of max)", 
        f"Radio: {radio_budget:.1f} ({radio_percent}% of max)",
        f"Newspaper: {newspaper_budget:.1f} ({newspaper_percent}% of max)"
    )

    st.markdown(
        """
        ### What is the model doing?
        The model estimates sales by learning a linear relationship between each advertising channel and sales.
        Each budget percentage contributes to the final prediction based on the model's learned coefficients.
        
        **Note**: Sliders show percentages of the maximum budget observed in the training data:
        - TV: 0-100% of $296.40
        - Radio: 0-100% of $49.60  
        - Newspaper: 0-100% of $114.00
        """
    )

    predicted_sales = predict_sales(model, tv_budget, radio_budget, newspaper_budget)
    st.metric("Predicted Sales", f"{predicted_sales:.2f}")

    st.markdown(
        """
        ### What does contribution mean?
        Each contribution shows how much each advertising channel adds to the predicted sales at the current budget levels.
        The bar chart displays the absolute contribution (coefficient × budget) for each channel.
        
        **Note**: The total prediction also includes the model's intercept (baseline sales).
        **Interpretation**: Higher bars indicate channels that have more impact on sales at the current budget percentages.
        Positive contributions increase sales, negative contributions decrease sales.
        """
    )

    contributions = compute_contributions(model, tv_budget, radio_budget, newspaper_budget)
    
    # Calculate total from contributions + intercept
    total_from_contributions = sum(contributions.values())
    intercept = float(model.intercept_)
    total_prediction = total_from_contributions + intercept
    
    st.write(f"**Model Intercept (Baseline Sales)**: {intercept:.2f}")
    st.write(f"**Total from Contributions**: {total_from_contributions:.2f}")
    st.write(f"**Final Prediction**: {total_prediction:.2f} (should match the metric above)")
    
    fig_bar = plot_contribution_bar(contributions)
    st.pyplot(fig_bar)
    st.markdown(describe_contribution_chart(contributions, intercept))

    st.markdown(
        """
        ### Visualizing the effect of each budget
        The line plots show how predicted sales change when one budget percentage moves from 0% to 100% 
        while the other budgets stay fixed at their current percentages.
        
        **Interpretation**: The slope of each line indicates how sensitive sales are to changes in that advertising channel.
        Steeper slopes mean the channel has more impact on sales prediction.
        """
    )

    fixed_values = {
        "TV": tv_budget,
        "Radio": radio_budget,
        "Newspaper": newspaper_budget,
    }

    # Create percentage ranges for plotting (0-100%)
    percent_range = np.linspace(0, 100, 101)
    
    # Convert percentage ranges to actual budget values for plotting
    tv_plot_range = percent_range * (TV_MAX / 100)
    radio_plot_range = percent_range * (RADIO_MAX / 100)
    newspaper_plot_range = percent_range * (NEWSPAPER_MAX / 100)

    st.subheader("TV Budget % vs Sales")
    fig_tv = plot_slice(model, fixed_values, "TV", tv_plot_range)
    st.pyplot(fig_tv)
    st.markdown(describe_slice_plot(model, fixed_values, "TV"))

    st.subheader("Radio Budget % vs Sales")
    fig_radio = plot_slice(model, fixed_values, "Radio", radio_plot_range)
    st.pyplot(fig_radio)
    st.markdown(describe_slice_plot(model, fixed_values, "Radio"))

    st.subheader("Newspaper Budget % vs Sales")
    fig_newspaper = plot_slice(model, fixed_values, "Newspaper", newspaper_plot_range)
    st.pyplot(fig_newspaper)
    st.markdown(describe_slice_plot(model, fixed_values, "Newspaper"))

    st.markdown(
        """
        ### How changing budget percentages affects prediction
        Increasing the percentage for any advertising channel increases its contribution to total sales.
        The model combines all contributions linearly to make the final sales prediction.
        
        **Key Insight**: Compare the slopes in the plots above - steeper lines indicate channels where 
        budget increases have more impact on sales.
        """
    )


if __name__ == "__main__":
    main()
