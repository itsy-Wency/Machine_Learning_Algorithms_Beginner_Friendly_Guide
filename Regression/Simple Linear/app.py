from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "student_model.pkl"
FALLBACK_MODEL_PATH = APP_DIR / "student_scores_model.pkl"
DATASET_PATH = APP_DIR / "Student_Performance.csv"
DATASET_URL = "https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression"
GITHUB_URL = "https://github.com/itsy-Wency/Machine_Learning_Algorithms_Beginner_Friendly_Guide"


def inject_styles():
    """Apply the same card-based Streamlit styling pattern used in the multiple linear app."""
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


def load_model():
    """Load the trained linear regression model from disk."""
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as file:
            return pickle.load(file)

    if FALLBACK_MODEL_PATH.exists():
        with FALLBACK_MODEL_PATH.open("rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(
        "Could not find 'student_model.pkl' or 'student_scores_model.pkl' in the app folder."
    )


def load_dataset():
    """Load the dataset and normalize the column names used in the app."""
    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)
    rename_map = {
        "Hours": "Study Hours",
        "Scores": "Exam Score",
    }
    return df.rename(columns=rename_map)


def predict_score(model, hours_value: float) -> float:
    """Predict the exam score for one study-hours input."""
    features = pd.DataFrame({"Study Hours": [hours_value]})
    prediction = model.predict(features)
    return float(prediction[0])


def build_regression_curve(model):
    """Generate a smooth regression curve from 0 to 10 study hours."""
    hours_range = np.linspace(0, 10, 200)
    line_input = pd.DataFrame({"Study Hours": hours_range})
    line_predictions = model.predict(line_input)
    return hours_range, np.asarray(line_predictions, dtype=float)


def make_regression_plot(model, hours_value: float, predicted_score: float):
    """Plot the regression line and the selected user input point."""
    hours_range, line_predictions = build_regression_curve(model)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(hours_range, line_predictions, label="Regression Line")
    ax.scatter([hours_value], [predicted_score], label="Selected Input")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Score")
    ax.set_title("Regression Line and Current Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def make_training_plot(model, df: pd.DataFrame, hours_value: float, predicted_score: float):
    """Plot the training data together with the fitted regression line."""
    hours_range, line_predictions = build_regression_curve(model)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(df["Study Hours"], df["Exam Score"], label="Training Data")
    ax.plot(hours_range, line_predictions, label="Regression Line")
    ax.scatter([hours_value], [predicted_score], label="Selected Input")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Score")
    ax.set_title("Training Data with Regression Line")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def make_residual_plot(model, df: pd.DataFrame, hours_value: float, predicted_score: float):
    """Show the model prediction against the nearest observed point to explain residual error."""
    hours_range, line_predictions = build_regression_curve(model)
    nearest_index = (df["Study Hours"] - hours_value).abs().idxmin()
    nearest_row = df.loc[nearest_index]
    actual_hours = float(nearest_row["Study Hours"])
    actual_score = float(nearest_row["Exam Score"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(df["Study Hours"], df["Exam Score"], label="Training Data")
    ax.plot(hours_range, line_predictions, label="Regression Line")
    ax.scatter([hours_value], [predicted_score], label="Predicted Point")
    ax.scatter([actual_hours], [actual_score], label="Nearest Actual Point")
    ax.plot([actual_hours, actual_hours], [predicted_score, actual_score], label="Residual")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Score")
    ax.set_title("Residual View")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, actual_hours, actual_score, abs(actual_score - predicted_score)


def main():
    st.set_page_config(
        page_title="Student Score Prediction (Simple Linear Regression)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    model = load_model()
    df = load_dataset()

    intercept = float(model.intercept_)
    coefficient = float(np.ravel(model.coef_)[0])

    st.sidebar.title("Study controls")
    st.sidebar.caption(
        "Adjust the study hours slider to see how the trained regression model responds in real time."
    )
    study_hours = st.sidebar.slider(
        "Study Hours",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
    )
    st.sidebar.divider()
    st.sidebar.caption("Model setup")
    st.sidebar.caption("Input range: 0 to 10 hours")
    st.sidebar.caption("Target: Exam Score")

    predicted_score = predict_score(model, study_hours)
    slope_direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "flat"

    if df is not None and {"Study Hours", "Exam Score"}.issubset(df.columns):
        nearest_index = (df["Study Hours"] - study_hours).abs().idxmin()
        nearest_row = df.loc[nearest_index]
        nearest_score = float(nearest_row["Exam Score"])
        residual_value = abs(nearest_score - predicted_score)
    else:
        nearest_score = None
        residual_value = None

    hero_col, snapshot_col = st.columns([1.45, 1], gap="large")
    with hero_col:
        st.markdown(
            """
            <div class="hero-card">
                <div class="eyebrow">Simple Linear Regression</div>
                <h1 class="hero-title">Student Score Prediction</h1>
                <p class="hero-copy">
                    Explore how one input, study hours, connects to one output, exam score,
                    through a trained linear regression model.
                </p>
                <div class="pill-row">
                    <span class="pill">Single-input model</span>
                    <span class="pill">Beginner-friendly visuals</span>
                    <span class="pill">Equation explained live</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snapshot_col:
        with st.container(border=True):
            st.caption("Prediction snapshot")
            st.metric(
                "Predicted score",
                f"{predicted_score:.2f}",
                delta=f"{study_hours:.1f} study hours",
                border=False,
            )
            st.markdown(
                f"""
                <div class="tiny-note">
                    Slope direction: <strong>{slope_direction}</strong><br>
                    Intercept: <strong>{intercept:.2f}</strong><br>
                    Coefficient: <strong>{coefficient:.2f}</strong><br>
                    Code repository: <a href="{GITHUB_URL}" target="_blank">Machine Learning Algorithms: Beginner Friendly Guide</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Current input")
    st.markdown(
        '<p class="section-copy">This card shows the selected study time and the score predicted by the model for that exact value.</p>',
        unsafe_allow_html=True,
    )

    current_cols = st.columns(3, gap="large")
    with current_cols[0]:
        with st.container(border=True):
            st.caption("Study input")
            st.metric("Study Hours", f"{study_hours:.1f}", border=False)
            st.progress(int((study_hours / 10) * 100))
            st.caption("Selected on a 0 to 10 hour scale.")
    with current_cols[1]:
        with st.container(border=True):
            st.caption("Predicted output")
            st.metric("Exam Score", f"{predicted_score:.2f}", border=False)
            st.caption("This is the model's estimated score for the selected hours.")
    with current_cols[2]:
        with st.container(border=True):
            st.caption("Model behavior")
            st.metric("Slope", f"{coefficient:.2f}", border=False)
            st.caption("A positive slope means the line rises as study hours increase.")

    st.subheader("Model summary")
    st.markdown(
        '<p class="section-copy">These summary cards make the equation easier to read before you look at the charts.</p>',
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(4, gap="medium")
    summary_cols[0].metric("Predicted score", f"{predicted_score:.2f}", border=True)
    summary_cols[1].metric("Intercept", f"{intercept:.2f}", border=True)
    summary_cols[2].metric("Coefficient", f"{coefficient:.2f}", border=True)
    summary_cols[3].metric(
        "Residual",
        f"{residual_value:.2f}" if residual_value is not None else "N/A",
        border=True,
    )

    formula_col, notes_col = st.columns([1.15, 0.95], gap="large")
    with formula_col:
        with st.container(border=True):
            st.caption("Regression formula")
            st.markdown(
                f"""
                <div class="formula-box">
                Score = {intercept:.2f}
                + ({coefficient:.2f} * Hours)
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"For the current input: Score = {intercept:.2f} + ({coefficient:.2f} * {study_hours:.1f}) = {predicted_score:.2f}"
            )

    with notes_col:
        with st.container(border=True):
            st.caption("How to interpret it")
            st.markdown(
                f"""
                - **Linear regression** fits a straight line through data.
                - The **coefficient** tells how much the score changes for one extra hour of study.
                - The **intercept** is the predicted score when study hours are 0.
                - Because the slope is `{coefficient:.2f}`, each extra hour changes the prediction by about `{coefficient:.2f}` points.
                """
            )

    st.subheader("Regression line view")
    st.markdown(
        '<p class="section-copy">This chart shows the full fitted line from 0 to 10 hours and highlights your current prediction.</p>',
        unsafe_allow_html=True,
    )

    chart_col, note_col = st.columns([1.55, 1], gap="large")
    with chart_col:
        regression_fig = make_regression_plot(model, study_hours, predicted_score)
        st.pyplot(regression_fig, use_container_width=True)
        plt.close(regression_fig)
    with note_col:
        with st.container(border=True):
            st.caption("How to interpret it")
            st.markdown(
                f"""
                - The line represents the model's predicted score for every study-hour value from 0 to 10.
                - The highlighted point marks your current input at `{study_hours:.1f}` hours.
                - If the line rises from left to right, more study time increases the predicted score.

                **Current takeaway:** At `{study_hours:.1f}` hours, the model predicts a score of `{predicted_score:.2f}`.
                """
            )

    st.subheader("Training data view")
    st.markdown(
        '<p class="section-copy">This comparison helps beginners see how the regression line sits across the original dataset points.</p>',
        unsafe_allow_html=True,
    )

    training_col, training_note_col = st.columns([1.55, 1], gap="large")
    with training_col:
        if df is not None and {"Study Hours", "Exam Score"}.issubset(df.columns):
            training_fig = make_training_plot(model, df, study_hours, predicted_score)
            st.pyplot(training_fig, use_container_width=True)
            plt.close(training_fig)
        else:
            st.info("Training dataset not found, so the scatter plot is unavailable.")
    with training_note_col:
        with st.container(border=True):
            st.caption("How to interpret it")
            if df is not None and {"Study Hours", "Exam Score"}.issubset(df.columns):
                st.markdown(
                    """
                    - Each dot is one example from the dataset.
                    - The regression line summarizes the average linear pattern across those points.
                    - Points close to the line are predicted well.
                    - Points farther from the line have larger errors.
                    """
                )
            else:
                st.markdown(
                    "Add `Student_Performance.csv` beside `app.py` if you want to display the original training scatter plot."
                )

    st.subheader("Residual view")
    st.markdown(
        '<p class="section-copy">Residuals show the difference between an actual observed score and the model prediction.</p>',
        unsafe_allow_html=True,
    )

    residual_col, residual_note_col = st.columns([1.55, 1], gap="large")
    with residual_col:
        if df is not None and {"Study Hours", "Exam Score"}.issubset(df.columns):
            residual_fig, actual_hours, actual_score, residual_gap = make_residual_plot(
                model, df, study_hours, predicted_score
            )
            st.pyplot(residual_fig, use_container_width=True)
            plt.close(residual_fig)
        else:
            st.info("Residual view needs the original training dataset.")
            actual_hours = None
            actual_score = None
            residual_gap = None
    with residual_note_col:
        with st.container(border=True):
            st.caption("How to interpret it")
            if residual_gap is not None:
                st.markdown(
                    f"""
                    - The predicted point is the score from the regression line.
                    - The nearest actual point comes from the original dataset.
                    - The vertical gap between them is the residual error.

                    **Current takeaway:** Near `{actual_hours:.1f}` hours, the actual score is `{actual_score:.2f}` and the residual size is about `{residual_gap:.2f}`.
                    """
                )
            else:
                st.markdown(
                    "Residual explanation becomes available once the original dataset is loaded."
                )

    st.markdown(
        f"""
        <div class="footer-links">
            Dataset source:
            <a href="{DATASET_URL}" target="_blank">Student Performance dataset on Kaggle</a><br>
            Code repository:
            <a href="{GITHUB_URL}" target="_blank">Machine Learning Algorithms: Beginner Friendly Guide</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
