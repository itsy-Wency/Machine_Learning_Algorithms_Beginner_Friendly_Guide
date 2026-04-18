# Machine Learning Algorithms Beginner Friendly Guide

<p align="center">
  <a href="#demo-links">
    <img src="./assets/ml-banner.svg" alt="Machine Learning beginner banner" width="100%" />
  </a>
</p>

<p align="center">
  Beginner-friendly guides and sample projects for practical machine learning algorithms.
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python badge" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter badge" />
  <img src="https://img.shields.io/badge/Level-Beginner-22C55E?style=for-the-badge" alt="Beginner badge" />
</p>

<p align="center">
  <img src="./assets/readme-divider.svg" alt="Animated divider" width="100%" />
</p>

## Demo Links

Current live Streamlit demos in this repository:

| App | Live Demo | Source |
| --- | --- | --- |
| Simple Linear Regression | [Open App](https://simple-linear-regression-model-studentscores.streamlit.app/) | [Regression/Simple Linear/app.py](./Regression/Simple%20Linear/app.py) |
| Multiple Linear Regression | [Open App](https://multiple-linear-regression-model-advertising-sales.streamlit.app/) | [Regression/Mutiple Linear/app.py](./Regression/Mutiple%20Linear/app.py) |

## App Previews

<p align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <a href="https://simple-linear-regression-model-studentscores.streamlit.app/">
          <img src="./assets/simple-linear-streamlit-preview.svg" alt="Simple Linear Regression Streamlit app preview in light and dark mode" width="100%" />
        </a>
        <br />
        <strong>Simple Linear Regression</strong>
      </td>
      <td align="center" width="50%">
        <a href="https://multiple-linear-regression-model-advertising-sales.streamlit.app/">
          <img src="./assets/multiple-linear-streamlit-preview.svg" alt="Multiple Linear Regression Streamlit app preview in light and dark mode" width="100%" />
        </a>
        <br />
        <strong>Multiple Linear Regression</strong>
      </td>
    </tr>
  </table>
</p>

<p align="center">
  Click any preview card in the gallery to open the live app.
</p>

## Purpose

This repository provides beginner-friendly, step-by-step machine learning walkthroughs with runnable notebooks.

Goals:
- learn one concept at a time
- run complete examples end-to-end
- understand outputs, metrics, and plots with plain-English interpretation

## Who This Is For

- Students starting machine learning
- Self-learners building practical intuition
- Anyone who wants guided notebooks with clear explanations

## Current Content

| Topic | Type | Path |
| --- | --- | --- |
| Simple Linear Regression | Guide | [Guide.txt](./Regression/Simple%20Linear/Guide.txt) |
| Simple Linear Regression | Notebook | [student_scores_regression.ipynb](./Regression/Simple%20Linear/student_scores_regression.ipynb) |
| Simple Linear Regression | Dataset | [Student_Performance.csv](./Regression/Simple%20Linear/Student_Performance.csv) |
| Simple Linear Regression | Live App | [Streamlit Demo](https://simple-linear-regression-model-studentscores.streamlit.app/) |
| Multiple Linear Regression | Guide | [Guide.txt](./Regression/Mutiple%20Linear/Guide.txt) |
| Multiple Linear Regression | Notebook | [advertising_regression.ipynb](./Regression/Mutiple%20Linear/advertising_regression.ipynb) |
| Multiple Linear Regression | Dataset | [advertising.csv](./Regression/Mutiple%20Linear/advertising.csv) |
| Multiple Linear Regression | Live App | [Streamlit Demo](https://multiple-linear-regression-model-advertising-sales.streamlit.app/) |

## Quick Start

1. Clone the repository.
2. Install dependencies:

```bash
pip install streamlit numpy altair matplotlib pandas scikit-learn seaborn notebook
```

3. Open one notebook:
- `Regression/Simple Linear/student_scores_regression.ipynb`
- `Regression/Mutiple Linear/advertising_regression.ipynb`

4. Run all cells in order.
5. Read the explanation markdown after each code block.

## Learning Roadmap

- [x] Simple Linear Regression
- [x] Multiple Linear Regression
- [ ] Logistic Regression
- [ ] Decision Tree
- [ ] Random Forest
- [ ] K-Nearest Neighbors
- [ ] Naive Bayes
- [ ] Support Vector Machine
- [ ] K-Means Clustering
- [ ] Principal Component Analysis

## Latest Updates

### April 18, 2026

- Added a new Simple Linear Regression Streamlit app for student score prediction using the exported pickle model.
- Added the Simple Linear Regression live demo link and a matching preview card to the README.
- Converted the app preview section into a gallery layout so both deployed regression apps are showcased together.
- Refined the Multiple Linear Regression Streamlit app into a more minimalist, theme-aware layout that adapts better to Streamlit light and dark mode.
- Added a dedicated README demo-links section and a clickable UI preview card for the live Streamlit app.

### April 16, 2026

- Enhanced both regression notebooks with education-focused markdown blocks after code cells.
- Added explicit output interpretation notes to explain tables, metrics, and printed values.
- Added figure interpretation notes for scatter plots, histograms, correlation heatmaps, pairplots, and residual plots.
- Improved wording in a few notebook interpretation lines for clearer beginner guidance.
- Updated this README to include Multiple Linear Regression content and revised project structure.
- Added all README visual assets from `assets/` to improve presentation and navigation flow.

## Project Structure

<p align="center">
  <img src="./assets/vscode-explorer-animated.svg" alt="VS Code style animated project explorer" width="100%" />
</p>

<p align="center">
  <img src="./assets/readme-divider.svg" alt="Animated divider" width="100%" />
</p>


## Contributing

Contributions are welcome, especially beginner-friendly examples that include:
- a short concept explanation
- a clean notebook with reproducible steps
- output and metric interpretation in plain language

## Notes

- Keep examples practical and simple first.
- Favor readable code over clever code.
- Design each new topic so a beginner can complete it in one sitting.
