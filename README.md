# Titanic Survival Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Status-Educational%20Project-blue?style=for-the-badge" alt="Educational project"/>
</p>

A machine-learning classification project that predicts whether a Titanic passenger survived using passenger attributes such as ticket class, sex, age, family aboard, fare, and port of embarkation. The repository includes a Jupyter notebook, a model-training script, a cleaned CSV dataset, and a Streamlit interface.

> **Important:** This is an educational demonstration using historical data. Predictions are not a validated risk assessment and should not be interpreted as certainty about an individual outcome.

## Features

- Exploratory analysis and Decision Tree classification in Jupyter.
- A Python training script that evaluates a Decision Tree and saves a trained model.
- Interactive Streamlit interface for trying passenger profiles.
- Reproducible local setup instructions.

## Repository contents

| File | Purpose |
|---|---|
| `Decision-Tree-Classification.ipynb` | Notebook for the classification workflow |
| `titanic_clean.csv` | Cleaned dataset used by the training script |
| `train_model.py` | Preprocesses data, trains the model, evaluates accuracy, and saves `titanic_model.pkl` |
| `app.py` | Streamlit prediction interface |
| `requirements.txt` | Python dependencies |

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/HBE-2206/titanic-survival-prediction-ml.git
cd titanic-survival-prediction-ml
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
```

Activate it:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Then install the packages:

```bash
python -m pip install -r requirements.txt
```

### 3. Train the model

Make sure `titanic_clean.csv` is in the repository root, then run:

```bash
python train_model.py
```

The script should create `titanic_model.pkl` after a successful training run.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

The app requires the trained `titanic_model.pkl` file to be available in the project root.

## Modeling workflow

1. Load and inspect the dataset.
2. Handle missing values and prepare categorical features.
3. Split the data into training and test subsets.
4. Fit a Decision Tree classifier.
5. Evaluate test accuracy.
6. Save the model for use by the Streamlit app.

## Evaluation and limitations

The training script reports accuracy for its configured train/test split. Accuracy alone does not describe all classification errors, and results may vary with the dataset and split. For a stronger evaluation, report precision, recall, F1-score, a confusion matrix, and cross-validation results.

The current training script and app should be reviewed together whenever preprocessing changes. In particular, categorical encoding must remain identical between training and inference. The saved model file is generated locally and is not required to be committed to GitHub.

## Author

**Hazem Elshafey**

- GitHub: [@HBE-2206](https://github.com/HBE-2206)
- LinkedIn: [Hazem Elshafey](https://www.linkedin.com/in/hazem-el-shafey-b56405331)
- Email: [hm2502630@gmail.com](mailto:hm2502630@gmail.com)

---

Suggestions and constructive feedback are welcome.
