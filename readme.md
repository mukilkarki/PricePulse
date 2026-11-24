# PricePulse — Adaptive Dynamic Pricing & Revenue Optimization (Module 1–3 MVP)

An interactive **Streamlit** web application for e-commerce dynamic pricing, flash-sale spike detection, and SKU segmentation—built strictly with **Modules 1–3** (Python, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn) and structured for instant deployment on **Streamlit Community Cloud**.

## Curriculum Mapping

| Module | Concepts Applied in PricePulse |
| :--- | :--- |
| **Module 1 – Python & GitHub** | Modular pricing rule engine (`apply_pricing_rules`), functions, control flow, dictionaries/tuples/lists, and GitHub-ready repository structure. |
| **Module 2 – Data Analysis** | Synthetic e-commerce dataset with missing-value imputation (median/mode), interactive filtering & sorting, descriptive statistics, and Matplotlib/Seaborn EDA charts (category revenue, regional sales, competitor vs. optimal price, correlation heatmap). |
| **Module 3 – Machine Learning** | `train_test_split` (with `stratify` for classification), `StandardScaler`, **Linear Regression** (coefficients, intercept, MAE, R²), **Decision Tree Regression** (`max_depth` overfitting control & `plot_tree`), **Logistic Regression** (`max_iter`, `predict_proba()`, Accuracy, Confusion Matrix, Classification Report), and **K-Means Clustering** (`n_clusters`, `n_init`, `kmeans.labels_`, `groupby().mean()`). |

## Repository Structure

Place these three files in the root of your GitHub repository (for example, `pricepulse-dynamic-pricing`):

```text
pricepulse-dynamic-pricing/
├── app.py
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Live on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a public GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), click **New app**, and select your repository and branch.
3. Set **Main file path** to `app.py` and click **Deploy**.

## Planned Phase 2 Upgrades

Once the Module 1–3 baseline is live, you can upgrade PricePulse with:
- **Random Forest & XGBoost Regressors** (targeting MAE < 8)
- **70/30 ML-to-Competitor Price Blending** via REST API
- **Sliding-Window Time-Series Demand Aggregation** & Flash-Sale CEP
- **Reinforcement Learning (DQN / PPO)** for adaptive pricing policies
