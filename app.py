"""
PricePulse — Adaptive Dynamic Pricing & Revenue Optimization (Module 1–3 MVP)
=============================================================================
Internship Minor Project | Industry: E-Commerce / Retail | Live via Streamlit

Strictly aligned with Modules 1–3:
- Module 1: Python fundamentals, functions, control flow, data structures, GitHub-ready layout
- Module 2: NumPy, Pandas, Matplotlib, Seaborn (data cleaning, EDA, business insights)
- Module 3: Scikit-Learn (Linear Regression, Decision Tree Regression, Logistic Regression,
            K-Means Clustering, StandardScaler, train_test_split, MAE, R², Accuracy,
            Confusion Matrix, Classification Report)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PricePulse | Dynamic Pricing Engine (MVP)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="deep")

# Module 1: Core Data Structures (Tuples, Lists, Dictionaries, Sets)
PRODUCT_CATEGORIES = ("Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports")
REGIONS = ("North", "South", "East", "West")
CUSTOMER_SEGMENTS = ("Budget", "Regular", "Premium")

CATEGORY_BASE_PRICE_RANGES = {
    "Electronics": (120.0, 650.0),
    "Fashion": (25.0, 180.0),
    "Home & Kitchen": (35.0, 260.0),
    "Beauty": (15.0, 110.0),
    "Sports": (30.0, 240.0),
}

SEGMENT_MULTIPLIERS = {
    "Budget": 0.94,
    "Regular": 1.00,
    "Premium": 1.08,
}

NUMERIC_FEATURES = [
    "base_price",
    "competitor_price",
    "inventory_level",
    "click_events",
    "historical_orders",
    "seasonal_index",
]


# ---------------------------------------------------------------------------
# Module 1 & 2: Synthetic E-Commerce Dataset Generator & Preprocessing
# ---------------------------------------------------------------------------
@st.cache_data
def generate_ecommerce_data(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic e-commerce dynamic pricing dataset with ~2% missing values."""
    rng = np.random.default_rng(seed)

    categories = rng.choice(PRODUCT_CATEGORIES, size=n_rows, p=[0.24, 0.22, 0.20, 0.18, 0.16])
    regions = rng.choice(REGIONS, size=n_rows, p=[0.28, 0.26, 0.24, 0.22])
    segments = rng.choice(CUSTOMER_SEGMENTS, size=n_rows, p=[0.30, 0.48, 0.22])

    base_prices = np.array(
        [
            rng.uniform(*CATEGORY_BASE_PRICE_RANGES[cat])
            for cat in categories
        ]
    )
    competitor_prices = base_prices * rng.normal(1.0, 0.08, size=n_rows)
    competitor_prices = np.clip(competitor_prices, 10.0, None)

    inventory_levels = rng.integers(5, 320, size=n_rows).astype(float)
    seasonal_index = np.round(rng.uniform(0.75, 1.45, size=n_rows), 2)

    # Click events and historical orders driven by seasonality and segment
    click_events = (
        rng.poisson(lam=140, size=n_rows) * seasonal_index
        + rng.normal(0, 25, size=n_rows)
    )
    click_events = np.clip(click_events, 15, 600).astype(float)

    historical_orders = np.clip(
        (click_events * rng.uniform(0.08, 0.22, size=n_rows)).astype(int),
        1,
        150,
    ).astype(float)

    # Demand score (0–100 scale)
    demand_score = np.clip(
        (click_events / 6.0) * 0.55 + (historical_orders / 1.5) * 0.30 + seasonal_index * 12.0,
        5.0,
        100.0,
    )

    # Flash-sale / stockout risk flag (binary target for Logistic Regression)
    spike_signal = (
        (click_events > 220).astype(int)
        + (inventory_levels < 65).astype(int)
        + (seasonal_index > 1.20).astype(int)
        + (historical_orders > 38).astype(int)
    )
    flash_sale_prob = 1.0 / (1.0 + np.exp(-(1.35 * spike_signal - 2.35)))
    flash_sale_flag = (rng.random(n_rows) < flash_sale_prob).astype(int)

    # Optimal dynamic price (continuous target for Linear & Decision Tree Regression)
    seg_mult = np.array([SEGMENT_MULTIPLIERS[s] for s in segments])
    scarcity_adj = np.where(inventory_levels < 50, 1.09, np.where(inventory_levels > 240, 0.92, 1.00))
    flash_adj = np.where(flash_sale_flag == 1, 1.06, 1.00)

    optimal_price = (
        0.58 * base_prices
        + 0.34 * competitor_prices
        + 0.08 * click_events
        + 0.14 * historical_orders
        - 0.05 * inventory_levels
        + 14.0 * (seasonal_index - 1.0)
    ) * seg_mult * scarcity_adj * flash_adj + rng.normal(0, 4.5, size=n_rows)
    optimal_price = np.round(np.clip(optimal_price, 10.0, None), 2)

    unit_cost = base_prices * rng.uniform(0.52, 0.68, size=n_rows)
    revenue = np.round(optimal_price * historical_orders, 2)
    profit_margin = np.round(((optimal_price - unit_cost) / optimal_price) * 100.0, 2)

    df = pd.DataFrame(
        {
            "sku_id": [f"SKU-{10000 + i}" for i in range(n_rows)],
            "product_category": categories,
            "region": regions,
            "customer_segment": segments,
            "base_price": np.round(base_prices, 2),
            "competitor_price": np.round(competitor_prices, 2),
            "inventory_level": inventory_levels,
            "click_events": np.round(click_events, 0),
            "historical_orders": historical_orders,
            "seasonal_index": seasonal_index,
            "demand_score": np.round(demand_score, 1),
            "flash_sale_flag": flash_sale_flag,
            "optimal_price": optimal_price,
            "revenue": revenue,
            "profit_margin_pct": profit_margin,
        }
    )

    # Inject ~2% realistic missing values to demonstrate Module 2 cleaning
    for col in ("competitor_price", "inventory_level", "click_events", "region"):
        missing_idx = rng.choice(df.index, size=int(0.02 * n_rows), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def clean_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle missing values using median (numeric) and mode (categorical) imputation."""
    df = raw_df.copy()
    missing_before = df.isnull().sum().to_dict()

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Recompute derived fields cleanly if needed
    if {"optimal_price", "historical_orders"}.issubset(df.columns):
        df["revenue"] = np.round(df["optimal_price"] * df["historical_orders"], 2)

    missing_after = df.isnull().sum().to_dict()
    report = {
        "total_missing_before": int(sum(missing_before.values())),
        "total_missing_after": int(sum(missing_after.values())),
        "missing_by_column": {k: int(v) for k, v in missing_before.items() if v > 0},
    }
    return df, report


def apply_pricing_rules(
    ml_price: float,
    base_price: float,
    competitor_price: float,
    inventory_level: int,
    flash_sale_prob: float,
) -> dict:
    """
    Module 1 business rule engine:
    Combines ML prediction with inventory guards, flash-sale multipliers,
    and competitor price boundaries.
    """
    adjusted_price = ml_price
    reasons = []

    # Rule 1: Inventory scarcity or excess clearance
    if inventory_level < 35:
        adjusted_price *= 1.07
        reasons.append("Low inventory (<35 units): +7% scarcity protection")
    elif inventory_level > 250:
        adjusted_price *= 0.93
        reasons.append("High inventory (>250 units): -7% clearance discount")
    else:
        reasons.append("Inventory healthy (35–250 units): no stock adjustment")

    # Rule 2: Flash-sale / surge probability multiplier
    if flash_sale_prob >= 0.65:
        adjusted_price *= 1.05
        reasons.append(f"High demand spike probability ({flash_sale_prob:.0%}): +5% surge multiplier")
    elif flash_sale_prob <= 0.25:
        reasons.append(f"Stable demand ({flash_sale_prob:.0%} spike risk): standard pricing")
    else:
        reasons.append(f"Moderate demand activity ({flash_sale_prob:.0%} spike risk): monitor closely")

    # Rule 3: Competitor price guardrail (stay within ±15% of competitor & above 75% of base)
    lower_bound = max(base_price * 0.75, competitor_price * 0.85)
    upper_bound = competitor_price * 1.15
    final_price = float(np.clip(adjusted_price, lower_bound, upper_bound))

    if final_price != adjusted_price:
        reasons.append(
            f"Competitor guardrail applied (bounded to ${lower_bound:.2f}–${upper_bound:.2f})"
        )

    uplift_pct = ((final_price - base_price) / base_price) * 100.0 if base_price > 0 else 0.0

    return {
        "final_price": round(final_price, 2),
        "uplift_pct": round(uplift_pct, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Module 3: Model Training Pipeline (Cached for Fast Streamlit Interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_ml_models(
    _df: pd.DataFrame,
    data_signature: tuple[int, float],
    test_size: float = 0.20,
    tree_max_depth: int = 5,
    log_max_iter: int = 500,
    n_clusters: int = 4,
    n_init: int = 10,
) -> dict:
    """Train Linear Regression, Decision Tree Regression, Logistic Regression, and K-Means."""
    df = _df
    # Feature engineering: encode categorical columns as numeric floats
    feature_cols = NUMERIC_FEATURES + ["product_category", "customer_segment"]
    X_encoded = pd.get_dummies(df[feature_cols], drop_first=True, dtype=float)
    feature_names = list(X_encoded.columns)

    # --- Regression Target: optimal_price ---
    y_reg = df["optimal_price"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_encoded, y_reg, test_size=test_size, random_state=42
    )

    reg_scaler = StandardScaler()
    X_train_r_scaled = reg_scaler.fit_transform(X_train_r)
    X_test_r_scaled = reg_scaler.transform(X_test_r)

    # 1. Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_r_scaled, y_train_r)
    lin_train_pred = lin_reg.predict(X_train_r_scaled)
    lin_test_pred = lin_reg.predict(X_test_r_scaled)

    # 2. Decision Tree Regression
    dt_reg = DecisionTreeRegressor(max_depth=tree_max_depth, random_state=42)
    dt_reg.fit(X_train_r, y_train_r)
    dt_train_pred = dt_reg.predict(X_train_r)
    dt_test_pred = dt_reg.predict(X_test_r)

    # --- Classification Target: flash_sale_flag ---
    y_clf = df["flash_sale_flag"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_encoded, y_clf, test_size=test_size, random_state=42, stratify=y_clf
    )

    clf_scaler = StandardScaler()
    X_train_c_scaled = clf_scaler.fit_transform(X_train_c)
    X_test_c_scaled = clf_scaler.transform(X_test_c)

    # 3. Logistic Regression
    log_reg = LogisticRegression(max_iter=log_max_iter, random_state=42)
    log_reg.fit(X_train_c_scaled, y_train_c)
    log_pred = log_reg.predict(X_test_c_scaled)
    log_proba = log_reg.predict_proba(X_test_c_scaled)[:, 1]

    # --- Unsupervised Learning: K-Means Clustering ---
    cluster_features = ["base_price", "inventory_level", "click_events", "historical_orders", "seasonal_index"]
    km_scaler = StandardScaler()
    X_cluster_scaled = km_scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    kmeans.fit(X_cluster_scaled)

    return {
        "feature_names": feature_names,
        "cluster_features": cluster_features,
        "reg_scaler": reg_scaler,
        "clf_scaler": clf_scaler,
        "km_scaler": km_scaler,
        "lin_reg": lin_reg,
        "dt_reg": dt_reg,
        "log_reg": log_reg,
        "kmeans": kmeans,
        "reg_eval": {
            "y_test": y_test_r,
            "lin_train_mae": mean_absolute_error(y_train_r, lin_train_pred),
            "lin_test_mae": mean_absolute_error(y_test_r, lin_test_pred),
            "lin_train_r2": r2_score(y_train_r, lin_train_pred),
            "lin_test_r2": r2_score(y_test_r, lin_test_pred),
            "lin_test_pred": lin_test_pred,
            "dt_train_mae": mean_absolute_error(y_train_r, dt_train_pred),
            "dt_test_mae": mean_absolute_error(y_test_r, dt_test_pred),
            "dt_train_r2": r2_score(y_train_r, dt_train_pred),
            "dt_test_r2": r2_score(y_test_r, dt_test_pred),
            "dt_test_pred": dt_test_pred,
        },
        "clf_eval": {
            "y_test": y_test_c,
            "y_pred": log_pred,
            "y_proba": log_proba,
            "accuracy": accuracy_score(y_test_c, log_pred),
            "conf_matrix": confusion_matrix(y_test_c, log_pred),
            "report_dict": classification_report(y_test_c, log_pred, output_dict=True, zero_division=0),
        },
        "cluster_labels": kmeans.labels_,
    }


def encode_single_observation(
    base_price: float,
    competitor_price: float,
    inventory_level: int,
    click_events: int,
    historical_orders: int,
    seasonal_index: float,
    category: str,
    segment: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Align a single user input row to the exact training feature columns."""
    row = {col: 0.0 for col in feature_names}
    row["base_price"] = base_price
    row["competitor_price"] = competitor_price
    row["inventory_level"] = float(inventory_level)
    row["click_events"] = float(click_events)
    row["historical_orders"] = float(historical_orders)
    row["seasonal_index"] = seasonal_index

    cat_col = f"product_category_{category}"
    seg_col = f"customer_segment_{segment}"
    if cat_col in row:
        row[cat_col] = 1.0
    if seg_col in row:
        row[seg_col] = 1.0

    return pd.DataFrame([row], columns=feature_names)


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.title("📈 PricePulse Controls")
st.sidebar.caption("Module 1–3 Dynamic Pricing MVP")

uploaded_file = st.sidebar.file_uploader(
    "Upload E-Commerce CSV (optional)",
    type=["csv"],
    help="Leave empty to use the built-in 1,200-row simulated e-commerce pricing dataset.",
)

st.sidebar.subheader("Model Hyperparameters")
test_size = st.sidebar.slider("Test Split (`test_size`)", 0.15, 0.35, 0.20, 0.05)
tree_max_depth = st.sidebar.slider("Decision Tree `max_depth`", 1, 15, 5, 1)
log_max_iter = st.sidebar.slider("Logistic Regression `max_iter`", 100, 1000, 500, 100)
n_clusters = st.sidebar.slider("K-Means `n_clusters`", 2, 6, 4, 1)
n_init = st.sidebar.slider("K-Means `n_init`", 5, 20, 10, 5)

# Load & clean data
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    required_cols = set(NUMERIC_FEATURES + ["product_category", "region", "customer_segment", "flash_sale_flag", "optimal_price"])
    if not required_cols.issubset(raw_df.columns):
        st.sidebar.warning(
            "Uploaded CSV is missing required PricePulse columns; falling back to the built-in dataset."
        )
        raw_df = generate_ecommerce_data()
else:
    raw_df = generate_ecommerce_data()

df, clean_report = clean_dataset(raw_df)
data_signature = (len(df), round(float(df["optimal_price"].sum()), 2))
models = train_ml_models(
    df,
    data_signature=data_signature,
    test_size=test_size,
    tree_max_depth=tree_max_depth,
    log_max_iter=log_max_iter,
    n_clusters=n_clusters,
    n_init=n_init,
)

# ---------------------------------------------------------------------------
# Main Header & Top KPIs
# ---------------------------------------------------------------------------
st.title("PricePulse — Adaptive Dynamic Pricing & Revenue Optimization")
st.caption(
    "Internship Minor Project (Module 1–3 MVP) | Python • Pandas • Matplotlib/Seaborn • Scikit-Learn • Streamlit"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total SKUs Analyzed", f"{len(df):,}")
kpi2.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
kpi3.metric("Avg Optimal Price", f"${df['optimal_price'].mean():.2f}")
kpi4.metric("Flash Sale / Spike Rate", f"{df['flash_sale_flag'].mean() * 100:.1f}%")
kpi5.metric("Best Regression MAE", f"${min(models['reg_eval']['lin_test_mae'], models['reg_eval']['dt_test_mae']):.2f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1. Live Price Engine",
        "2. Data & Cleaning",
        "3. E-Commerce EDA",
        "4. Regression Models",
        "5. Flash Sale Classifier",
        "6. K-Means Segments",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: Live PricePulse Simulator (Real-Time Inference)
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Real-Time Dynamic Price Recommendation")
    st.write(
        "Adjust SKU signals below to generate an instant price recommendation using our trained "
        "Scikit-Learn models plus Module 1 inventory, flash-sale, and competitor guardrails."
    )

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        sim_category = st.selectbox("Product Category", PRODUCT_CATEGORIES, index=0)
        sim_segment = st.selectbox("Customer Segment", CUSTOMER_SEGMENTS, index=1)
        sim_model_choice = st.radio(
            "Primary Price Model",
            ["Linear Regression", "Decision Tree Regression"],
            horizontal=True,
        )
    with col_in2:
        sim_base_price = st.number_input("Base Price ($)", min_value=10.0, max_value=1000.0, value=250.0, step=5.0)
        sim_comp_price = st.number_input("Competitor Price ($)", min_value=10.0, max_value=1000.0, value=260.0, step=5.0)
        sim_seasonal = st.slider("Seasonal Demand Index", 0.75, 1.45, 1.15, 0.05)
    with col_in3:
        sim_inventory = st.slider("Inventory Level (Units)", 5, 320, 45, 5)
        sim_clicks = st.slider("Recent Click Events", 15, 600, 240, 5)
        sim_orders = st.slider("Historical Orders", 1, 150, 42, 1)

    obs_df = encode_single_observation(
        base_price=sim_base_price,
        competitor_price=sim_comp_price,
        inventory_level=sim_inventory,
        click_events=sim_clicks,
        historical_orders=sim_orders,
        seasonal_index=sim_seasonal,
        category=sim_category,
        segment=sim_segment,
        feature_names=models["feature_names"],
    )

    # Predictions across all 4 models
    lin_price_pred = float(models["lin_reg"].predict(models["reg_scaler"].transform(obs_df))[0])
    dt_price_pred = float(models["dt_reg"].predict(obs_df)[0])
    chosen_ml_price = lin_price_pred if sim_model_choice == "Linear Regression" else dt_price_pred

    flash_prob = float(models["log_reg"].predict_proba(models["clf_scaler"].transform(obs_df))[0, 1])

    km_input = pd.DataFrame(
        [[sim_base_price, sim_inventory, sim_clicks, sim_orders, sim_seasonal]],
        columns=models["cluster_features"],
    )
    assigned_cluster = int(models["kmeans"].predict(models["km_scaler"].transform(km_input))[0])

    rule_result = apply_pricing_rules(
        ml_price=chosen_ml_price,
        base_price=sim_base_price,
        competitor_price=sim_comp_price,
        inventory_level=sim_inventory,
        flash_sale_prob=flash_prob,
    )

    st.markdown("#### PricePulse Recommendation")
    res1, res2, res3, res4 = st.columns(4)
    res1.metric(
        "Recommended Dynamic Price",
        f"${rule_result['final_price']:.2f}",
        delta=f"{rule_result['uplift_pct']:+.1f}% vs Base",
    )
    res2.metric("Raw ML Price Prediction", f"${chosen_ml_price:.2f}", help=f"Linear: ${lin_price_pred:.2f} | Tree: ${dt_price_pred:.2f}")
    res3.metric(
        "Flash Sale / Spike Probability",
        f"{flash_prob * 100:.1f}%",
        delta="Alert" if flash_prob >= 0.65 else "Normal",
        delta_color="inverse" if flash_prob >= 0.65 else "normal",
    )
    res4.metric("Assigned SKU Segment", f"Cluster {assigned_cluster}")

    if flash_prob >= 0.65:
        st.warning(
            f"Flash-sale / stockout spike detected ({flash_prob:.0%} probability). "
            "Emergency pricing multiplier has been applied to protect margin and inventory."
        )
    else:
        st.success("Pricing is within normal operating bounds and competitor guardrails.")

    with st.expander("View Pricing Rule Breakdown", expanded=True):
        for r in rule_result["reasons"]:
            st.write(f"- {r}")
        st.caption(
            f"Guardrail range: ${rule_result['lower_bound']:.2f} – ${rule_result['upper_bound']:.2f} | "
            f"Linear Regression raw: ${lin_price_pred:.2f} | Decision Tree raw: ${dt_price_pred:.2f}"
        )

# ---------------------------------------------------------------------------
# TAB 2: Data Handling, Cleaning & Filtering (Module 2)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Data Cleaning, Filtering & Descriptive Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows × Columns", f"{df.shape[0]:,} × {df.shape[1]}")
    c2.metric("Missing Values Before Cleaning", clean_report["total_missing_before"])
    c3.metric("Missing Values After Cleaning", clean_report["total_missing_after"])

    if clean_report["missing_by_column"]:
        st.info(
            "Imputed missing values (median for numeric, mode for categorical): "
            + ", ".join(f"`{col}` ({cnt})" for col, cnt in clean_report["missing_by_column"].items())
        )

    f1, f2, f3 = st.columns(3)
    selected_cats = f1.multiselect("Filter by Category", sorted(df["product_category"].unique()), default=sorted(df["product_category"].unique()))
    selected_regions = f2.multiselect("Filter by Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    sort_col = f3.selectbox("Sort Table By", ["revenue", "optimal_price", "click_events", "inventory_level", "profit_margin_pct"])

    filtered_df = df[
        df["product_category"].isin(selected_cats) & df["region"].isin(selected_regions)
    ].sort_values(by=sort_col, ascending=False)

    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.markdown("#### Descriptive Statistics (`df.describe()`)")
    st.dataframe(filtered_df[NUMERIC_FEATURES + ["optimal_price", "revenue", "profit_margin_pct"]].describe().round(2), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Exploratory Data Analysis & Business Insights (Module 2)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("E-Commerce Sales, Demand & Pricing EDA")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        cat_summary = (
            df.groupby("product_category", as_index=False)
            .agg(total_revenue=("revenue", "sum"), avg_margin=("profit_margin_pct", "mean"))
            .sort_values("total_revenue", ascending=False)
        )
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
        sns.barplot(data=cat_summary, x="product_category", y="total_revenue", hue="product_category", legend=False, ax=ax1)
        ax1.set_title("Total Revenue by Product Category")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Total Revenue ($)")
        ax1.tick_params(axis="x", rotation=20)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with eda_col2:
        reg_summary = (
            df.groupby("region", as_index=False)
            .agg(total_revenue=("revenue", "sum"), avg_orders=("historical_orders", "mean"))
            .sort_values("total_revenue", ascending=False)
        )
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
        sns.barplot(data=reg_summary, x="region", y="total_revenue", hue="region", legend=False, ax=ax2)
        ax2.set_title("Regional Sales Performance")
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Total Revenue ($)")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        fig3, ax3 = plt.subplots(figsize=(6.5, 4.2))
        sns.scatterplot(
            data=df.sample(min(400, len(df)), random_state=42),
            x="competitor_price",
            y="optimal_price",
            hue="customer_segment",
            alpha=0.7,
            ax=ax3,
        )
        ax3.set_title("Competitor Price vs. Optimal Dynamic Price")
        ax3.set_xlabel("Competitor Price ($)")
        ax3.set_ylabel("Optimal Price ($)")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with eda_col4:
        corr_cols = NUMERIC_FEATURES + ["optimal_price", "flash_sale_flag"]
        fig4, ax4 = plt.subplots(figsize=(6.5, 4.2))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax4, annot_kws={"size": 8})
        ax4.set_title("Pricing Signal Correlation Heatmap")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

# ---------------------------------------------------------------------------
# TAB 4: Regression Models (Linear & Decision Tree — Module 3)
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Optimal Price Prediction: Linear vs. Decision Tree Regression")

    reg_eval = models["reg_eval"]
    comp_df = pd.DataFrame(
        {
            "Model": ["Linear Regression (StandardScaler)", f"Decision Tree Regressor (max_depth={tree_max_depth})"],
            "Train MAE ($)": [round(reg_eval["lin_train_mae"], 2), round(reg_eval["dt_train_mae"], 2)],
            "Test MAE ($)": [round(reg_eval["lin_test_mae"], 2), round(reg_eval["dt_test_mae"], 2)],
            "Train R²": [round(reg_eval["lin_train_r2"], 4), round(reg_eval["dt_train_r2"], 4)],
            "Test R²": [round(reg_eval["lin_test_r2"], 4), round(reg_eval["dt_test_r2"], 4)],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if reg_eval["dt_train_r2"] - reg_eval["dt_test_r2"] > 0.08:
        st.info(
            "Overfitting check: Decision Tree Train R² is noticeably higher than Test R². "
            "Try lowering `max_depth` in the sidebar to improve generalization."
        )

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        fig_lr, ax_lr = plt.subplots(figsize=(6.2, 4.2))
        ax_lr.scatter(reg_eval["y_test"], reg_eval["lin_test_pred"], alpha=0.55, label="Predictions")
        min_v, max_v = reg_eval["y_test"].min(), reg_eval["y_test"].max()
        ax_lr.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Ideal Fit (y = x)")
        ax_lr.set_title("Linear Regression: Actual vs. Predicted Price")
        ax_lr.set_xlabel("Actual Optimal Price ($)")
        ax_lr.set_ylabel("Predicted Optimal Price ($)")
        ax_lr.legend()
        fig_lr.tight_layout()
        st.pyplot(fig_lr)
        plt.close(fig_lr)

    with rcol2:
        coef_df = pd.DataFrame(
            {
                "Feature": models["feature_names"],
                "Slope (Coefficient)": np.round(models["lin_reg"].coef_, 3),
            }
        ).sort_values("Slope (Coefficient)", key=abs, ascending=False)
        st.markdown(f"**Linear Regression Intercept:** `${models['lin_reg'].intercept_:.2f}`")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.markdown("#### Decision Tree Visualization (`plot_tree`, top 3 levels)")
    fig_tree, ax_tree = plt.subplots(figsize=(14, 5.2))
    plot_tree(
        models["dt_reg"],
        feature_names=models["feature_names"],
        filled=True,
        rounded=True,
        max_depth=min(tree_max_depth, 3),
        fontsize=8,
        ax=ax_tree,
    )
    fig_tree.tight_layout()
    st.pyplot(fig_tree)
    plt.close(fig_tree)

# ---------------------------------------------------------------------------
# TAB 5: Classification Model (Logistic Regression — Module 3)
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Flash Sale & Stockout Spike Detection (Logistic Regression)")

    clf_eval = models["clf_eval"]
    st.metric("Test Accuracy (`accuracy_score`)", f"{clf_eval['accuracy'] * 100:.2f}%")

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=clf_eval["conf_matrix"],
            display_labels=["Normal (0)", "Flash Spike (1)"],
        )
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title("Confusion Matrix")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with ccol2:
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(clf_eval["report_dict"]).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("#### Sample Predicted Class Probabilities (`predict_proba()`)")
    prob_preview = pd.DataFrame(
        {
            "Actual Flag": clf_eval["y_test"].values[:12],
            "Predicted Flag": clf_eval["y_pred"][:12],
            "Spike Probability": np.round(clf_eval["y_proba"][:12], 3),
        }
    )
    st.dataframe(prob_preview, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 6: Unsupervised Segmentation (K-Means Clustering — Module 3)
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("SKU & Demand Tier Segmentation (K-Means Clustering)")

    clustered_df = df.copy()
    clustered_df["cluster"] = models["cluster_labels"]

    cluster_profile = (
        clustered_df.groupby("cluster")[
            ["base_price", "optimal_price", "inventory_level", "click_events", "historical_orders", "revenue"]
        ]
        .mean()
        .round(2)
    )
    cluster_profile["sku_count"] = clustered_df.groupby("cluster").size()
    st.markdown("#### Cluster Profiles (`groupby('cluster').mean()`)")
    st.dataframe(cluster_profile, use_container_width=True)

    kcol1, kcol2 = st.columns(2)
    with kcol1:
        fig_km1, ax_km1 = plt.subplots(figsize=(6.2, 4.2))
        sns.scatterplot(
            data=clustered_df.sample(min(500, len(clustered_df)), random_state=42),
            x="click_events",
            y="inventory_level",
            hue="cluster",
            palette="tab10",
            alpha=0.75,
            ax=ax_km1,
        )
        ax_km1.set_title("Clusters: Click Events vs. Inventory Level")
        ax_km1.set_xlabel("Click Events")
        ax_km1.set_ylabel("Inventory Level (Units)")
        fig_km1.tight_layout()
        st.pyplot(fig_km1)
        plt.close(fig_km1)

    with kcol2:
        fig_km2, ax_km2 = plt.subplots(figsize=(6.2, 4.2))
        sns.boxplot(data=clustered_df, x="cluster", y="optimal_price", hue="cluster", legend=False, palette="tab10", ax=ax_km2)
        ax_km2.set_title("Optimal Price Distribution Across Clusters")
        ax_km2.set_xlabel("Cluster ID")
        ax_km2.set_ylabel("Optimal Price ($)")
        fig_km2.tight_layout()
        st.pyplot(fig_km2)
        plt.close(fig_km2)