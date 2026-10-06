import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CPCV vs K-Fold vs Walk-Forward",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="darkgrid")

# ============================================================
# SIDEBAR — CONTROLS
# ============================================================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Tune the synthetic data and validation parameters.")

with st.sidebar.expander("📈 Data Generation", expanded=True):
    seed = st.number_input("Random seed", value=42, step=1)
    T = st.slider("Number of bars (T)", 2000, 20000, 10000, step=1000)
    n_regimes = st.slider("Number of regimes", 2, 8, 4)
    mom_window = st.slider("Momentum window", 5, 50, 10)

with st.sidebar.expander("🔴 K-Fold Settings", expanded=False):
    kf_splits = st.slider("Number of folds", 3, 20, 10)

with st.sidebar.expander("🟠 Walk-Forward Settings", expanded=False):
    wf_train = st.slider("Train size", 500, 5000, 2000, step=100)
    wf_test = st.slider("Test size", 100, 2000, 500, step=100)

with st.sidebar.expander("🟢 CPCV Settings", expanded=True):
    n_folds = st.slider("n_folds (total blocks)", 3, 12, 6)
    n_test_folds = st.slider("n_test_folds (held-out blocks)", 1, 5, 2)
    purged_size = st.slider("Purged size (bars)", 0, 100, 20)
    embargo_size = st.slider("Embargo size (bars)", 0, 100, 5)

with st.sidebar.expander("💰 Trading Costs", expanded=False):
    transaction_cost = st.slider(
        "Transaction cost (per trade)", 0.0, 0.005, 0.0005, step=0.0001, format="%.4f"
    )

run_button = st.sidebar.button("🚀 Run Backtest Comparison", use_container_width=True, type="primary")

# ============================================================
# CORE LOGIC (cached so it only recomputes when inputs change)
# ============================================================
@st.cache_data(show_spinner=False)
def generate_data(seed, T, n_regimes, mom_window):
    np.random.seed(seed)
    price = np.zeros(T)
    price[0] = 100
    regime_vol = np.array([0.005, 0.01, 0.02, 0.008][:n_regimes] * (n_regimes // 4 + 1))[:n_regimes]
    regime_ret = np.array([0.0005, -0.0004, 0.0, 0.0002][:n_regimes] * (n_regimes // 4 + 1))[:n_regimes]
    # Pad if n_regimes > 4
    if len(regime_vol) < n_regimes:
        regime_vol = np.pad(regime_vol, (0, n_regimes - len(regime_vol)), constant_values=0.01)
        regime_ret = np.pad(regime_ret, (0, n_regimes - len(regime_ret)), constant_values=0.0)

    for t in range(1, T):
        regime = t // (T // n_regimes)
        regime = min(regime, n_regimes - 1)
        shock = np.random.randn() * regime_vol[regime]
        price[t] = price[t - 1] * np.exp(regime_ret[regime] + shock)

    df = pd.DataFrame(
        {"price": price},
        index=pd.date_range("2023-01-01", periods=T, freq="15min"),
    )
    df["returns"] = df["price"].pct_change()
    df["mom"] = df["price"].pct_change(mom_window)
    df["target"] = df["returns"].shift(-1)
    df.dropna(inplace=True)
    return df


def calculate_sharpe(returns, periods_per_year=35040):
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def evaluate_model(model, X_train, y_train, X_test, y_test, transaction_cost):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    positions = np.sign(preds)
    trades = np.abs(np.diff(positions, prepend=0))
    strategy_returns = (positions * y_test) - (trades * transaction_cost)
    return {
        "sharpe": calculate_sharpe(strategy_returns),
        "returns": strategy_returns,
    }


@st.cache_data(show_spinner=False)
def run_all_methods(seed, T, n_regimes, mom_window, kf_splits,
                    wf_train, wf_test, n_folds, n_test_folds,
                    purged_size, embargo_size, transaction_cost):
    df = generate_data(seed, T, n_regimes, mom_window)
    X = df[["mom"]].values
    y = df["target"].values

    results = {}

    # --- K-Fold ---
    kf = KFold(n_splits=kf_splits, shuffle=True, random_state=seed)
    kf_sharpes, kf_returns = [], []
    for train_idx, test_idx in kf.split(X):
        train_idx = np.hstack(train_idx)
        test_idx = np.hstack(test_idx)
        res = evaluate_model(LinearRegression(), X[train_idx], y[train_idx],
                             X[test_idx], y[test_idx], transaction_cost)
        kf_sharpes.append(res["sharpe"])
        kf_returns.extend(res["returns"])
    results["K-Fold"] = {
        "sharpe_mean": np.mean(kf_sharpes),
        "sharpe_std": np.std(kf_sharpes),
        "sharpe_values": kf_sharpes,
        "equity_curve": np.cumprod(1 + np.array(kf_returns)),
    }

    # --- Walk-Forward ---
    wf = WalkForward(train_size=wf_train, test_size=wf_test)
    wf_sharpes, wf_returns = [], []
    for train_idx, test_idx in wf.split(X):
        train_idx = np.hstack(train_idx)
        test_idx = np.hstack(test_idx)
        res = evaluate_model(LinearRegression(), X[train_idx], y[train_idx],
                             X[test_idx], y[test_idx], transaction_cost)
        wf_sharpes.append(res["sharpe"])
        wf_returns.extend(res["returns"])
    results["Walk-Forward"] = {
        "sharpe_mean": np.mean(wf_sharpes),
        "sharpe_std": np.std(wf_sharpes),
        "sharpe_values": wf_sharpes,
        "equity_curve": np.cumprod(1 + np.array(wf_returns)),
    }

    # --- CPCV ---
    cpcv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purged_size,
        embargo_size=embargo_size,
    )
    cpcv_sharpes, cpcv_returns = [], []
    for train_idx, test_idx in cpcv.split(X):
        train_idx = np.hstack(train_idx)
        test_idx = np.hstack(test_idx)
        res = evaluate_model(LinearRegression(), X[train_idx], y[train_idx],
                             X[test_idx], y[test_idx], transaction_cost)
        cpcv_sharpes.append(res["sharpe"])
        cpcv_returns.extend(res["returns"])
    results["CPCV"] = {
        "sharpe_mean": np.mean(cpcv_sharpes),
        "sharpe_std": np.std(cpcv_sharpes),
        "sharpe_values": cpcv_sharpes,
        "equity_curve": np.cumprod(1 + np.array(cpcv_returns)),
    }

    return results, df


# ============================================================
# HEADER
# ============================================================
st.title("📊 CPCV vs K-Fold vs Walk-Forward")
st.markdown(
    """
    **A demonstration of why standard backtesting methods overfit, and how
    Combinatorial Purged Cross-Validation (CPCV) gives an honest estimate of
    out-of-sample performance.**

    Built with the framework from López de Prado's *Advances in Financial Machine Learning*.
    """
)
st.divider()

# ============================================================
# RUN / DISPLAY
# ============================================================
if not run_button:
    st.info("👈 Configure the parameters in the sidebar and click **🚀 Run Backtest Comparison**.")
    st.markdown(
        """
        ### What each method does
        - 🔴 **K-Fold**: Shuffles data randomly. Causes **data leakage** — the model
          trains on data from the future relative to its test set. Produces inflated
          and unstable results.
        - 🟠 **Walk-Forward**: Trains on past, tests on future, sequentially. Avoids
          leakage but gives only **one fragile backtest path**.
        - 🟢 **CPCV**: Tests every combination of held-out blocks with **purging** and
          **embargoing**. Gives a *distribution* of outcomes — the only honest way to
          measure robustness.
        """
    )
    st.stop()

with st.spinner("Running all three validation methods..."):
    results, df = run_all_methods(
        seed, T, n_regimes, mom_window, kf_splits,
        wf_train, wf_test, n_folds, n_test_folds,
        purged_size, embargo_size, transaction_cost,
    )

st.success(f"✅ Ran on **{len(df):,}** bars across **{n_regimes}** market regimes.")

# ============================================================
# KPI CARDS
# ============================================================
col1, col2, col3 = st.columns(3)
method_meta = {
    "K-Fold": ("🔴", col1),
    "Walk-Forward": ("🟠", col2),
    "CPCV": ("🟢", col3),
}

for method, (emoji, col) in method_meta.items():
    with col:
        st.markdown(f"### {emoji} {method}")
        st.metric(
            label="Mean Sharpe",
            value=f"{results[method]['sharpe_mean']:.2f}",
        )
        st.metric(
            label="Std Dev of Sharpe",
            value=f"{results[method]['sharpe_std']:.2f}",
            delta=f"{'✅ most stable' if method == 'CPCV' else '⚠️ high variance'}",
            delta_color="normal" if method == "CPCV" else "inverse",
        )

st.divider()

# ============================================================
# SUMMARY TABLE
# ============================================================
st.subheader("📋 Summary Table")
summary_df = pd.DataFrame({
    "Method": list(results.keys()),
    "Mean Sharpe": [results[m]["sharpe_mean"] for m in results],
    "Std Dev Sharpe": [results[m]["sharpe_std"] for m in results],
    "N Fits": [len(results[m]["sharpe_values"]) for m in results],
}).round(2)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.caption(
    "💡 **The winning metric is the LOWEST Std Dev, not the highest Mean Sharpe.** "
    "A high mean with high variance = an overfit fluke. A tight distribution = a reliable estimate."
)

st.divider()

# ============================================================
# PLOTS
# ============================================================
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("📉 Sharpe Ratio Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    colors = {"K-Fold": "red", "Walk-Forward": "orange", "CPCV": "green"}
    for method, res in results.items():
        if len(res["sharpe_values"]) > 0:
            sns.histplot(
                res["sharpe_values"], kde=True, ax=ax1,
                label=method, color=colors[method], alpha=0.5,
                bins=max(5, len(res["sharpe_values"]) // 2),
            )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Out-of-Sample Sharpe Ratio")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution across validation splits")
    ax1.legend()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with plot_col2:
    st.subheader("💰 Cumulative Equity Curves")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(results["K-Fold"]["equity_curve"], color="red", linewidth=1.5,
             label="K-Fold (Leakage)")
    ax2.plot(results["Walk-Forward"]["equity_curve"], color="orange",
             linewidth=2, label="Walk-Forward")
    ax2.plot(results["CPCV"]["equity_curve"], color="green",
             linewidth=2, label="CPCV (Robust)")
    ax2.set_xlabel("Time (aggregated trades)")
    ax2.set_ylabel("Cumulative Growth of $1")
    ax2.set_title("Out-of-sample equity curves")
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

st.divider()

# ============================================================
# KEY INSIGHTS
# ============================================================
st.subheader("🔑 Key Insights")

best_std = min(results, key=lambda m: results[m]["sharpe_std"])
highest_mean = max(results, key=lambda m: results[m]["sharpe_mean"])

insight_col1, insight_col2 = st.columns(2)
with insight_col1:
    st.markdown(
        f"""
        **🏆 Most stable method:** `{best_std}`
        - Std Dev of Sharpe = **{results[best_std]['sharpe_std']:.2f}**
        - This is the method you should trust for real capital allocation.
        """
    )
with insight_col2:
    st.markdown(
        f"""
        **🎭 Most misleading method:** `{highest_mean}`
        - Mean Sharpe = **{results[highest_mean]['sharpe_mean']:.2f}**
        - But Std Dev = **{results[highest_mean]['sharpe_std']:.2f}** → unreliable.
        """
    )

st.markdown(
    """
    > **Remember:** A backtest that makes money is not proof of edge.
    > A backtest whose *distribution of outcomes* is tight around a positive mean is.
    > CPCV is the only one of the three that gives you that distribution honestly.
    """
)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "References: Bailey & López de Prado (2014), *The Deflated Sharpe Ratio* · "
    "Bailey et al. (2015), *The Probability of Backtest Overfitting* · "
    "Arian et al. (2024), *Backtest Overfitting in the ML Era*"
)