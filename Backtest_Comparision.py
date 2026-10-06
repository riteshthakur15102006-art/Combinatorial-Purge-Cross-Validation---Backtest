import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA GENERATION (Clearer Trend + Noise)
# ============================================================
print("[*] Generating synthetic 15-minute BTCUSDT-style data with a clear trend...")
np.random.seed(42)
T = 10000 # More data for CPCV to shine
n_regimes = 4

price = np.zeros(T)
price[0] = 100
# Regime 0: Strong Up, 1: Strong Down, 2: Choppy, 3: Mild Up
regime_vol = np.array([0.005, 0.01, 0.02, 0.008])
regime_ret = np.array([0.0005, -0.0004, 0.0, 0.0002])

for t in range(1, T):
    regime = t // (T // n_regimes)
    regime = min(regime, n_regimes - 1)
    shock = np.random.randn() * regime_vol[regime]
    price[t] = price[t-1] * np.exp(regime_ret[regime] + shock)

df = pd.DataFrame({'price': price}, index=pd.date_range('2023-01-01', periods=T, freq='15min'))
df['returns'] = df['price'].pct_change()

# ============================================================
# 2. FEATURE & TARGET ENGINEERING
# ============================================================
# A slightly better signal: 10-period momentum
df['mom'] = df['price'].pct_change(10)
# Target: next bar's return
df['target'] = df['returns'].shift(-1)

df.dropna(inplace=True)

X = df[['mom']].values
y = df['target'].values

# ============================================================
# 3. BACKTESTING FRAMEWORK (With Transaction Costs)
# ============================================================
def calculate_sharpe(returns):
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    # Annualization factor for 15-min bars (365 days * 24 hours * 4)
    periods_per_year = 35040
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Strategy: Long if pred > 0, Short if pred < 0
    positions = np.sign(preds)
    
    # FIX: Add transaction costs (0.05% per trade)
    # This punishes K-Fold for overtrading on leaked information
    trades = np.abs(np.diff(positions, prepend=0))
    transaction_cost = 0.0005
    
    strategy_returns = (positions * y_test) - (trades * transaction_cost)
    
    return {
        'sharpe': calculate_sharpe(strategy_returns),
        'returns': strategy_returns,
    }

results = {}

# ============================================================
# 4. METHOD 1: STANDARD K-FOLD CV (The Trap)
# ============================================================
print("[*] Running Standard K-Fold CV...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf_sharpes = []
kf_returns = []

for train_idx, test_idx in kf.split(X):
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)
    res = evaluate_model(LinearRegression(), X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    kf_sharpes.append(res['sharpe'])
    kf_returns.extend(res['returns'])

results['K-Fold'] = {
    'sharpe_mean': np.mean(kf_sharpes),
    'sharpe_std': np.std(kf_sharpes),
    'sharpe_values': kf_sharpes,
    'equity_curve': np.cumprod(1 + np.array(kf_returns))
}

# ============================================================
# 5. METHOD 2: WALK-FORWARD
# ============================================================
print("[*] Running Walk-Forward...")
wf = WalkForward(train_size=2000, test_size=500)
wf_sharpes = []
wf_returns = []

for train_idx, test_idx in wf.split(X):
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)
    res = evaluate_model(LinearRegression(), X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    wf_sharpes.append(res['sharpe'])
    wf_returns.extend(res['returns'])

results['Walk-Forward'] = {
    'sharpe_mean': np.mean(wf_sharpes),
    'sharpe_std': np.std(wf_sharpes),
    'sharpe_values': wf_sharpes,
    'equity_curve': np.cumprod(1 + np.array(wf_returns))
}

# ============================================================
# 6. METHOD 3: CPCV
# ============================================================
print("[*] Running Combinatorial Purged Cross-Validation (CPCV)...")
cpcv = CombinatorialPurgedCV(
    n_folds=6, 
    n_test_folds=2, 
    purged_size=20,
    embargo_size=5
)

cpcv_sharpes = []
cpcv_returns = []

for train_idx, test_idx in cpcv.split(X):
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)
    res = evaluate_model(LinearRegression(), X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    cpcv_sharpes.append(res['sharpe'])
    cpcv_returns.extend(res['returns'])

results['CPCV'] = {
    'sharpe_mean': np.mean(cpcv_sharpes),
    'sharpe_std': np.std(cpcv_sharpes),
    'sharpe_values': cpcv_sharpes,
    'equity_curve': np.cumprod(1 + np.array(cpcv_returns))
}

# ============================================================
# 7. PRINT SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("BACKTESTING METHOD COMPARISON")
print("="*60)
print(f"{'Method':<20} | {'Mean Sharpe':<12} | {'Std Dev Sharpe':<15}")
print("-"*60)
for method, res in results.items():
    print(f"{method:<20} | {res['sharpe_mean']:>10.2f}   | {res['sharpe_std']:>13.2f}")
print("="*60)

# ============================================================
# 8. VISUALIZATIONS
# ============================================================
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Plot 1: Sharpe Ratio Distribution
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'K-Fold': 'red', 'Walk-Forward': 'orange', 'CPCV': 'green'}

for method, res in results.items():
    if len(res['sharpe_values']) > 0:
        sns.histplot(res['sharpe_values'], kde=True, ax=ax, label=method, color=colors[method], alpha=0.5, bins=15)

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_title("Out-of-Sample Sharpe Ratio Distribution by Method", fontsize=16)
ax.set_xlabel("Sharpe Ratio", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('sharpe_distribution.png', dpi=200)
plt.show()

# Plot 2: Equity Curves
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(results['K-Fold']['equity_curve'], color='red', linewidth=1.5, label='K-Fold (Misleading / Leakage)')
ax.plot(results['Walk-Forward']['equity_curve'], color='orange', linewidth=2, label='Walk-Forward')
ax.plot(results['CPCV']['equity_curve'], color='green', linewidth=2, label='CPCV (Robust)')

ax.set_title("Cumulative Out-of-Sample Equity Curves", fontsize=16)
ax.set_xlabel("Time (aggregated trades)", fontsize=12)
ax.set_ylabel("Cumulative Growth of $1", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('equity_curves.png', dpi=200)
plt.show()
