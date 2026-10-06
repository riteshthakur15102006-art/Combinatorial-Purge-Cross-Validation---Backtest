import numpy as np                          
import pandas as pd                         
import matplotlib.pyplot as plt             
import seaborn as sns                       
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import KFold           
from skfolio.model_selection import (
    CombinatorialPurgedCV,                  
    WalkForward,                            
)
import warnings
warnings.filterwarnings('ignore')           


# 1. DATA GENERATION
# We build a synthetic price series that mimics 15-minute BTC bars.
# It has 4 distinct market "regimes" (up, down, choppy, mild up)
# so that any strategy has to generalise across regimes, not just ride one lucky trend.
print("[*] Generating synthetic 15-minute BTCUSDT-style data with a clear trend...")

np.random.seed(42)          
T = 10000                   # total number of 15-min bars (~104 days)
n_regimes = 4               

price = np.zeros(T)         
price[0] = 100              

# Each regime has its own volatility and drift (average return per bar)
regime_vol = np.array([0.005, 0.01, 0.02, 0.008])   # low, med, high, med-low
regime_ret = np.array([0.0005, -0.0004, 0.0, 0.0002])  # up, down, flat, mild up

for t in range(1, T):
    regime = t // (T // n_regimes)
    regime = min(regime, n_regimes - 1)             # clamp at the last regime

    # Random shock scaled by regime volatility
    shock = np.random.randn() * regime_vol[regime]

    # Geometric random walk: new price = old price * exp(drift + shock)
    price[t] = price[t-1] * np.exp(regime_ret[regime] + shock)


df = pd.DataFrame(
    {'price': price},
    index=pd.date_range('2023-01-01', periods=T, freq='15min')
)

# Compute simple bar-to-bar returns (% change)
df['returns'] = df['price'].pct_change()


# 2. FEATURE & TARGET ENGINEERING
# Feature (X): 10-bar momentum — how much price has moved in the last 10 bars. This is our only predictor.
# Target (y):  the NEXT bar's return — what we want to predict.

df['mom'] = df['price'].pct_change(10)      
df['target'] = df['returns'].shift(-1)      # next bar's return (look-ahead by design)

df.dropna(inplace=True)                     

X = df[['mom']].values                      
y = df['target'].values                     


# 3. BACKTESTING HELPERS

def calculate_sharpe(returns):
    """
    Annualised Sharpe ratio.
    For 15-minute bars there are ~35040 bars per year:
        365 days * 24 hours * 4 bars/hour = 35040
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    periods_per_year = 35040
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Fit model on the training slice, predict on the test slice,
    convert predictions into long/short positions, apply transaction
    costs, and return the Sharpe + the raw strategy returns.
    """
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Turn predictions into positions: +1 (long) or -1 (short)
    positions = np.sign(preds)


    # A "trade" occurs whenever the position changes from the previous bar
    trades = np.abs(np.diff(positions, prepend=0))
    transaction_cost = 0.0005            

    # Net return = position * actual return  -  cost * trades_this_bar
    strategy_returns = (positions * y_test) - (trades * transaction_cost)

    return {
        'sharpe': calculate_sharpe(strategy_returns),
        'returns': strategy_returns,
    }


results = {}


# 4. METHOD 1 — STANDARD K-FOLD CV  (The Trap)

# KFold randomly shuffles the data and splits into 10 folds.
# Because financial labels span time (y[t] = return from t to t+1),
# shuffled folds leak future information into the training set.
# This inflates performance and is a classic overfitting bug.

print("[*] Running Standard K-Fold CV...")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

kf_sharpes = []      
kf_returns = []      

for train_idx, test_idx in kf.split(X):
    # skfolio sometimes returns nested arrays — flatten with hstack
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)

    res = evaluate_model(
        LinearRegression(),
        X[train_idx], y[train_idx],
        X[test_idx],  y[test_idx]
    )

    kf_sharpes.append(res['sharpe'])
    kf_returns.extend(res['returns'])

results['K-Fold'] = {
    'sharpe_mean':  np.mean(kf_sharpes),
    'sharpe_std':   np.std(kf_sharpes),
    'sharpe_values': kf_sharpes,
    'equity_curve': np.cumprod(1 + np.array(kf_returns)),  
}


# 5. METHOD 2 — WALK-FORWARD
# The traditional fix: train on a rolling window of the PAST,
# test on the immediate FUTURE. No leakage — but only ONE path.
# We get a single fragile equity curve, not a distribution.

print("[*] Running Walk-Forward...")

wf = WalkForward(train_size=2000, test_size=500)

wf_sharpes = []
wf_returns = []

for train_idx, test_idx in wf.split(X):
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)

    res = evaluate_model(
        LinearRegression(),
        X[train_idx], y[train_idx],
        X[test_idx],  y[test_idx]
    )

    wf_sharpes.append(res['sharpe'])
    wf_returns.extend(res['returns'])

results['Walk-Forward'] = {
    'sharpe_mean':  np.mean(wf_sharpes),
    'sharpe_std':   np.std(wf_sharpes),
    'sharpe_values': wf_sharpes,
    'equity_curve': np.cumprod(1 + np.array(wf_returns)),
}


# 6. METHOD 3 — COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)

# The rigorous method:
#   - Split the timeline into n_folds contiguous blocks.
#   - For every combination of n_test_folds blocks used as TEST,
#     the rest are TRAINING.
#   - PURGE training rows whose label horizon overlaps the test.
#   - EMBARGO a buffer after the test set to kill serial correlation.
# This creates many train/test paths -> a DISTRIBUTION of outcomes.

print("[*] Running Combinatorial Purged Cross-Validation (CPCV)...")

cpcv = CombinatorialPurgedCV(
    n_folds=6,          # split the timeline into 6 blocks
    n_test_folds=2,     # in each split, hold out 2 blocks as test
    purged_size=20,     # remove 20 training bars around each test block
    embargo_size=5      # additionally skip 5 bars right after the test
)

cpcv_sharpes = []
cpcv_returns = []

for train_idx, test_idx in cpcv.split(X):
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)

    res = evaluate_model(
        LinearRegression(),
        X[train_idx], y[train_idx],
        X[test_idx],  y[test_idx]
    )

    cpcv_sharpes.append(res['sharpe'])
    cpcv_returns.extend(res['returns'])

results['CPCV'] = {
    'sharpe_mean':  np.mean(cpcv_sharpes),
    'sharpe_std':   np.std(cpcv_sharpes),
    'sharpe_values': cpcv_sharpes,
    'equity_curve': np.cumprod(1 + np.array(cpcv_returns)),
}


# 7. SUMMARY TABLE

# The most informative column is Std Dev, NOT Mean Sharpe.
# A low std = a reliable estimate. A high std = a fragile fluke.

print("\n" + "=" * 60)
print("BACKTESTING METHOD COMPARISON")
print("=" * 60)
print(f"{'Method':<20} | {'Mean Sharpe':<12} | {'Std Dev Sharpe':<15}")
print("-" * 60)
for method, res in results.items():
    print(f"{method:<20} | {res['sharpe_mean']:>10.2f}   | {res['sharpe_std']:>13.2f}")
print("=" * 60)


# 8. VISUALISATIONS

sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)

#  Plot 1: Sharpe Ratio Distribution across splits
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'K-Fold': 'red', 'Walk-Forward': 'orange', 'CPCV': 'green'}

for method, res in results.items():
    if len(res['sharpe_values']) > 0:
        sns.histplot(
            res['sharpe_values'],
            kde=True, ax=ax,
            label=method,
            color=colors[method],
            alpha=0.5,
            bins=15,
        )

ax.axvline(0, color='black', linestyle='--', linewidth=1)   # zero line
ax.set_title("Out-of-Sample Sharpe Ratio Distribution by Method", fontsize=16)
ax.set_xlabel("Sharpe Ratio", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('sharpe_distribution.png', dpi=200)
plt.show()

# Plot 2: Cumulative OOS Equity Curves 
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(results['K-Fold']['equity_curve'],       color='red',    linewidth=1.5,
        label='K-Fold (Misleading / Leakage)')
ax.plot(results['Walk-Forward']['equity_curve'], color='orange', linewidth=2,
        label='Walk-Forward')
ax.plot(results['CPCV']['equity_curve'],         color='green',  linewidth=2,
        label='CPCV (Robust)')

ax.set_title("Cumulative Out-of-Sample Equity Curves", fontsize=16)
ax.set_xlabel("Time (aggregated trades)", fontsize=12)
ax.set_ylabel("Cumulative Growth of $1", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('equity_curves.png', dpi=200)
plt.show()
