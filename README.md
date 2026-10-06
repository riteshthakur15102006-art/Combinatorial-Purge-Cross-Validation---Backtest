# Backtesting Validation Methods: A Comparative Study of K-Fold, Walk-Forward, and Combinatorial Purged Cross-Validation

A quantitative research project demonstrating the impact of validation methodology on out-of-sample performance estimation in algorithmic trading. The project implements and compares three cross-validation approaches on synthetic 15-minute BTCUSDT-style market data, quantifying the degree to which each method mitigates backtest overfitting.

## Table of Contents

- Overview
- Motivation
- Methodology
- Key Results
- Interactive Demo
- Installation
- Usage
- Project Structure
- Technical Implementation
- Research Context
- References
- Author

## Overview

Standard machine learning workflows employ K-Fold cross-validation for model selection. In financial time series, this approach is fundamentally flawed: it shuffles observations across time, causing the model's training set to contain information that is temporally posterior to its test set. This is commonly referred to as data leakage.

This project provides an empirical comparison of three validation frameworks:

1. K-Fold Cross-Validation - the standard machine learning approach, incompatible with time series due to data leakage.
2. Walk-Forward Validation - a sequential approach that eliminates leakage but yields only a single backtest path.
3. Combinatorial Purged Cross-Validation (CPCV) - a method developed by Marcos Lopez de Prado that generates multiple backtest paths while applying purging and embargoing to eliminate leakage from overlapping label horizons.

The experiment applies all three methods to an identical linear strategy on identical data, then evaluates the resulting Sharpe ratio distributions.

## Motivation

Quantitative research faces a reproducibility crisis. A 2023 study by Harvey, Liu, and Zhu found that a substantial fraction of published empirical asset pricing findings fail to replicate out-of-sample. A principal cause is the misuse of statistical validation: researchers optimize on a single historical sample and report the resulting Sharpe ratio without controlling for the number of trials attempted or the temporal structure of the underlying data.

CPCV addresses both issues:

- Temporal integrity - purging removes training observations whose label periods overlap with the test set; embargoing removes a buffer after each test block to account for serial correlation.
- Multiple paths - by systematically holding out every combination of k groups out of N total groups, CPCV produces a distribution of backtest outcomes rather than a single number.

This project demonstrates these properties empirically.

## Methodology

### Data

A synthetic price series is generated to simulate 15-minute BTCUSDT bars across four distinct market regimes:

| Regime | Volatility | Drift   | Interpretation    |
|--------|------------|---------|-------------------|
| 0      | 0.005      | +0.0005 | Calm uptrend      |
| 1      | 0.010      | -0.0004 | Volatile downtrend|
| 2      | 0.020      | 0.0000  | Choppy sideways   |
| 3      | 0.008      | +0.0002 | Steady uptrend    |

The price process follows a geometric random walk:

    P_t = P_(t-1) * exp(drift_regime + epsilon_t * volatility_regime)

Total length: 10,000 bars. Fixed random seed for reproducibility.

### Feature and Target

- Feature: 10-bar rolling momentum (pct_change(10))
- Target: Next-bar return (returns.shift(-1))
- Model: Ordinary least squares linear regression

### Strategy

Predictions are converted to positions via the sign function (+/-1). Transaction costs of 5 basis points are applied per position change. Net strategy returns are computed as:

    r_strategy = position * r_actual - trades * 0.0005

### Validation Splitters

| Method       | Configuration                                  | Number of Fits |
|--------------|------------------------------------------------|----------------|
| K-Fold       | 10 folds, shuffled                             | 10             |
| Walk-Forward | 2,000-bar train, 500-bar test                  | 15             |
| CPCV         | 6 folds, 2 test folds, purge 20, embargo 5     | 15             |

### Metrics

- Mean Sharpe ratio across all out-of-sample folds
- Standard deviation of Sharpe ratios across all folds - the primary indicator of estimator reliability
- Cumulative out-of-sample equity curve

## Key Results

| Method       | Mean Sharpe | Std Dev Sharpe | Interpretation                                        |
|--------------|-------------|----------------|-------------------------------------------------------|
| K-Fold       | -2.68       | 7.30           | Data leakage produces unstable, random estimates      |
| Walk-Forward | 3.46        | 10.98          | Single-path fragility; extreme variance across windows|
| CPCV         | -4.32       | 4.34           | Lowest variance; most reliable estimator              |

### Interpretation

The critical finding is that the standard deviation of the Sharpe estimator is the relevant metric of trustworthiness, not the mean. A high mean Sharpe with high variance indicates a fragile strategy that will not generalize. CPCV produces the tightest distribution, indicating a reliable estimate of true out-of-sample performance - even when that estimate is unfavorable.

In this experiment, CPCV correctly identifies that the strategy is unprofitable after transaction costs. K-Fold and Walk-Forward both produce misleading Sharpe ratios due to, respectively, temporal leakage and single-path variance.

### Visualizations

Sharpe Ratio Distribution across Validation Splits

The K-Fold distribution is wide and erratic, characteristic of leakage. The Walk-Forward distribution exhibits extreme outliers (standard deviation of nearly 11), indicating that the estimate is highly sensitive to window boundaries. The CPCV distribution is tightly clustered, providing a stable estimate.

Cumulative Out-of-Sample Equity Curves

K-Fold produces a piecewise equity curve that appears profitable in isolation but represents a patchwork of leaked information. Walk-Forward exhibits a classic overfitting signature: rapid ascent followed by collapse. CPCV produces a flat equity curve, honestly reflecting the strategy's lack of edge after costs.

## Interactive Demo

An interactive Streamlit application is available for exploring the parameters and reproducing the experiment.

Live demo: [Link to deployed app]

Features:
- Adjustable data generation parameters (seed, length, number of regimes)
- Configurable splitter settings (folds, purge size, embargo size)
- Real-time comparison of all three validation methods
- Automatic detection of the most and least stable estimators

## Installation

### Requirements

- Python 3.9 or higher
- See requirements.txt for the complete dependency list

### Setup

    git clone https://github.com/riteshthakur15102006-art/Combinatorial-Purge-Cross-Validation--Backtest.git
    cd Combinatorial-Purge-Cross-Validation--Backtest
    pip install -r requirements.txt

## Usage

### Running the Comparison Script

    python Backtest_Comparison.py

This produces:
- A terminal summary table
- sharpe_distribution.png - histogram of Sharpe ratios by method
- equity_curves.png - cumulative out-of-sample equity curves

### Running the Interactive App

    streamlit run app.py

The app is accessible at http://localhost:8501.

## Project Structure

    .
    |-- app.py                     # Streamlit interactive demo
    |-- Backtest_Comparison.py                   # Main comparison script
    |-- requirements.txt           # Python dependencies
    |-- README.md                  # This file
    |-- Results/
        |-- sharpe_distribution.png
        |-- equity_curves.png
        |-- Terminal Result.png

## Technical Implementation

### Key Design Decisions

Shared evaluation function. All three validation methods call the same evaluate_model function, ensuring that differences in results are attributable solely to the splitter, not to implementation discrepancies.

Transaction costs. A 5 basis point cost per trade is applied uniformly. This is critical: without costs, K-Fold's leakage would produce artificially inflated returns. Costs expose strategies that churn positions to capture noise.

Trade counting via np.diff(positions, prepend=0). The prepend argument ensures the output array has the same length as the position array, preventing broadcasting errors during cost application.

Reproducibility. np.random.seed(42) and random_state=42 are set at the top of the script. Anyone running the code obtains identical results.

### Purging and Embargoing

The CPCV splitter applies two leak-prevention mechanisms:

- Purge size of 20 bars. Removes training observations whose 10-bar feature window or 1-bar label horizon overlaps with the test block.
- Embargo size of 5 bars. Removes a further 5 bars of training data immediately following each test block to account for serial correlation.

### Library Note

The skfolio library returns split indices as lists of arrays for CPCV because test blocks may be non-contiguous. The np.hstack function flattens these into a single 1D array required by NumPy and pandas indexing.

## Research Context

This project is directly informed by the following literature:

- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. - Chapter 7 covers purged K-Fold CV; Chapter 12 introduces CPCV.
- Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2015). The Probability of Backtest Overfitting. Journal of Computational Finance.
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. Journal of Portfolio Management.
- Arian, H. R., Norouzi, D., & Seco, L. (2024). Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment. Knowledge-Based Systems.

## References

1. Lopez de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.

2. Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. Notices of the American Mathematical Society, 61(5), 458-471.

3. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. Review of Financial Studies, 29(1), 5-68.

4. Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. Journal of Portfolio Management, 40(5), 94-107.

5. Arian, H. R., Norouzi, D., & Seco, L. (2024). Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment. Knowledge-Based Systems.

## Author

Ritesh Thakur

GitHub: https://github.com/riteshthakur15102006-art

## License

This project is released for educational and research purposes. See repository for details.
