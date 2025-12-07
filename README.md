# Bayesian Online Changepoint Detection (BOCPD) for Financial Time Series

A Python implementation of Bayesian Online Changepoint Detection for identifying volatility regime changes in financial markets, based on the methodology from Adams & MacKay (2007).

## Overview

This repository implements a robust BOCPD algorithm designed to detect volatility regime changes in stock market data. The system provides real-time changepoint detection with comprehensive evaluation metrics against ground truth, making it suitable for both research and practical applications in financial risk management and trading strategies.

### Key Features

- **Real-time changepoint detection** using Bayesian inference with Student-t predictive distributions
- **Conjugate priors** for unknown variance estimation with online parameter updates
- **Time-varying hazard functions** that model realistic regime persistence patterns
- **Comprehensive evaluation framework** with precision, recall, F1 scores, ROC/AUC analysis
- **Multiple visualization tools** including run-length distributions, changepoint probabilities, and performance metrics
- **Support for both real and synthetic data** for controlled testing and validation

## Repository Contents

- **`bocpd_real_data.py`** - Production-ready Python script for analyzing real financial data (AAPL, SPY, etc.)
- **`bocpd_simulated_data.ipynb`** - Jupyter notebook for controlled testing with synthetic data containing known regime changes

## Installation

### Requirements

```bash
pip install numpy pandas scipy matplotlib yfinance scikit-learn
```

### Dependencies

- Python 3.7+
- numpy >= 1.19.0
- pandas >= 1.2.0
- scipy >= 1.6.0
- matplotlib >= 3.3.0
- yfinance >= 0.1.63
- scikit-learn >= 0.24.0

## Quick Start

### Running with Real Data

```python
# Run the standalone script with default parameters
python bocpd_real_data.py
```

This will:
1. Download stock data (default: SPY from 2010-present)
2. Run BOCPD algorithm with optimized parameters
3. Generate comprehensive visualizations
4. Evaluate performance against ground truth changepoints
5. Display detailed performance metrics

### Customizing Parameters

Edit the user configuration section in `bocpd_real_data.py`:

```python
# Stock parameters
ticker = "AAPL"  # Change to any valid Yahoo Finance ticker
start_date = "2010-01-01"
end_date = date.today()

# BOCPD Algorithm Parameters
hazard_rate = 1/30        # Expected regime duration (days)
alpha0 = 3.0              # Prior degrees of freedom
beta0 = 0.000288          # Prior variance scale
threshold = 0.15          # Changepoint detection threshold
```

### Using the Jupyter Notebook

For exploratory analysis with synthetic data:

```bash
jupyter notebook bocpd_simulated_data.ipynb
```

The notebook includes:
- Synthetic data generation with multiple volatility regimes
- Interactive parameter tuning
- Controlled testing environment with known ground truth

## Algorithm Details

### BOCPD Methodology

The implementation follows the Adams & MacKay (2007) approach:

1. **Predictive Distribution**: Student-t distribution for modeling returns with unknown variance
2. **Run-length Distribution**: Posterior probability over the time since the last changepoint
3. **Hazard Function**: Models the probability of regime changes as a function of regime age
4. **Bayesian Update**: Online parameter updates using conjugate priors (Gamma distribution for precision)

### Key Parameters

- **`hazard_rate`**: Controls expected regime duration. Set to `1/expected_duration_days`. Lower values reduce false positives but may miss short regimes.

- **`alpha0`**: Prior shape parameter for variance. Higher values (3-5) provide robustness against noise; lower values increase sensitivity.

- **`beta0`**: Prior scale parameter for variance. Should reflect expected market volatility (typically 0.0001-0.001 for daily returns).

- **`threshold`**: Changepoint probability threshold for detection (typical range: 0.1-0.3). Higher values reduce false positives but may miss genuine changepoints.

- **`detection_delay_tolerance`**: Time window for matching predicted changepoints to ground truth (typically 50-100 days for daily data).

## Evaluation Framework

### Performance Metrics

The implementation provides comprehensive evaluation against ground truth changepoints:

- **Detection Performance**:
  - Precision: Proportion of detected changepoints that are true positives
  - Recall: Proportion of true changepoints successfully detected
  - F1 Score: Harmonic mean of precision and recall

- **Timing Metrics**:
  - Mean/Median Detection Delay
  - Maximum Detection Delay
  - Missed Detections Count

- **ROC/AUC Analysis**:
  - ROC Curve and AUC score
  - Precision-Recall Curve and AUC
  - Confusion Matrix

### Ground Truth Changepoints

For AAPL stock (2010-2025), pre-identified ground truth changepoints correspond to major market events:
- Flash Crash (2010)
- European Debt Crisis (2011-2012)
- China Devaluation (2015)
- Brexit Referendum (2016)
- COVID-19 Pandemic (2020)
- Fed Rate Hikes (2022)
- Banking Crisis (2023)

## Visualization Outputs

The system generates multiple visualization types:

1. **Returns Time Series** - Shows actual log returns with detected changepoints
2. **Changepoint Probability** - Full series and zoomed view with threshold overlay
3. **MAP Run Length** - Most likely time since last changepoint
4. **Estimated Variance** - Real-time variance estimation
5. **ROC Curve** - True positive vs false positive rate
6. **Precision-Recall Curve** - Trade-off between precision and recall
7. **Performance Metrics Bar Chart** - Visual comparison of detection metrics
8. **Confusion Matrix** - Classification performance breakdown

## Parameter Optimization Guidelines

### Data-Driven Approach

1. **Analyze actual regime spacing**: Calculate average time between real changepoints
2. **Set hazard_rate accordingly**: `hazard_rate = 1 / average_spacing`
3. **Tune threshold iteratively**: Start at 0.15, adjust based on precision/recall trade-off
4. **Validate with synthetic data**: Test on controlled scenarios before real data

### Common Pitfall: Over-Detection

BOCPD algorithms tend to over-detect changepoints. If you observe significantly more detected than true changepoints:
- Increase `threshold` (0.2-0.3 for conservative detection)
- Increase `hazard_rate` denominator (e.g., 1/50 instead of 1/30)
- Increase `alpha0` for more robust variance estimation

## Example Output

```
BOCPD MODEL EVALUATION REPORT
================================================================================

Detection Performance:
--------------------------------------------------------------------------------
  Precision........................................ 0.7647
  Recall........................................... 0.7647
  F1 Score......................................... 0.7647
  True Positives................................... 13
  False Positives.................................. 4
  False Negatives.................................. 4

Timing Metrics:
--------------------------------------------------------------------------------
  Mean Detection Delay............................. 8.46 steps
  Median Detection Delay........................... 3.00 steps
  Max Detection Delay.............................. 41.00 steps
  Missed Detections................................ 4

ROC/AUC Metrics:
--------------------------------------------------------------------------------
  ROC AUC.......................................... 0.9547
  Precision-Recall AUC............................. 0.8342
```

## Mathematical Background

### Predictive Distribution

For a zero-mean model with unknown variance, the predictive distribution is Student-t:

```
p(x_t | r_{t-1}, Data) ~ t_{2α}(0, β/α)
```

where α and β are the posterior Gamma parameters for the precision.

### Update Equations

**Run-length distribution update**:
```
P(r_t = 0 | x_t) ∝ h(r_t) × p(x_t | r_t = 0)
P(r_t = r | x_t) ∝ (1 - h(r)) × P(r_{t-1} = r-1) × p(x_t | r_{t-1} = r-1)
```

**Posterior parameter updates**:
```
α_new = α_old + 0.5
β_new = β_old + 0.5 × x_t²
```

## Use Cases

- **Risk Management**: Identify regime changes for dynamic portfolio allocation
- **Trading Strategies**: Detect volatility shifts for strategy adaptation
- **Market Analysis**: Study historical volatility patterns and market microstructure
- **Research**: Validate theoretical models with controlled synthetic data

## Citation


```
Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. 
arXiv preprint arXiv:0710.3742.
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Additional evaluation metrics
- Alternative hazard function implementations
- Support for multivariate time series
- Performance optimizations
- Documentation improvements




## Acknowledgments

- Original BOCPD methodology by Ryan Prescott Adams and David MacKay
- Yahoo Finance API for market data access
- Financial data science community for evaluation best practices

