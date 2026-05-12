# Bayesian Online Changepoint Detection for Financial Time Series

**Course:** CODS 620 — Advanced Statistical Inference  
**Institution:** Khalifa University, MSc Computational Data Science, 2026  
**Collaborators:** Mohammed Musthafa, Zeinabou Ahmed

---

## Overview

Financial time series exhibit volatility regime changes triggered by macroeconomic shocks — events that permanently alter the statistical properties of return distributions. Detecting these changes in real time, without revisiting historical data, is essential for dynamic risk management and portfolio rebalancing.

This project implements **Bayesian Online Changepoint Detection (BOCPD)** (Adams & MacKay, 2007) for real-time volatility regime identification in financial return series, using a conjugate Inverse-Gamma/Student-t likelihood model with fully closed-form recursive updates.

---

## Method

### Core Idea

BOCPD maintains a posterior distribution over the **run length** r_t — the number of observations since the last changepoint. At each time step, two possibilities are updated:

- **Regime continues:** run length grows by 1, likelihood conditioned on current regime parameters
- **Changepoint occurs:** run length resets to 0, parameters reset to prior

### Likelihood Model

Within each regime, returns are modelled as zero-mean Gaussian with unknown variance:

```
x_t | σ² ~ N(0, σ²)
```

### Conjugate Prior

```
σ² ~ InvGamma(α₀, β₀)     α₀ = 3.0,  β₀ = 0.0002
```

This gives a closed-form **Student-t predictive distribution**, enabling exact online inference without approximation.

### Recursive Update Equations

```
α_r^(t) = α_r^(t-1) + 1/2
β_r^(t) = β_r^(t-1) + x_t² / 2
```

### Run-Length Posterior

```
P(r_t = 0  | x_t) ∝ h · p(x_t | prior)
P(r_t = r  | x_t) ∝ (1 − h) · P(r_{t-1} = r−1) · p(x_t | r_{t-1} = r−1)
```

where `h = 1/30` (constant hazard rate, expected regime duration = 30 trading days).

---

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC (real equity data) | 0.955 |
| Precision | 0.765 |
| Recall | 0.765 |
| F1 Score | 0.765 |
| Mean detection delay | 8.5 steps |
| Median detection delay | 3.0 steps |

Tested against manually identified ground-truth changepoints on AAPL and SPY (2010–2025), corresponding to: Flash Crash (2010), European Debt Crisis (2011–12), China Devaluation (2015), Brexit (2016), COVID-19 (2020), Fed Rate Hikes (2022), Banking Crisis (2023).

---

## Comparative Analysis

The project includes a systematic literature review of 7 changepoint detection methods across 8 evaluation dimensions:

| Method | Online | Uncertainty | Computational Cost | Financial Applicability |
|---|---|---|---|---|
| **BOCPD** | ✅ | Full posterior | O(T²) | High |
| CUSUM | ✅ | None | O(T) | Medium |
| PELT | ❌ | None | O(T) | Medium |
| GLRT | ❌ | None | O(T²) | Medium |
| NOUGAT | ✅ | None | O(T·m) | Medium |
| HMM | ✅ | Partial | O(T·K²) | Medium |
| DAE-LSTM | ✅ | None | O(T·N) | High |

BOCPD advantage: full uncertainty quantification and online operation. BOCPD limitation vs. NOUGAT: requires hazard rate tuning; parametric distributional assumptions.

---

## Repository Contents

```
BOCPD-For-Financial-Time-Series/
├── bocpd_real_data.py             # Production script: real financial data (SPY, AAPL)
├── bocpd_simulated_data.ipynb     # Notebook: controlled synthetic data with known regimes
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/safurapalekar2643/Bayesian-Online-Changepoint-Detection-BOCPD-For-Financial-Time-Series.git
cd Bayesian-Online-Changepoint-Detection-BOCPD-For-Financial-Time-Series
pip install -r requirements.txt
```

Python 3.7+

---

## Quick Start

**Real financial data:**

```bash
python bocpd_real_data.py
```

Downloads SPY from 2010–present, runs BOCPD, outputs visualisations and evaluation metrics.

**Custom ticker / parameters:**

```python
ticker       = "AAPL"
hazard_rate  = 1/30      # Expected regime duration (trading days)
alpha0       = 3.0       # Prior shape — higher = more robust to noise
beta0        = 0.000288  # Prior scale — reflects expected daily return variance
threshold    = 0.15      # Changepoint probability threshold
```

**Synthetic data (controlled experiments):**

```bash
jupyter notebook bocpd_simulated_data.ipynb
```

---

## Visualisation Outputs

1. Log returns time series with detected changepoints overlaid
2. Changepoint probability series with threshold line
3. MAP run-length (most likely time since last changepoint)
4. Real-time variance estimate
5. ROC curve and PR curve
6. Performance metrics bar chart
7. Confusion matrix

---

## Applications

- **Dynamic VaR recalibration** — update risk model parameters as volatility regime shifts
- **Volatility targeting** — detect transitions between low/high volatility environments
- **Regime-aware portfolio rebalancing** — trigger exposure adjustments on detected breaks
- **Strategy performance attribution** — separate in-regime vs. cross-regime returns

---

## References

- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
- Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution.

---

## Author

Safura Palekar · [GitHub](https://github.com/safurapalekar2643) · [LinkedIn](https://linkedin.com/in/safurapalekar2643)
