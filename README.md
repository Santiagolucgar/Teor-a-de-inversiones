# Investment Allocation & Decision-Support Tool

A comprehensive Python CLI tool for portfolio optimization and investment advisory.
Built on Markowitz mean-variance theory with investor profiling, fixed-income recommendations, VaR analysis, valuation signals, and professional recommendation reports.

## Features

| Module | Description |
|--------|-------------|
| **Investor Questionnaire** | 10-question structured survey → 5 risk profiles |
| **Horizon Classification** | Short / Medium / Long term → influences all decisions |
| **Fixed-Income Diversification** | Rule-based allocation across 7 debt instrument categories |
| **Markowitz Optimization** | Max Sharpe + Min Variance portfolios (long-only) |
| **Monte Carlo Simulation** | 5,000 random portfolios for efficient frontier |
| **Value-at-Risk (VaR)** | Parametric VaR at 90%, 95%, 99% confidence |
| **Target Return Mode** | Check if portfolio meets user-defined return goals |
| **Valuation Signals** | P/E, P/B, 52-week range → under/overvalued classification |
| **Rebalancing Engine** | Drift, volatility, valuation-based suggestions |
| **Benchmark Comparison** | Return, volatility, Sharpe, max drawdown, cumulative return |
| **Professional Report** | Advisory memo with 8 structured sections |
| **CSV Exports** | Correlation, covariance, stats, weights, FI recommendation |
| **Dashboard Charts** | Normalized prices, efficient frontier, pie charts, heatmap |

## Requirements

```
pip install yfinance pandas numpy matplotlib scipy seaborn
```

Or using the requirements file:

```
pip install -r requirements.txt
```

## Quick Start

```bash
python portafolio_optimizer.py
```

The program will guide you through:
1. **Investor questionnaire** — 10 questions about your objectives, horizon, and risk tolerance
2. **Profile result** — computed risk profile with option to override
3. **Asset allocation** — recommended FI/RV split based on profile + horizon
4. **Fixed-income diversification** — optional recommendation for debt instruments
5. **Market configuration** — tickers, benchmark, dates, periodicity, amount
6. **Analysis** — optimization, VaR, valuation, rebalancing
7. **Results** — full dashboard, professional recommendation memo, CSV exports

## Investor Profiles

| Profile | Equity Range | Optimization | Max Volatility |
|---------|-------------|--------------|----------------|
| Conservador | 10-30% | Min Variance | 12% |
| Moderadamente Conservador | 20-40% | Min Variance | 16% |
| Moderado | 30-60% | Max Sharpe | 20% |
| Moderadamente Agresivo | 50-75% | Max Sharpe | 28% |
| Agresivo | 70-95% | Max Sharpe | 35% |

## Model Assumptions

- Long-only portfolios (no short selling)
- No leverage, derivatives, or margin
- Parametric VaR assumes normal distribution
- Valuation signals are indicative, not definitive
- Fixed-income recommendations are rule-based (qualitative)

## Output Files

All outputs are saved in the `outputs/` directory:
- `correlation_matrix.csv`
- `covariance_matrix.csv`
- `summary_statistics.csv`
- `portfolio_weights.csv`
- `fixed_income_recommendation.csv` (if FI module activated)
- `portfolio_analysis_charts.png` (in project root)
