# =============================================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — WITH YAHOO FINANCE & INVESTOR PROFILE
# =============================================================================
# Requirements:
#   pip install yfinance pandas numpy matplotlib scipy
#
# Run:
#   python portfolio_optimizer.py
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is not installed.")
    print("Please run:  pip install yfinance")
    sys.exit(1)

try:
    from scipy.optimize import minimize
except ImportError:
    print("ERROR: scipy is not installed.")
    print("Please run:  pip install scipy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CONFIGURATION (edit defaults here)
# ─────────────────────────────────────────────────────────────────────────────

# Risk-free rate used in Sharpe ratio (annualized, e.g. 0.045 = 4.5%)
RISK_FREE_RATE = 0.045

# Number of random portfolios to simulate for the efficient frontier
N_SIMULATIONS = 5000

# Maximum number of tickers a user can enter
MAX_TICKERS = 25

# ── Investor profile thresholds (variable income %) — edit as needed ────────
PROFILE_THRESHOLDS = {
    "Conservative": (0,  30),   # variable income 0–30%
    "Moderate":     (31, 60),   # variable income 31–60%
    "Aggressive":   (61, 100),  # variable income 61–100%
}

# ── Periodicity → (yfinance interval, annualization factor) ─────────────────
PERIODICITY_MAP = {
    "daily":   ("1d",  252),
    "weekly":  ("1wk", 52),
    "monthly": ("1mo", 12),
}

# ── Investment horizon → yfinance period string ──────────────────────────────
HORIZON_MAP = {
    "6 months": "6mo",
    "1 year":   "1y",
    "3 years":  "3y",
    "5 years":  "5y",
}

# Default benchmark ticker
DEFAULT_BENCHMARK = "SPY"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — USER INPUT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def prompt(msg, valid_options=None, allow_blank=False):
    """
    General-purpose prompt helper.
    Keeps asking until the user gives valid input.
    valid_options: list of accepted strings (case-insensitive), or None for free text.
    """
    while True:
        answer = input(msg).strip()
        if not answer and allow_blank:
            return answer
        if not answer:
            print("  ✗ Input cannot be empty. Please try again.")
            continue
        if valid_options:
            if answer.lower() in [v.lower() for v in valid_options]:
                # Return the matching option in its original case
                return valid_options[[v.lower() for v in valid_options].index(answer.lower())]
            else:
                print(f"  ✗ Invalid choice. Options are: {', '.join(valid_options)}")
        else:
            return answer


def get_user_inputs():
    """
    Collect all inputs from the user interactively.
    Returns a dict with all settings for the analysis.
    """
    print("\n" + "=" * 65)
    print("  MARKOWITZ PORTFOLIO OPTIMIZER")
    print("=" * 65)

    # ── Tickers ──────────────────────────────────────────────────────────────
    print("\n[1/6]  STOCK TICKERS")
    print(f"  Enter up to {MAX_TICKERS} tickers separated by spaces or commas.")
    print("  Examples: AAPL MSFT KO WMT SPY\n")

    while True:
        raw = input("  Tickers: ").strip()
        if not raw:
            print("  ✗ You must enter at least 1 ticker.")
            continue
        # Accept both spaces and commas as separators
        tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
        tickers = list(dict.fromkeys(tickers))   # remove duplicates while keeping order

        if len(tickers) > MAX_TICKERS:
            print(f"  ✗ Too many tickers. Maximum is {MAX_TICKERS}. You entered {len(tickers)}.")
            continue
        if len(tickers) < 1:
            print("  ✗ Please enter at least 1 ticker.")
            continue
        break

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\n[2/6]  BENCHMARK")
    print(f"  Suggested: SPY, ^GSPC, ^DJI, ^IXIC")
    print(f"  Press Enter to use default ({DEFAULT_BENCHMARK}).\n")
    bench_raw = input("  Benchmark ticker: ").strip().upper()
    benchmark = bench_raw if bench_raw else DEFAULT_BENCHMARK

    # ── Periodicity ───────────────────────────────────────────────────────────
    print("\n[3/6]  DATA PERIODICITY")
    print("  Options: daily, weekly, monthly\n")
    periodicity = prompt("  Periodicity: ", valid_options=list(PERIODICITY_MAP.keys()))

    # ── Investment horizon ────────────────────────────────────────────────────
    print("\n[4/6]  INVESTMENT HORIZON")
    horizon_options = list(HORIZON_MAP.keys())
    print(f"  Options: {', '.join(horizon_options)}\n")
    horizon = prompt("  Horizon: ", valid_options=horizon_options)

    # ── Fixed vs variable income allocation ───────────────────────────────────
    print("\n[5/6]  ASSET ALLOCATION")
    print("  Enter the % you want in fixed income and variable income.")
    print("  These must sum to 100.\n")

    while True:
        try:
            fi = float(input("  Fixed income (%):    ").strip())
            vi = float(input("  Variable income (%): ").strip())
            if abs(fi + vi - 100) > 0.01:
                print(f"  ✗ Fixed + variable must equal 100. You entered {fi} + {vi} = {fi+vi}.")
                continue
            if fi < 0 or vi < 0:
                print("  ✗ Percentages cannot be negative.")
                continue
            break
        except ValueError:
            print("  ✗ Please enter numeric values.")

    # ── Investor profile (auto-determined, shown to user) ─────────────────────
    profile = determine_investor_profile(vi)

    print(f"\n[6/6]  RISK-FREE RATE")
    print(f"  Press Enter to use default ({RISK_FREE_RATE*100:.1f}% annual).\n")
    rf_raw = input("  Risk-free rate (%): ").strip()
    if rf_raw:
        try:
            risk_free = float(rf_raw) / 100
        except ValueError:
            print("  ✗ Invalid input, using default.")
            risk_free = RISK_FREE_RATE
    else:
        risk_free = RISK_FREE_RATE

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  INPUT SUMMARY")
    print("-" * 65)
    print(f"  Tickers       : {', '.join(tickers)}")
    print(f"  Benchmark     : {benchmark}")
    print(f"  Periodicity   : {periodicity}")
    print(f"  Horizon       : {horizon}")
    print(f"  Fixed income  : {fi:.1f}%")
    print(f"  Variable inc. : {vi:.1f}%")
    print(f"  Investor type : {profile}")
    print(f"  Risk-free rate: {risk_free*100:.2f}%")
    print("-" * 65)

    confirm = prompt("\n  Proceed with these settings? (yes/no): ",
                     valid_options=["yes", "no"])
    if confirm.lower() == "no":
        print("  Restarting input...\n")
        return get_user_inputs()   # restart if user wants to change

    return {
        "tickers":    tickers,
        "benchmark":  benchmark,
        "periodicity": periodicity,
        "horizon":    horizon,
        "fixed_income":    fi,
        "variable_income": vi,
        "profile":    profile,
        "risk_free":  risk_free,
        "interval":   PERIODICITY_MAP[periodicity][0],
        "annualize":  PERIODICITY_MAP[periodicity][1],
        "period":     HORIZON_MAP[horizon],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INVESTOR PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def determine_investor_profile(variable_income_pct):
    """
    Classify investor profile based on variable income percentage.
    Thresholds are defined in PROFILE_THRESHOLDS at the top of the file.
    """
    for profile, (lo, hi) in PROFILE_THRESHOLDS.items():
        if lo <= variable_income_pct <= hi:
            return profile
    return "Aggressive"   # fallback


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DATA DOWNLOAD & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_download(tickers, benchmark, period, interval):
    """
    Downloads adjusted close prices for all tickers + benchmark from Yahoo Finance.
    Validates each ticker and removes any that return no data.
    Returns:
        prices     : DataFrame of valid stock prices
        bench_prices: Series of benchmark prices
        valid_tickers: list of tickers that passed validation
    """
    print("\n  Downloading data from Yahoo Finance...")

    all_tickers = list(set(tickers + [benchmark]))

    # Download all at once for efficiency
    raw = yf.download(
        tickers=all_tickers,
        period=period,
        interval=interval,
        auto_adjust=True,    # gives adjusted close directly in 'Close'
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices_all = raw["Close"]
    else:
        # Single ticker — wrap in DataFrame
        prices_all = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    # ── Validate each ticker ──────────────────────────────────────────────────
    valid_tickers = []
    invalid_tickers = []

    for t in tickers:
        if t not in prices_all.columns:
            invalid_tickers.append(t)
            continue
        col = prices_all[t].dropna()
        if len(col) < 10:
            print(f"  ⚠  {t}: too little data returned (only {len(col)} rows) — skipping.")
            invalid_tickers.append(t)
        else:
            valid_tickers.append(t)

    if invalid_tickers:
        print(f"  ⚠  Skipped invalid/empty tickers: {', '.join(invalid_tickers)}")

    if not valid_tickers:
        print("\n  ✗ ERROR: No valid tickers found. Exiting.")
        sys.exit(1)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if benchmark not in prices_all.columns or prices_all[benchmark].dropna().shape[0] < 10:
        print(f"  ⚠  Benchmark '{benchmark}' returned no data. Falling back to {DEFAULT_BENCHMARK}.")
        bench_data = yf.download(DEFAULT_BENCHMARK, period=period,
                                 interval=interval, auto_adjust=True, progress=False)
        bench_prices = bench_data["Close"].dropna()
        benchmark = DEFAULT_BENCHMARK
    else:
        bench_prices = prices_all[benchmark].dropna()

    prices = prices_all[valid_tickers].dropna(how="all")

    print(f"  ✓ Loaded {len(prices)} rows × {len(valid_tickers)} stocks.")
    print(f"  ✓ Benchmark '{benchmark}': {len(bench_prices)} rows.")

    return prices, bench_prices, valid_tickers, benchmark


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RETURN CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_returns(prices):
    """Simple percentage returns from prices."""
    return prices.pct_change().dropna()


def annualized_stats(returns, annualize_factor, risk_free):
    """
    For each asset compute annualized mean return, volatility, and Sharpe ratio.
    annualize_factor: 252 (daily), 52 (weekly), 12 (monthly)
    """
    mean = returns.mean() * annualize_factor
    vol  = returns.std()  * np.sqrt(annualize_factor)
    sharpe = (mean - risk_free) / vol
    return pd.DataFrame({"Annual Return": mean, "Annual Volatility": vol, "Sharpe": sharpe})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MARKOWITZ PORTFOLIO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_portfolios(returns, annualize_factor, risk_free, n=N_SIMULATIONS):
    """
    Monte Carlo simulation: generate N random weight combinations.
    Returns a DataFrame with columns: weights, return, volatility, sharpe.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * annualize_factor
    cov_matrix   = returns.cov()  * annualize_factor

    results = {
        "return":     np.zeros(n),
        "volatility": np.zeros(n),
        "sharpe":     np.zeros(n),
        "weights":    [None] * n,
    }

    for i in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()                                # normalize to sum = 1
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(w @ cov_matrix.values @ w)
        sr  = (ret - risk_free) / vol

        results["return"][i]     = ret
        results["volatility"][i] = vol
        results["sharpe"][i]     = sr
        results["weights"][i]    = w

    return pd.DataFrame(results)


def optimize_portfolio(returns, annualize_factor, risk_free, target="sharpe"):
    """
    Use scipy.optimize to find the exact:
      - Maximum Sharpe ratio portfolio  (target='sharpe')
      - Minimum variance portfolio      (target='minvol')

    Returns: (weights_array, portfolio_return, portfolio_vol, portfolio_sharpe)
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * annualize_factor
    cov_matrix   = returns.cov()  * annualize_factor

    # Weights must sum to 1, and each weight >= 0 (long-only)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, 1.0) for _ in range(n_assets))
    w0          = np.array([1 / n_assets] * n_assets)  # equal-weight starting point

    if target == "sharpe":
        # Minimize negative Sharpe (= maximize Sharpe)
        def neg_sharpe(w):
            ret = np.dot(w, mean_returns)
            vol = np.sqrt(w @ cov_matrix.values @ w)
            return -(ret - risk_free) / vol

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-9})

    elif target == "minvol":
        # Minimize portfolio variance
        def portfolio_vol(w):
            return np.sqrt(w @ cov_matrix.values @ w)

        result = minimize(portfolio_vol, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-9})

    if not result.success:
        print(f"  ⚠  Optimization warning ({target}): {result.message}")

    w_opt = result.x
    ret   = np.dot(w_opt, mean_returns)
    vol   = np.sqrt(w_opt @ cov_matrix.values @ w_opt)
    sr    = (ret - risk_free) / vol

    return w_opt, ret, vol, sr


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — BENCHMARK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_benchmark_metrics(bench_prices, annualize_factor, risk_free):
    """Compute annualized return, volatility, and Sharpe for the benchmark."""
    bench_returns = bench_prices.pct_change().dropna()
    ret = bench_returns.mean() * annualize_factor
    vol = bench_returns.std()  * np.sqrt(annualize_factor)
    sr  = (ret - risk_free) / vol
    return {"return": ret, "volatility": vol, "sharpe": sr,
            "returns_series": bench_returns}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def display_results(cfg, valid_tickers, asset_stats,
                    max_sharpe_w, max_sharpe_ret, max_sharpe_vol, max_sharpe_sr,
                    min_vol_w,    min_vol_ret,    min_vol_vol,    min_vol_sr,
                    bench_metrics, benchmark):
    """Print a formatted summary of all results to the console."""

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  ANALYSIS RESULTS")
    print(SEP)

    # ── Session summary ───────────────────────────────────────────────────────
    print(f"\n  Investor profile  : {cfg['profile']}")
    print(f"  Horizon           : {cfg['horizon']}")
    print(f"  Periodicity       : {cfg['periodicity']}")
    print(f"  Fixed income      : {cfg['fixed_income']:.1f}%")
    print(f"  Variable income   : {cfg['variable_income']:.1f}%")
    print(f"  Risk-free rate    : {cfg['risk_free']*100:.2f}%")

    # ── Individual asset stats ────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("  INDIVIDUAL ASSET STATS (annualized)")
    print(f"{'-'*65}")
    print(f"  {'Ticker':<10} {'Ann. Return':>12} {'Ann. Volatility':>16} {'Sharpe':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*16} {'-'*8}")
    for ticker in valid_tickers:
        r  = asset_stats.loc[ticker, "Annual Return"]   * 100
        v  = asset_stats.loc[ticker, "Annual Volatility"] * 100
        sh = asset_stats.loc[ticker, "Sharpe"]
        print(f"  {ticker:<10} {r:>11.2f}%  {v:>15.2f}%  {sh:>8.4f}")

    # ── Maximum Sharpe portfolio ──────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("  OPTIMAL PORTFOLIO — Maximum Sharpe Ratio")
    print(f"{'-'*65}")
    print(f"  Expected Annual Return     : {max_sharpe_ret*100:>8.2f}%")
    print(f"  Expected Annual Volatility : {max_sharpe_vol*100:>8.2f}%")
    print(f"  Sharpe Ratio               : {max_sharpe_sr:>8.4f}")
    print("\n  Stock Weights (within variable income allocation):")
    for t, w in zip(valid_tickers, max_sharpe_w):
        if w > 0.001:
            print(f"    {t:<10}: {w*100:>7.2f}%")

    # ── Minimum variance portfolio ────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("  OPTIMAL PORTFOLIO — Minimum Variance")
    print(f"{'-'*65}")
    print(f"  Expected Annual Return     : {min_vol_ret*100:>8.2f}%")
    print(f"  Expected Annual Volatility : {min_vol_vol*100:>8.2f}%")
    print(f"  Sharpe Ratio               : {min_vol_sr:>8.4f}")
    print("\n  Stock Weights (within variable income allocation):")
    for t, w in zip(valid_tickers, min_vol_w):
        if w > 0.001:
            print(f"    {t:<10}: {w*100:>7.2f}%")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print(f"  BENCHMARK: {benchmark} (annualized)")
    print(f"{'-'*65}")
    print(f"  Return     : {bench_metrics['return']*100:>8.2f}%")
    print(f"  Volatility : {bench_metrics['volatility']*100:>8.2f}%")
    print(f"  Sharpe     : {bench_metrics['sharpe']:>8.4f}")

    # ── Full portfolio allocation ─────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("  FINAL PORTFOLIO ALLOCATION SUMMARY")
    print(f"{'-'*65}")
    print(f"  Fixed income                 : {cfg['fixed_income']:.1f}%")
    print(f"  Variable income (total)      : {cfg['variable_income']:.1f}%")
    print(f"\n  Within variable income — Max Sharpe weights:")
    vi = cfg["variable_income"] / 100
    for t, w in zip(valid_tickers, max_sharpe_w):
        total_allocation = w * vi * 100
        print(f"    {t:<10}: {w*100:>6.2f}% of equity  →  {total_allocation:>6.2f}% of total portfolio")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(prices, returns, bench_prices, bench_metrics, benchmark,
                 sim_df, valid_tickers,
                 max_sharpe_w, max_sharpe_ret, max_sharpe_vol,
                 min_vol_w,    min_vol_ret,    min_vol_vol,
                 asset_stats, cfg):
    """
    Generate a 4-panel dashboard:
      1. Normalized price history (stocks + benchmark)
      2. Risk-Return scatter (efficient frontier)
      3. Pie chart of Max Sharpe weights
      4. Asset return vs volatility bar comparison
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Portfolio Analysis Dashboard  |  {cfg['profile']} Investor  |  {cfg['horizon']}",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── 1. Normalized price chart ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    # Align benchmark to the same date range as stocks
    combined_prices = prices.copy()
    bench_aligned = bench_prices.reindex(prices.index, method="ffill")
    combined_prices[benchmark] = bench_aligned

    normalized = combined_prices / combined_prices.iloc[0] * 100
    colors_stocks = plt.cm.tab10(np.linspace(0, 1, len(valid_tickers)))

    for i, t in enumerate(valid_tickers):
        ax1.plot(normalized.index, normalized[t], label=t,
                 color=colors_stocks[i], linewidth=1.4, alpha=0.85)

    # Benchmark in bold black dashed line
    if benchmark in normalized.columns:
        ax1.plot(normalized.index, normalized[benchmark],
                 label=f"{benchmark} (benchmark)", color="black",
                 linewidth=2.2, linestyle="--", alpha=0.9)

    ax1.set_title("Normalized Price History (Base = 100)", fontsize=12)
    ax1.set_ylabel("Indexed Price")
    ax1.legend(loc="upper left", fontsize=8, ncol=min(5, len(valid_tickers)+1))
    ax1.grid(True, alpha=0.25)
    ax1.axhline(100, color="gray", linewidth=0.7, linestyle=":")

    # ── 2. Efficient frontier (Monte Carlo scatter) ────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    sc = ax2.scatter(
        sim_df["volatility"] * 100,
        sim_df["return"]     * 100,
        c=sim_df["sharpe"], cmap="viridis",
        alpha=0.35, s=6, zorder=1
    )
    plt.colorbar(sc, ax=ax2, label="Sharpe Ratio", fraction=0.046, pad=0.04)

    # Max Sharpe
    ax2.scatter(max_sharpe_vol * 100, max_sharpe_ret * 100,
                color="red", s=120, zorder=5, marker="*",
                label=f"Max Sharpe ({max_sharpe_ret*100:.1f}%, {max_sharpe_vol*100:.1f}%)")

    # Min Vol
    ax2.scatter(min_vol_vol * 100, min_vol_ret * 100,
                color="blue", s=120, zorder=5, marker="D",
                label=f"Min Vol ({min_vol_ret*100:.1f}%, {min_vol_vol*100:.1f}%)")

    # Benchmark
    ax2.scatter(bench_metrics["volatility"] * 100, bench_metrics["return"] * 100,
                color="black", s=120, zorder=5, marker="^",
                label=f"{benchmark} ({bench_metrics['return']*100:.1f}%, {bench_metrics['volatility']*100:.1f}%)")

    # Individual assets
    for t in valid_tickers:
        ax2.scatter(asset_stats.loc[t, "Annual Volatility"] * 100,
                    asset_stats.loc[t, "Annual Return"] * 100,
                    color="orange", s=60, zorder=4, marker="o", alpha=0.8)
        ax2.annotate(t,
                     xy=(asset_stats.loc[t, "Annual Volatility"] * 100,
                         asset_stats.loc[t, "Annual Return"] * 100),
                     fontsize=7, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")

    ax2.set_title("Efficient Frontier (Monte Carlo)", fontsize=12)
    ax2.set_xlabel("Annual Volatility (%)")
    ax2.set_ylabel("Annual Return (%)")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.25)

    # ── 3. Pie chart — Max Sharpe weights ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    # Only show assets with weight > 0.5%
    labels   = [t for t, w in zip(valid_tickers, max_sharpe_w) if w > 0.005]
    sizes    = [w for t, w in zip(valid_tickers, max_sharpe_w) if w > 0.005]
    explode  = [0.04] * len(labels)

    if sizes:
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            explode=explode, startangle=90,
            colors=plt.cm.tab10(np.linspace(0, 1, len(labels)))
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax3.set_title(
            f"Max Sharpe Portfolio Weights\n"
            f"(variable income = {cfg['variable_income']:.0f}% of total)",
            fontsize=11
        )
    else:
        ax3.text(0.5, 0.5, "No weights > 0.5%", ha="center", va="center")

    plt.savefig("portfolio_analysis_charts.png", dpi=150, bbox_inches="tight")
    print("  Chart saved → portfolio_analysis_charts.png")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Collect all inputs from user
    cfg = get_user_inputs()

    # 2. Download and validate data
    prices, bench_prices, valid_tickers, benchmark = validate_and_download(
        cfg["tickers"], cfg["benchmark"], cfg["period"], cfg["interval"]
    )

    # 3. Calculate returns
    print("\n  Calculating returns...")
    returns = calculate_returns(prices)

    # 4. Asset-level statistics
    asset_stats = annualized_stats(returns, cfg["annualize"], cfg["risk_free"])

    # 5. Benchmark metrics
    print("  Calculating benchmark metrics...")
    bench_metrics = calculate_benchmark_metrics(
        bench_prices, cfg["annualize"], cfg["risk_free"]
    )

    # 6. Portfolio optimization — Max Sharpe
    print("  Optimizing portfolio (Max Sharpe)...")
    max_sharpe_w, max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = optimize_portfolio(
        returns, cfg["annualize"], cfg["risk_free"], target="sharpe"
    )

    # 7. Portfolio optimization — Min Volatility
    print("  Optimizing portfolio (Min Volatility)...")
    min_vol_w, min_vol_ret, min_vol_vol, min_vol_sr = optimize_portfolio(
        returns, cfg["annualize"], cfg["risk_free"], target="minvol"
    )

    # 8. Monte Carlo simulation for the frontier chart
    print(f"  Running {N_SIMULATIONS} portfolio simulations...")
    sim_df = simulate_portfolios(returns, cfg["annualize"], cfg["risk_free"])

    # 9. Display text results
    display_results(
        cfg, valid_tickers, asset_stats,
        max_sharpe_w, max_sharpe_ret, max_sharpe_vol, max_sharpe_sr,
        min_vol_w,    min_vol_ret,    min_vol_vol,    min_vol_sr,
        bench_metrics, benchmark
    )

    # 10. Generate charts
    print("  Generating charts...")
    plot_results(
        prices, returns, bench_prices, bench_metrics, benchmark,
        sim_df, valid_tickers,
        max_sharpe_w, max_sharpe_ret, max_sharpe_vol,
        min_vol_w,    min_vol_ret,    min_vol_vol,
        asset_stats, cfg
    )

    print("  Done! ✓\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()