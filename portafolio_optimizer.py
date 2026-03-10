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
from datetime import datetime

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
# ── Periodicity → (yfinance interval, annualization factor) ─────────────────
PERIODICITY_MAP = {
    "diaria":  ("1d",  252),
    "mensual": ("1mo", 12),
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
            print("  [X] Input cannot be empty. Please try again.")
            continue
        if valid_options:
            if answer.lower() in [v.lower() for v in valid_options]:
                # Return the matching option in its original case
                return valid_options[[v.lower() for v in valid_options].index(answer.lower())]
            else:
                print(f"  [X] Invalid choice. Options are: {', '.join(valid_options)}")
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
    
    while True:
        try:
            num_tickers = int(input(f"  How many tickers do you want to analyze? (Max {MAX_TICKERS}): ").strip())
            if 1 <= num_tickers <= MAX_TICKERS:
                break
            else:
                print(f"  [X] Please enter a number between 1 and {MAX_TICKERS}.")
        except ValueError:
            print("  [X] Invalid number. Please enter an integer.")

    tickers = []
    print("\n  Enter the ticker symbols one by one (e.g., AAPL, MSFT, SPY):")
    for i in range(num_tickers):
        while True:
            ticker = input(f"    Ticker {i+1}/{num_tickers}: ").strip().upper()
            if not ticker:
                print("  [X] Ticker cannot be empty.")
            elif ticker in tickers:
                print("  [X] Ticker already added. Please enter a different one.")
            else:
                tickers.append(ticker)
                break

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\n[2/6]  BENCHMARK")
    print(f"  Suggested: SPY, ^GSPC, ^DJI, ^IXIC")
    print(f"  Press Enter to use default ({DEFAULT_BENCHMARK}).\n")
    bench_raw = input("  Benchmark ticker: ").strip().upper()
    benchmark = bench_raw if bench_raw else DEFAULT_BENCHMARK

    # ── Periodicity ───────────────────────────────────────────────────────────
    print("\n[3/6]  DATA PERIODICITY")
    print("  Opciones: diaria, mensual\n")
    periodicity = prompt("  Periodicidad: ", valid_options=list(PERIODICITY_MAP.keys()))

    # ── Custom Dates ─────────────────────────────────────────────────────────
    print("\n[4/6]  RANGO DE FECHAS")
    print("  Ingresa la fecha de inicio y cierre (formato: YYYY-MM-DD)")
    while True:
        start_date_str = input("  Fecha de Inicio (YYYY-MM-DD): ").strip()
        end_date_str   = input("  Fecha de Cierre (YYYY-MM-DD): ").strip()
        
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date   = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            if end_date <= start_date:
                print("  [X] La fecha de cierre debe ser posterior a la fecha de inicio.")
                continue
            
            delta = (end_date - start_date).days
            if delta < 30:
                print(f"  [X] El rango debe ser de al menos 30 días para garantizar estabilidad estadística. (Ingresaste {delta} días).")
                continue
                
            break
        except ValueError:
            print("  [X] Formato de fecha inválido. Por favor, usa YYYY-MM-DD.")

    # ── Total Invertido & Pesos Manuales ───────────────────────────────────────
    print("\n[5/6]  ASIGNACIÓN DE PORTAFOLIO MANUAL")
    while True:
        try:
            monto_total = float(input("  Monto Total a Invertir (ej. 1000000): ").strip().replace(",",""))
            if monto_total <= 0:
                print("  [X] El monto debe ser mayor a 0.")
                continue
            break
        except ValueError:
            print("  [X] Ingresa un número válido.")
            
    # ── Tasa Libre de Riesgo ──────────────────────────────────────────────────
    print(f"\n[6/6]  TASA LIBRE DE RIESGO")
    print(f"  Presiona Enter para usar por defecto ({RISK_FREE_RATE*100:.1f}% anual).\n")
    rf_raw = input("  Tasa libre de riesgo (%): ").strip()
    if rf_raw:
        try:
            risk_free = float(rf_raw) / 100
        except ValueError:
            print("  [X] Entrada inválida, usando por defecto.")
            risk_free = RISK_FREE_RATE
    else:
        risk_free = RISK_FREE_RATE

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  RESUMEN DE ENTRADAS")
    print("-" * 65)
    print(f"  Tickers       : {', '.join(tickers)}")
    print(f"  Benchmark     : {benchmark}")
    print(f"  Periodicidad  : {periodicity}")
    print(f"  Rango fechas  : {start_date_str} a {end_date_str}")
    print(f"  Inversión     : ${monto_total:,.2f}")
    print(f"  Tasa libre r. : {risk_free*100:.2f}%")
    print("-" * 65)

    confirm = prompt("\n  ¿Proceder con esta configuración? (si/no): ",
                     valid_options=["si", "no"])
    if confirm.lower() == "no":
        print("  Reiniciando entradas...\n")
        return get_user_inputs()   # restart if user wants to change

    return {
        "tickers":    tickers,
        "benchmark":  benchmark,
        "periodicity": periodicity,
        "start_date": start_date_str,
        "end_date":   end_date_str,
        "monto_total": monto_total,
        "risk_free":  risk_free,
        "interval":   PERIODICITY_MAP[periodicity][0],
        "annualize":  PERIODICITY_MAP[periodicity][1],
    }

def get_manual_weights(valid_tickers):
    """
    Prompt the user for manual weights AFTER seeing the optimal portfolios.
    Returns a numpy array of weights.
    """
    print(f"\n{'-'*75}")
    print("  DISEÑA TU PORTAFOLIO MANUAL")
    print(f"{'-'*75}")
    print("  Ya has visto las combinaciones óptimas matemáticas.")
    print("  Ahora, asigna tus propios pesos (%).")
    print("  Puedes ingresar todos de golpe (ej. AAPL=10, MSFT=90)")
    print("  O presiona Enter para ingresarlos uno por uno.")
    print("  Escribe 'equitativo' para distribuir por partes iguales.")
    
    pesos = {}
    ingreso_bloque = input("\n  Pesos en bloque (opcional): ").strip()
    
    if ingreso_bloque.lower() == "equitativo":
        for t in valid_tickers:
            pesos[t] = 1.0 / len(valid_tickers)
    elif ingreso_bloque:
        pares = ingreso_bloque.split(",")
        for par in pares:
            if "=" in par:
                t, w = par.split("=")
                t = t.strip().upper()
                try:
                    w = float(w.strip())
                    if t in valid_tickers:
                        pesos[t] = w / 100.0
                except ValueError:
                    pass
    
    # Fill in any missing tickers
    for t in valid_tickers:
        if t not in pesos:
            while True:
                try:
                    w = float(input(f"  Peso para {t} (%): ").strip())
                    if w < 0 or w > 100:
                         print("  [X] El peso debe estar entre 0 y 100.")
                         continue
                    pesos[t] = w / 100.0
                    break
                except ValueError:
                    print("  [X] Ingresa un número válido.")

    suma_pesos = sum(pesos.values())
    if abs(suma_pesos - 1.0) > 0.001:
        print(f"  [!] ADVERTENCIA: La suma es {suma_pesos*100:.1f}%. Normalizando a 100%...")
        for t in pesos:
            pesos[t] = pesos[t] / suma_pesos

    return np.array([pesos[t] for t in valid_tickers])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DATA DOWNLOAD & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_download(tickers, benchmark, start_date_str, end_date_str, interval):
    """
    Downloads adjusted close prices for all tickers + benchmark from Yahoo Finance.
    Validates each ticker and removes any that return no data.
    Returns:
        prices     : DataFrame of valid stock prices
        bench_prices: Series of benchmark prices
        valid_tickers: list of tickers that passed validation
    """
    print("\n  Descargando datos desde Yahoo Finance...")

    all_tickers = list(set(tickers + [benchmark]))

    # Download all at once for efficiency
    raw = yf.download(
        tickers=all_tickers,
        start=start_date_str,
        end=end_date_str,
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
            print(f"  [!] {t}: too little data returned (only {len(col)} rows) — skipping.")
            invalid_tickers.append(t)
        else:
            valid_tickers.append(t)

    if invalid_tickers:
        print(f"  [!] Skipped invalid/empty tickers: {', '.join(invalid_tickers)}")

    if not valid_tickers:
        print("\n  [X] ERROR: No valid tickers found. Exiting.")
        sys.exit(1)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if benchmark not in prices_all.columns or prices_all[benchmark].dropna().shape[0] < 10:
        print(f"  [!] Benchmark '{benchmark}' returned no data. Falling back to {DEFAULT_BENCHMARK}.")
        bench_data = yf.download(DEFAULT_BENCHMARK, start=start_date_str, end=end_date_str,
                                 interval=interval, auto_adjust=True, progress=False)
        bench_prices = bench_data["Close"].dropna()
        benchmark = DEFAULT_BENCHMARK
    else:
        bench_prices = prices_all[benchmark].dropna()

    prices = prices_all[valid_tickers].dropna(how="all")

    print(f"  [OK] Loaded {len(prices)} rows × {len(valid_tickers)} stocks.")
    print(f"  [OK] Benchmark '{benchmark}': {len(bench_prices)} rows.")

    return prices, bench_prices, valid_tickers, benchmark


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RETURN CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_returns(prices):
    """Simple percentage returns from prices."""
    return prices.pct_change().dropna()


def annualized_stats(returns, bench_returns, annualize_factor, risk_free):
    """
    For each asset compute annualized mean return, volatility, Sharpe ratio,
    Beta, Jensen's Alpha, and Treynor Ratio.
    annualize_factor: 252 (daily), 12 (monthly)
    """
    mean_ret = returns.mean() * annualize_factor
    vol  = returns.std()  * np.sqrt(annualize_factor)
    sharpe = (mean_ret - risk_free) / vol

    bench_mean = bench_returns.mean() * annualize_factor
    bench_var = bench_returns.var()

    stats = {
        "Annual Return": mean_ret,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Beta": pd.Series(dtype=float),
        "Alpha": pd.Series(dtype=float),
        "Treynor": pd.Series(dtype=float),
    }

    # Iterate calculating Beta, Alpha, and Treynor
    for t in returns.columns:
        # Avoid issues where asset data might have slightly different alignment
        merged = pd.concat([returns[t], bench_returns], axis=1).dropna()
        if merged.empty or bench_var == 0:
            b, a, tr = 0.0, 0.0, 0.0
        else:
            cov = merged.cov().iloc[0, 1]
            b = cov / bench_var
            # Jensen's Alpha using actual historical returns
            a = mean_ret[t] - (risk_free + b * (bench_mean - risk_free))
            tr = (mean_ret[t] - risk_free) / b if b != 0 else np.nan
        
        stats["Beta"].loc[t] = b
        stats["Alpha"].loc[t] = a
        stats["Treynor"].loc[t] = tr

    return pd.DataFrame(stats)


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


def evaluate_portfolio(weights, returns, bench_returns, asset_stats, annualize_factor, risk_free):
    """
    Given a set of asset weights, evaluates the portfolio performance:
    Return, Volatility, Sharpe, Beta, Jensen's Alpha, and Treynor Ratio.
    """
    mean_returns = returns.mean() * annualize_factor
    cov_matrix   = returns.cov()  * annualize_factor
    
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights @ cov_matrix.values @ weights)
    sr  = (ret - risk_free) / vol
    
    # Portfolio Beta is weighted sum of individual Betas
    port_beta = np.dot(weights, asset_stats["Beta"].values)
    
    bench_mean = bench_returns.mean() * annualize_factor
    port_alpha = ret - (risk_free + port_beta * (bench_mean - risk_free))
    port_treynor = (ret - risk_free) / port_beta if port_beta != 0 else np.nan
    
    return ret, vol, sr, port_beta, port_alpha, port_treynor


def optimize_portfolio(returns, bench_returns, asset_stats, annualize_factor, risk_free, target="sharpe"):
    """
    Use scipy.optimize to find the exact:
      - Maximum Sharpe ratio portfolio  (target='sharpe')
      - Minimum variance portfolio      (target='minvol')

    Returns: (weights_array, ret, vol, sr, beta, alpha, treynor)
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
        print(f"  [!] Optimization warning ({target}): {result.message}")

    w_opt = result.x
    return (w_opt,) + evaluate_portfolio(w_opt, returns, bench_returns, asset_stats, annualize_factor, risk_free)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — BENCHMARK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_benchmark_metrics(bench_returns, annualize_factor, risk_free):
    """Compute annualized return, volatility, and Sharpe for the benchmark."""
    ret = bench_returns.mean() * annualize_factor
    vol = bench_returns.std()  * np.sqrt(annualize_factor)
    sr  = (ret - risk_free) / vol
    return {"return": ret, "volatility": vol, "sharpe": sr,
            "beta": 1.0, "alpha": 0.0, "treynor": (ret - risk_free)/1.0}


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
    print(f"\n  Inversión Total   : ${cfg['monto_total']:,.2f}")
    print(f"  Rango fechas      : {cfg['start_date']} a {cfg['end_date']}")
    print(f"  Periodicidad      : {cfg['periodicity']}")
    print(f"  Tasa libre riesgo : {cfg['risk_free']*100:.2f}%")

    # ── Individual asset stats ────────────────────────────────────────────────
    print(f"\n{'-'*95}")
    print("  INDIVIDUAL ASSET STATS (annualized)")
    print(f"{'-'*95}")
    print(f"  {'Ticker':<8} {'Ann. Return':>12} {'Ann. Volatility':>16} {'Sharpe':>8} {'Beta':>8} {'Alpha':>8} {'Treynor':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ticker in valid_tickers:
        r  = asset_stats.loc[ticker, "Annual Return"]   * 100
        v  = asset_stats.loc[ticker, "Annual Volatility"] * 100
        sh = asset_stats.loc[ticker, "Sharpe"]
        b  = asset_stats.loc[ticker, "Beta"]
        a  = asset_stats.loc[ticker, "Alpha"] * 100
        tr = asset_stats.loc[ticker, "Treynor"]
        print(f"  {ticker:<8} {r:>11.2f}%  {v:>15.2f}%  {sh:>8.4f} {b:>8.4f} {a:>7.2f}% {tr:>8.4f}")

    # ── Maximum Sharpe & Min Vol Options ──────────────────────────────────────
    opts = [
        ("OPTIMAL PORTFOLIO — Maximum Sharpe Ratio", max_sharpe_w, max_sharpe_ret, max_sharpe_vol, max_sharpe_sr),
        ("OPTIMAL PORTFOLIO — Minimum Variance",     min_vol_w, min_vol_ret, min_vol_vol, min_vol_sr)
    ]
    for title, w_opt, ret_opt, vol_opt, sr_opt in opts:
        print(f"\n{'-'*75}")
        print(f"  {title}")
        print(f"{'-'*75}")
        print(f"  Expected Annual Return     : {ret_opt*100:>8.2f}%")
        print(f"  Expected Annual Volatility : {vol_opt*100:>8.2f}%")
        print(f"  Sharpe Ratio               : {sr_opt:>8.4f}")
        print("\n  Stock Weights:")
        for t, w in zip(valid_tickers, w_opt):
            if w > 0.001:
                amt = w * cfg['monto_total']
                print(f"    {t:<8}: {w*100:>6.2f}%  ->  ${amt:,.2f}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*75}")
    print(f"  BENCHMARK: {benchmark} (annualized)")
    print(f"{'-'*75}")
    print(f"  Return     : {bench_metrics['return']*100:>8.2f}%")
    print(f"  Volatility : {bench_metrics['volatility']*100:>8.2f}%")
    print(f"  Sharpe     : {bench_metrics['sharpe']:>8.4f}")

    print(f"\n{SEP}\n")

def display_manual_portfolio(cfg, valid_tickers, manual_w, returns, bench_returns, asset_stats):
    # ── Full portfolio allocation ─────────────────────────────────────────────
    print(f"\n{'-'*75}")
    print("  TU PORTAFOLIO MANUAL")
    print(f"{'-'*75}")
    
    # Evaluate manual portfolio
    man_ret, man_vol, man_sr, man_beta, man_alpha, man_treynor = evaluate_portfolio(
        manual_w, returns, bench_returns, asset_stats, cfg["annualize"], cfg["risk_free"]
    )
    
    print(f"  Expected Annual Return     : {man_ret*100:>8.2f}%")
    print(f"  Expected Annual Volatility : {man_vol*100:>8.2f}%")
    print(f"  Sharpe Ratio               : {man_sr:>8.4f}")
    print(f"  Beta                       : {man_beta:>8.4f}")
    print(f"  Jensen's Alpha             : {man_alpha*100:>8.2f}%")
    print(f"  Treynor Ratio              : {man_treynor:>8.4f}")
    
    print("\n  Stock Weights:")
    for t, w in zip(valid_tickers, manual_w):
        if w > 0.001:
            amt = w * cfg['monto_total']
            print(f"    {t:<8}: {w*100:>6.2f}%  ->  ${amt:,.2f}")

    print(f"\n{'=' * 65}\n")
    
    return man_ret, man_vol, man_sr


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(prices, returns, bench_prices, bench_metrics, benchmark,
                 sim_df, valid_tickers,
                 max_sharpe_w, max_sharpe_ret, max_sharpe_vol,
                 min_vol_w,    min_vol_ret,    min_vol_vol,
                 man_ret,      man_vol,
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
        f"Portfolio Analysis Dashboard  |  Rango: {cfg['start_date']} - {cfg['end_date']}  |  {cfg['periodicity']}",
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

    # Manual Custom Portfolio
    if man_ret is not None and man_vol is not None:
        ax2.scatter(man_vol * 100, man_ret * 100,
                    color="magenta", s=150, zorder=6, marker="P",
                    label=f"Manual Port ({man_ret*100:.1f}%, {man_vol*100:.1f}%)")

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
            f"(Proporción sobre inversión en RV)",
            fontsize=11
        )
    else:
        ax3.text(0.5, 0.5, "No weights > 0.5%", ha="center", va="center")

    plt.savefig("portfolio_analysis_charts.png", dpi=150, bbox_inches="tight")
    print("  Chart saved -> portfolio_analysis_charts.png")
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
        cfg["tickers"], cfg["benchmark"], cfg["start_date"], cfg["end_date"], cfg["interval"]
    )

    # 3. Calculate returns
    print("\n  Calculating returns...")
    returns = calculate_returns(prices)
    bench_returns = bench_prices.pct_change().dropna()

    # 4. Benchmark metrics
    print("  Calculating benchmark metrics...")
    bench_metrics = calculate_benchmark_metrics(
        bench_returns, cfg["annualize"], cfg["risk_free"]
    )

    # 5. Asset-level statistics
    asset_stats = annualized_stats(returns, bench_returns, cfg["annualize"], cfg["risk_free"])

    # 6. Portfolio optimization — Max Sharpe
    print("  Optimizing portfolio (Max Sharpe)...")
    max_sharpe_w, max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, ms_beta, ms_alpha, ms_treynor = optimize_portfolio(
        returns, bench_returns, asset_stats, cfg["annualize"], cfg["risk_free"], target="sharpe"
    )

    # 7. Portfolio optimization — Min Volatility
    print("  Optimizing portfolio (Min Volatility)...")
    min_vol_w, min_vol_ret, min_vol_vol, min_vol_sr, mv_beta, mv_alpha, mv_treynor = optimize_portfolio(
        returns, bench_returns, asset_stats, cfg["annualize"], cfg["risk_free"], target="minvol"
    )

    # 8. Monte Carlo simulation for the frontier chart
    print(f"  Running {N_SIMULATIONS} portfolio simulations...")
    sim_df = simulate_portfolios(returns, cfg["annualize"], cfg["risk_free"])

    # 9. Display text results optimized
    display_results(
        cfg, valid_tickers, asset_stats,
        max_sharpe_w, max_sharpe_ret, max_sharpe_vol, max_sharpe_sr,
        min_vol_w,    min_vol_ret,    min_vol_vol,    min_vol_sr,
        bench_metrics, benchmark
    )

    # 10. Ask user for manual weights now that they have seen optimal portfolios
    manual_w = get_manual_weights(valid_tickers)

    # 11. Evaluate and display it
    man_ret, man_vol, man_sr = display_manual_portfolio(
        cfg, valid_tickers, manual_w, returns, bench_returns, asset_stats
    )

    # 12. Generate charts
    print("  Generating charts...")
    plot_results(
        prices, returns, bench_prices, bench_metrics, benchmark,
        sim_df, valid_tickers,
        max_sharpe_w, max_sharpe_ret, max_sharpe_vol,
        min_vol_w,    min_vol_ret,    min_vol_vol,
        man_ret,      man_vol,
        asset_stats, cfg
    )

    print("  Done! [OK]\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()