# =============================================================================
# INVESTMENT ALLOCATION & DECISION-SUPPORT TOOL
# =============================================================================
# A comprehensive portfolio optimizer that supports:
#   - Investor profiling via structured questionnaire
#   - Investment horizon classification
#   - Fixed-income diversification recommendations
#   - Markowitz equity optimization (Max Sharpe / Min Variance)
#   - Value-at-Risk (VaR) estimation
#   - Optional target-return analysis
#   - Valuation signals for rebalancing
#   - Professional recommendation reports
#
# Requirements:
#   pip install yfinance pandas numpy matplotlib scipy seaborn
#
# Run:
#   python portafolio_optimizer.py
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is not installed.  Run:  pip install yfinance")
    sys.exit(1)

try:
    from scipy.optimize import minimize
except ImportError:
    print("ERROR: scipy is not installed.  Run:  pip install scipy")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CONFIGURATION (edit defaults here)
# ─────────────────────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.045
N_SIMULATIONS = 5000
MAX_TICKERS = 100

PERIODICITY_MAP = {
    "diaria":  ("1d",  252),
    "semanal": ("1wk",  52),
    "mensual": ("1mo",  12),
}

DEFAULT_BENCHMARK = "SPY"
OUTPUT_DIR = "outputs"

# Horizon classification (years)
HORIZON_RULES = {
    "Corto plazo":   (0, 3),    # 0-3 years
    "Mediano plazo": (4, 10),   # 4-10 years
    "Largo plazo":   (11, 100), # 11+ years
}

# Profile → suggested equity % range and optimization target
PROFILE_SETTINGS = {
    "Conservador":              {"equity_range": (10, 30),  "opt_target": "minvol",  "max_vol": 0.12},
    "Moderadamente Conservador":{"equity_range": (20, 40),  "opt_target": "minvol",  "max_vol": 0.16},
    "Moderado":                 {"equity_range": (30, 60),  "opt_target": "sharpe",  "max_vol": 0.20},
    "Moderadamente Agresivo":   {"equity_range": (50, 75),  "opt_target": "sharpe",  "max_vol": 0.28},
    "Agresivo":                 {"equity_range": (70, 95),  "opt_target": "sharpe",  "max_vol": 0.35},
}

# Fixed-income bucket names (generic, not country-specific)
FI_BUCKETS = [
    "Efectivo / Mercado de dinero",
    "Deuda gubernamental corto plazo",
    "Bonos gubernamentales plazo intermedio",
    "Bonos gubernamentales largo plazo",
    "Deuda corporativa grado de inversion",
    "Valores protegidos contra inflacion",
    "Fondos de deuda / bonos",
]

# Fixed-income allocation rules: (profile_group, horizon_group) → bucket weights
# profile_group: "conservative", "moderate", "aggressive"
# horizon_group: "short", "medium", "long"
FI_ALLOCATION_RULES = {
    ("conservative", "short"):  [0.40, 0.35, 0.10, 0.00, 0.05, 0.05, 0.05],
    ("conservative", "medium"): [0.15, 0.20, 0.30, 0.05, 0.10, 0.10, 0.10],
    ("conservative", "long"):   [0.05, 0.10, 0.25, 0.20, 0.15, 0.15, 0.10],
    ("moderate", "short"):      [0.35, 0.30, 0.15, 0.00, 0.10, 0.05, 0.05],
    ("moderate", "medium"):     [0.10, 0.15, 0.25, 0.10, 0.15, 0.10, 0.15],
    ("moderate", "long"):       [0.05, 0.05, 0.20, 0.20, 0.20, 0.15, 0.15],
    ("aggressive", "short"):    [0.30, 0.25, 0.15, 0.00, 0.15, 0.05, 0.10],
    ("aggressive", "medium"):   [0.10, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
    ("aggressive", "long"):     [0.05, 0.05, 0.15, 0.15, 0.25, 0.15, 0.20],
}

# Valuation thresholds (configurable)
VALUATION_THRESHOLDS = {
    "pe_low": 12,   "pe_high": 25,
    "pb_low": 1.0,  "pb_high": 3.5,
    "52w_low_pct": 0.25,  "52w_high_pct": 0.75,
}

# Rebalancing drift threshold (percentage points)
REBALANCE_DRIFT_PCT = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HELPER: generic prompt
# ─────────────────────────────────────────────────────────────────────────────

def prompt(msg, valid_options=None, allow_blank=False):
    """General-purpose prompt helper. Keeps asking until valid input."""
    while True:
        answer = input(msg).strip()
        if not answer and allow_blank:
            return answer
        if not answer:
            print("  [X] La entrada no puede estar vacia. Intenta de nuevo.")
            continue
        if valid_options:
            lower_opts = [v.lower() for v in valid_options]
            if answer.lower() in lower_opts:
                return valid_options[lower_opts.index(answer.lower())]
            else:
                print(f"  [X] Opcion invalida. Opciones: {', '.join(valid_options)}")
        else:
            return answer


def prompt_int(msg, min_val=None, max_val=None):
    """Prompt for an integer within optional bounds."""
    while True:
        try:
            val = int(input(msg).strip())
            if min_val is not None and val < min_val:
                print(f"  [X] El valor minimo es {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [X] El valor maximo es {max_val}.")
                continue
            return val
        except ValueError:
            print("  [X] Ingresa un numero entero valido.")


def prompt_float(msg, min_val=None, max_val=None, allow_blank=False, default=None):
    """Prompt for a float within optional bounds."""
    while True:
        raw = input(msg).strip().replace(",", "")
        if not raw and allow_blank:
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  [X] El valor minimo es {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [X] El valor maximo es {max_val}.")
                continue
            return val
        except ValueError:
            print("  [X] Ingresa un numero valido.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INVESTOR QUESTIONNAIRE & PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def get_investor_questionnaire():
    """
    Ask 10 structured questions to understand the investor.
    Returns a dict of raw answers and a computed profile score.
    """
    print("\n" + "=" * 65)
    print("  CUESTIONARIO DEL INVERSIONISTA")
    print("=" * 65)

    answers = {}
    score = 0  # Higher score → more aggressive

    # Q1 — Investment objective
    print("\n  [1/10] Cual es tu objetivo de inversion?")
    print("    1. Preservacion de capital")
    print("    2. Generacion de ingreso / renta")
    print("    3. Crecimiento moderado")
    print("    4. Crecimiento agresivo / maximizar rendimiento")
    q1 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["objective"] = q1
    score += (q1 - 1) * 3  # 0, 3, 6, 9

    obj_labels = {1: "Preservacion de capital", 2: "Generacion de ingreso",
                  3: "Crecimiento moderado", 4: "Crecimiento agresivo"}
    answers["objective_label"] = obj_labels[q1]

    # Q2 — Time horizon
    print("\n  [2/10] Horizonte de inversion (en anos)")
    print("    Cuantos anos planeas mantener esta inversion?")
    q2 = prompt_int("    Anos: ", 1, 60)
    answers["horizon_years"] = q2
    answers["horizon_category"] = determine_horizon_category(q2)
    if q2 <= 3:
        score += 0
    elif q2 <= 10:
        score += 4
    else:
        score += 8

    # Q3 — Experience
    print("\n  [3/10] Experiencia previa en inversiones")
    print("    1. Ninguna")
    print("    2. Basica (cuentas de ahorro, depositos)")
    print("    3. Intermedia (fondos, bonos, algunas acciones)")
    print("    4. Avanzada (acciones, derivados, trading activo)")
    q3 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["experience"] = q3
    score += (q3 - 1) * 2  # 0, 2, 4, 6

    exp_labels = {1: "Ninguna", 2: "Basica", 3: "Intermedia", 4: "Avanzada"}
    answers["experience_label"] = exp_labels[q3]

    # Q4 — Instruments used before
    print("\n  [4/10] Instrumentos que has utilizado antes (puedes elegir varios)")
    print("    1. Solo ahorro bancario")
    print("    2. Bonos / deuda gubernamental")
    print("    3. Fondos de inversion")
    print("    4. Acciones individuales")
    print("    5. Derivados / futuros / opciones")
    q4_raw = input("    Tu respuesta (ej. 1,3,4): ").strip()
    q4_items = [x.strip() for x in q4_raw.split(",") if x.strip().isdigit()]
    q4_items = [int(x) for x in q4_items if 1 <= int(x) <= 5]
    if not q4_items:
        q4_items = [1]
    answers["instruments"] = q4_items
    score += min(max(q4_items) - 1, 4)  # 0-4

    # Q5 — Reaction to 10% drop
    print("\n  [5/10] Si tu portafolio pierde 10% en un mes, tu reaccion seria:")
    print("    1. Vender todo inmediatamente")
    print("    2. Vender una parte para reducir riesgo")
    print("    3. Mantener y esperar la recuperacion")
    print("    4. Comprar mas aprovechando precios bajos")
    q5 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["reaction_10pct"] = q5
    score += (q5 - 1) * 3  # 0, 3, 6, 9

    # Q6 — Reaction to 20% drop
    print("\n  [6/10] Si tu portafolio pierde 20% en tres meses, tu reaccion seria:")
    print("    1. Vender todo inmediatamente")
    print("    2. Vender una parte para reducir riesgo")
    print("    3. Mantener y esperar")
    print("    4. Comprar mas")
    q6 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["reaction_20pct"] = q6
    score += (q6 - 1) * 3  # 0, 3, 6, 9

    # Q7 — Liquidity needs
    print("\n  [7/10] Necesidades de liquidez")
    print("    1. Puedo necesitar el dinero en cualquier momento")
    print("    2. Puedo necesitarlo dentro de 1 ano")
    print("    3. No necesito el dinero en los proximos 3+ anos")
    q7 = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["liquidity"] = q7
    score += (q7 - 1) * 3  # 0, 3, 6

    # Q8 — Withdrawal timing
    print("\n  [8/10] Cuando planeas comenzar a retirar fondos?")
    print("    1. Ya estoy retirando / muy pronto")
    print("    2. Dentro de 1-5 anos")
    print("    3. En mas de 5 anos")
    q8 = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["withdrawal_timing"] = q8
    score += (q8 - 1) * 2  # 0, 2, 4

    # Q9 — Comfort with volatility
    print("\n  [9/10] Que tan comodo te sientes con la volatilidad?")
    print("    1 = Muy incomodo (prefiero estabilidad total)")
    print("    5 = Muy comodo (acepto fluctuaciones fuertes)")
    q9 = prompt_int("    Tu respuesta (1-5): ", 1, 5)
    answers["volatility_comfort"] = q9
    score += (q9 - 1) * 2  # 0, 2, 4, 6, 8

    # Q10 — Desired FI/RV split (guidance)
    print("\n  [10/10] Distribucion deseada entre renta fija y renta variable")
    print("    Indica el % que deseas en renta fija (el resto sera renta variable).")
    print("    Si no tienes preferencia, presiona Enter para usar la recomendacion.\n")
    q10 = prompt_float("    % Renta Fija (0-100, o Enter para auto): ",
                       min_val=0, max_val=100, allow_blank=True, default=None)
    answers["desired_pct_fixed"] = q10

    answers["raw_score"] = score
    return answers


def determine_horizon_category(years):
    """Classify investment horizon into short / medium / long term."""
    for category, (lo, hi) in HORIZON_RULES.items():
        if lo <= years <= hi:
            return category
    return "Largo plazo"


def determine_investor_profile(answers):
    """
    Compute investor profile from questionnaire score.
    Score ranges (0-59 max):
      0-10  → Conservador
      11-18 → Moderadamente Conservador
      19-30 → Moderado
      31-42 → Moderadamente Agresivo
      43+   → Agresivo
    """
    score = answers["raw_score"]
    if score <= 10:
        profile = "Conservador"
    elif score <= 18:
        profile = "Moderadamente Conservador"
    elif score <= 30:
        profile = "Moderado"
    elif score <= 42:
        profile = "Moderadamente Agresivo"
    else:
        profile = "Agresivo"
    return profile


def get_recommended_split(profile, horizon_cat):
    """
    Based on profile and horizon, suggest a fixed/variable income split.
    Returns (pct_fixed, pct_variable).
    """
    settings = PROFILE_SETTINGS[profile]
    eq_lo, eq_hi = settings["equity_range"]
    # Horizon adjusts within the range
    if horizon_cat == "Corto plazo":
        eq_pct = eq_lo
    elif horizon_cat == "Mediano plazo":
        eq_pct = (eq_lo + eq_hi) // 2
    else:
        eq_pct = eq_hi
    return (100 - eq_pct, eq_pct)


def _profile_group(profile):
    """Map 5-level profile to 3-level group for FI allocation rules."""
    if profile in ("Conservador", "Moderadamente Conservador"):
        return "conservative"
    elif profile in ("Moderado",):
        return "moderate"
    else:
        return "aggressive"


def _horizon_group(horizon_cat):
    """Map horizon category to rule key."""
    mapping = {"Corto plazo": "short", "Mediano plazo": "medium", "Largo plazo": "long"}
    return mapping.get(horizon_cat, "medium")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FIXED-INCOME RECOMMENDATION MODULE
# ─────────────────────────────────────────────────────────────────────────────

def recommend_fixed_income_mix(profile, horizon_cat, pct_fixed, monto_total):
    """
    Rule-based fixed-income diversification recommendation.
    Returns a DataFrame with bucket names, weights, and dollar amounts.
    """
    pg = _profile_group(profile)
    hg = _horizon_group(horizon_cat)
    weights = FI_ALLOCATION_RULES.get((pg, hg), FI_ALLOCATION_RULES[("moderate", "medium")])

    monto_fi = monto_total * pct_fixed / 100.0
    df = pd.DataFrame({
        "Instrumento": FI_BUCKETS,
        "Peso (%)": [w * 100 for w in weights],
        "Monto ($)": [w * monto_fi for w in weights],
    })
    df = df[df["Peso (%)"] > 0].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — USER INPUT COLLECTION (tickers, dates, benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def get_market_inputs(profile):
    """
    Collect market-specific inputs: tickers, benchmark, periodicity,
    date range, investment amount, risk-free rate, target return.
    """
    print("\n" + "=" * 65)
    print("  CONFIGURACION DE MERCADO")
    print("=" * 65)

    # Number of assets
    print(f"\n  [1/7] NUMERO DE ACTIVOS (max {MAX_TICKERS})")
    num_tickers = prompt_int(f"  Cuantos tickers deseas analizar? (1-{MAX_TICKERS}): ", 1, MAX_TICKERS)

    # Ticker symbols
    print(f"\n  [2/7] TICKERS (simbolos de Yahoo Finance)")
    print("  Ingresa los tickers uno por uno (ej. AAPL, MSFT, AMZN):\n")
    tickers = []
    for i in range(num_tickers):
        while True:
            ticker = input(f"    Ticker {i+1}/{num_tickers}: ").strip().upper()
            if not ticker:
                print("    [X] El ticker no puede estar vacio.")
            elif ticker in tickers:
                print("    [X] Ticker duplicado.")
            else:
                tickers.append(ticker)
                break

    # Benchmark
    print(f"\n  [3/7] BENCHMARK")
    print(f"  Sugeridos: SPY, ^GSPC, ^DJI, ^IXIC, QQQ")
    print(f"  Presiona Enter para usar ({DEFAULT_BENCHMARK}).\n")
    bench_raw = input("  Benchmark ticker: ").strip().upper()
    benchmark = bench_raw if bench_raw else DEFAULT_BENCHMARK

    # Periodicity
    print("\n  [4/7] PERIODICIDAD DE DATOS")
    print("  Opciones: diaria, semanal, mensual\n")
    periodicity = prompt("  Periodicidad: ", valid_options=list(PERIODICITY_MAP.keys()))

    # Date range
    print("\n  [5/7] RANGO DE FECHAS (formato: YYYY-MM-DD)")
    while True:
        start_date_str = input("  Fecha de Inicio (YYYY-MM-DD): ").strip()
        end_date_str   = input("  Fecha de Cierre (YYYY-MM-DD): ").strip()
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date   = datetime.strptime(end_date_str,   "%Y-%m-%d")
            if end_date <= start_date:
                print("  [X] La fecha de cierre debe ser posterior a la de inicio.")
                continue
            if (end_date - start_date).days < 30:
                print("  [X] El rango debe ser de al menos 30 dias.")
                continue
            break
        except ValueError:
            print("  [X] Formato invalido. Usa YYYY-MM-DD.")

    # Investment amount
    print("\n  [6/7] MONTO TOTAL A INVERTIR")
    monto_total = prompt_float("  Monto total (ej. 1000000): ", min_val=0.01)

    # Risk-free rate
    print(f"\n  [7/7] TASA LIBRE DE RIESGO")
    print(f"  Presiona Enter para usar ({RISK_FREE_RATE*100:.1f} % anual).\n")
    risk_free = prompt_float("  Tasa libre de riesgo (%): ",
                             allow_blank=True, default=RISK_FREE_RATE * 100)
    if risk_free is not None and risk_free != RISK_FREE_RATE * 100:
        risk_free = risk_free / 100.0
    else:
        risk_free = RISK_FREE_RATE

    # Target return (optional)
    print("\n  MODO RENDIMIENTO OBJETIVO (opcional)")
    print("  Puedes especificar un rendimiento anual objetivo.")
    print("  Presiona Enter para omitir.\n")
    target_return = prompt_float("  Rendimiento objetivo anual (%): ",
                                 allow_blank=True, default=None)
    if target_return is not None:
        target_return = target_return / 100.0

    return {
        "tickers": tickers,
        "benchmark": benchmark,
        "periodicity": periodicity,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "monto_total": monto_total,
        "risk_free": risk_free,
        "interval": PERIODICITY_MAP[periodicity][0],
        "annualize": PERIODICITY_MAP[periodicity][1],
        "target_return": target_return,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MANUAL WEIGHTS ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def get_manual_weights(valid_tickers):
    """Let the user assign custom weights. Returns a numpy array (sums to 1)."""
    print(f"\n{'-'*75}")
    print("  DISENA TU PORTAFOLIO MANUAL")
    print(f"{'-'*75}")
    print("  Opciones rapidas:")
    print("    'equitativo' para distribuir por partes iguales.")
    print("    Ingresa todos: AAPL=40, MSFT=60")
    print("    Presiona Enter para ingresarlos uno por uno.\n")

    pesos = {}
    ingreso_bloque = input("  Pesos en bloque (opcional): ").strip()

    if ingreso_bloque.lower() == "equitativo":
        for t in valid_tickers:
            pesos[t] = 1.0 / len(valid_tickers)
    elif ingreso_bloque:
        pares = ingreso_bloque.split(",")
        for par in pares:
            if "=" in par:
                t, w = par.split("=", 1)
                t = t.strip().upper()
                try:
                    w = float(w.strip())
                    if t in valid_tickers:
                        pesos[t] = w / 100.0
                except ValueError:
                    pass

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
                    print("  [X] Ingresa un numero valido.")

    suma_pesos = sum(pesos.values())
    if abs(suma_pesos - 1.0) > 0.001:
        print(f"  [!] ADVERTENCIA: La suma es {suma_pesos*100:.1f} %. Normalizando...")
        for t in pesos:
            pesos[t] = pesos[t] / suma_pesos

    return np.array([pesos[t] for t in valid_tickers])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — DATA DOWNLOAD & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def download_asset_data(tickers, benchmark, start_date_str, end_date_str, interval):
    """Download adjusted close prices from Yahoo Finance. Validates each ticker."""
    print("\n  Descargando datos desde Yahoo Finance...")
    all_tickers = list(set(tickers + [benchmark]))

    try:
        raw = yf.download(
            tickers=all_tickers, start=start_date_str, end=end_date_str,
            interval=interval, auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:
        print(f"  [X] Error al descargar datos: {e}")
        sys.exit(1)

    if raw.empty:
        print("  [X] No se obtuvieron datos. Verifica los tickers y fechas.")
        sys.exit(1)

    if isinstance(raw.columns, pd.MultiIndex):
        prices_all = raw["Close"]
    else:
        prices_all = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    valid_tickers = []
    invalid_tickers = []
    for t in tickers:
        if t not in prices_all.columns:
            invalid_tickers.append(t)
            continue
        col = prices_all[t].dropna()
        if len(col) < 10:
            print(f"  [!] {t}: datos insuficientes ({len(col)} filas) — omitiendo.")
            invalid_tickers.append(t)
        else:
            valid_tickers.append(t)

    if invalid_tickers:
        print(f"  [!] Tickers omitidos: {', '.join(invalid_tickers)}")
    if not valid_tickers:
        print("\n  [X] ERROR: Ningun ticker valido. Saliendo.")
        sys.exit(1)

    if benchmark not in prices_all.columns or prices_all[benchmark].dropna().shape[0] < 10:
        print(f"  [!] Benchmark '{benchmark}' sin datos. Usando {DEFAULT_BENCHMARK}.")
        bench_data = yf.download(DEFAULT_BENCHMARK, start=start_date_str, end=end_date_str,
                                 interval=interval, auto_adjust=True, progress=False)
        bench_prices = bench_data["Close"].dropna()
        benchmark = DEFAULT_BENCHMARK
    else:
        bench_prices = prices_all[benchmark].dropna()

    prices = prices_all[valid_tickers].dropna(how="all")
    print(f"  [OK] {len(prices)} filas x {len(valid_tickers)} activos cargados.")
    print(f"  [OK] Benchmark '{benchmark}': {len(bench_prices)} filas.")
    return prices, bench_prices, valid_tickers, benchmark


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RETURN & STATISTICS CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_returns(prices):
    """Compute simple percentage returns."""
    return prices.pct_change().dropna()


def calculate_summary_statistics(returns, bench_returns, annualize_factor, risk_free):
    """Per-asset annualized stats: return, volatility, Sharpe, Beta, Alpha, Treynor."""
    mean_ret = returns.mean() * annualize_factor
    vol      = returns.std()  * np.sqrt(annualize_factor)
    sharpe   = (mean_ret - risk_free) / vol

    bench_mean = bench_returns.mean() * annualize_factor
    bench_var  = bench_returns.var()

    stats = {
        "Annual Return": mean_ret, "Annual Volatility": vol, "Sharpe": sharpe,
        "Beta": pd.Series(dtype=float), "Alpha": pd.Series(dtype=float),
        "Treynor": pd.Series(dtype=float),
    }

    for t in returns.columns:
        merged = pd.concat([returns[t], bench_returns], axis=1).dropna()
        if merged.empty or bench_var == 0:
            b, a, tr = 0.0, 0.0, 0.0
        else:
            cov_val = merged.cov().iloc[0, 1]
            b  = cov_val / bench_var
            a  = mean_ret[t] - (risk_free + b * (bench_mean - risk_free))
            tr = (mean_ret[t] - risk_free) / b if b != 0 else np.nan
        stats["Beta"].loc[t]    = b
        stats["Alpha"].loc[t]   = a
        stats["Treynor"].loc[t] = tr

    return pd.DataFrame(stats)


def calculate_correlation_matrix(returns):
    return returns.corr()


def calculate_covariance_matrix(returns, annualize_factor):
    return returns.cov() * annualize_factor


def display_matrix(matrix, title):
    print(f"\n{'─'*75}")
    print(f"  {title}")
    print(f"{'─'*75}")
    formatted = matrix.round(4).to_string()
    for line in formatted.split("\n"):
        print(f"  {line}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MARKOWITZ PORTFOLIO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_portfolios(returns, annualize_factor, risk_free, n=N_SIMULATIONS):
    """Monte Carlo: N random long-only portfolios."""
    n_assets     = returns.shape[1]
    mean_returns = returns.mean() * annualize_factor
    cov_matrix   = returns.cov()  * annualize_factor

    results = {"return": np.zeros(n), "volatility": np.zeros(n),
               "sharpe": np.zeros(n), "weights": [None] * n}

    for i in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(w @ cov_matrix.values @ w)
        sr  = (ret - risk_free) / vol if vol > 0 else 0
        results["return"][i]     = ret
        results["volatility"][i] = vol
        results["sharpe"][i]     = sr
        results["weights"][i]    = w

    return pd.DataFrame(results)


def evaluate_portfolio(weights, returns, bench_returns, asset_stats,
                       annualize_factor, risk_free):
    """Evaluate a portfolio: (ret, vol, sharpe, beta, alpha, treynor)."""
    mean_returns = returns.mean() * annualize_factor
    cov_matrix   = returns.cov()  * annualize_factor

    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights @ cov_matrix.values @ weights)
    sr  = (ret - risk_free) / vol if vol > 0 else 0

    port_beta    = np.dot(weights, asset_stats["Beta"].values)
    bench_mean   = bench_returns.mean() * annualize_factor
    port_alpha   = ret - (risk_free + port_beta * (bench_mean - risk_free))
    port_treynor = (ret - risk_free) / port_beta if port_beta != 0 else np.nan

    return ret, vol, sr, port_beta, port_alpha, port_treynor


def optimize_portfolio(returns, bench_returns, asset_stats,
                       annualize_factor, risk_free, target="sharpe"):
    """Scipy optimization for Max Sharpe or Min Variance. Long-only."""
    n_assets     = returns.shape[1]
    mean_returns = returns.mean() * annualize_factor
    cov_matrix   = returns.cov()  * annualize_factor

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, 1.0) for _ in range(n_assets))
    w0          = np.array([1 / n_assets] * n_assets)

    try:
        if target == "sharpe":
            def neg_sharpe(w):
                r = np.dot(w, mean_returns)
                v = np.sqrt(w @ cov_matrix.values @ w)
                return -(r - risk_free) / v if v > 0 else 0
            result = minimize(neg_sharpe, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-9})
        elif target == "minvol":
            def portfolio_vol(w):
                return np.sqrt(w @ cov_matrix.values @ w)
            result = minimize(portfolio_vol, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-9})
        else:
            raise ValueError(f"Target desconocido: {target}")

        if not result.success:
            print(f"  [!] Advertencia optimizacion ({target}): {result.message}")
        w_opt = result.x
    except Exception as e:
        print(f"  [!] Optimizacion fallida ({target}): {e}")
        w_opt = np.array([1 / n_assets] * n_assets)

    metrics = evaluate_portfolio(w_opt, returns, bench_returns,
                                  asset_stats, annualize_factor, risk_free)
    return (w_opt,) + metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — VaR CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_var(port_return, port_vol, confidence, horizon_days, investment_amount):
    """
    Parametric Value-at-Risk assuming normal distribution.
    Returns the absolute VaR in dollars (positive number = potential loss).
    Assumptions: returns are normally distributed, parameters are annualized.
    """
    # Convert annualized to daily
    daily_ret = port_return / 252
    daily_vol = port_vol / np.sqrt(252)

    z_score = norm.ppf(1 - confidence)  # negative
    var_pct = -(daily_ret * horizon_days + z_score * daily_vol * np.sqrt(horizon_days))
    var_dollar = var_pct * investment_amount
    return max(var_dollar, 0)


def calculate_var_multiple(port_return, port_vol, investment_amount):
    """Calculate VaR at 90%, 95%, 99% confidence for 1-day and annual horizons."""
    results = {}
    for conf in [0.90, 0.95, 0.99]:
        var_1d = calculate_var(port_return, port_vol, conf, 1, investment_amount)
        var_annual = calculate_var(port_return, port_vol, conf, 252, investment_amount)
        results[conf] = {"1_day": var_1d, "annual": var_annual}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — VALUATION SIGNALS MODULE
# ─────────────────────────────────────────────────────────────────────────────

def estimate_valuation_signals(tickers):
    """
    Fetch fundamental data from yfinance and classify each ticker.
    Returns a dict: ticker → {metrics..., signal}
    """
    signals = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            pe = info.get("trailingPE")
            fwd_pe = info.get("forwardPE")
            pb = info.get("priceToBook")
            div_yield = info.get("dividendYield")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            hi52 = info.get("fiftyTwoWeekHigh")
            lo52 = info.get("fiftyTwoWeekLow")

            signal = classify_valuation(pe, fwd_pe, pb, price, hi52, lo52)
            signals[t] = {
                "trailing_PE": pe, "forward_PE": fwd_pe, "P/B": pb,
                "div_yield": div_yield, "price": price,
                "52w_high": hi52, "52w_low": lo52, "signal": signal,
            }
        except Exception:
            signals[t] = {"signal": "Datos insuficientes"}
    return signals


def classify_valuation(pe, fwd_pe, pb, price, hi52, lo52):
    """
    Classify a stock as potentially undervalued, neutral, or overvalued.
    Uses a simple scoring system based on available metrics.
    NOTE: This is an INDICATIVE signal, not a definitive valuation.
    """
    score = 0
    n_signals = 0

    # P/E analysis
    if pe is not None and pe > 0:
        n_signals += 1
        if pe < VALUATION_THRESHOLDS["pe_low"]:
            score += 1   # cheap
        elif pe > VALUATION_THRESHOLDS["pe_high"]:
            score -= 1   # expensive

    # Forward P/E confirmation
    if fwd_pe is not None and fwd_pe > 0 and pe is not None and pe > 0:
        n_signals += 1
        if fwd_pe < pe * 0.85:
            score += 1   # earnings expected to grow
        elif fwd_pe > pe * 1.15:
            score -= 1

    # P/B analysis
    if pb is not None and pb > 0:
        n_signals += 1
        if pb < VALUATION_THRESHOLDS["pb_low"]:
            score += 1
        elif pb > VALUATION_THRESHOLDS["pb_high"]:
            score -= 1

    # 52-week range position
    if price is not None and hi52 is not None and lo52 is not None and hi52 > lo52:
        n_signals += 1
        position = (price - lo52) / (hi52 - lo52)
        if position < VALUATION_THRESHOLDS["52w_low_pct"]:
            score += 1   # near lows
        elif position > VALUATION_THRESHOLDS["52w_high_pct"]:
            score -= 1   # near highs

    if n_signals == 0:
        return "Datos insuficientes"
    elif score >= 2:
        return "Potencialmente subvaluado"
    elif score <= -2:
        return "Potencialmente sobrevaluado"
    else:
        return "Neutral / valor justo"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — REBALANCING RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_rebalancing_suggestions(valid_tickers, opt_weights, valuation_signals,
                                      port_vol, profile, horizon_cat):
    """
    Produce rebalancing suggestions based on valuation signals,
    portfolio drift, and profile constraints.
    """
    suggestions = []
    settings = PROFILE_SETTINGS.get(profile, PROFILE_SETTINGS["Moderado"])
    max_vol = settings["max_vol"]

    # Volatility check
    if port_vol > max_vol:
        suggestions.append(
            f"⚠ La volatilidad del portafolio ({port_vol*100:.1f}%) excede el limite "
            f"sugerido para perfil {profile} ({max_vol*100:.1f}%). "
            f"Considere reducir exposicion a renta variable o seleccionar activos menos volatiles."
        )

    # Per-stock valuation + weight suggestions
    for t, w in zip(valid_tickers, opt_weights):
        if w < 0.01:
            continue  # skip negligible positions
        sig = valuation_signals.get(t, {}).get("signal", "Datos insuficientes")
        if sig == "Potencialmente sobrevaluado" and w > 0.15:
            suggestions.append(
                f"📉 {t}: posicion significativa ({w*100:.1f}%) y senal de sobrevaluacion. "
                f"Considere reducir exposicion."
            )
        elif sig == "Potencialmente subvaluado" and w < 0.10:
            suggestions.append(
                f"📈 {t}: senal de subvaluacion con posicion baja ({w*100:.1f}%). "
                f"Considere incrementar si es consistente con su perfil."
            )
        elif sig == "Potencialmente sobrevaluado":
            suggestions.append(
                f"🔍 {t}: monitorear — senal de sobrevaluacion detectada."
            )

    # Horizon-based suggestion
    if horizon_cat == "Corto plazo" and port_vol > 0.15:
        suggestions.append(
            "⏰ Horizonte corto con volatilidad elevada. Considere aumentar "
            "la proporcion de renta fija para proteger el capital."
        )

    if not suggestions:
        suggestions.append(
            "✅ No se detectan senales urgentes de rebalanceo. "
            "El portafolio se encuentra dentro de los parametros esperados."
        )

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — TARGET RETURN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def check_target_return(target_ret, port_ret, port_vol, risk_free):
    """
    Evaluate whether the portfolio can meet a target return.
    Returns a dict with analysis results.
    """
    meets_target = port_ret >= target_ret
    gap = port_ret - target_ret

    # Probability of meeting target under normal assumption (annual)
    if port_vol > 0:
        z = (port_ret - target_ret) / port_vol
        prob_meet = norm.cdf(z) * 100
    else:
        prob_meet = 100.0 if meets_target else 0.0

    return {
        "target": target_ret,
        "portfolio_return": port_ret,
        "gap": gap,
        "meets_target": meets_target,
        "probability_pct": prob_meet,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — BENCHMARK & MAX DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────

def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown from a returns series."""
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()  # most negative value


def calculate_benchmark_metrics(bench_returns, annualize_factor, risk_free):
    """Annualized metrics for the benchmark including max drawdown."""
    ret = bench_returns.mean() * annualize_factor
    vol = bench_returns.std()  * np.sqrt(annualize_factor)
    sr  = (ret - risk_free) / vol if vol > 0 else 0
    cum_ret = (1 + bench_returns).prod() - 1
    max_dd = calculate_max_drawdown(bench_returns)
    return {"return": ret, "volatility": vol, "sharpe": sr,
            "beta": 1.0, "alpha": 0.0, "treynor": (ret - risk_free),
            "cumulative_return": cum_ret, "max_drawdown": max_dd}


def calculate_portfolio_drawdown(weights, returns):
    """Calculate max drawdown for a weighted portfolio."""
    port_returns = returns @ weights
    return calculate_max_drawdown(port_returns)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def display_results(cfg, valid_tickers, asset_stats,
                    ms_w, ms_ret, ms_vol, ms_sr,
                    mv_w, mv_ret, mv_vol, mv_sr,
                    bench_metrics, benchmark):
    """Print formatted summary of all analysis results."""
    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  RESULTADOS DEL ANALISIS")
    print(SEP)
    print(f"\n  Perfil inversion  : {cfg['profile']}")
    print(f"  Horizonte         : {cfg['horizon_years']} anos ({cfg['horizon_category']})")
    print(f"  Renta Fija        : {cfg['pct_fixed']:.0f} %")
    print(f"  Renta Variable    : {cfg['pct_variable']:.0f} %")
    print(f"  Inversion total   : ${cfg['monto_total']:,.2f}")
    monto_rv = cfg['monto_total'] * cfg['pct_variable'] / 100
    print(f"  Monto en RV       : ${monto_rv:,.2f}")
    print(f"  Rango fechas      : {cfg['start_date']} a {cfg['end_date']}")
    print(f"  Periodicidad      : {cfg['periodicity']}")
    print(f"  Tasa libre riesgo : {cfg['risk_free']*100:.2f} %")

    # Individual asset stats
    print(f"\n{'_'*95}")
    print("  ESTADISTICAS INDIVIDUALES POR ACTIVO (anualizadas)")
    print(f"{'_'*95}")
    hdr = f"  {'Ticker':<8} {'Rend.Anual':>11} {'Vol.Anual':>11} {'Sharpe':>8} {'Beta':>8} {'Alpha':>8} {'Treynor':>8}"
    print(hdr)
    print(f"  {'-'*8} {'-'*11} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ticker in valid_tickers:
        r  = asset_stats.loc[ticker, "Annual Return"] * 100
        v  = asset_stats.loc[ticker, "Annual Volatility"] * 100
        sh = asset_stats.loc[ticker, "Sharpe"]
        b  = asset_stats.loc[ticker, "Beta"]
        a  = asset_stats.loc[ticker, "Alpha"] * 100
        tr = asset_stats.loc[ticker, "Treynor"]
        print(f"  {ticker:<8} {r:>10.2f}% {v:>10.2f}% {sh:>8.4f} {b:>8.4f} {a:>7.2f}% {tr:>8.4f}")

    # Optimal portfolios
    opts = [
        ("PORTAFOLIO OPTIMO — Maximo Sharpe", ms_w, ms_ret, ms_vol, ms_sr),
        ("PORTAFOLIO OPTIMO — Minima Varianza", mv_w, mv_ret, mv_vol, mv_sr),
    ]
    for title, w_opt, ret_opt, vol_opt, sr_opt in opts:
        print(f"\n{'_'*75}")
        print(f"  {title}")
        print(f"{'_'*75}")
        print(f"  Rendimiento Anual Esperado : {ret_opt*100:>8.2f} %")
        print(f"  Volatilidad Anual Esperada : {vol_opt*100:>8.2f} %")
        print(f"  Ratio de Sharpe            : {sr_opt:>8.4f}")
        print("\n  Pesos de los activos (sobre la porcion de Renta Variable):")
        for t, w in zip(valid_tickers, w_opt):
            if w > 0.001:
                amt = w * monto_rv
                print(f"    {t:<8}: {w*100:>6.2f} %  ->  ${amt:,.2f}")

    # Benchmark
    print(f"\n{'_'*75}")
    print(f"  BENCHMARK: {benchmark} (anualizado)")
    print(f"{'_'*75}")
    print(f"  Rendimiento       : {bench_metrics['return']*100:>8.2f} %")
    print(f"  Volatilidad       : {bench_metrics['volatility']*100:>8.2f} %")
    print(f"  Sharpe            : {bench_metrics['sharpe']:>8.4f}")
    print(f"  Retorno acumulado : {bench_metrics['cumulative_return']*100:>8.2f} %")
    print(f"  Max Drawdown      : {bench_metrics['max_drawdown']*100:>8.2f} %")
    print(f"\n{SEP}\n")


def display_manual_portfolio(cfg, valid_tickers, manual_w, returns,
                              bench_returns, asset_stats):
    """Evaluate and display the user's manual portfolio."""
    print(f"\n{'_'*75}")
    print("  TU PORTAFOLIO MANUAL")
    print(f"{'_'*75}")
    man_ret, man_vol, man_sr, man_beta, man_alpha, man_treynor = evaluate_portfolio(
        manual_w, returns, bench_returns, asset_stats,
        cfg["annualize"], cfg["risk_free"]
    )
    monto_rv = cfg['monto_total'] * cfg['pct_variable'] / 100
    print(f"  Rendimiento Anual Esperado : {man_ret*100:>8.2f} %")
    print(f"  Volatilidad Anual Esperada : {man_vol*100:>8.2f} %")
    print(f"  Ratio de Sharpe            : {man_sr:>8.4f}")
    print(f"  Beta                       : {man_beta:>8.4f}")
    print(f"  Alpha de Jensen            : {man_alpha*100:>8.2f} %")
    print(f"  Ratio de Treynor           : {man_treynor:>8.4f}")
    print("\n  Pesos de los activos:")
    for t, w in zip(valid_tickers, manual_w):
        if w > 0.001:
            amt = w * monto_rv
            print(f"    {t:<8}: {w*100:>6.2f} %  ->  ${amt:,.2f}")
    print(f"\n{'='*75}\n")
    return man_ret, man_vol, man_sr


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — PROFESSIONAL RECOMMENDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendation_report(cfg, valid_tickers, corr_matrix,
                                    ms_w, ms_ret, ms_vol, ms_sr,
                                    mv_w, mv_ret, mv_vol, mv_sr,
                                    bench_metrics, benchmark,
                                    var_results, valuation_signals,
                                    rebalancing_suggestions,
                                    fi_recommendation, target_analysis,
                                    port_max_dd):
    """
    Generate a professional advisory-style investment recommendation memo.
    """
    monto_rv = cfg['monto_total'] * cfg['pct_variable'] / 100
    monto_rf = cfg['monto_total'] * cfg['pct_fixed'] / 100

    # Average pairwise correlation
    n = len(corr_matrix)
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        avg_corr = corr_matrix.values[mask].mean()
    else:
        avg_corr = 1.0

    if avg_corr < 0.3:
        div_text = "excelente (correlacion promedio baja)"
    elif avg_corr < 0.6:
        div_text = "buena (correlacion promedio moderada)"
    else:
        div_text = "limitada (correlacion promedio alta)"

    # Pick primary portfolio based on profile
    settings = PROFILE_SETTINGS.get(cfg['profile'], PROFILE_SETTINGS["Moderado"])
    if settings["opt_target"] == "minvol":
        prim_w, prim_ret, prim_vol, prim_sr = mv_w, mv_ret, mv_vol, mv_sr
        prim_label = "Minima Varianza (sugerido para perfil conservador)"
    else:
        prim_w, prim_ret, prim_vol, prim_sr = ms_w, ms_ret, ms_vol, ms_sr
        prim_label = "Maximo Sharpe (sugerido para perfil moderado/agresivo)"

    SEP = "=" * 75

    print(f"\n{SEP}")
    print("  MEMORANDUM DE RECOMENDACION DE INVERSION")
    print(SEP)

    # 1. Client Profile Summary
    print(f"\n  1. PERFIL DEL CLIENTE")
    print(f"  {'─'*40}")
    print(f"  Objetivo de inversion : {cfg.get('objective_label', 'N/A')}")
    print(f"  Horizonte             : {cfg['horizon_years']} anos ({cfg['horizon_category']})")
    print(f"  Perfil de riesgo      : {cfg['profile']}")
    print(f"  Experiencia           : {cfg.get('experience_label', 'N/A')}")
    print(f"  Tolerancia volatilidad: {cfg.get('volatility_comfort', 'N/A')}/5")

    # 2. Asset Allocation Overview
    print(f"\n  2. ASIGNACION DE ACTIVOS")
    print(f"  {'─'*40}")
    print(f"  Inversion Total       : ${cfg['monto_total']:>14,.2f}")
    print(f"  Renta Fija            : ${monto_rf:>14,.2f}  ({cfg['pct_fixed']:.0f} %)")
    print(f"  Renta Variable        : ${monto_rv:>14,.2f}  ({cfg['pct_variable']:.0f} %)")

    # 3. Fixed-Income Structure
    if fi_recommendation is not None and not fi_recommendation.empty:
        print(f"\n  3. ESTRUCTURA DE RENTA FIJA (recomendacion basada en reglas)")
        print(f"  {'─'*40}")
        print(f"  NOTA: Estas son recomendaciones cualitativas basadas en perfil/horizonte.")
        for _, row in fi_recommendation.iterrows():
            print(f"    - {row['Instrumento']:<45} {row['Peso (%)']:>5.1f}%  ${row['Monto ($)']:>12,.2f}")
    else:
        print(f"\n  3. RENTA FIJA")
        print(f"  {'─'*40}")
        print(f"  Modulo de diversificacion de renta fija no activado.")

    # 4. Equity Portfolio
    print(f"\n  4. PORTAFOLIO DE RENTA VARIABLE — {prim_label}")
    print(f"  {'─'*40}")
    for t, w in zip(valid_tickers, prim_w):
        if w > 0.001:
            amt = w * monto_rv
            print(f"    {t:<8}: {w*100:>6.2f} %  ->  ${amt:>12,.2f}")

    # 5. Portfolio Metrics
    print(f"\n  5. METRICAS DEL PORTAFOLIO")
    print(f"  {'─'*40}")
    print(f"  Rendimiento anual esperado : {prim_ret*100:>8.2f} %")
    print(f"  Volatilidad anual          : {prim_vol*100:>8.2f} %")
    print(f"  Ratio de Sharpe            : {prim_sr:>8.4f}")
    print(f"  Max Drawdown historico     : {port_max_dd*100:>8.2f} %")
    print(f"  Diversificacion            : {div_text}")
    print(f"  Correlacion promedio       : {avg_corr:>8.4f}")

    # VaR display
    if var_results:
        print(f"\n  VALUE AT RISK (VaR) — Metodo Parametrico")
        print(f"  Supuestos: distribucion normal, parametros historicos")
        print(f"  {'Confianza':>12} {'VaR 1 dia':>16} {'VaR Anual':>16}")
        print(f"  {'-'*12} {'-'*16} {'-'*16}")
        for conf, vals in var_results.items():
            print(f"  {conf*100:>11.0f}% ${vals['1_day']:>14,.2f} ${vals['annual']:>14,.2f}")

    # 6. Benchmark Comparison
    print(f"\n  6. COMPARACION CON BENCHMARK ({benchmark})")
    print(f"  {'─'*40}")
    print(f"  {'Metrica':<28} {'Portafolio':>12} {'Benchmark':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}")
    print(f"  {'Rendimiento anual':<28} {prim_ret*100:>11.2f}% {bench_metrics['return']*100:>11.2f}%")
    print(f"  {'Volatilidad anual':<28} {prim_vol*100:>11.2f}% {bench_metrics['volatility']*100:>11.2f}%")
    print(f"  {'Sharpe':<28} {prim_sr:>12.4f} {bench_metrics['sharpe']:>12.4f}")
    print(f"  {'Max Drawdown':<28} {port_max_dd*100:>11.2f}% {bench_metrics['max_drawdown']*100:>11.2f}%")

    if target_analysis is not None:
        print(f"\n  {'Rendimiento objetivo':<28} {target_analysis['target']*100:>11.2f}%")
        status = "✅ CUMPLE" if target_analysis['meets_target'] else "❌ NO CUMPLE"
        print(f"  {'Estatus vs objetivo':<28} {status:>12}")
        print(f"  {'Probabilidad de alcanzarlo':<28} {target_analysis['probability_pct']:>11.1f}%")

    # 7. Valuation & Rebalancing
    if valuation_signals:
        print(f"\n  7. SENALES DE VALUACION Y REBALANCEO")
        print(f"  {'─'*40}")
        print(f"  NOTA: Senales indicativas, no constituyen consejo de inversion definitivo.\n")
        print(f"  {'Ticker':<8} {'P/E':>8} {'Fwd P/E':>8} {'P/B':>6} {'Senal':<30}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*30}")
        for t in valid_tickers:
            vs = valuation_signals.get(t, {})
            pe = vs.get("trailing_PE")
            fpe = vs.get("forward_PE")
            pb = vs.get("P/B")
            sig = vs.get("signal", "N/A")
            pe_str = f"{pe:.1f}" if pe else "N/A"
            fpe_str = f"{fpe:.1f}" if fpe else "N/A"
            pb_str = f"{pb:.2f}" if pb else "N/A"
            print(f"  {t:<8} {pe_str:>8} {fpe_str:>8} {pb_str:>6} {sig:<30}")

    if rebalancing_suggestions:
        print(f"\n  SUGERENCIAS DE REBALANCEO:")
        for sug in rebalancing_suggestions:
            print(f"    {sug}")

    # 8. Professional Conclusion
    print(f"\n  8. CONCLUSION PROFESIONAL")
    print(f"  {'─'*40}")

    beats_return = prim_ret > bench_metrics["return"]
    beats_sharpe = prim_sr  > bench_metrics["sharpe"]

    conclusion_parts = []
    conclusion_parts.append(
        f"  Para un inversionista con perfil {cfg['profile']} y horizonte de "
        f"{cfg['horizon_years']} anos ({cfg['horizon_category']}), se recomienda "
        f"una asignacion de {cfg['pct_fixed']:.0f}% renta fija / {cfg['pct_variable']:.0f}% "
        f"renta variable."
    )

    if beats_return and beats_sharpe:
        conclusion_parts.append(
            "  El portafolio optimizado supera al benchmark tanto en rendimiento "
            "como en eficiencia ajustada por riesgo, lo cual lo posiciona como una "
            "alternativa atractiva."
        )
    elif beats_sharpe:
        conclusion_parts.append(
            "  Aunque el rendimiento absoluto es menor que el benchmark, la eficiencia "
            "ajustada por riesgo (Sharpe) es superior, adecuado para inversores que "
            "priorizan la relacion riesgo-rendimiento."
        )
    elif beats_return:
        conclusion_parts.append(
            "  El portafolio ofrece mayor rendimiento que el benchmark, aunque con "
            "menor eficiencia ajustada por riesgo."
        )
    else:
        conclusion_parts.append(
            "  El benchmark supera al portafolio en las metricas principales. "
            "Considere ajustar la seleccion de activos o el horizonte."
        )

    profile_advice = {
        "Conservador":
            "  Dado el perfil conservador, se prioriza estabilidad y preservacion "
            "de capital. La porcion de renta fija actua como ancla protectora.",
        "Moderadamente Conservador":
            "  El perfil moderadamente conservador busca seguridad con algo de crecimiento. "
            "La combinacion sugerida balancea proteccion y rendimiento modesto.",
        "Moderado":
            "  El perfil moderado busca equilibrio entre crecimiento y proteccion. "
            "La asignacion optimizada se alinea con este objetivo.",
        "Moderadamente Agresivo":
            "  El perfil moderadamente agresivo prioriza crecimiento aceptando mayor "
            "volatilidad. La mayor exposicion a renta variable maximiza oportunidades.",
        "Agresivo":
            "  El perfil agresivo prioriza maximizar rendimiento a largo plazo. "
            "La alta exposicion a renta variable asume volatilidades significativas "
            "en busca de mayor retorno.",
    }
    conclusion_parts.append(profile_advice.get(cfg['profile'], ''))

    for part in conclusion_parts:
        print(part)

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — SAVE CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(corr_matrix, cov_matrix, asset_stats, valid_tickers,
                 ms_w, mv_w, fi_recommendation=None):
    """Save analysis outputs to CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path_corr = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr_matrix.to_csv(path_corr)
    print(f"  [OK] Guardado: {path_corr}")

    path_cov = os.path.join(OUTPUT_DIR, "covariance_matrix.csv")
    cov_matrix.to_csv(path_cov)
    print(f"  [OK] Guardado: {path_cov}")

    path_stats = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
    asset_stats.to_csv(path_stats)
    print(f"  [OK] Guardado: {path_stats}")

    weights_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Max_Sharpe_Weight": ms_w,
        "Min_Vol_Weight": mv_w,
    })
    path_w = os.path.join(OUTPUT_DIR, "portfolio_weights.csv")
    weights_df.to_csv(path_w, index=False)
    print(f"  [OK] Guardado: {path_w}")

    if fi_recommendation is not None and not fi_recommendation.empty:
        path_fi = os.path.join(OUTPUT_DIR, "fixed_income_recommendation.csv")
        fi_recommendation.to_csv(path_fi, index=False)
        print(f"  [OK] Guardado: {path_fi}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(prices, returns, bench_prices, bench_metrics, benchmark,
                 sim_df, valid_tickers, corr_matrix,
                 ms_w, ms_ret, ms_vol,
                 mv_w, mv_ret, mv_vol,
                 man_ret, man_vol,
                 asset_stats, cfg, fi_recommendation=None):
    """Generate a multi-panel dashboard with optional FI pie chart."""
    n_panels_bottom = 4 if (fi_recommendation is not None and not fi_recommendation.empty) else 3
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"Dashboard de Analisis  |  {cfg['start_date']} - {cfg['end_date']}  |  "
        f"{cfg['periodicity']}  |  Perfil: {cfg['profile']}  |  "
        f"Horizonte: {cfg['horizon_years']}a ({cfg['horizon_category']})",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, n_panels_bottom, figure=fig, hspace=0.40, wspace=0.35)

    # 1. Normalized price chart
    ax1 = fig.add_subplot(gs[0, :])
    combined = prices.copy()
    bench_aligned = bench_prices.reindex(prices.index, method="ffill")
    combined[benchmark] = bench_aligned
    normalized = combined / combined.iloc[0] * 100
    colors_stocks = plt.cm.tab10(np.linspace(0, 1, len(valid_tickers)))

    for i, t in enumerate(valid_tickers):
        ax1.plot(normalized.index, normalized[t], label=t,
                 color=colors_stocks[i], linewidth=1.4, alpha=0.85)
    if benchmark in normalized.columns:
        ax1.plot(normalized.index, normalized[benchmark],
                 label=f"{benchmark} (benchmark)", color="black",
                 linewidth=2.2, linestyle="--", alpha=0.9)
    ax1.set_title("Precio Normalizado (Base = 100)", fontsize=12)
    ax1.set_ylabel("Precio Indexado")
    ax1.legend(loc="upper left", fontsize=7, ncol=min(6, len(valid_tickers)+1))
    ax1.grid(True, alpha=0.25)
    ax1.axhline(100, color="gray", linewidth=0.7, linestyle=":")

    # 2. Efficient frontier
    ax2 = fig.add_subplot(gs[1, 0])
    sc = ax2.scatter(
        sim_df["volatility"] * 100, sim_df["return"] * 100,
        c=sim_df["sharpe"], cmap="viridis", alpha=0.35, s=6, zorder=1
    )
    plt.colorbar(sc, ax=ax2, label="Sharpe", fraction=0.046, pad=0.04)
    ax2.scatter(ms_vol*100, ms_ret*100, color="red", s=130, zorder=5,
                marker="*", label=f"Max Sharpe ({ms_ret*100:.1f}%)")
    ax2.scatter(mv_vol*100, mv_ret*100, color="blue", s=130, zorder=5,
                marker="D", label=f"Min Vol ({mv_ret*100:.1f}%)")
    if man_ret is not None and man_vol is not None:
        ax2.scatter(man_vol*100, man_ret*100, color="magenta", s=150,
                    zorder=6, marker="P", label="Manual")
    ax2.scatter(bench_metrics["volatility"]*100, bench_metrics["return"]*100,
                color="black", s=130, zorder=5, marker="^", label=benchmark)
    for t in valid_tickers:
        ax2.scatter(asset_stats.loc[t, "Annual Volatility"]*100,
                    asset_stats.loc[t, "Annual Return"]*100,
                    color="orange", s=50, zorder=4, marker="o", alpha=0.8)
        ax2.annotate(t, xy=(asset_stats.loc[t, "Annual Volatility"]*100,
                            asset_stats.loc[t, "Annual Return"]*100),
                     fontsize=6, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")
    ax2.set_title("Frontera Eficiente (Monte Carlo)", fontsize=11)
    ax2.set_xlabel("Volatilidad Anual (%)")
    ax2.set_ylabel("Rendimiento Anual (%)")
    ax2.legend(fontsize=6, loc="upper left")
    ax2.grid(True, alpha=0.25)

    # 3. Max Sharpe pie chart
    ax3 = fig.add_subplot(gs[1, 1])
    labels = [t for t, w in zip(valid_tickers, ms_w) if w > 0.005]
    sizes  = [w for w in ms_w if w > 0.005]
    if sizes:
        explode = [0.04] * len(labels)
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            explode=explode, startangle=90,
            colors=plt.cm.tab10(np.linspace(0, 1, len(labels)))
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax3.set_title("Pesos - Max Sharpe", fontsize=11)
    else:
        ax3.text(0.5, 0.5, "Sin pesos > 0.5%", ha="center", va="center")

    # 4. Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-1, vmax=1, ax=ax4, cbar_kws={"shrink": 0.8},
                    linewidths=0.5, square=True, annot_kws={"fontsize": 7})
    else:
        im = ax4.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_yticks(range(len(corr_matrix.columns)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7)
        ax4.set_yticklabels(corr_matrix.columns, fontsize=7)
        for i_r in range(len(corr_matrix)):
            for j_c in range(len(corr_matrix)):
                ax4.text(j_c, i_r, f"{corr_matrix.values[i_r, j_c]:.2f}",
                         ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_title("Matriz de Correlacion", fontsize=11)

    # 5. Optional FI pie chart
    if n_panels_bottom == 4 and fi_recommendation is not None:
        ax5 = fig.add_subplot(gs[1, 3])
        fi_labels = fi_recommendation["Instrumento"].tolist()
        fi_sizes = fi_recommendation["Peso (%)"].tolist()
        # Shorten labels for the pie chart
        short_labels = [l[:25] + "..." if len(l) > 28 else l for l in fi_labels]
        explode_fi = [0.03] * len(fi_labels)
        ax5.pie(fi_sizes, labels=short_labels, autopct="%1.0f%%",
                explode=explode_fi, startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(fi_labels))))
        ax5.set_title("Renta Fija Sugerida", fontsize=11)

    chart_path = "portfolio_analysis_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Graficas guardadas -> {chart_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 — MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Investor questionnaire ────────────────────────────────────────
    answers = get_investor_questionnaire()
    profile = determine_investor_profile(answers)
    horizon_cat = answers["horizon_category"]

    print(f"\n{'='*65}")
    print(f"  RESULTADO DEL CUESTIONARIO")
    print(f"{'='*65}")
    print(f"  Perfil determinado : {profile}")
    print(f"  Horizonte          : {answers['horizon_years']} anos ({horizon_cat})")
    print(f"  Puntuacion         : {answers['raw_score']}")

    # Allow override
    override = prompt("\n  Deseas cambiar el perfil manualmente? (si/no): ",
                       valid_options=["si", "no"])
    if override.lower() == "si":
        profile = prompt("  Selecciona perfil: ",
                          valid_options=list(PROFILE_SETTINGS.keys()))

    # ── Step 2: Determine FI/RV split ─────────────────────────────────────────
    if answers["desired_pct_fixed"] is not None:
        pct_fixed = answers["desired_pct_fixed"]
    else:
        rec_fi, rec_rv = get_recommended_split(profile, horizon_cat)
        print(f"\n  Asignacion recomendada para perfil {profile} + horizonte {horizon_cat}:")
        print(f"    Renta Fija    : {rec_fi}%")
        print(f"    Renta Variable: {rec_rv}%")
        use_rec = prompt("  Usar esta recomendacion? (si/no): ",
                          valid_options=["si", "no"])
        if use_rec.lower() == "si":
            pct_fixed = rec_fi
        else:
            pct_fixed = prompt_float("  % Renta Fija (0-100): ", min_val=0, max_val=100)

    pct_variable = 100 - pct_fixed

    # ── Step 3: Fixed-income diversification ──────────────────────────────────
    fi_recommendation = None
    wants_fi = prompt("\n  Deseas que el programa recomiende diversificacion de renta fija? (si/no): ",
                       valid_options=["si", "no"])

    # ── Step 4: Market inputs ────────────────────────────────────────────────
    market_cfg = get_market_inputs(profile)

    # Build unified config dict
    cfg = {
        **market_cfg,
        "profile": profile,
        "pct_fixed": pct_fixed,
        "pct_variable": pct_variable,
        "horizon_years": answers["horizon_years"],
        "horizon_category": horizon_cat,
        "objective_label": answers.get("objective_label", "N/A"),
        "experience_label": answers.get("experience_label", "N/A"),
        "volatility_comfort": answers.get("volatility_comfort", "N/A"),
    }

    # Generate FI recommendation now that we have monto_total
    if wants_fi.lower() == "si":
        fi_recommendation = recommend_fixed_income_mix(
            profile, horizon_cat, pct_fixed, cfg["monto_total"]
        )
        print(f"\n{'─'*65}")
        print("  RECOMENDACION DE RENTA FIJA")
        print(f"{'─'*65}")
        monto_fi = cfg['monto_total'] * pct_fixed / 100
        print(f"  Monto asignado a renta fija: ${monto_fi:,.2f}\n")
        for _, row in fi_recommendation.iterrows():
            print(f"    {row['Instrumento']:<45} {row['Peso (%)']:>5.1f}%  ${row['Monto ($)']:>12,.2f}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("  RESUMEN DE CONFIGURACION")
    print(f"{'-'*65}")
    print(f"  Perfil            : {profile}")
    print(f"  Horizonte         : {answers['horizon_years']} anos ({horizon_cat})")
    print(f"  Renta Fija        : {pct_fixed:.0f} %")
    print(f"  Renta Variable    : {pct_variable:.0f} %")
    print(f"  Tickers           : {', '.join(cfg['tickers'])}")
    print(f"  Benchmark         : {cfg['benchmark']}")
    print(f"  Periodicidad      : {cfg['periodicity']}")
    print(f"  Rango fechas      : {cfg['start_date']} a {cfg['end_date']}")
    print(f"  Inversion total   : ${cfg['monto_total']:,.2f}")
    print(f"  Tasa libre riesgo : {cfg['risk_free']*100:.2f} %")
    if cfg.get("target_return") is not None:
        print(f"  Rendimiento obj.  : {cfg['target_return']*100:.2f} %")
    print(f"{'-'*65}")

    confirm = prompt("\n  Proceder con esta configuracion? (si/no): ",
                     valid_options=["si", "no"])
    if confirm.lower() == "no":
        print("  Reiniciando...\n")
        return main()

    # ── Step 5: Download data ─────────────────────────────────────────────────
    prices, bench_prices, valid_tickers, benchmark = download_asset_data(
        cfg["tickers"], cfg["benchmark"],
        cfg["start_date"], cfg["end_date"], cfg["interval"]
    )
    cfg["benchmark"] = benchmark

    # ── Step 6: Calculate returns ─────────────────────────────────────────────
    print("\n  Calculando rendimientos...")
    returns       = calculate_returns(prices)
    bench_returns = bench_prices.pct_change().dropna()

    # ── Step 7: Statistics ────────────────────────────────────────────────────
    asset_stats = calculate_summary_statistics(
        returns, bench_returns, cfg["annualize"], cfg["risk_free"]
    )

    # ── Step 8: Matrices ──────────────────────────────────────────────────────
    print("  Calculando matrices de correlacion y covarianza...")
    corr_matrix = calculate_correlation_matrix(returns)
    cov_matrix  = calculate_covariance_matrix(returns, cfg["annualize"])
    display_matrix(corr_matrix, "MATRIZ DE CORRELACION")
    display_matrix(cov_matrix,  "MATRIZ DE COVARIANZA (anualizada)")

    # ── Step 9: Benchmark metrics ─────────────────────────────────────────────
    print("  Calculando metricas del benchmark...")
    bench_metrics = calculate_benchmark_metrics(
        bench_returns, cfg["annualize"], cfg["risk_free"]
    )

    # ── Step 10: Optimization ─────────────────────────────────────────────────
    print("  Optimizando portafolio (Maximo Sharpe)...")
    ms_w, ms_ret, ms_vol, ms_sr, ms_beta, ms_alpha, ms_treynor = \
        optimize_portfolio(returns, bench_returns, asset_stats,
                           cfg["annualize"], cfg["risk_free"], target="sharpe")

    print("  Optimizando portafolio (Minima Varianza)...")
    mv_w, mv_ret, mv_vol, mv_sr, mv_beta, mv_alpha, mv_treynor = \
        optimize_portfolio(returns, bench_returns, asset_stats,
                           cfg["annualize"], cfg["risk_free"], target="minvol")

    # ── Step 11: Monte Carlo ──────────────────────────────────────────────────
    print(f"  Simulando {N_SIMULATIONS} portafolios aleatorios...")
    sim_df = simulate_portfolios(returns, cfg["annualize"], cfg["risk_free"])

    # ── Step 12: VaR ──────────────────────────────────────────────────────────
    print("  Calculando Value at Risk (VaR)...")
    monto_rv = cfg['monto_total'] * pct_variable / 100
    var_results = calculate_var_multiple(ms_ret, ms_vol, monto_rv)

    # ── Step 13: Valuation signals ────────────────────────────────────────────
    print("  Obteniendo senales de valuacion...")
    valuation_signals = estimate_valuation_signals(valid_tickers)

    # ── Step 14: Portfolio max drawdown ────────────────────────────────────────
    port_max_dd = calculate_portfolio_drawdown(ms_w, returns)

    # ── Step 15: Rebalancing suggestions ──────────────────────────────────────
    rebalancing_suggestions = generate_rebalancing_suggestions(
        valid_tickers, ms_w, valuation_signals, ms_vol, profile, horizon_cat
    )

    # ── Step 16: Target return analysis ───────────────────────────────────────
    target_analysis = None
    if cfg.get("target_return") is not None:
        target_analysis = check_target_return(
            cfg["target_return"], ms_ret, ms_vol, cfg["risk_free"]
        )

    # ── Step 17: Display results ──────────────────────────────────────────────
    display_results(
        cfg, valid_tickers, asset_stats,
        ms_w, ms_ret, ms_vol, ms_sr,
        mv_w, mv_ret, mv_vol, mv_sr,
        bench_metrics, benchmark
    )

    # ── Step 18: Optional manual weights ──────────────────────────────────────
    wants_manual = prompt(
        "  Deseas ingresar pesos manuales para comparar? (si/no): ",
        valid_options=["si", "no"]
    )
    if wants_manual.lower() == "si":
        manual_w = get_manual_weights(valid_tickers)
        man_ret, man_vol, man_sr = display_manual_portfolio(
            cfg, valid_tickers, manual_w, returns, bench_returns, asset_stats
        )
    else:
        man_ret, man_vol = None, None

    # ── Step 19: Professional recommendation report ───────────────────────────
    generate_recommendation_report(
        cfg, valid_tickers, corr_matrix,
        ms_w, ms_ret, ms_vol, ms_sr,
        mv_w, mv_ret, mv_vol, mv_sr,
        bench_metrics, benchmark,
        var_results, valuation_signals,
        rebalancing_suggestions,
        fi_recommendation, target_analysis,
        port_max_dd
    )

    # ── Step 20: Save CSV outputs ─────────────────────────────────────────────
    print("  Guardando archivos CSV...")
    save_outputs(corr_matrix, cov_matrix, asset_stats,
                 valid_tickers, ms_w, mv_w, fi_recommendation)

    # ── Step 21: Generate charts ──────────────────────────────────────────────
    print("  Generando graficas...")
    plot_results(
        prices, returns, bench_prices, bench_metrics, benchmark,
        sim_df, valid_tickers, corr_matrix,
        ms_w, ms_ret, ms_vol,
        mv_w, mv_ret, mv_vol,
        man_ret, man_vol,
        asset_stats, cfg, fi_recommendation
    )

    print("  Listo! [OK]\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()