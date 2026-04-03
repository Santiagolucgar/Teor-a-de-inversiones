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
#   pip install yfinance pandas numpy matplotlib scipy seaborn scikit-learn
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
from scipy.stats import norm, t as t_dist  # [MODIFIED — PHASE 3] added t_dist

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

# [NEW — PHASE 2] Ledoit-Wolf shrinkage estimator for robust covariance
try:
    from sklearn.covariance import LedoitWolf
    HAS_LEDOIT_WOLF = True
except ImportError:
    HAS_LEDOIT_WOLF = False


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

# Monte Carlo future scenario simulation defaults
N_FUTURE_SIMULATIONS = 1000   # Number of future scenarios to simulate
N_PLOT_PATHS = 100            # Number of paths shown on trajectory chart
FIXED_INCOME_GROWTH_MODE = "risk_free_rate"  # How FI portion is projected
FIXED_INCOME_CUSTOM_RATE = None              # Optional custom FI annual rate

# [NEW — PHASE 3] Advanced modeling defaults
N_BOOTSTRAP_SAMPLES = 200       # Resampling iterations for weight sensitivity
DEFAULT_T_DF = 5               # Degrees of freedom for Student-t MC (lower = fatter tails)
REBALANCE_FREQ_MAP = {          # Rebalancing frequency options
    "none": None,
    "annual": 252,               # Trading days per year
    "quarterly": 63,            # ~252/4
}

# [NEW — PHASE 3] Macro scenarios for scenario-based optimization
# Each scenario adjusts expected returns and volatility relative to base case.
# These are MULTIPLICATIVE factors applied to historical estimates.
MACRO_SCENARIOS = {
    "Caso Base": {
        "return_adj": 1.0,      # no change
        "vol_adj": 1.0,         # no change
        "description": "Parametros historicos sin ajuste.",
    },
    "Recesion": {
        "return_adj": 0.4,      # returns drop to 40% of historical
        "vol_adj": 1.5,         # volatility increases 50%
        "description": "Contraccion economica: rendimientos bajos, volatilidad alta.",
    },
    "Tasas Altas": {
        "return_adj": 0.7,      # moderate return compression
        "vol_adj": 1.3,         # moderate vol increase
        "description": "Politica monetaria restrictiva: valuaciones presionadas.",
    },
    "Estanflacion": {
        "return_adj": 0.3,      # severe return compression
        "vol_adj": 1.6,         # high volatility
        "description": "Inflacion alta + crecimiento bajo: el peor escenario.",
    },
}

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

# ── [NEW — PHASE 1] Position-size limits by profile ─────────────────────────
# Maximum weight any single asset can have in the optimized portfolio.
# Prevents excessive concentration based on investor profile.
MAX_WEIGHT_BY_PROFILE = {
    "Conservador":              0.20,
    "Moderadamente Conservador": 0.25,
    "Moderado":                 0.30,
    "Moderadamente Agresivo":   0.35,
    "Agresivo":                 0.45,
}

# ── [NEW — PHASE 1] 4-Pillar profiling level system ─────────────────────────
# Maps pillar names to (score_ranges → level_name).
# Used by score_pillar_to_level() and synthesize_final_profile().
PILLAR_LEVELS = [
    "Conservador",
    "Moderadamente Conservador",
    "Moderado",
    "Moderadamente Agresivo",
    "Agresivo",
]

# Pillar score boundaries: (max_score_for_level, level_name)
# Each pillar has its own scale; these are the DEFAULT boundaries.
# tolerance:  0-31,  capacity: 0-47,  liquidity: 0-24
PILLAR_BOUNDARIES = {
    "tolerance":  [(7, 0), (15, 1), (22, 2), (27, 3), (31, 4)],
    "capacity":   [(11, 0), (22, 1), (33, 2), (41, 3), (47, 4)],
    "liquidity":  [(7, 0), (14, 1), (24, 2)],  # 3 levels → mapped to 5
    "objective":  [(1, 0), (2, 1), (3, 2), (4, 4)],  # direct question mapping
}

# ── [NEW — PHASE 2] Stress testing scenarios ────────────────────────────────
# Each scenario models a macro shock to equity and fixed-income portions.
# These are stylized approximations for portfolio-level impact estimation,
# NOT precise replicas of historical events.
STRESS_SCENARIOS = {
    "Crisis Financiera (2008)": {
        "equity_shock": -0.50,       # S&P500 peak-to-trough approx
        "bond_change": +0.05,        # Flight to quality
        "correlation_spike": 0.85,   # Correlations converge in crises
        "duration_months": 17,
        "description": "Colapso crediticio global, caida severa de activos de riesgo.",
    },
    "Crash COVID (2020)": {
        "equity_shock": -0.34,
        "bond_change": +0.03,
        "correlation_spike": 0.80,
        "duration_months": 2,
        "description": "Caida rapida y profunda seguida de recuperacion acelerada.",
    },
    "Shock de Tasas (2022)": {
        "equity_shock": -0.25,
        "bond_change": -0.13,        # Bonds ALSO fell — unusual
        "correlation_spike": 0.60,
        "duration_months": 10,
        "description": "Alzas agresivas de tasas — renta fija y variable cayeron simultaneamente.",
    },
    "Estanflacion (1970s)": {
        "equity_shock": -0.40,
        "bond_change": -0.08,
        "correlation_spike": 0.50,
        "duration_months": 24,
        "description": "Crecimiento bajo + inflacion alta. Proteccion limitada en bonos.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2B — MODULO EDUCATIVO (conceptos, ejemplos, tabla de empresas)
# ─────────────────────────────────────────────────────────────────────────────
# Este modulo agrega funciones educativas opcionales al flujo del programa.
# No modifica la logica de optimizacion; solo presenta informacion
# que ayuda al usuario a entender lo que esta seleccionando.
# ─────────────────────────────────────────────────────────────────────────────

# Tabla de empresas de ejemplo agrupadas por sector.
# Esta lista es SOLO de referencia educativa — el usuario puede ingresar
# cualquier ticker valido de Yahoo Finance, no esta limitado a estos.
EXAMPLE_COMPANIES = [
    # Tecnologia
    {"sector": "Tecnologia",                "company": "Apple",                 "ticker": "AAPL"},
    {"sector": "Tecnologia",                "company": "Microsoft",             "ticker": "MSFT"},
    {"sector": "Tecnologia",                "company": "Alphabet (Google)",     "ticker": "GOOGL"},
    {"sector": "Tecnologia",                "company": "NVIDIA",                "ticker": "NVDA"},
    {"sector": "Tecnologia",                "company": "Meta Platforms",        "ticker": "META"},
    # Financiero
    {"sector": "Financiero",                "company": "JPMorgan Chase",        "ticker": "JPM"},
    {"sector": "Financiero",                "company": "Bank of America",       "ticker": "BAC"},
    {"sector": "Financiero",                "company": "Goldman Sachs",         "ticker": "GS"},
    {"sector": "Financiero",                "company": "Visa",                  "ticker": "V"},
    # Consumo Basico
    {"sector": "Consumo Basico",            "company": "Coca-Cola",             "ticker": "KO"},
    {"sector": "Consumo Basico",            "company": "Procter & Gamble",      "ticker": "PG"},
    {"sector": "Consumo Basico",            "company": "Walmart",               "ticker": "WMT"},
    {"sector": "Consumo Basico",            "company": "Costco",                "ticker": "COST"},
    # Consumo Discrecional
    {"sector": "Consumo Discrecional",      "company": "Amazon",                "ticker": "AMZN"},
    {"sector": "Consumo Discrecional",      "company": "Tesla",                 "ticker": "TSLA"},
    {"sector": "Consumo Discrecional",      "company": "Nike",                  "ticker": "NKE"},
    {"sector": "Consumo Discrecional",      "company": "McDonald's",            "ticker": "MCD"},
    # Salud
    {"sector": "Salud",                     "company": "Johnson & Johnson",     "ticker": "JNJ"},
    {"sector": "Salud",                     "company": "UnitedHealth",          "ticker": "UNH"},
    {"sector": "Salud",                     "company": "Pfizer",                "ticker": "PFE"},
    {"sector": "Salud",                     "company": "Abbott Labs",           "ticker": "ABT"},
    # Energia
    {"sector": "Energia",                   "company": "Exxon Mobil",           "ticker": "XOM"},
    {"sector": "Energia",                   "company": "Chevron",               "ticker": "CVX"},
    {"sector": "Energia",                   "company": "ConocoPhillips",        "ticker": "COP"},
    # Industriales
    {"sector": "Industriales",              "company": "Caterpillar",           "ticker": "CAT"},
    {"sector": "Industriales",              "company": "Honeywell",             "ticker": "HON"},
    {"sector": "Industriales",              "company": "Union Pacific",         "ticker": "UNP"},
    {"sector": "Industriales",              "company": "3M",                    "ticker": "MMM"},
    # Comunicaciones
    {"sector": "Comunicaciones",            "company": "Walt Disney",           "ticker": "DIS"},
    {"sector": "Comunicaciones",            "company": "Netflix",               "ticker": "NFLX"},
    {"sector": "Comunicaciones",            "company": "Comcast",               "ticker": "CMCSA"},
    {"sector": "Comunicaciones",            "company": "T-Mobile US",           "ticker": "TMUS"},
    # Materiales
    {"sector": "Materiales",                "company": "Linde",                 "ticker": "LIN"},
    {"sector": "Materiales",                "company": "Freeport-McMoRan",      "ticker": "FCX"},
    {"sector": "Materiales",                "company": "Newmont Mining",        "ticker": "NEM"},
    # Servicios Publicos
    {"sector": "Servicios Publicos",        "company": "NextEra Energy",        "ticker": "NEE"},
    {"sector": "Servicios Publicos",        "company": "Duke Energy",           "ticker": "DUK"},
    {"sector": "Servicios Publicos",        "company": "Southern Company",      "ticker": "SO"},
    # ETFs / Benchmarks
    {"sector": "ETFs / Benchmarks",         "company": "SPDR S&P 500 ETF",     "ticker": "SPY"},
    {"sector": "ETFs / Benchmarks",         "company": "Invesco QQQ (Nasdaq)",  "ticker": "QQQ"},
    {"sector": "ETFs / Benchmarks",         "company": "iShares MSCI ACWI",    "ticker": "ACWI"},
    {"sector": "ETFs / Benchmarks",         "company": "Vanguard Total Stock",  "ticker": "VTI"},
    {"sector": "ETFs / Benchmarks",         "company": "iShares Russell 2000",  "ticker": "IWM"},
    {"sector": "ETFs / Benchmarks",         "company": "iShares Core US Agg Bond", "ticker": "AGG"},
]


# ── Funciones educativas ─────────────────────────────────────────────────────

def explain_financial_concepts():
    """
    Muestra una explicacion clara y breve de los conceptos financieros clave,
    pensada para estudiantes o usuarios con conocimiento basico/intermedio.
    """
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  CONCEPTOS FINANCIEROS CLAVE")
    print(SEP)

    print("""
  1. TASA FIJA (Renta Fija)
  ─────────────────────────
  Es un instrumento donde la tasa de rendimiento se conoce o se pacta
  desde el inicio. El inversionista sabe cuanto va a ganar.

  ¿Por que importa?
    Te da previsibilidad — sabes exactamente cuanto recibiras.

  Ejemplo practico:
    Si compras un bono gubernamental que paga 8% anual fijo, ese 8%
    no cambia sin importar lo que pase en el mercado.

  2. TASA VARIABLE (Renta Variable)
  ──────────────────────────────────
  Es un instrumento cuyo rendimiento cambia dependiendo de las condiciones
  del mercado o de una tasa de referencia (como TIIE, SOFR o inflacion).

  ¿Por que importa?
    Puede dar rendimientos mayores, pero tambien puede bajar.

  Ejemplo practico:
    Si inviertes en acciones, tu rendimiento depende de como se muevan
    los precios. Si la accion sube 15%, ganas; si baja 10%, pierdes.

  3. PORTAFOLIO DE MAXIMO SHARPE
  ──────────────────────────────
  Es la combinacion de activos que ofrece la mejor relacion entre
  rendimiento y riesgo. Busca maximizar cuanto ganas por cada unidad
  de riesgo que aceptas.

  ¿Por que importa?
    Es la opcion mas "eficiente" — no necesariamente la de mayor
    rendimiento, pero si la que mejor compensa el riesgo tomado.

  Ejemplo practico:
    Si dos portafolios rinden 12%, pero uno tiene volatilidad de 10%
    y otro de 20%, el de menor volatilidad tiene mejor Sharpe.

  4. PORTAFOLIO DE MINIMA VARIANZA
  ────────────────────────────────
  Es la combinacion de activos que busca la menor volatilidad posible.
  No necesariamente da el mayor rendimiento, pero reduce al maximo
  las fluctuaciones del portafolio.

  ¿Por que importa?
    Es ideal para inversionistas conservadores que prefieren
    estabilidad sobre rendimiento alto.

  Ejemplo practico:
    Un portafolio con 60% en bonos y 40% en acciones de baja
    volatilidad podria tener mucha menos variacion que uno con
    100% en acciones tecnologicas.

  5. BENCHMARK (Referencia de mercado)
  ────────────────────────────────────
  Un benchmark es un indice o ETF que se usa como punto de comparacion
  para saber si tu portafolio lo esta haciendo bien o mal respecto
  al mercado general.

  ¿Por que importa?
    Sin un punto de referencia, no sabes si tu rendimiento es bueno.
    Superar al benchmark significa que estas generando valor adicional.

  Ejemplo practico:
    Si tu portafolio rinde 12% anual y el S&P 500 rindio 10%,
    tu portafolio supero al benchmark por 2 puntos porcentuales.
""")
    print(f"{SEP}\n")


def explain_fixed_vs_variable_rate():
    """
    Explicacion corta de renta fija vs variable, usada inline
    antes de pedir la distribucion FI/RV al usuario.
    """
    print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║  ¿Que es Renta Fija y Renta Variable?                       ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  Renta Fija: instrumentos con rendimiento predecible         ║
  ║  (bonos, CETES, depositos). Menor riesgo, menor retorno.    ║
  ║                                                               ║
  ║  Renta Variable: instrumentos cuyo rendimiento fluctua        ║
  ║  (acciones, ETFs). Mayor potencial de ganancia, pero          ║
  ║  tambien mayor riesgo.                                        ║
  ║                                                               ║
  ║  Tu distribucion entre ambas depende de tu perfil de riesgo   ║
  ║  y tu horizonte de inversion.                                 ║
  ╚═══════════════════════════════════════════════════════════════╝
""")


def explain_max_sharpe():
    """Explicacion inline del portafolio de maximo Sharpe."""
    print("""
  📊 ¿Que es el portafolio de Maximo Sharpe?
     Busca la MEJOR relacion entre rendimiento y riesgo.
     No es necesariamente el que mas gana, sino el que mejor
     compensa cada unidad de riesgo asumida.
""")


def explain_min_variance():
    """Explicacion inline del portafolio de minima varianza."""
    print("""
  📊 ¿Que es el portafolio de Minima Varianza?
     Busca REDUCIR AL MAXIMO la volatilidad (las fluctuaciones).
     Es ideal para inversionistas conservadores que prefieren
     estabilidad sobre rendimiento alto.
""")


def explain_benchmark():
    """Explicacion inline del benchmark, usada antes de pedir el ticker."""
    print("""
  📊 ¿Que es un Benchmark?
     Es un indice de referencia (como el S&P 500) que se usa para
     comparar el desempeno de tu portafolio contra el mercado.
     Si tu portafolio supera al benchmark, estas generando valor.
""")


def show_example_tickers_by_sector():
    """
    Muestra una tabla de empresas de ejemplo organizadas por sector.
    Sirve como guia para que el usuario sepa que tickers puede elegir.
    NOTA: El usuario puede ingresar CUALQUIER ticker de Yahoo Finance,
    no esta limitado a esta lista.
    """
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  EMPRESAS DE EJEMPLO POR SECTOR")
    print(f"  (Puedes usar CUALQUIER ticker de Yahoo Finance, no solo estos)")
    print(SEP)

    # Agrupar por sector manteniendo el orden
    current_sector = None
    print(f"\n  {'Sector':<24} {'Empresa':<28} {'Ticker':<8}")
    print(f"  {'-'*24} {'-'*28} {'-'*8}")

    for entry in EXAMPLE_COMPANIES:
        sector = entry["sector"]
        company = entry["company"]
        ticker = entry["ticker"]
        # Mostrar el nombre del sector solo en la primera fila de cada grupo
        if sector != current_sector:
            if current_sector is not None:
                print()  # linea en blanco entre sectores
            sector_display = sector
            current_sector = sector
        else:
            sector_display = ""
        print(f"  {sector_display:<24} {company:<28} {ticker:<8}")

    print(f"\n{SEP}")
    print("  RECUERDA: Esta lista es solo de referencia. Puedes buscar")
    print("  cualquier accion o ETF en finance.yahoo.com para obtener")
    print("  su ticker y usarlo en este programa.")
    print(f"{SEP}\n")





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
# SECTION 4 — INVESTOR QUESTIONNAIRE & PROFILE (PHASE 1 REDESIGN)
# ─────────────────────────────────────────────────────────────────────────────
# [MODIFIED — PHASE 1]
# Redesigned into 4 independent pillars:
#   Pillar 1: Risk Tolerance (psychological — how you FEEL about risk)
#   Pillar 2: Risk Capacity (financial — can you AFFORD to lose money?)
#   Pillar 3: Liquidity Profile (when do you NEED the money?)
#   Pillar 4: Investment Objective (what is your GOAL?)
#
# Each pillar produces its own score and level.
# The final profile uses binding constraint logic:
#   final = min(tolerance, capacity, liquidity)
# Contradictions are detected and explained to the user.
# ─────────────────────────────────────────────────────────────────────────────


def _score_to_level_index(score, boundaries):
    """
    [NEW — PHASE 1] Convert a raw pillar score to a level index (0-4).
    boundaries is a list of (max_score, level_index) tuples, ordered ascending.
    """
    for max_score, level_idx in boundaries:
        if score <= max_score:
            return level_idx
    # If score exceeds all boundaries, return the last level
    return boundaries[-1][1]


def _level_index_to_name(idx):
    """[NEW — PHASE 1] Convert level index (0-4) to profile name."""
    idx = max(0, min(idx, len(PILLAR_LEVELS) - 1))
    return PILLAR_LEVELS[idx]


def get_investor_questionnaire():
    """
    [MODIFIED — PHASE 1] 4-pillar investor profiling questionnaire.
    Asks 16 structured questions across 4 pillars + allocation preference.
    Returns a dict with raw answers, per-pillar scores, and pillar levels.

    Backward-compatible: still produces 'raw_score', 'objective_label',
    'experience_label', 'horizon_years', 'horizon_category', etc.
    """
    print("\n" + "=" * 65)
    print("  CUESTIONARIO DEL INVERSIONISTA")
    print("=" * 65)
    print("  Responde las siguientes preguntas para determinar tu perfil.")
    print("  El cuestionario evalua 4 dimensiones independientes:\n")
    print("    1. Objetivo de inversion y horizonte")
    print("    2. Capacidad financiera (situacion economica)")
    print("    3. Experiencia y tolerancia al riesgo (psicologica)")
    print("    4. Necesidades de liquidez\n")

    answers = {}
    legacy_score = 0  # Backward-compatible aggregate score

    # ══════════════════════════════════════════════════════════════════════
    # PILAR 1: OBJETIVO DE INVERSION
    # ══════════════════════════════════════════════════════════════════════
    print(f"  {'═'*60}")
    print("  PILAR 1 de 4 — OBJETIVO DE INVERSION")
    print(f"  {'═'*60}")

    # Q1 — Investment objective
    print("\n  [1/16] Cual es tu objetivo de inversion?")
    print("    1. Preservacion de capital")
    print("    2. Generacion de ingreso / renta")
    print("    3. Crecimiento moderado")
    print("    4. Crecimiento agresivo / maximizar rendimiento")
    q_obj = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["objective"] = q_obj
    legacy_score += (q_obj - 1) * 3

    obj_labels = {1: "Preservacion de capital", 2: "Generacion de ingreso",
                  3: "Crecimiento moderado", 4: "Crecimiento agresivo"}
    answers["objective_label"] = obj_labels[q_obj]

    # Q2 — Time horizon
    print("\n  [2/16] Horizonte de inversion (en anos)")
    print("    Cuantos anos planeas mantener esta inversion?")
    q_horizon = prompt_int("    Anos: ", 1, 60)
    answers["horizon_years"] = q_horizon
    answers["horizon_category"] = determine_horizon_category(q_horizon)
    if q_horizon <= 3:
        legacy_score += 0
    elif q_horizon <= 10:
        legacy_score += 4
    else:
        legacy_score += 8

    # Pillar 1 score: objective level index (direct mapping)
    obj_level_idx = _score_to_level_index(q_obj, PILLAR_BOUNDARIES["objective"])
    answers["pillar_objective_level"] = obj_level_idx
    answers["pillar_objective_name"] = _level_index_to_name(obj_level_idx)

    # ══════════════════════════════════════════════════════════════════════
    # PILAR 2: CAPACIDAD FINANCIERA (NEW QUESTIONS)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'═'*60}")
    print("  PILAR 2 de 4 — CAPACIDAD FINANCIERA")
    print(f"  {'═'*60}")
    print("  Estas preguntas evaluan tu situacion economica actual.\n")

    capacity_score = 0

    # C1 — Income stability
    print("  [3/16] Que tan estable es tu ingreso principal?")
    print("    1. Muy inestable (freelance, comisiones variables)")
    print("    2. Algo estable (contrato temporal, negocio propio)")
    print("    3. Estable (salario fijo)")
    print("    4. Muy estable (multiples fuentes de ingreso)")
    q_c1 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["income_stability"] = q_c1
    capacity_score += [0, 3, 6, 9][q_c1 - 1]

    # C2 — Debt level
    print("\n  [4/16] Cual es tu nivel de deuda respecto a tu ingreso?")
    print("    1. Alto (deuda > 50% del ingreso)")
    print("    2. Moderado (deuda 25-50% del ingreso)")
    print("    3. Bajo (deuda 10-25% del ingreso)")
    print("    4. Minimo o sin deuda (< 10%)")
    q_c2 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["debt_level"] = q_c2
    capacity_score += [0, 2, 5, 8][q_c2 - 1]

    # C3 — Dependents
    print("\n  [5/16] Cuantas personas dependen de ti economicamente?")
    print("    1. Tres o mas")
    print("    2. Una o dos")
    print("    3. Ninguna")
    q_c3 = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["dependents"] = q_c3
    capacity_score += [0, 3, 7][q_c3 - 1]

    # C4 — Emergency fund
    print("\n  [6/16] Tienes un fondo de emergencia (3-6 meses de gastos)?")
    print("    1. No tengo fondo de emergencia")
    print("    2. Parcialmente (cubro 1-2 meses)")
    print("    3. Si (3-6 meses)")
    print("    4. Amplio (mas de 12 meses)")
    q_c4 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["emergency_fund"] = q_c4
    capacity_score += [0, 2, 5, 8][q_c4 - 1]

    # C5 — Percentage of net worth
    print("\n  [7/16] Que porcentaje de tu patrimonio total representa esta inversion?")
    print("    1. Mas del 50%")
    print("    2. Entre 25% y 50%")
    print("    3. Entre 10% y 25%")
    print("    4. Menos del 10%")
    q_c5 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["pct_net_worth"] = q_c5
    capacity_score += [0, 2, 5, 8][q_c5 - 1]

    # C6 — Other investments
    print("\n  [8/16] Tienes otras inversiones o cuentas de retiro?")
    print("    1. No, esta es mi unica inversion")
    print("    2. Algo pequeno")
    print("    3. Si, tengo inversiones significativas adicionales")
    print("    4. Si, estoy ampliamente diversificado")
    q_c6 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["other_investments"] = q_c6
    capacity_score += [0, 2, 5, 7][q_c6 - 1]

    answers["pillar_capacity_score"] = capacity_score
    cap_level_idx = _score_to_level_index(capacity_score, PILLAR_BOUNDARIES["capacity"])
    answers["pillar_capacity_level"] = cap_level_idx
    answers["pillar_capacity_name"] = _level_index_to_name(cap_level_idx)

    # ══════════════════════════════════════════════════════════════════════
    # PILAR 3: TOLERANCIA AL RIESGO (Psicologica)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'═'*60}")
    print("  PILAR 3 de 4 — TOLERANCIA AL RIESGO (PSICOLOGICA)")
    print(f"  {'═'*60}")
    print("  Estas preguntas evaluan como reaccionas ante las perdidas.\n")

    tolerance_score = 0

    # Q — Experience level
    print("  [9/16] Experiencia previa en inversiones")
    print("    1. Ninguna")
    print("    2. Basica (cuentas de ahorro, depositos)")
    print("    3. Intermedia (fondos, bonos, algunas acciones)")
    print("    4. Avanzada (acciones, derivados, trading activo)")
    q_exp = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["experience"] = q_exp
    legacy_score += (q_exp - 1) * 2
    tolerance_score += (q_exp - 1) * 2  # 0, 2, 4, 6

    exp_labels = {1: "Ninguna", 2: "Basica", 3: "Intermedia", 4: "Avanzada"}
    answers["experience_label"] = exp_labels[q_exp]

    # Q — Instruments used before
    print("\n  [10/16] Instrumentos que has utilizado antes (puedes elegir varios)")
    print("    1. Solo ahorro bancario")
    print("    2. Bonos / deuda gubernamental")
    print("    3. Fondos de inversion")
    print("    4. Acciones individuales")
    print("    5. Derivados / futuros / opciones")
    q_instr_raw = input("    Tu respuesta (ej. 1,3,4): ").strip()
    q_instr_items = [x.strip() for x in q_instr_raw.split(",") if x.strip().isdigit()]
    q_instr_items = [int(x) for x in q_instr_items if 1 <= int(x) <= 5]
    if not q_instr_items:
        q_instr_items = [1]
    answers["instruments"] = q_instr_items
    legacy_score += min(max(q_instr_items) - 1, 4)

    # Q — Reaction to 10% drop
    print("\n  [11/16] Si tu portafolio pierde 10% en un mes, tu reaccion seria:")
    print("    1. Vender todo inmediatamente")
    print("    2. Vender una parte para reducir riesgo")
    print("    3. Mantener y esperar la recuperacion")
    print("    4. Comprar mas aprovechando precios bajos")
    q_r10 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["reaction_10pct"] = q_r10
    legacy_score += (q_r10 - 1) * 3
    tolerance_score += (q_r10 - 1) * 3  # 0, 3, 6, 9

    # Q — Reaction to 20% drop
    print("\n  [12/16] Si tu portafolio pierde 20% en tres meses, tu reaccion seria:")
    print("    1. Vender todo inmediatamente")
    print("    2. Vender una parte para reducir riesgo")
    print("    3. Mantener y esperar")
    print("    4. Comprar mas")
    q_r20 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["reaction_20pct"] = q_r20
    legacy_score += (q_r20 - 1) * 3
    tolerance_score += (q_r20 - 1) * 3  # 0, 3, 6, 9

    # Q — Comfort with volatility
    print("\n  [13/16] Que tan comodo te sientes con la volatilidad?")
    print("    1 = Muy incomodo (prefiero estabilidad total)")
    print("    5 = Muy comodo (acepto fluctuaciones fuertes)")
    q_vol = prompt_int("    Tu respuesta (1-5): ", 1, 5)
    answers["volatility_comfort"] = q_vol
    legacy_score += (q_vol - 1) * 2
    tolerance_score += (q_vol - 1) * 2  # 0, 2, 4, 6, 8

    answers["pillar_tolerance_score"] = tolerance_score
    tol_level_idx = _score_to_level_index(tolerance_score, PILLAR_BOUNDARIES["tolerance"])
    answers["pillar_tolerance_level"] = tol_level_idx
    answers["pillar_tolerance_name"] = _level_index_to_name(tol_level_idx)

    # ══════════════════════════════════════════════════════════════════════
    # PILAR 4: NECESIDADES DE LIQUIDEZ
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'═'*60}")
    print("  PILAR 4 de 4 — NECESIDADES DE LIQUIDEZ")
    print(f"  {'═'*60}")

    liquidity_score = 0

    # Q — When withdraw
    print("\n  [14/16] Cuando planeas comenzar a retirar fondos?")
    print("    1. Ya estoy retirando / muy pronto")
    print("    2. Dentro de 1-5 anos")
    print("    3. En mas de 5 anos")
    q_withdraw = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["withdrawal_timing"] = q_withdraw
    legacy_score += (q_withdraw - 1) * 2
    liquidity_score += (q_withdraw - 1) * 4  # 0, 4, 8

    # Q — Immediate liquidity needs
    print("\n  [15/16] Necesidades de liquidez")
    print("    1. Puedo necesitar el dinero en cualquier momento")
    print("    2. Puedo necesitarlo dentro de 1 ano")
    print("    3. No necesito el dinero en los proximos 3+ anos")
    q_liq = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["liquidity"] = q_liq
    legacy_score += (q_liq - 1) * 3
    liquidity_score += (q_liq - 1) * 4  # 0, 4, 8

    # Q — Other liquid reserves
    print("\n  [16/16] Tienes otras reservas liquidas para emergencias?")
    print("    1. No")
    print("    2. Limitadas")
    print("    3. Si, suficientes")
    q_reserves = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["liquid_reserves"] = q_reserves
    liquidity_score += (q_reserves - 1) * 4  # 0, 4, 8

    answers["pillar_liquidity_score"] = liquidity_score
    liq_level_idx = _score_to_level_index(liquidity_score, PILLAR_BOUNDARIES["liquidity"])
    # Map 3-level liquidity to 5-level scale
    liq_to_5 = {0: 0, 1: 2, 2: 4}  # LOW→Conservador, MED→Moderado, HIGH→Agresivo
    liq_5_idx = liq_to_5.get(liq_level_idx, 2)
    answers["pillar_liquidity_level"] = liq_5_idx
    answers["pillar_liquidity_name"] = _level_index_to_name(liq_5_idx)

    # ── Allocation preference ────────────────────────────────────────────
    print(f"\n  {'═'*60}")
    print("  PREFERENCIA DE ASIGNACION")
    print(f"  {'═'*60}")

    explain_fixed_vs_variable_rate()

    print("\n  Distribucion deseada entre renta fija y renta variable")
    print("    Indica el % que deseas en renta fija (el resto sera renta variable).")
    print("    Si no tienes preferencia, presiona Enter para usar la recomendacion.\n")
    q_fi_pref = prompt_float("    % Renta Fija (0-100, o Enter para auto): ",
                             min_val=0, max_val=100, allow_blank=True, default=None)
    answers["desired_pct_fixed"] = q_fi_pref

    # Backward-compatible aggregate score
    answers["raw_score"] = legacy_score
    return answers


def determine_horizon_category(years):
    """Classify investment horizon into short / medium / long term."""
    for category, (lo, hi) in HORIZON_RULES.items():
        if lo <= years <= hi:
            return category
    return "Largo plazo"


def determine_investor_profile(answers):
    """
    [MODIFIED — PHASE 1] Compute investor profile using 4-pillar synthesis.
    Falls back to legacy single-score method if pillar data is missing
    (backward compatibility).
    """
    # If pillar data is available, use the new system
    if "pillar_tolerance_level" in answers:
        return synthesize_final_profile(answers)["final_profile"]
    # Legacy fallback
    score = answers["raw_score"]
    if score <= 10:
        return "Conservador"
    elif score <= 18:
        return "Moderadamente Conservador"
    elif score <= 30:
        return "Moderado"
    elif score <= 42:
        return "Moderadamente Agresivo"
    else:
        return "Agresivo"


def detect_contradictions(answers):
    """
    [NEW — PHASE 1] Cross-validate the 4 pillar scores.
    Returns a list of dicts: {severity, message, recommendation}
    severity: 'CRITICO', 'ADVERTENCIA', 'NOTA'
    """
    contradictions = []

    tol_level = answers.get("pillar_tolerance_level", 2)
    cap_level = answers.get("pillar_capacity_level", 2)
    liq_level = answers.get("pillar_liquidity_level", 2)
    obj_level = answers.get("pillar_objective_level", 2)
    horizon = answers.get("horizon_years", 5)
    pct_nw = answers.get("pct_net_worth", 3)

    tol_name = _level_index_to_name(tol_level)
    cap_name = _level_index_to_name(cap_level)

    # ── Rule 1: High tolerance + Low capacity ────────────────────────────
    if tol_level >= 3 and cap_level <= 1:
        contradictions.append({
            "severity": "CRITICO",
            "message": (
                f"Tu tolerancia psicologica al riesgo es alta ({tol_name}), "
                f"pero tu capacidad financiera es baja ({cap_name}). "
                "Te sientes comodo con el riesgo, pero tu situacion economica "
                "indica que no puedes permitirte perdidas grandes."
            ),
            "recommendation": (
                "El perfil se ajustara a tu capacidad financiera (la restriccion "
                "vinculante). Esto protege tu patrimonio en caso de perdidas."
            ),
        })

    # ── Rule 2: High tolerance + High liquidity need ─────────────────────
    if tol_level >= 3 and liq_level <= 1:
        contradictions.append({
            "severity": "CRITICO",
            "message": (
                "Expresas alta tolerancia al riesgo, pero necesitas acceso "
                "al dinero a corto plazo. Un portafolio agresivo podria estar "
                "en perdida justo cuando necesites retirar."
            ),
            "recommendation": (
                "Tu necesidad de liquidez limita el nivel de riesgo aceptable. "
                "Se priorizara proteccion del capital a corto plazo."
            ),
        })

    # ── Rule 3: Preservation goal + High tolerance answers ───────────────
    if obj_level <= 1 and tol_level >= 3:
        contradictions.append({
            "severity": "ADVERTENCIA",
            "message": (
                "Tu objetivo es preservacion de capital, pero tus respuestas "
                "de tolerancia al riesgo indican comodidad con la volatilidad. "
                "Esto podria ser inconsistente."
            ),
            "recommendation": (
                "Si tu prioridad es realmente preservar el capital, el perfil "
                "se alineara con ese objetivo. Si deseas crecimiento, considera "
                "cambiar tu objetivo."
            ),
        })

    # ── Rule 4: Short horizon + Aggressive goal ──────────────────────────
    if horizon <= 3 and obj_level >= 3:
        contradictions.append({
            "severity": "ADVERTENCIA",
            "message": (
                f"Tu horizonte es de {horizon} anos (corto plazo), pero tu "
                "objetivo es crecimiento agresivo. Es estadisticamente dificil "
                "lograr rendimientos agresivos en periodos cortos sin asumir "
                "riesgo extremo."
            ),
            "recommendation": (
                "Considera extender tu horizonte o ajustar tus expectativas "
                "de rendimiento."
            ),
        })

    # ── Rule 5: Large % of net worth + aggressive tolerance ──────────────
    if pct_nw <= 1 and tol_level >= 3 and cap_level <= 2:
        contradictions.append({
            "severity": "CRITICO",
            "message": (
                "Esta inversion representa mas del 50% de tu patrimonio total, "
                "con capacidad financiera limitada. Esto es un riesgo de "
                "concentracion a nivel de patrimonio personal."
            ),
            "recommendation": (
                "Se limitara el perfil a Moderado como maximo. "
                "Con una proporcion tan alta de tu patrimonio en una sola "
                "inversion, la proteccion del capital es prioritaria."
            ),
        })

    # ── Rule 6: Low experience + High tolerance ──────────────────────────
    if answers.get("experience", 2) <= 1 and tol_level >= 3:
        contradictions.append({
            "severity": "NOTA",
            "message": (
                "No tienes experiencia previa en inversiones pero expresas "
                "alta tolerancia al riesgo. Esto puede deberse a que no has "
                "experimentado perdidas reales. La investigacion muestra que "
                "la tolerancia auto-reportada suele sobreestimar la real."
            ),
            "recommendation": (
                "Se aplicara un factor de cautela al perfil. Considera empezar "
                "con una asignacion mas conservadora e ir ajustando con el tiempo."
            ),
        })

    return contradictions


def synthesize_final_profile(answers):
    """
    [NEW — PHASE 1] Determine the final investor profile using the
    binding constraint principle: final = min(tolerance, capacity, liquidity).
    The most restrictive pillar determines the ceiling.

    Returns a dict with:
      - final_profile: str (one of PILLAR_LEVELS)
      - final_level: int (0-4)
      - pillar_levels: dict {pillar_name: level_index}
      - binding_pillar: str (which pillar constrained the result)
      - contradictions: list from detect_contradictions()
      - explanation: str (why this profile was chosen)
    """
    tol_level = answers.get("pillar_tolerance_level", 2)
    cap_level = answers.get("pillar_capacity_level", 2)
    liq_level = answers.get("pillar_liquidity_level", 2)
    obj_level = answers.get("pillar_objective_level", 2)

    pillar_levels = {
        "Tolerancia psicologica": tol_level,
        "Capacidad financiera": cap_level,
        "Liquidez": liq_level,
        "Objetivo": obj_level,
    }

    # Binding constraint: minimum of the three core pillars
    # (Objective informs, but tolerance/capacity/liquidity constrain)
    constrained_pillars = {
        "Tolerancia psicologica": tol_level,
        "Capacidad financiera": cap_level,
        "Liquidez": liq_level,
    }
    binding_pillar = min(constrained_pillars, key=constrained_pillars.get)
    raw_final_level = min(constrained_pillars.values())

    # Experience discount: if zero experience + high tolerance, cap at Moderado
    experience = answers.get("experience", 2)
    experience_capped = False
    if experience <= 1 and raw_final_level >= 3:
        raw_final_level = 2  # cap at Moderado
        binding_pillar = "Cautela por inexperiencia"
        experience_capped = True

    # % net worth hard cap
    pct_nw_capped = False
    if answers.get("pct_net_worth", 3) <= 1 and raw_final_level >= 3:
        raw_final_level = 2  # cap at Moderado
        binding_pillar = "Concentracion patrimonial"
        pct_nw_capped = True

    final_profile = _level_index_to_name(raw_final_level)
    contradictions = detect_contradictions(answers)

    # Build explanation
    explanation_parts = []
    explanation_parts.append(
        f"Perfil determinado: {final_profile} (nivel {raw_final_level}/4)"
    )
    explanation_parts.append(f"Razon: Restriccion vinculante = {binding_pillar}")

    if raw_final_level < tol_level:
        explanation_parts.append(
            f"Tu tolerancia psicologica ({_level_index_to_name(tol_level)}) "
            f"sugiere mayor agresividad, pero tu {binding_pillar.lower()} "
            f"limita el riesgo aceptable."
        )

    if experience_capped:
        explanation_parts.append(
            "Se aplico un tope por falta de experiencia en inversiones."
        )
    if pct_nw_capped:
        explanation_parts.append(
            "Se aplico un tope porque esta inversion representa >50% de tu patrimonio."
        )

    return {
        "final_profile": final_profile,
        "final_level": raw_final_level,
        "pillar_levels": pillar_levels,
        "binding_pillar": binding_pillar,
        "contradictions": contradictions,
        "explanation": "\n".join(explanation_parts),
    }


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
# SECTION 6 — INVESTMENT CONFIGURATION (tickers, dates, benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def configure_investment(profile):
    """
    Collect investment configuration inputs, organized in 3 logical blocks:
      Block A — Assets (tickers + benchmark)
      Block B — Market Parameters (periodicity, dates, risk-free rate)
      Block C — Investment Parameters (amount, target return)
    """
    print("\n" + "=" * 65)
    print("  CONFIGURACION DE LA INVERSION")
    print("=" * 65)

    # ── BLOQUE A: Activos ────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  BLOQUE A — SELECCION DE ACTIVOS")
    print(f"  {'─'*60}")

    print(f"\n  [1/7] NUMERO DE ACTIVOS (max {MAX_TICKERS})")
    num_tickers = prompt_int(f"  Cuantos tickers deseas analizar? (1-{MAX_TICKERS}): ", 1, MAX_TICKERS)

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

    print(f"\n  [3/7] BENCHMARK (indice de referencia)")
    print(f"  Sugeridos: SPY, ^GSPC, ^DJI, ^IXIC, QQQ")
    print(f"  Presiona Enter para usar ({DEFAULT_BENCHMARK}).\n")
    bench_raw = input("  Benchmark ticker: ").strip().upper()
    benchmark = bench_raw if bench_raw else DEFAULT_BENCHMARK

    # ── BLOQUE B: Parametros de Mercado ──────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  BLOQUE B — PARAMETROS DE MERCADO")
    print(f"  {'─'*60}")

    print("\n  [4/7] PERIODICIDAD DE DATOS")
    print("  Opciones: diaria, semanal, mensual\n")
    periodicity = prompt("  Periodicidad: ", valid_options=list(PERIODICITY_MAP.keys()))

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

    print(f"\n  [6/7] TASA LIBRE DE RIESGO")
    print(f"  Presiona Enter para usar ({RISK_FREE_RATE*100:.1f} % anual).\n")
    risk_free = prompt_float("  Tasa libre de riesgo (%): ",
                             allow_blank=True, default=RISK_FREE_RATE * 100)
    if risk_free is not None and risk_free != RISK_FREE_RATE * 100:
        risk_free = risk_free / 100.0
    else:
        risk_free = RISK_FREE_RATE

    # ── BLOQUE C: Parametros de Inversion ────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  BLOQUE C — PARAMETROS DE INVERSION")
    print(f"  {'─'*60}")

    print("\n  [7/7] MONTO TOTAL A INVERTIR")
    monto_total = prompt_float("  Monto total (ej. 1000000): ", min_val=0.01)

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


def calculate_shrunk_covariance(returns, annualize_factor):
    """
    [NEW — PHASE 2] Ledoit-Wolf shrinkage covariance estimator.

    WHY SHRINKAGE?
    The raw sample covariance matrix has two problems:
      1. It is noisy — small changes in data produce large changes in the matrix.
      2. It can be poorly conditioned (near-singular) when the number of
         observations is small relative to the number of assets.

    Ledoit-Wolf shrinkage "blends" the sample covariance with a structured
    target (typically a scaled identity matrix), producing a more stable
    and reliable estimate. The shrinkage intensity (0-1) controls how much
    regularization is applied:
      - intensity ≈ 0 → trusts the raw sample covariance
      - intensity ≈ 1 → ignores sample data, uses structured target

    This makes optimization results more stable and less sensitive to
    small perturbations in the input data.

    Returns: (cov_shrunk_df, shrinkage_intensity)
    """
    if not HAS_LEDOIT_WOLF:
        print("  [!] sklearn no disponible. Usando covarianza muestral estandar.")
        return returns.cov() * annualize_factor, 0.0

    try:
        lw = LedoitWolf().fit(returns.values)
        cov_shrunk = lw.covariance_ * annualize_factor
        intensity = lw.shrinkage_

        cov_df = pd.DataFrame(cov_shrunk,
                              index=returns.columns,
                              columns=returns.columns)
        return cov_df, intensity
    except Exception as e:
        print(f"  [!] Ledoit-Wolf fallo ({e}). Usando covarianza muestral.")
        return returns.cov() * annualize_factor, 0.0


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
                       annualize_factor, risk_free, target="sharpe",
                       max_weight=1.0, cov_override=None):
    """
    [MODIFIED — PHASE 2] Scipy optimization for Max Sharpe or Min Variance.
    Long-only, with profile-based position-size limits.

    max_weight:   Maximum weight any single asset can have (0-1).
    cov_override: Optional covariance matrix (e.g., Ledoit-Wolf shrunk).
                  If provided, used for optimization instead of raw sample cov.
                  evaluate_portfolio still uses raw returns for unbiased metrics.
    """
    n_assets     = returns.shape[1]
    mean_returns = returns.mean() * annualize_factor
    if cov_override is not None:
        cov_matrix = cov_override
    else:
        cov_matrix = returns.cov() * annualize_factor

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, min(max_weight, 1.0)) for _ in range(n_assets))
    w0          = np.array([1 / n_assets] * n_assets)

    if max_weight < 1.0:
        print(f"    [Restriccion] Peso maximo por activo: {max_weight*100:.0f}%")

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
# SECTION 11B — [NEW — PHASE 2] CVaR, STRESS TESTING, & RISK SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def calculate_cvar_historical(returns_series, confidence=0.95):
    """
    [NEW — PHASE 2] Conditional Value-at-Risk (CVaR / Expected Shortfall).
    Uses historical return distribution — no normality assumption.

    VaR  = "What is the worst loss at the X% threshold?"
    CVaR = "If we breach that threshold, how bad is it ON AVERAGE?"

    CVaR is always >= VaR. It captures tail risk that VaR misses.

    Args:
        returns_series: pandas Series of periodic portfolio returns
        confidence: float (e.g. 0.95 = 95%)

    Returns:
        dict with 'var_pct', 'cvar_pct', 'n_tail_obs'
    """
    if len(returns_series) < 10:
        return {"var_pct": 0.0, "cvar_pct": 0.0, "n_tail_obs": 0}

    cutoff = np.percentile(returns_series, (1 - confidence) * 100)
    tail = returns_series[returns_series <= cutoff]

    var_pct = -cutoff      # positive = loss magnitude
    cvar_pct = -tail.mean() if len(tail) > 0 else var_pct

    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "n_tail_obs": len(tail),
    }


def calculate_cvar_multiple(portfolio_returns):
    """
    [NEW — PHASE 2] Compute historical CVaR at multiple confidence levels.
    Returns a dict: {confidence → {var_pct, cvar_pct, n_tail_obs}}
    """
    results = {}
    for conf in [0.90, 0.95, 0.99]:
        results[conf] = calculate_cvar_historical(portfolio_returns, conf)
    return results


def run_stress_tests(monto_total, pct_equity, pct_fixed):
    """
    [NEW — PHASE 2] Apply predefined stress scenarios to the portfolio.
    Uses STRESS_SCENARIOS config dict.

    Args:
        monto_total: total investment amount
        pct_equity:  % allocated to equity (0-100)
        pct_fixed:   % allocated to fixed income (0-100)

    Returns:
        list of dicts, one per scenario, with:
          - name, description
          - equity_loss, bond_change, total_loss
          - stressed_value, pct_loss
          - duration_months
    """
    equity_portion = monto_total * pct_equity / 100
    bond_portion = monto_total * pct_fixed / 100
    results = []

    for name, scenario in STRESS_SCENARIOS.items():
        eq_loss = equity_portion * scenario["equity_shock"]
        bond_chg = bond_portion * scenario["bond_change"]
        total_change = eq_loss + bond_chg  # negative = loss

        stressed_value = monto_total + total_change
        pct_loss = total_change / monto_total if monto_total > 0 else 0

        results.append({
            "name": name,
            "description": scenario["description"],
            "equity_loss": eq_loss,
            "bond_change": bond_chg,
            "total_loss": -total_change,  # positive = loss amount
            "stressed_value": stressed_value,
            "pct_loss": pct_loss,         # negative = loss
            "duration_months": scenario["duration_months"],
        })

    return results


def display_stress_tests(stress_results, monto_total):
    """
    [NEW — PHASE 2] Display stress test results in a formatted table.
    """
    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  PRUEBAS DE ESTRES — ESCENARIOS DE CRISIS")
    print(SEP)
    print(f"  Capital total: ${monto_total:,.2f}\n")

    print(f"  {'Escenario':<28} {'Perdida ($)':>14} {'Perdida (%)':>12} "
          f"{'Valor Final':>14} {'Recup.':>8}")
    print(f"  {'-'*28} {'-'*14} {'-'*12} {'-'*14} {'-'*8}")

    worst_loss_pct = 0
    for r in stress_results:
        loss_pct = abs(r["pct_loss"]) * 100
        if loss_pct > worst_loss_pct:
            worst_loss_pct = loss_pct
        print(f"  {r['name']:<28} "
              f"${r['total_loss']:>13,.0f} "
              f"{r['pct_loss']*100:>+11.1f}% "
              f"${r['stressed_value']:>13,.0f} "
              f"~{r['duration_months']:>3.0f} meses")

    # Key insights
    print(f"\n  {'─'*70}")
    # Check if bonds helped
    rate_shock = next((r for r in stress_results
                       if "Tasas" in r["name"]), None)
    if rate_shock and rate_shock["bond_change"] < 0:
        print(f"  ⚠ En el escenario de shock de tasas, la renta fija TAMBIEN perdio")
        print(f"    valor. La diversificacion tradicional (acciones + bonos) NO")
        print(f"    siempre protege. Esto ocurrio en 2022.")

    if worst_loss_pct > 35:
        print(f"\n  🔴 El peor escenario implica una perdida de {worst_loss_pct:.0f}%.")
        print(f"    Asegurate de que puedes absorber este nivel de perdida")
        print(f"    sin necesidad de vender en el peor momento.")
    elif worst_loss_pct > 20:
        print(f"\n  🟡 El peor escenario implica una perdida del {worst_loss_pct:.0f}%.")
        print(f"    Esto esta dentro del rango esperado para portafolios diversificados,")
        print(f"    pero requiere tolerancia a la volatilidad.")
    else:
        print(f"\n  ✅ La exposicion a crisis es moderada (max {worst_loss_pct:.0f}%).")

    print()


def generate_risk_summary(cfg, sharpe, var_results, cvar_results,
                           max_drawdown, stress_results, bench_metrics):
    """
    [NEW — PHASE 2] Synthesize all risk metrics into a unified risk assessment.
    Returns a dict with the synthesis AND prints a formatted summary.

    Analyzes:
      - Whether portfolio risk is acceptable for the user's profile
      - Whether tail risk is significantly worse than normal volatility
      - Whether stress scenarios reveal dangerous exposures
      - How the portfolio's risk compares to the benchmark
    """
    profile = cfg.get("profile", "Moderado")
    settings = PROFILE_SETTINGS.get(profile, PROFILE_SETTINGS["Moderado"])
    max_vol = settings["max_vol"]
    horizon = cfg.get("horizon_years", 5)
    monto_rv = cfg["monto_total"] * cfg["pct_variable"] / 100

    # ── Gather metrics ───────────────────────────────────────────────────
    cvar_95 = cvar_results.get(0.95, {})
    cvar_pct = cvar_95.get("cvar_pct", 0)
    var_pct = cvar_95.get("var_pct", 0)
    var_95_dollar = var_results.get(0.95, {}).get("annual", 0)

    worst_stress = max(stress_results, key=lambda r: r["total_loss"])
    worst_stress_pct = abs(worst_stress["pct_loss"])

    bench_vol = bench_metrics.get("volatility", 0)
    bench_dd = bench_metrics.get("max_drawdown", 0)

    # ── Risk assessment flags ────────────────────────────────────────────
    warnings_list = []
    overall_level = "ACEPTABLE"

    # 1. Profile-vol alignment
    prim_vol = var_results.get(0.95, {}).get("annual", 0) / monto_rv if monto_rv > 0 else 0
    # Use sharpe as proxy for reward-per-risk

    # 2. Tail risk analysis: CVaR vs VaR ratio
    tail_ratio = cvar_pct / var_pct if var_pct > 0 else 1.0
    if tail_ratio > 1.5:
        warnings_list.append({
            "level": "ALTO",
            "message": (
                f"El riesgo de cola es significativo: la perdida promedio "
                f"en el peor {5}% de periodos (CVaR={cvar_pct*100:.2f}%) "
                f"es {tail_ratio:.1f}x peor que el umbral VaR ({var_pct*100:.2f}%). "
                "Esto sugiere que los eventos extremos son mas severos "
                "de lo que la volatilidad normal indicaria."
            ),
        })
        overall_level = "ELEVADO"
    elif tail_ratio > 1.25:
        warnings_list.append({
            "level": "MODERADO",
            "message": (
                f"El riesgo de cola es moderado: CVaR/VaR ratio = {tail_ratio:.2f}. "
                "Los eventos extremos son algo peores que el umbral VaR."
            ),
        })

    # 3. Drawdown severity
    if abs(max_drawdown) > 0.35:
        warnings_list.append({
            "level": "ALTO",
            "message": (
                f"El drawdown maximo historico ({max_drawdown*100:.1f}%) es severo. "
                "La mayoria de los inversionistas con perfiles conservadores o "
                "moderados no toleran caidas de esta magnitud."
            ),
        })
        overall_level = "ELEVADO"
    elif abs(max_drawdown) > 0.25:
        warnings_list.append({
            "level": "MODERADO",
            "message": (
                f"El drawdown maximo historico ({max_drawdown*100:.1f}%) es "
                "significativo pero dentro del rango esperado para portafolios "
                "con exposicion a renta variable."
            ),
        })

    # 4. Stress test severity
    if worst_stress_pct > 0.35:
        warnings_list.append({
            "level": "ALTO",
            "message": (
                f"En el escenario de '{worst_stress['name']}', el portafolio "
                f"perderia {worst_stress_pct*100:.1f}%. "
                f"Con un horizonte de {horizon} anos, esto podria requerir "
                f"~{worst_stress['duration_months']} meses de recuperacion."
            ),
        })
        if overall_level == "ACEPTABLE":
            overall_level = "MODERADO"

    # 5. Sharpe assessment
    if sharpe < 0:
        warnings_list.append({
            "level": "CRITICO",
            "message": (
                "El Sharpe Ratio es negativo. El portafolio no compensa "
                "el riesgo asumido frente a la tasa libre de riesgo."
            ),
        })
        overall_level = "ELEVADO"

    # 6. Benchmark risk comparison
    port_vol_est = var_pct * np.sqrt(252) if var_pct > 0 else 0  # rough annual
    if port_vol_est > bench_vol * 1.3 and sharpe <= bench_metrics.get("sharpe", 0):
        warnings_list.append({
            "level": "MODERADO",
            "message": (
                "El portafolio toma significativamente mas riesgo que el benchmark "
                "pero no logra mejor eficiencia (Sharpe). El riesgo adicional "
                "no esta siendo recompensado."
            ),
        })

    # ── Display ──────────────────────────────────────────────────────────
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  RESUMEN INTEGRADO DE RIESGO")
    print(SEP)

    level_icon = {
        "ACEPTABLE": "✅",
        "MODERADO": "🟡",
        "ELEVADO": "🔴",
    }

    print(f"\n  Nivel de riesgo general: {level_icon.get(overall_level, '⚠')} {overall_level}")
    print(f"  Perfil del inversionista: {profile}")
    print(f"  Horizonte: {horizon} anos")

    # Metric synthesis table
    print(f"\n  {'Metrica':<32} {'Valor':>14} {'Evaluacion':<20}")
    print(f"  {'-'*32} {'-'*14} {'-'*20}")

    sr_eval = ("Optimo" if sharpe > 0.7 else
               "Aceptable" if sharpe > 0.3 else
               "Bajo" if sharpe > 0 else "NEGATIVO")
    print(f"  {'Sharpe Ratio':<32} {sharpe:>14.4f} {sr_eval:<20}")

    print(f"  {'VaR 95% (hist.)':<32} {var_pct*100:>13.2f}% {'por periodo':<20}")
    print(f"  {'CVaR 95% (Expected Shortfall)':<32} {cvar_pct*100:>13.2f}% "
          f"{'promedio en cola':<20}")
    print(f"  {'Ratio CVaR/VaR':<32} {tail_ratio:>14.2f} "
          f"{'Normal' if tail_ratio < 1.3 else 'Colas pesadas':<20}")
    print(f"  {'Max Drawdown historico':<32} {max_drawdown*100:>13.2f}% "
          f"{'Severo' if abs(max_drawdown) > 0.35 else 'Moderado' if abs(max_drawdown) > 0.20 else 'Bajo':<20}")

    var_95_annual_val = var_results.get(0.95, {}).get("annual", 0)
    print(f"  {'VaR 95% anual ($)':<32} ${var_95_annual_val:>13,.0f} "
          f"{'perdida potencial':<20}")

    # VaR vs CVaR explanation
    print(f"\n  {'─'*60}")
    print(f"  VaR vs CVaR (explicacion):")
    print(f"    VaR  = 'En el 95% de los periodos, no perderas mas de {var_pct*100:.2f}%'")
    print(f"    CVaR = 'Pero si pierdes mas, la perdida PROMEDIO sera {cvar_pct*100:.2f}%'")
    if tail_ratio > 1.3:
        print(f"    ⚠ La diferencia entre VaR y CVaR es grande ({tail_ratio:.1f}x).")
        print(f"      Esto significa que cuando las cosas van mal, van MUCHO peor")
        print(f"      de lo que la volatilidad normal sugiere.")

    # Warnings
    if warnings_list:
        print(f"\n  {'═'*60}")
        print(f"  ALERTAS DE RIESGO ({len(warnings_list)})")
        print(f"  {'═'*60}")
        for w in warnings_list:
            icon = "🔴" if w["level"] in ("ALTO", "CRITICO") else "🟡"
            print(f"\n  {icon} [{w['level']}]")
            lines = _wrap_text(w["message"], width=58, indent=5)
            for line in lines:
                print(line)

    # Profile suitability
    print(f"\n  {'─'*60}")
    print(f"  ADECUACION AL PERFIL '{profile}':")
    if overall_level == "ACEPTABLE":
        print(f"  ✅ El nivel de riesgo del portafolio es consistente con tu perfil.")
    elif overall_level == "MODERADO":
        print(f"  🟡 El portafolio presenta riesgos moderados. Asegurate de que")
        print(f"     puedes tolerar las fluctuaciones proyectadas sin vender")
        print(f"     en momentos de panico.")
    else:
        print(f"  🔴 El perfil de riesgo del portafolio EXCEDE lo recomendado")
        print(f"     para un inversionista '{profile}'. Considera:")
        print(f"     - Reducir la exposicion a renta variable")
        print(f"     - Aumentar la diversificacion")
        print(f"     - Ajustar las expectativas de rendimiento")

    print()

    return {
        "overall_level": overall_level,
        "tail_ratio": tail_ratio,
        "warnings": warnings_list,
        "sharpe": sharpe,
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "max_drawdown": max_drawdown,
        "worst_stress_name": worst_stress["name"],
        "worst_stress_pct": worst_stress_pct,
    }


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
# SECTION 15B — ANALYSIS INTELLIGENCE MODULE
# ─────────────────────────────────────────────────────────────────────────────
# Pure-analysis functions that detect portfolio issues and generate
# professional-grade warnings and insights. Nothing is blocked or restricted —
# all output is advisory.

def analyze_diversification(corr_matrix, valid_tickers):
    """
    Assess portfolio diversification quality using correlation analysis
    and optional sector detection via yfinance.

    Returns a dict with:
      - classification: 'fuerte', 'moderada', 'debil'
      - avg_corr: float
      - high_corr_pairs: list of (ticker_a, ticker_b, corr)
      - sectors: dict {ticker: sector_name}
      - sector_concentration: bool (many assets in same sector)
      - explanation: str
    """
    n = len(corr_matrix)
    result = {
        "classification": "fuerte",
        "avg_corr": 0.0,
        "high_corr_pairs": [],
        "sectors": {},
        "sector_concentration": False,
        "explanation": "",
    }

    # Average pairwise correlation (exclude diagonal)
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        avg_corr = corr_matrix.values[mask].mean()
    else:
        avg_corr = 1.0
    result["avg_corr"] = avg_corr

    # Identify highly correlated pairs (> 0.7)
    high_pairs = []
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                c = corr_matrix.iloc[i, j]
                if abs(c) > 0.7:
                    high_pairs.append((valid_tickers[i], valid_tickers[j], c))
    result["high_corr_pairs"] = high_pairs

    # Sector and country detection (best-effort via yfinance)
    sectors = {}
    countries = {}
    for t in valid_tickers:
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector", "Desconocido")
            sectors[t] = sector if sector else "Desconocido"
            country = info.get("country", "Desconocido")
            countries[t] = country if country else "Desconocido"
        except Exception:
            sectors[t] = "Desconocido"
            countries[t] = "Desconocido"
    result["sectors"] = sectors
    result["countries"] = countries

    # Sector concentration check
    from collections import Counter
    known_sectors = [s for s in sectors.values() if s != "Desconocido"]
    if known_sectors:
        sector_counts = Counter(known_sectors)
        most_common_count = sector_counts.most_common(1)[0][1]
        n_known = len(known_sectors)
        if n_known > 1 and most_common_count / n_known >= 0.6:
            result["sector_concentration"] = True

    # Country / market concentration check
    known_countries = [c for c in countries.values() if c != "Desconocido"]
    country_concentration = False
    if known_countries:
        country_counts = Counter(known_countries)
        most_common_country, most_common_cnt = country_counts.most_common(1)[0]
        if len(known_countries) > 1 and most_common_cnt / len(known_countries) >= 0.8:
            country_concentration = True
    result["country_concentration"] = country_concentration

    # Asset class check — all equities?
    all_equities = all(
        s not in ("Desconocido",) and s != "" for s in sectors.values()
    ) and len(valid_tickers) > 1
    result["all_same_asset_class"] = all_equities

    # Classification
    warnings_list = []
    if avg_corr >= 0.7:
        result["classification"] = "debil"
        warnings_list.append(
            "La correlacion promedio entre activos es alta (>0.7), lo que "
            "limita significativamente los beneficios de diversificacion."
        )
    elif avg_corr >= 0.4:
        result["classification"] = "moderada"
        warnings_list.append(
            "La correlacion promedio es moderada. La diversificacion "
            "ofrece ciertos beneficios pero podria mejorarse."
        )
    else:
        result["classification"] = "fuerte"
        warnings_list.append(
            "La correlacion promedio es baja, lo que indica buena "
            "diversificacion entre los activos seleccionados."
        )

    if high_pairs:
        pair_str = ", ".join([f"{a}-{b} ({c:.2f})" for a, b, c in high_pairs[:5]])
        warnings_list.append(
            f"Pares altamente correlacionados (>0.7): {pair_str}. "
            "Estos activos tienden a moverse juntos y ofrecen poca "
            "proteccion mutua en periodos de estres."
        )

    if result["sector_concentration"]:
        warnings_list.append(
            "Se detecto concentracion sectorial significativa. Aunque se "
            "seleccionaron multiples activos, varios pertenecen al mismo "
            "sector, lo que limita la diversificacion real."
        )

    if country_concentration:
        warnings_list.append(
            f"Todos o casi todos los activos provienen del mismo mercado "
            f"({most_common_country}). Considere incluir activos de "
            f"mercados internacionales para mejorar la diversificacion geografica."
        )

    if all_equities:
        warnings_list.append(
            "Aunque la diversificacion estadistica puede ser aceptable, "
            "el portafolio permanece concentrado en una sola clase de activo "
            "(renta variable). Incluir bonos, commodities o REITs podria "
            "mejorar la diversificacion estructural."
        )

    if n == 1:
        result["classification"] = "debil"
        warnings_list.append(
            "Portafolio de un solo activo — no hay diversificacion. "
            "Toda la exposicion depende de un unico instrumento."
        )

    result["explanation"] = " ".join(warnings_list)
    return result


def check_data_quality(returns, cfg):
    """
    Evaluate whether the dataset is sufficient for reliable analysis.

    Returns a dict with:
      - n_observations: int
      - is_small_sample: bool (< 30 observations)
      - horizon_mismatch: bool
      - warnings: list of str
    """
    n_obs = len(returns)
    horizon_years = cfg.get("horizon_years", 5)

    # Data span in years (approximate)
    if not returns.empty:
        data_span_days = (returns.index[-1] - returns.index[0]).days
        data_span_years = data_span_days / 365.25
    else:
        data_span_days = 0
        data_span_years = 0

    is_small = n_obs < 30
    horizon_mismatch = (horizon_years > 5 and data_span_years < 2) or \
                       (horizon_years > 10 and data_span_years < 5)

    warnings_list = []

    if is_small:
        warnings_list.append(
            f"El analisis se basa en solo {n_obs} observaciones. "
            "Con menos de 30 puntos de datos, las estimaciones de riesgo "
            "y rendimiento tienen baja confiabilidad estadistica."
        )

    if horizon_mismatch:
        warnings_list.append(
            f"Horizonte de inversion: {horizon_years} anos, pero datos "
            f"historicos disponibles: {data_span_years:.1f} anos. "
            "Esta ventana puede no representar completamente el "
            "comportamiento del mercado a largo plazo."
        )

    if n_obs < 60 and not is_small:
        warnings_list.append(
            f"La muestra tiene {n_obs} observaciones. Aunque supera el "
            "minimo, una ventana mas amplia mejoraria la robustez "
            "de las estimaciones."
        )

    return {
        "n_observations": n_obs,
        "data_span_years": round(data_span_years, 1),
        "is_small_sample": is_small,
        "horizon_mismatch": horizon_mismatch,
        "warnings": warnings_list,
    }


def analyze_concentration(weights, tickers, label=""):
    """
    Detect portfolio concentration risk.

    Returns a dict with:
      - max_weight: float
      - max_ticker: str
      - top2_weight: float
      - hhi: float (Herfindahl-Hirschman Index, 0-1)
      - is_concentrated: bool (>50% in one asset)
      - is_highly_concentrated: bool (>60% in one asset)
      - warnings: list of str
    """
    max_w = max(weights) if len(weights) > 0 else 0
    max_idx = int(np.argmax(weights)) if len(weights) > 0 else 0
    max_ticker = tickers[max_idx] if len(tickers) > 0 else "N/A"

    # Top 2 weight
    sorted_w = sorted(weights, reverse=True)
    top2_w = sum(sorted_w[:2]) if len(sorted_w) >= 2 else max_w

    # HHI — sum of squared weights (1.0 = fully concentrated)
    hhi = sum(w**2 for w in weights)

    is_concentrated = max_w > 0.50
    is_highly_concentrated = max_w > 0.60

    warnings_list = []

    if is_highly_concentrated:
        warnings_list.append(
            f"⚠ ADVERTENCIA FUERTE: El portafolio {label} muestra alta concentracion: "
            f"{max_ticker} tiene un peso de {max_w*100:.1f}%. "
            "Esto incrementa significativamente el riesgo especifico del emisor. "
            "Considere limitar la exposicion a un solo activo para "
            "mejorar la diversificacion."
        )
    elif is_concentrated:
        warnings_list.append(
            f"⚠ ADVERTENCIA: El portafolio {label} tiene concentracion significativa: "
            f"{max_ticker} tiene un peso de {max_w*100:.1f}%. "
            "Considere limitar la exposicion a un solo activo para "
            "mejorar la diversificacion."
        )

    if top2_w > 0.80 and len(weights) > 2:
        warnings_list.append(
            f"Los 2 activos principales representan {top2_w*100:.1f}% del "
            f"portafolio. El resto de activos contribuyen marginalmente."
        )

    return {
        "max_weight": max_w,
        "max_ticker": max_ticker,
        "top2_weight": top2_w,
        "hhi": hhi,
        "is_concentrated": is_concentrated,
        "is_highly_concentrated": is_highly_concentrated,
        "warnings": warnings_list,
    }


def check_profile_consistency(profile, pct_variable, port_vol, port_ret, cfg):
    """
    Compare the actual portfolio characteristics against the expected
    parameters for the investor profile. Advisory only — no restrictions.

    Returns a dict with:
      - aligned: bool (True if portfolio matches profile expectations)
      - warnings: list of str
      - severity: 'ok', 'mild', 'significant'
    """
    settings = PROFILE_SETTINGS.get(profile, PROFILE_SETTINGS["Moderado"])
    eq_lo, eq_hi = settings["equity_range"]
    max_vol = settings["max_vol"]

    warnings_list = []
    severity = "ok"

    # Equity allocation vs profile range
    if pct_variable > eq_hi + 10:  # Allow 10pp flexibility
        warnings_list.append(
            f"La asignacion actual de renta variable ({pct_variable:.0f}%) excede "
            f"significativamente el rango sugerido para perfil {profile} "
            f"({eq_lo}-{eq_hi}%). Esta asignacion es mas agresiva que lo "
            f"sugerido por el perfil."
        )
        severity = "significant"
    elif pct_variable > eq_hi:
        warnings_list.append(
            f"La asignacion de renta variable ({pct_variable:.0f}%) esta ligeramente "
            f"por encima del rango sugerido para perfil {profile} ({eq_lo}-{eq_hi}%)."
        )
        severity = "mild"

    # Portfolio volatility vs profile max
    if port_vol > max_vol:
        vol_excess = (port_vol - max_vol) * 100
        warnings_list.append(
            f"La volatilidad del portafolio ({port_vol*100:.1f}%) excede el "
            f"umbral tipico para perfil {profile} ({max_vol*100:.1f}%) "
            f"en {vol_excess:.1f} puntos porcentuales. El portafolio actual "
            f"asume mas riesgo de lo recomendado para este perfil."
        )
        if severity != "significant":
            severity = "significant" if vol_excess > 5 else "mild"

    aligned = len(warnings_list) == 0

    return {
        "aligned": aligned,
        "warnings": warnings_list,
        "severity": severity,
    }


def analyze_objective_alignment(objective_label, valid_tickers, diversification_info):
    """
    Check whether the selected assets are compatible with the stated
    investment objective using heuristic sector + dividend analysis.

    Returns a dict with:
      - aligned: bool
      - warnings: list of str
    """
    warnings_list = []
    obj_lower = objective_label.lower() if objective_label else ""

    # Classify objective intent
    is_income = "ingreso" in obj_lower or "renta" in obj_lower
    is_preservation = "preserv" in obj_lower
    is_growth = "crecimiento" in obj_lower and "agresivo" in obj_lower

    sectors = diversification_info.get("sectors", {})

    # Growth-oriented sectors
    growth_sectors = {"Technology", "Communication Services",
                      "Consumer Cyclical", "Consumer Discretionary"}
    # Income-oriented sectors
    income_sectors = {"Utilities", "Real Estate", "Financial Services",
                      "Consumer Defensive", "Energy"}

    known = {t: s for t, s in sectors.items() if s != "Desconocido"}

    if not known:
        return {"aligned": True, "warnings": []}

    growth_count = sum(1 for s in known.values() if s in growth_sectors)
    income_count = sum(1 for s in known.values() if s in income_sectors)
    total = len(known)

    if is_income and total > 0 and growth_count / total >= 0.6:
        warnings_list.append(
            "El objetivo declarado es generacion de ingreso/renta, pero la "
            "mayoria de los activos seleccionados pertenecen a sectores "
            "orientados al crecimiento (tecnologia, consumo discrecional). "
            "Considere incluir activos con mayor rendimiento por dividendos "
            "(utilities, REITs, acciones de consumo basico)."
        )

    if is_preservation and total > 0 and growth_count / total >= 0.5:
        warnings_list.append(
            "El objetivo declarado es preservacion de capital, pero una "
            "porcion significativa del portafolio esta en sectores de alto "
            "crecimiento que tienden a ser mas volatiles. Considere una "
            "mayor proporcion en renta fija o sectores defensivos."
        )

    if is_growth and total > 0 and income_count / total >= 0.7:
        warnings_list.append(
            "El objetivo declarado es crecimiento agresivo, pero la mayoria "
            "de los activos son de sectores defensivos/de ingreso. Considere "
            "incluir activos de mayor potencial de apreciacion."
        )

    aligned = len(warnings_list) == 0
    return {"aligned": aligned, "warnings": warnings_list}


def enhance_benchmark_analysis(ms_ret, ms_vol, port_max_dd,
                                bench_metrics, ms_sr):
    """
    Produce a balanced, professional benchmark comparison that captures
    both sides of the tradeoff (return AND risk).

    Returns a dict with:
      - beats_return, beats_vol, beats_sharpe, beats_dd: bool
      - tradeoff_text: str (balanced interpretation)
    """
    b_ret = bench_metrics["return"]
    b_vol = bench_metrics["volatility"]
    b_sr  = bench_metrics["sharpe"]
    b_dd  = bench_metrics["max_drawdown"]

    beats_return = ms_ret > b_ret
    beats_vol    = ms_vol < b_vol   # lower vol is better
    beats_sharpe = ms_sr > b_sr
    beats_dd     = port_max_dd > b_dd  # less negative is better

    parts = []

    # --- Return + risk balanced assessment ---
    if beats_return and beats_vol and beats_sharpe:
        parts.append(
            "El portafolio supera al benchmark en rendimiento, volatilidad "
            "y eficiencia ajustada por riesgo (Sharpe). Este es un resultado "
            "muy favorable."
        )
    elif beats_return and not beats_vol:
        vol_diff = (ms_vol - b_vol) * 100
        parts.append(
            f"El portafolio supera al benchmark en rendimiento, pero tambien "
            f"exhibe mayor volatilidad (+{vol_diff:.1f} pp). El mayor "
            f"rendimiento viene acompanado de mayor riesgo."
        )
    elif not beats_return and beats_vol:
        parts.append(
            "El portafolio tiene menor volatilidad que el benchmark, pero "
            "tambien menor rendimiento. Es una opcion mas defensiva."
        )
    elif not beats_return and not beats_vol:
        parts.append(
            "El benchmark supera al portafolio tanto en rendimiento como en "
            "volatilidad. Considere ajustar la seleccion de activos o "
            "evaluar un ETF indexado como alternativa."
        )

    # --- Drawdown comparison ---
    if beats_return and not beats_dd:
        parts.append(
            f"Sin embargo, el max drawdown historico del portafolio "
            f"({port_max_dd*100:.1f}%) es mas profundo que el del benchmark "
            f"({b_dd*100:.1f}%), lo que implica mayor exposicion a "
            f"perdidas extremas."
        )

    # --- Sharpe nuance ---
    if beats_return and not beats_sharpe:
        parts.append(
            "A pesar de superar en rendimiento absoluto, la eficiencia "
            "ajustada por riesgo (Sharpe) es menor que la del benchmark, "
            "lo que sugiere que el rendimiento adicional no compensa "
            "proporcionalmente el riesgo extra asumido."
        )

    return {
        "beats_return": beats_return,
        "beats_vol": beats_vol,
        "beats_sharpe": beats_sharpe,
        "beats_dd": beats_dd,
        "tradeoff_text": " ".join(parts),
    }


def interpret_risk_metrics(var_results, returns, cfg):
    """
    Add interpretive caveats to VaR and probability estimates.

    Returns a dict with:
      - warnings: list of str
    """
    n_obs = len(returns)
    warnings_list = []

    # Standard caveat — always present
    warnings_list.append(
        "Estas estimaciones se basan en datos historicos y suponen "
        "distribucion normal de rendimientos. Los resultados no "
        "garantizan desempeno futuro. En condiciones de estres, las "
        "perdidas reales pueden superar las estimaciones del VaR."
    )

    # Small sample
    if n_obs < 60:
        warnings_list.append(
            f"La muestra utilizada ({n_obs} observaciones) es limitada. "
            "Las estimaciones de VaR pueden tener menor precision "
            "que las calculadas con historiales mas amplios."
        )

    return {"warnings": warnings_list}


def check_fi_realism(fi_recommendation, cfg):
    """
    Evaluate whether the fixed-income allocation is realistic.

    Returns a dict with:
      - is_small_amount: bool
      - warning: str or None
    """
    if fi_recommendation is None or fi_recommendation.empty:
        return {"is_small_amount": False, "warning": None}

    monto_fi = cfg["monto_total"] * cfg["pct_fixed"] / 100
    n_instruments = len(fi_recommendation)

    if monto_fi < 1000 and n_instruments > 1:
        return {
            "is_small_amount": True,
            "warning": (
                f"El monto asignado a renta fija (${monto_fi:,.2f}) es "
                "relativamente pequeno para dividirlo entre multiples "
                "instrumentos. Para montos reducidos, puede ser mas "
                "practico utilizar un solo instrumento diversificado "
                "(ej. un fondo de bonos o CETES) en lugar de distribuir "
                "entre varias categorias."
            ),
        }
    return {"is_small_amount": False, "warning": None}


def generate_insights(corr_matrix, valid_tickers, returns, cfg,
                      ms_w, ms_ret, ms_vol, ms_sr,
                      mv_w, mv_ret, mv_vol, mv_sr,
                      bench_metrics, port_max_dd,
                      var_results, valuation_signals,
                      fi_recommendation):
    """
    Master analysis function. Calls all sub-analyzers and returns a
    unified insights dict consumed by display and report functions.
    """
    print("  Generando analisis critico del portafolio...")

    diversification = analyze_diversification(corr_matrix, valid_tickers)
    data_quality = check_data_quality(returns, cfg)
    concentration_ms = analyze_concentration(ms_w, valid_tickers, "Max Sharpe")
    concentration_mv = analyze_concentration(mv_w, valid_tickers, "Min Varianza")
    benchmark_tradeoff = enhance_benchmark_analysis(
        ms_ret, ms_vol, port_max_dd, bench_metrics, ms_sr
    )
    risk_interpretation = interpret_risk_metrics(var_results, returns, cfg)
    fi_realism = check_fi_realism(fi_recommendation, cfg)

    # Profile consistency check (Part 2)
    profile_consistency = check_profile_consistency(
        cfg["profile"], cfg["pct_variable"], ms_vol, ms_ret, cfg
    )

    # Objective vs asset alignment (Part 3)
    objective_alignment = analyze_objective_alignment(
        cfg.get("objective_label", ""), valid_tickers, diversification
    )

    # Min-variance efficiency check
    mv_below_rf = mv_ret < cfg["risk_free"]

    # Valuation signal reliability
    all_neutral = True
    for t in valid_tickers:
        sig = valuation_signals.get(t, {}).get("signal", "")
        if sig not in ("Neutral / valor justo", "Datos insuficientes", ""):
            all_neutral = False
            break

    # Build actionable recommendations
    recommendations = []

    # Profile mismatch recommendations
    if not profile_consistency["aligned"]:
        for w in profile_consistency["warnings"]:
            recommendations.append(w)

    # Objective alignment recommendations
    if not objective_alignment["aligned"]:
        for w in objective_alignment["warnings"]:
            recommendations.append(w)

    if diversification["classification"] == "debil":
        recommendations.append(
            "Considere incluir activos de sectores o regiones diferentes "
            "para mejorar la diversificacion."
        )
    if diversification["sector_concentration"]:
        recommendations.append(
            "Considere reducir la concentracion sectorial agregando "
            "activos de otros sectores (ej. consumo, salud, energia)."
        )
    if diversification.get("country_concentration"):
        recommendations.append(
            "Considere incluir activos de mercados internacionales "
            "para mejorar la diversificacion geografica."
        )
    if data_quality["is_small_sample"]:
        recommendations.append(
            "Considere utilizar un periodo historico mas amplio "
            "para mejorar la confiabilidad del analisis."
        )
    if data_quality["horizon_mismatch"]:
        recommendations.append(
            "Considere usar datos historicos que cubran al menos "
            "la mitad de su horizonte de inversion."
        )
    if concentration_ms["is_highly_concentrated"]:
        recommendations.append(
            f"Considere limitar la exposicion maxima a un solo activo "
            f"(actualmente {concentration_ms['max_ticker']} tiene "
            f"{concentration_ms['max_weight']*100:.0f}%)."
        )
    if mv_below_rf:
        recommendations.append(
            "El portafolio de Minima Varianza rinde por debajo de la "
            "tasa libre de riesgo. Considere el portafolio Max Sharpe "
            "como alternativa mas eficiente."
        )
    if fi_realism.get("is_small_amount"):
        recommendations.append(
            "Considere simplificar la asignacion de renta fija a un "
            "solo instrumento diversificado para montos pequenos."
        )

    return {
        "diversification": diversification,
        "data_quality": data_quality,
        "concentration_ms": concentration_ms,
        "concentration_mv": concentration_mv,
        "benchmark_tradeoff": benchmark_tradeoff,
        "risk_interpretation": risk_interpretation,
        "fi_realism": fi_realism,
        "profile_consistency": profile_consistency,
        "objective_alignment": objective_alignment,
        "mv_below_rf": mv_below_rf,
        "all_valuation_neutral": all_neutral,
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — MONTE CARLO FUTURE SCENARIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
# This module is SEPARATE from the existing Monte Carlo in Section 10.
#
# Existing Monte Carlo (Section 10, simulate_portfolios):
#   → Generates random portfolio WEIGHTS to map the efficient frontier.
#   → Purpose: identify optimal portfolios.
#
# New Monte Carlo (this section):
#   → Simulates future RETURN PATHS using correlated multivariate normal.
#   → Purpose: estimate the distribution of future portfolio outcomes.
#
# Assumptions:
#   - Returns are modeled with multivariate normal distribution (simplification)
#   - Per-period parameters are derived via arithmetic de-annualization
#   - Fixed income is modeled deterministically (not stochastically)
#   - Historical parameters do not guarantee future performance
# ─────────────────────────────────────────────────────────────────────────────

def simulate_future_portfolio_paths(weights, mean_returns_ann, cov_matrix_ann,
                                     annualize_factor, horizon_years,
                                     initial_value,
                                     n_sims=N_FUTURE_SIMULATIONS):
    """
    Simulate future portfolio value paths using correlated multivariate
    normal returns. Weights are FIXED (recommended portfolio) and we
    simulate RETURNS over time — conceptually different from
    simulate_portfolios() which randomizes weights.

    Uses simple returns (not log returns) for consistency with the rest
    of the script.  Per-period parameters use arithmetic de-annualization.
    """
    n_steps = horizon_years * annualize_factor

    # De-annualize (arithmetic approximation — stated in output)
    period_mean = np.array(mean_returns_ann) / annualize_factor
    cov_vals = cov_matrix_ann.values if hasattr(cov_matrix_ann, 'values') \
        else cov_matrix_ann
    period_cov = cov_vals / annualize_factor

    # Simulate correlated asset returns: shape (n_sims, n_steps, n_assets)
    sim_asset_rets = np.random.multivariate_normal(
        mean=period_mean, cov=period_cov, size=(n_sims, n_steps)
    )

    # Portfolio returns per period: shape (n_sims, n_steps)
    port_rets = sim_asset_rets @ weights

    # Cumulative value paths: shape (n_sims, n_steps+1)
    cum_factors = np.cumprod(1 + port_rets, axis=1)
    paths = np.hstack([np.ones((n_sims, 1)), cum_factors]) * initial_value

    return {"paths": paths, "returns": port_rets, "n_steps": n_steps}


def project_fixed_income_path(initial_value, annual_rate, annualize_factor,
                               horizon_years):
    """
    Deterministically project fixed-income portion over the horizon
    using compound growth at the given annual rate converted to the
    selected periodicity.
    """
    n_steps = horizon_years * annualize_factor
    period_rate = (1 + annual_rate) ** (1.0 / annualize_factor) - 1 \
        if annual_rate > 0 else 0.0
    return initial_value * (1 + period_rate) ** np.arange(n_steps + 1)


def combine_total_portfolio_paths(vi_paths, fi_path):
    """Combine stochastic VI paths with deterministic FI path."""
    return vi_paths + fi_path[np.newaxis, :]


def calculate_simulation_statistics(final_values, initial_value,
                                     target_return=None,
                                     horizon_years=None, label=""):
    """Compute distribution statistics from simulated final values."""
    stats = {
        "label": label,
        "initial_value": initial_value,
        "mean": np.mean(final_values),
        "median": np.median(final_values),
        "min": np.min(final_values),
        "max": np.max(final_values),
        "std": np.std(final_values),
        "p5": np.percentile(final_values, 5),
        "p25": np.percentile(final_values, 25),
        "p50": np.percentile(final_values, 50),
        "p75": np.percentile(final_values, 75),
        "p95": np.percentile(final_values, 95),
        "prob_loss": np.mean(final_values < initial_value) * 100,
    }
    # Target return probability
    if target_return is not None and horizon_years is not None \
            and horizon_years > 0:
        target_value = initial_value * (1 + target_return) ** horizon_years
        stats["target_value"] = target_value
        stats["target_return"] = target_return
        stats["prob_target"] = np.mean(final_values >= target_value) * 100
    else:
        stats["target_value"] = None
        stats["target_return"] = None
        stats["prob_target"] = None
    # Annualized geometric return
    if initial_value > 0 and horizon_years and horizon_years > 0:
        geo = (final_values / initial_value) ** (1.0 / horizon_years) - 1
        stats["mean_ann_return"] = np.mean(geo)
        stats["median_ann_return"] = np.median(geo)
    else:
        stats["mean_ann_return"] = 0.0
        stats["median_ann_return"] = 0.0
    return stats


def calculate_simulated_drawdowns(paths):
    """
    Compute max drawdown per simulated path.  Each path is treated
    independently; drawdown = (value - running_max) / running_max.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = np.where(running_max > 0,
                      (paths - running_max) / running_max, 0.0)
    max_dd = dd.min(axis=1)
    return {
        "max_dd_per_path": max_dd,
        "avg_max_dd": np.mean(max_dd),
        "worst_max_dd": np.min(max_dd),
        "p5_max_dd": np.percentile(max_dd, 5),
        "p50_max_dd": np.median(max_dd),
        "p95_max_dd": np.percentile(max_dd, 95),
    }


def _generate_simulation_interpretation(stats_vi, stats_total,
                                         drawdown_stats, n_obs,
                                         horizon_years):
    """Generate plain-language interpretation in Spanish."""
    lines = []
    primary = stats_total if stats_total is not None else stats_vi

    # General direction
    if primary["prob_loss"] < 15:
        lines.append(
            "En la gran mayoria de los escenarios simulados, el portafolio "
            "crece a lo largo del horizonte de inversion.")
    elif primary["prob_loss"] < 40:
        lines.append(
            "En la mayoria de los escenarios el portafolio crece, pero "
            "existe una probabilidad relevante de terminar por debajo "
            "del capital inicial.")
    else:
        lines.append(
            "Existe una probabilidad significativa de terminar por debajo "
            "del capital inicial. El portafolio enfrenta riesgo considerable.")

    # Dispersion
    if primary["initial_value"] > 0:
        spread = (primary["p95"] - primary["p5"]) / primary["initial_value"]
        if spread > 1.5:
            lines.append(
                "La distribucion de resultados finales es muy amplia, "
                "lo que sugiere alta incertidumbre sobre el valor terminal.")
        elif spread > 0.5:
            lines.append(
                "La distribucion de resultados finales muestra variabilidad "
                "moderada, tipica de portafolios diversificados.")
        else:
            lines.append(
                "La distribucion de resultados finales es relativamente "
                "estrecha, lo que sugiere menor incertidumbre.")

    # Upside
    if primary["median"] > primary["initial_value"] * 1.1:
        lines.append(
            "El portafolio tiene potencial de crecimiento importante "
            "en el escenario mediano.")

    # Drawdown warning
    if drawdown_stats["avg_max_dd"] < -0.20:
        lines.append(
            f"El drawdown maximo promedio simulado es "
            f"{drawdown_stats['avg_max_dd']*100:.1f}%, lo que implica que "
            "caidas temporales significativas son probables durante "
            "el horizonte de inversion.")

    # Target return
    if primary.get("prob_target") is not None:
        prob = primary["prob_target"]
        if prob > 70:
            lines.append(
                f"La probabilidad simulada de alcanzar el rendimiento "
                f"objetivo es alta ({prob:.1f}%).")
        elif prob > 40:
            lines.append(
                f"La probabilidad simulada de alcanzar el rendimiento "
                f"objetivo es moderada ({prob:.1f}%).")
        else:
            lines.append(
                f"La probabilidad simulada de alcanzar el rendimiento "
                f"objetivo es baja ({prob:.1f}%). Considere ajustar "
                "expectativas o la composicion del portafolio.")

    # Data quality caveat
    if n_obs < 60:
        lines.append(
            f"La confiabilidad de la simulacion puede ser limitada "
            f"debido al tamano de la muestra historica "
            f"({n_obs} observaciones).")
    return lines


def display_monte_carlo_simulation_results(future_mc, cfg):
    """Display Monte Carlo future simulation results."""
    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  SIMULACION MONTE CARLO — ESCENARIOS FUTUROS")
    print(SEP)
    print(f"  Escenarios simulados : {N_FUTURE_SIMULATIONS:,}")
    print(f"  Horizonte            : {cfg['horizon_years']} anos")
    print(f"  Periodicidad         : {cfg['periodicity']}")
    print(f"  Pasos por escenario  : {future_mc['n_steps']:,}")

    # Assumptions
    print(f"\n  {'─'*60}")
    print("  SUPUESTOS DE LA SIMULACION")
    print(f"  {'─'*60}")
    print("  • Simulacion base: distribucion normal multivariada")
    print("  • Simulacion adicional: Student-t con colas pesadas (Phase 3)")
    print("  • Parametros derivados de datos historicos (media y covarianza)")
    print("  • De-anualizacion aritmetica (simplificacion)")
    print("  • Renta fija proyectada deterministicamente a tasa libre de riesgo")
    print("  • Comparacion con/sin rebalanceo anual (Phase 3)")
    print("  • Los resultados NO garantizan desempeno futuro")

    # Variable-Income
    s = future_mc["stats_variable_income"]
    print(f"\n  {'─'*60}")
    print(f"  RESULTADOS — RENTA VARIABLE")
    print(f"  {'─'*60}")
    print(f"  Capital inicial (RV)       : ${s['initial_value']:>14,.2f}")
    print(f"  Valor final medio          : ${s['mean']:>14,.2f}")
    print(f"  Valor final mediano        : ${s['median']:>14,.2f}")
    print(f"  Valor final minimo         : ${s['min']:>14,.2f}")
    print(f"  Valor final maximo         : ${s['max']:>14,.2f}")
    print(f"\n  Percentiles del valor final:")
    print(f"    P5  (pesimista)          : ${s['p5']:>14,.2f}")
    print(f"    P25                      : ${s['p25']:>14,.2f}")
    print(f"    P50 (mediana)            : ${s['p50']:>14,.2f}")
    print(f"    P75                      : ${s['p75']:>14,.2f}")
    print(f"    P95 (optimista)          : ${s['p95']:>14,.2f}")
    print(f"\n  Rendimiento anualizado simulado:")
    print(f"    Medio                    : {s['mean_ann_return']*100:>8.2f} %")
    print(f"    Mediano                  : {s['median_ann_return']*100:>8.2f} %")
    print(f"\n  Probabilidad de perdida (RV): {s['prob_loss']:>6.1f} %")
    if s.get("prob_target") is not None:
        print(f"  Prob. de alcanzar objetivo : {s['prob_target']:>6.1f} %")
        print(f"  (Objetivo: {s['target_return']*100:.2f}% anual "
              f"-> ${s['target_value']:,.2f})")

    # Total Portfolio (if FI exists)
    st = future_mc.get("stats_total_portfolio")
    if st is not None:
        print(f"\n  {'─'*60}")
        print(f"  RESULTADOS — PORTAFOLIO TOTAL (RV + RF)")
        print(f"  {'─'*60}")
        print(f"  Capital inicial total      : ${st['initial_value']:>14,.2f}")
        print(f"  Valor final medio          : ${st['mean']:>14,.2f}")
        print(f"  Valor final mediano        : ${st['median']:>14,.2f}")
        print(f"  Valor final minimo         : ${st['min']:>14,.2f}")
        print(f"  Valor final maximo         : ${st['max']:>14,.2f}")
        print(f"\n  Percentiles del valor final total:")
        print(f"    P5  (pesimista)          : ${st['p5']:>14,.2f}")
        print(f"    P50 (mediana)            : ${st['p50']:>14,.2f}")
        print(f"    P95 (optimista)          : ${st['p95']:>14,.2f}")
        print(f"\n  Probabilidad de perdida total: {st['prob_loss']:>6.1f} %")
        if st.get("prob_target") is not None:
            print(f"  Prob. de alcanzar objetivo : {st['prob_target']:>6.1f} %")

    # Drawdown
    dd = future_mc["drawdown_stats"]
    print(f"\n  {'─'*60}")
    print(f"  DRAWDOWN SIMULADO")
    print(f"  {'─'*60}")
    print(f"  Max drawdown promedio      : {dd['avg_max_dd']*100:>8.2f} %")
    print(f"  Max drawdown peor caso     : {dd['worst_max_dd']*100:>8.2f} %")
    print(f"  Max drawdown P5 (severo)   : {dd['p5_max_dd']*100:>8.2f} %")
    print(f"  Max drawdown P50 (mediano) : {dd['p50_max_dd']*100:>8.2f} %")
    print(f"  Max drawdown P95 (leve)    : {dd['p95_max_dd']*100:>8.2f} %")

    # Interpretation
    interp = future_mc.get("interpretation", [])
    if interp:
        print(f"\n  {'─'*60}")
        print(f"  INTERPRETACION")
        print(f"  {'─'*60}")
        for line in interp:
            print(f"  -> {line}")
    print(f"\n{SEP}\n")

    # ── [NEW — PHASE 3] Fat-tail vs Normal comparison ────────────────────
    stats_ft = future_mc.get("stats_fat_tail")
    stats_normal = future_mc.get("stats_variable_income")
    if stats_ft and stats_normal:
        display_fat_tail_comparison(
            stats_normal, stats_ft, future_mc.get("fat_tail_df", DEFAULT_T_DF))

    # ── [NEW — PHASE 3] Rebalancing comparison ──────────────────────────
    stats_rebal = future_mc.get("stats_rebalanced")
    stats_no_rebal = future_mc.get("stats_no_rebalanced")
    if stats_rebal and stats_no_rebal:
        display_rebalancing_comparison(stats_no_rebal, stats_rebal, "annual")


def plot_monte_carlo_paths(future_mc, cfg):
    """Plot subset of simulated portfolio paths — standalone figure."""
    if future_mc.get("total_portfolio_paths") is not None:
        all_paths = future_mc["total_portfolio_paths"]
        title_lbl = "Portafolio Total (RV + RF)"
        init_val = future_mc["stats_total_portfolio"]["initial_value"]
    else:
        all_paths = future_mc["variable_income_paths"]
        title_lbl = "Renta Variable"
        init_val = future_mc["stats_variable_income"]["initial_value"]

    C_DARK, C_GREEN, C_RED = "#1D3557", "#2A9D8F", "#E63946"
    BG = "#FAFAFA"
    n_sims, n_cols = all_paths.shape
    n_plot = min(N_PLOT_PATHS, n_sims)
    time_ax = np.arange(n_cols) / cfg["annualize"]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)

    idxs = np.random.choice(n_sims, size=n_plot, replace=False)
    for i in idxs:
        ax.plot(time_ax, all_paths[i], color=C_DARK, alpha=0.08, linewidth=0.5)

    p5  = np.percentile(all_paths, 5,  axis=0)
    p25 = np.percentile(all_paths, 25, axis=0)
    p50 = np.percentile(all_paths, 50, axis=0)
    p75 = np.percentile(all_paths, 75, axis=0)
    p95 = np.percentile(all_paths, 95, axis=0)

    ax.fill_between(time_ax, p5, p95, alpha=0.15, color=C_GREEN, label="P5–P95")
    ax.fill_between(time_ax, p25, p75, alpha=0.25, color=C_GREEN, label="P25–P75")
    ax.plot(time_ax, p50, color=C_GREEN, linewidth=2.5,
            label="Mediana (P50)", zorder=4)
    ax.axhline(init_val, color=C_RED, linewidth=1.2, linestyle="--",
               alpha=0.7, label="Capital inicial")

    ax.set_title(
        f"Simulacion Monte Carlo: Trayectorias del {title_lbl}\n"
        f"{N_FUTURE_SIMULATIONS:,} escenarios · Horizonte: "
        f"{cfg['horizon_years']} anos",
        fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Tiempo (anos)", fontsize=10)
    ax.set_ylabel("Valor del Portafolio ($)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.20, linestyle="--")
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    path = os.path.join(OUTPUT_DIR, "monte_carlo_trajectories.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  [OK] Trayectorias Monte Carlo -> {path}")
    plt.show()
    plt.close()


def plot_monte_carlo_terminal_distribution(future_mc, cfg):
    """Histogram of simulated final portfolio values — standalone figure."""
    C_DARK, C_GREEN, C_RED = "#1D3557", "#2A9D8F", "#E63946"
    C_BLUE, C_GOLD, BG = "#457B9D", "#E9C46A", "#FAFAFA"

    if future_mc.get("stats_total_portfolio") is not None:
        stats = future_mc["stats_total_portfolio"]
        fv = future_mc["final_values_total_portfolio"]
        title_lbl = "Portafolio Total (RV + RF)"
    else:
        stats = future_mc["stats_variable_income"]
        fv = future_mc["final_values_variable_income"]
        title_lbl = "Renta Variable"

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)

    n_bins = max(min(int(1 + 3.322 * np.log10(max(len(fv), 1))), 80), 25)
    ax.hist(fv, bins=n_bins, alpha=0.65, color=C_GREEN,
            edgecolor="white", linewidth=0.4, density=True)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(fv)
        xr = np.linspace(fv.min() * 0.9, fv.max() * 1.05, 300)
        ax.plot(xr, kde(xr), color=C_GREEN, linewidth=2.0, alpha=0.9)
    except Exception:
        pass

    ax.axvline(stats["mean"], color=C_BLUE, linewidth=2.0, linestyle="--",
               alpha=0.9, label=f"Media: ${stats['mean']:,.0f}")
    ax.axvline(stats["median"], color=C_DARK, linewidth=2.0, linestyle="-.",
               alpha=0.9, label=f"Mediana: ${stats['median']:,.0f}")
    ax.axvline(stats["p5"], color=C_RED, linewidth=1.5, linestyle=":",
               alpha=0.8, label=f"P5: ${stats['p5']:,.0f}")
    ax.axvline(stats["p95"], color=C_GOLD, linewidth=1.5, linestyle=":",
               alpha=0.8, label=f"P95: ${stats['p95']:,.0f}")
    ax.axvline(stats["initial_value"], color="black", linewidth=1.5,
               linestyle="-", alpha=0.6,
               label=f"Capital inicial: ${stats['initial_value']:,.0f}")

    ax.set_title(
        f"Distribucion del Valor Final del {title_lbl}\n"
        f"{N_FUTURE_SIMULATIONS:,} escenarios · Horizonte: "
        f"{cfg['horizon_years']} anos",
        fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Valor Final del Portafolio ($)", fontsize=10)
    ax.set_ylabel("Densidad", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.20, linestyle="--")
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    ax.text(0.02, 0.95,
            f"Prob. perdida: {stats['prob_loss']:.1f}%\n"
            f"Rend. anual medio: {stats['mean_ann_return']*100:.2f}%\n"
            f"Rango P5-P95: ${stats['p5']:,.0f} – ${stats['p95']:,.0f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            color=C_DARK, bbox=dict(boxstyle="round,pad=0.4",
                                    facecolor="white", edgecolor="#DDDDDD",
                                    alpha=0.9))

    path = os.path.join(OUTPUT_DIR, "monte_carlo_distribution.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  [OK] Distribucion Monte Carlo -> {path}")
    plt.show()
    plt.close()


def plot_monte_carlo_percentile_bands(future_mc, cfg):
    """Plot P5/P25/P50/P75/P95 bands through time — standalone figure."""
    C_DARK, C_GREEN, C_RED = "#1D3557", "#2A9D8F", "#E63946"
    BG = "#FAFAFA"

    if future_mc.get("total_portfolio_paths") is not None:
        paths = future_mc["total_portfolio_paths"]
        title_lbl = "Portafolio Total"
        init_val = future_mc["stats_total_portfolio"]["initial_value"]
    else:
        paths = future_mc["variable_income_paths"]
        title_lbl = "Renta Variable"
        init_val = future_mc["stats_variable_income"]["initial_value"]

    n_steps = paths.shape[1] - 1
    time_ax = np.arange(n_steps + 1) / cfg["annualize"]

    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)

    ax.fill_between(time_ax, p5, p95, alpha=0.15, color=C_GREEN,
                     label="P5 – P95")
    ax.fill_between(time_ax, p25, p75, alpha=0.30, color=C_GREEN,
                     label="P25 – P75")
    ax.plot(time_ax, p50, color=C_GREEN, linewidth=2.5,
            label="Mediana (P50)", zorder=4)
    ax.plot(time_ax, p5, color=C_RED, linewidth=1.0, linestyle=":",
            alpha=0.7, label="P5")
    ax.plot(time_ax, p95, color=C_DARK, linewidth=1.0, linestyle=":",
            alpha=0.7, label="P95")
    ax.axhline(init_val, color="black", linewidth=1.2, linestyle="--",
               alpha=0.5, label="Capital inicial")

    ax.set_title(
        f"Bandas de Percentiles — {title_lbl}\n"
        f"{N_FUTURE_SIMULATIONS:,} escenarios · Horizonte: "
        f"{cfg['horizon_years']} anos",
        fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Tiempo (anos)", fontsize=10)
    ax.set_ylabel("Valor del Portafolio ($)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.20, linestyle="--")
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    path = os.path.join(OUTPUT_DIR, "monte_carlo_percentile_bands.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  [OK] Bandas de percentiles -> {path}")
    plt.show()
    plt.close()


def run_future_monte_carlo(cfg, returns, prim_w, prim_label, n_obs):
    """
    Orchestrator: runs the complete future Monte Carlo simulation.
    Returns a dict stored under analysis['future_monte_carlo'].
    """
    print(f"\n{'─'*65}")
    print("  EJECUTANDO SIMULACION MONTE CARLO DE ESCENARIOS FUTUROS...")
    print(f"{'─'*65}")
    print(f"  Portafolio base       : {prim_label}")
    print(f"  Escenarios            : {N_FUTURE_SIMULATIONS:,}")
    print(f"  Horizonte             : {cfg['horizon_years']} anos")
    print(f"  Periodicidad          : {cfg['periodicity']}")

    af = cfg["annualize"]
    hy = cfg["horizon_years"]
    monto_rv = cfg["monto_total"] * cfg["pct_variable"] / 100.0
    monto_fi = cfg["monto_total"] * cfg["pct_fixed"] / 100.0
    rf = cfg["risk_free"]
    target_ret = cfg.get("target_return")
    n_steps = hy * af

    mean_ann = returns.mean() * af
    cov_ann = returns.cov() * af

    # Step 1: Simulate variable-income paths
    print(f"  Simulando {N_FUTURE_SIMULATIONS:,} trayectorias de RV...")
    vi = simulate_future_portfolio_paths(
        prim_w, mean_ann, cov_ann, af, hy, monto_rv, N_FUTURE_SIMULATIONS)
    vi_paths = vi["paths"]
    fv_vi = vi_paths[:, -1]

    # Step 2: Project fixed-income path
    fi_rate = FIXED_INCOME_CUSTOM_RATE if FIXED_INCOME_CUSTOM_RATE else rf
    fi_path = project_fixed_income_path(monto_fi, fi_rate, af, hy)

    # Step 3: Combine total portfolio
    has_fi = cfg["pct_fixed"] > 0 and monto_fi > 0
    if has_fi:
        total_paths = combine_total_portfolio_paths(vi_paths, fi_path)
        fv_total = total_paths[:, -1]
        print(f"  RF proyectada deterministicamente ({fi_rate*100:.2f}% anual)")
    else:
        total_paths = None
        fv_total = None

    # Step 4: Statistics
    print("  Calculando estadisticas de simulacion...")
    stats_vi = calculate_simulation_statistics(
        fv_vi, monto_rv, target_ret, hy, "Renta Variable")
    stats_total = None
    if has_fi:
        stats_total = calculate_simulation_statistics(
            fv_total, cfg["monto_total"], target_ret, hy, "Portafolio Total")

    # Step 5: Drawdowns (on total if available, else VI)
    dd_paths = total_paths if total_paths is not None else vi_paths
    dd_stats = calculate_simulated_drawdowns(dd_paths)

    # Step 6: Interpretation
    interp = _generate_simulation_interpretation(
        stats_vi, stats_total, dd_stats, n_obs, hy)

    print("  [OK] Simulacion normal completada.")

    # ── [NEW — PHASE 3] Fat-tail simulation ──────────────────────────────
    print(f"  Simulando trayectorias con colas pesadas (Student-t, df={DEFAULT_T_DF})...")
    ft = simulate_future_paths_fat_tail(
        prim_w, mean_ann, cov_ann, af, hy, monto_rv,
        N_FUTURE_SIMULATIONS, df=DEFAULT_T_DF)
    ft_paths = ft["paths"]
    fv_ft = ft_paths[:, -1]
    stats_ft = calculate_simulation_statistics(
        fv_ft, monto_rv, target_ret, hy, "Fat-Tail RV")

    # ── [NEW — PHASE 3] Rebalancing simulation ──────────────────────────
    print("  Simulando trayectorias con rebalanceo anual...")
    rebal = simulate_with_rebalancing(
        prim_w, mean_ann, cov_ann, af, hy, monto_rv,
        rebal_freq="annual", n_sims=N_FUTURE_SIMULATIONS)
    rebal_paths = rebal["paths"]
    fv_rebal = rebal_paths[:, -1]
    stats_rebal = calculate_simulation_statistics(
        fv_rebal, monto_rv, target_ret, hy, "Rebalanceo Anual")

    # No-rebalancing stats (same as normal VI for comparison)
    stats_no_rebal = stats_vi

    print("  [OK] Todas las simulaciones completadas.")
    return {
        "variable_income_paths": vi_paths,
        "fixed_income_path": fi_path,
        "total_portfolio_paths": total_paths,
        "final_values_variable_income": fv_vi,
        "final_values_total_portfolio": fv_total,
        "stats_variable_income": stats_vi,
        "stats_total_portfolio": stats_total,
        "drawdown_stats": dd_stats,
        "interpretation": interp,
        "n_steps": n_steps,
        "prim_label": prim_label,
        # [NEW — PHASE 3] additional simulation results
        "stats_fat_tail": stats_ft,
        "fat_tail_df": DEFAULT_T_DF,
        "stats_rebalanced": stats_rebal,
        "stats_no_rebalanced": stats_no_rebal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16B — [NEW — PHASE 3] ADVANCED MODELING
# ─────────────────────────────────────────────────────────────────────────────
# This section adds:
#   1. Fat-tail Monte Carlo (Student-t distribution)
#   2. Rebalancing inside simulation
#   3. Bootstrap weight sensitivity analysis
#   4. Scenario-based optimization framework
# ─────────────────────────────────────────────────────────────────────────────

def simulate_future_paths_fat_tail(weights, mean_returns_ann, cov_matrix_ann,
                                    annualize_factor, horizon_years,
                                    initial_value, n_sims=N_FUTURE_SIMULATIONS,
                                    df=DEFAULT_T_DF):
    """
    [NEW — PHASE 3] Simulate future portfolio paths using Student-t returns.

    WHY FAT TAILS?
    Financial returns exhibit "fat tails" — extreme events (crashes, spikes)
    occur MORE frequently than a normal distribution predicts. The Student-t
    distribution captures this by having heavier tails controlled by the
    degrees-of-freedom parameter (df):
      - df = 3-5:  very heavy tails (realistic for equities)
      - df = 10:   moderate tails
      - df = 30+:  approaches normal distribution

    METHOD:
    1. Compute Cholesky decomposition of the covariance matrix
    2. Generate independent Student-t random variables
    3. Apply Cholesky to introduce correlations
    4. Scale to match desired mean/variance

    This preserves the correlation structure while making individual
    returns heavier-tailed than a multivariate normal.
    """
    n_assets = len(weights)
    n_steps = horizon_years * annualize_factor

    # De-annualize
    period_mean = np.array(mean_returns_ann) / annualize_factor
    cov_vals = cov_matrix_ann.values if hasattr(cov_matrix_ann, 'values') \
        else cov_matrix_ann
    period_cov = cov_vals / annualize_factor

    # Cholesky decomposition — lower triangular matrix L such that L @ L.T = cov
    try:
        L = np.linalg.cholesky(period_cov)
    except np.linalg.LinAlgError:
        # If cov is not positive definite, add small diagonal perturbation
        eps = 1e-8 * np.eye(n_assets)
        L = np.linalg.cholesky(period_cov + eps)

    # Generate independent Student-t samples: shape (n_sims, n_steps, n_assets)
    # Student-t variance = df/(df-2) for df>2, so we scale to unit variance
    t_samples = t_dist.rvs(df=df, size=(n_sims, n_steps, n_assets))
    scale_factor = np.sqrt((df - 2) / df) if df > 2 else 1.0
    t_samples *= scale_factor

    # Apply Cholesky to introduce correlations: Z @ L.T + mean
    # shape: (n_sims, n_steps, n_assets) @ (n_assets, n_assets) → same shape
    sim_asset_rets = t_samples @ L.T + period_mean[np.newaxis, np.newaxis, :]

    # Portfolio returns per period
    port_rets = sim_asset_rets @ weights

    # Cumulative value paths
    cum_factors = np.cumprod(1 + port_rets, axis=1)
    paths = np.hstack([np.ones((n_sims, 1)), cum_factors]) * initial_value

    return {"paths": paths, "returns": port_rets, "n_steps": n_steps,
            "df": df, "distribution": "Student-t"}


def simulate_with_rebalancing(weights, mean_returns_ann, cov_matrix_ann,
                               annualize_factor, horizon_years,
                               initial_value, rebal_freq="annual",
                               n_sims=N_FUTURE_SIMULATIONS,
                               use_fat_tails=False, df=DEFAULT_T_DF):
    """
    [NEW — PHASE 3] Simulate portfolio paths WITH periodic rebalancing.

    Without rebalancing, winning assets grow in weight and losing assets
    shrink, causing the portfolio to DRIFT away from the target allocation.
    With rebalancing, weights are reset to the target at fixed intervals.

    Args:
        rebal_freq: 'annual', 'quarterly', or 'none'
        use_fat_tails: if True, use Student-t returns instead of normal
    """
    n_assets = len(weights)
    n_steps = horizon_years * annualize_factor
    rebal_period = REBALANCE_FREQ_MAP.get(rebal_freq)

    # De-annualize
    period_mean = np.array(mean_returns_ann) / annualize_factor
    cov_vals = cov_matrix_ann.values if hasattr(cov_matrix_ann, 'values') \
        else cov_matrix_ann
    period_cov = cov_vals / annualize_factor

    # Generate returns
    if use_fat_tails:
        try:
            L = np.linalg.cholesky(period_cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(period_cov + 1e-8 * np.eye(n_assets))
        t_samples = t_dist.rvs(df=df, size=(n_sims, n_steps, n_assets))
        scale_factor = np.sqrt((df - 2) / df) if df > 2 else 1.0
        t_samples *= scale_factor
        sim_asset_rets = t_samples @ L.T + period_mean[np.newaxis, np.newaxis, :]
    else:
        sim_asset_rets = np.random.multivariate_normal(
            mean=period_mean, cov=period_cov, size=(n_sims, n_steps)
        )

    # Simulate with rebalancing
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = initial_value

    if rebal_period is None:
        # No rebalancing — static weights, simple compound
        port_rets = sim_asset_rets @ weights
        cum_factors = np.cumprod(1 + port_rets, axis=1)
        paths[:, 1:] = cum_factors * initial_value
    else:
        # Rebalancing: track per-asset values, reset weights periodically
        # asset_values: (n_sims, n_assets) — current dollar value per asset
        asset_values = np.outer(np.ones(n_sims), weights * initial_value)

        for step in range(n_steps):
            # Apply returns to each asset
            asset_values = asset_values * (1 + sim_asset_rets[:, step, :])
            # Total portfolio value
            total = asset_values.sum(axis=1)
            paths[:, step + 1] = total

            # Rebalance at fixed intervals
            if (step + 1) % rebal_period == 0:
                asset_values = np.outer(total, weights)

    return {
        "paths": paths,
        "n_steps": n_steps,
        "rebal_freq": rebal_freq,
        "use_fat_tails": use_fat_tails,
    }


def display_fat_tail_comparison(normal_stats, fat_tail_stats, df):
    """
    [NEW — PHASE 3] Display side-by-side Normal vs Fat-Tail MC comparison.
    """
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  COMPARACION: NORMAL vs COLAS PESADAS (Student-t)")
    print(SEP)
    print(f"  Grados de libertad (df): {df}")
    print(f"  (Menor df = colas mas pesadas = eventos extremos mas frecuentes)")

    n = normal_stats
    f = fat_tail_stats

    print(f"\n  {'Metrica':<30} {'Normal':>14} {'Student-t':>14} {'Difer.':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*10}")

    rows = [
        ("Valor final medio", n["mean"], f["mean"]),
        ("Valor final mediano", n["median"], f["median"]),
        ("P5 (pesimista)", n["p5"], f["p5"]),
        ("P95 (optimista)", n["p95"], f["p95"]),
        ("Prob. de perdida (%)", n["prob_loss"], f["prob_loss"]),
    ]

    for label, nv, fv in rows:
        if "Prob" in label:
            diff = fv - nv
            print(f"  {label:<30} {nv:>13.1f}% {fv:>13.1f}% {diff:>+9.1f}%")
        else:
            diff_pct = (fv - nv) / nv * 100 if nv != 0 else 0
            print(f"  {label:<30} ${nv:>13,.0f} ${fv:>13,.0f} {diff_pct:>+9.1f}%")

    # Interpretation
    p5_diff = (f["p5"] - n["p5"]) / n["p5"] * 100 if n["p5"] != 0 else 0
    print(f"\n  Interpretacion:")
    if p5_diff < -5:
        print(f"  ⚠ El escenario pesimista (P5) empeora {abs(p5_diff):.0f}% con colas pesadas.")
        print(f"    Los modelos normales SUBESTIMAN el riesgo de eventos extremos.")
        print(f"    Esto es importante para la planificacion de proteccion patrimonial.")
    elif p5_diff < -1:
        print(f"  ℹ El escenario pesimista cambia moderadamente ({p5_diff:.1f}%).")
        print(f"    Las colas pesadas tienen un impacto limitado en este portafolio.")
    else:
        print(f"  ✅ Diferencia minima entre modelos. El portafolio no es")
        print(f"     particularmente sensible a eventos de cola.")
    print()


def display_rebalancing_comparison(no_rebal_stats, rebal_stats, rebal_freq):
    """
    [NEW — PHASE 3] Display comparison of drift vs rebalanced portfolios.
    """
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  IMPACTO DEL REBALANCEO EN LA SIMULACION")
    print(SEP)
    print(f"  Frecuencia de rebalanceo: {rebal_freq}")
    print(f"  Sin rebalanceo: los pesos derivan con el mercado")
    print(f"  Con rebalanceo: los pesos se restauran periodicamente\n")

    n = no_rebal_stats
    r = rebal_stats

    print(f"  {'Metrica':<30} {'Sin Rebal.':>14} {'Con Rebal.':>14} {'Difer.':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*10}")

    rows = [
        ("Valor final medio", n["mean"], r["mean"]),
        ("Valor final mediano", n["median"], r["median"]),
        ("P5 (pesimista)", n["p5"], r["p5"]),
        ("Prob. de perdida (%)", n["prob_loss"], r["prob_loss"]),
    ]

    for label, nv, rv in rows:
        if "Prob" in label:
            diff = rv - nv
            print(f"  {label:<30} {nv:>13.1f}% {rv:>13.1f}% {diff:>+9.1f}%")
        else:
            diff_pct = (rv - nv) / nv * 100 if nv != 0 else 0
            print(f"  {label:<30} ${nv:>13,.0f} ${rv:>13,.0f} {diff_pct:>+9.1f}%")

    print(f"\n  Interpretacion:")
    p5_n = n["p5"]
    p5_r = r["p5"]
    if p5_r > p5_n * 1.02:
        print(f"  ✅ El rebalanceo mejora el escenario pesimista, reduciendo")
        print(f"     el riesgo de deriva concentrada en activos perdedores.")
    elif p5_r < p5_n * 0.98:
        print(f"  ℹ En este caso, el rebalanceo reduce ligeramente el upside.")
        print(f"    Esto puede ocurrir cuando activos ganadores siguen subiendo.")
    else:
        print(f"  ✅ Impacto del rebalanceo es menor en este portafolio.")
    print()


def bootstrap_optimal_weights(returns, bench_returns, asset_stats,
                               annualize_factor, risk_free,
                               target="sharpe", max_weight=1.0,
                               cov_override=None,
                               n_samples=N_BOOTSTRAP_SAMPLES):
    """
    [NEW — PHASE 3] Bootstrap resampling for weight sensitivity analysis.

    WHY BOOTSTRAP?
    The optimizer produces a single "optimal" solution, but that solution
    depends on the specific sample of historical returns. Small changes
    in the data can produce very different weights.

    Bootstrap resampling creates N_BOOTSTRAP_SAMPLES alternative datasets
    by sampling WITH REPLACEMENT from historical returns, then re-optimizes
    each one. The dispersion of resulting weights tells you:
      - Which weights are STABLE (low variance → robust allocation)
      - Which weights are FRAGILE (high variance → sensitive to sample)

    Returns:
        dict with weight distributions, means, stds, confidence intervals
    """
    n_obs, n_assets = returns.shape
    tickers = list(returns.columns)
    all_weights = np.zeros((n_samples, n_assets))
    successful = 0

    for i in range(n_samples):
        # Resample rows WITH replacement
        idx = np.random.choice(n_obs, size=n_obs, replace=True)
        resampled = returns.iloc[idx].reset_index(drop=True)
        resampled_bench = bench_returns.iloc[idx].reset_index(drop=True) \
            if bench_returns is not None else None

        try:
            w, *_ = optimize_portfolio(
                resampled, resampled_bench, asset_stats,
                annualize_factor, risk_free, target=target,
                max_weight=max_weight, cov_override=cov_override
            )
            all_weights[successful] = w
            successful += 1
        except Exception:
            continue  # skip failed optimizations

    if successful < 10:
        return None  # not enough successful samples

    all_weights = all_weights[:successful]

    # Compute statistics
    w_mean = np.mean(all_weights, axis=0)
    w_std = np.std(all_weights, axis=0)
    w_p5 = np.percentile(all_weights, 5, axis=0)
    w_p95 = np.percentile(all_weights, 95, axis=0)
    w_range = w_p95 - w_p5

    # Stability classification
    stability = []
    for j in range(n_assets):
        if w_std[j] < 0.03:
            stability.append("ESTABLE")
        elif w_std[j] < 0.08:
            stability.append("MODERADO")
        else:
            stability.append("FRAGIL")

    return {
        "tickers": tickers,
        "all_weights": all_weights,
        "mean": w_mean,
        "std": w_std,
        "p5": w_p5,
        "p95": w_p95,
        "range_90": w_range,
        "stability": stability,
        "n_successful": successful,
        "n_attempted": n_samples,
    }


def display_weight_sensitivity(sensitivity, original_weights):
    """
    [NEW — PHASE 3] Display weight sensitivity analysis results.
    Shows which weights are robust vs fragile.
    """
    if sensitivity is None:
        print("\n  [!] Analisis de sensibilidad no disponible (pocas muestras exitosas).\n")
        return

    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  ANALISIS DE SENSIBILIDAD / ESTABILIDAD DE PESOS")
    print(SEP)
    print(f"  Muestras bootstrap: {sensitivity['n_successful']}"
          f" / {sensitivity['n_attempted']} exitosas")
    print(f"  Los pesos 'optimos' son ESTIMACIONES, no certezas.\n")

    print(f"  {'Ticker':<8} {'Peso Orig.':>10} {'Media Boot.':>12} "
          f"{'Desv.Est.':>10} {'[P5':>8} {'P95]':>8} {'Estabilidad':<12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")

    n_fragile = 0
    for j, t in enumerate(sensitivity["tickers"]):
        icon = "✅" if sensitivity["stability"][j] == "ESTABLE" \
            else "🟡" if sensitivity["stability"][j] == "MODERADO" else "🔴"
        if sensitivity["stability"][j] == "FRAGIL":
            n_fragile += 1
        print(f"  {t:<8} {original_weights[j]*100:>9.1f}% "
              f"{sensitivity['mean'][j]*100:>11.1f}% "
              f"{sensitivity['std'][j]*100:>9.1f}% "
              f"{sensitivity['p5'][j]*100:>7.1f}% "
              f"{sensitivity['p95'][j]*100:>7.1f}%  "
              f"{icon} {sensitivity['stability'][j]}")

    # Summary interpretation
    print(f"\n  {'─'*70}")
    if n_fragile == 0:
        print(f"  ✅ Todos los pesos son razonablemente estables.")
        print(f"     La optimizacion es robusta a cambios en la muestra.")
    elif n_fragile <= 2:
        print(f"  🟡 {n_fragile} activo(s) tienen pesos fragiles.")
        print(f"     Estos pesos cambian significativamente con datos diferentes.")
        print(f"     Considera fijar estos pesos manualmente o usar restricciones.")
    else:
        print(f"  🔴 {n_fragile} activos tienen pesos fragiles.")
        print(f"     La optimizacion es MUY sensible a la muestra de datos.")
        print(f"     Los pesos 'optimos' no deben tomarse como definitivos.")
        print(f"     Considera usar la covarianza shrunk (Ledoit-Wolf) y/o")
        print(f"     restricciones mas estrictas de peso maximo.")
    print()


def scenario_based_optimization(returns, bench_returns, asset_stats,
                                 annualize_factor, risk_free,
                                 target="sharpe", max_weight=1.0,
                                 cov_override=None):
    """
    [NEW — PHASE 3] Optimize under different macro scenarios.

    For each scenario in MACRO_SCENARIOS, we:
      1. Adjust expected returns (multiply by return_adj factor)
      2. Adjust covariance (multiply by vol_adj^2 factor)
      3. Re-optimize to find new optimal weights
      4. Compare how the allocation changes

    This helps the investor understand:
      - How robust is the recommended allocation across scenarios?
      - How would a recession or inflation change the optimal portfolio?

    Returns:
        list of dicts, one per scenario, with weights and metrics
    """
    n_assets = returns.shape[1]
    tickers = list(returns.columns)
    mean_ann = returns.mean() * annualize_factor
    cov_ann = cov_override if cov_override is not None \
        else returns.cov() * annualize_factor

    results = []

    for scenario_name, params in MACRO_SCENARIOS.items():
        adj_mean = mean_ann * params["return_adj"]
        adj_cov = cov_ann * (params["vol_adj"] ** 2)

        # Build a modified returns-like input for the optimizer
        # Instead of passing modified returns, we override cov and use adjusted means
        try:
            # Create a temp optimizer call with adjusted params
            n = n_assets
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            bounds = tuple((0.0, min(max_weight, 1.0)) for _ in range(n))
            w0 = np.array([1 / n] * n)

            cov_vals = adj_cov.values if hasattr(adj_cov, 'values') else adj_cov
            adj_mean_vals = adj_mean.values if hasattr(adj_mean, 'values') else adj_mean

            if target == "sharpe":
                def neg_sharpe(w):
                    r = np.dot(w, adj_mean_vals)
                    v = np.sqrt(w @ cov_vals @ w)
                    return -(r - risk_free) / v if v > 0 else 0
                result = minimize(neg_sharpe, w0, method="SLSQP",
                                  bounds=bounds, constraints=constraints,
                                  options={"maxiter": 1000, "ftol": 1e-9})
            else:
                def portfolio_vol(w):
                    return np.sqrt(w @ cov_vals @ w)
                result = minimize(portfolio_vol, w0, method="SLSQP",
                                  bounds=bounds, constraints=constraints,
                                  options={"maxiter": 1000, "ftol": 1e-9})

            w_opt = result.x if result.success else w0
            opt_ret = np.dot(w_opt, adj_mean_vals)
            opt_vol = np.sqrt(w_opt @ cov_vals @ w_opt)
            opt_sr = (opt_ret - risk_free) / opt_vol if opt_vol > 0 else 0

        except Exception:
            w_opt = np.array([1 / n_assets] * n_assets)
            opt_ret, opt_vol, opt_sr = 0, 0, 0

        results.append({
            "scenario": scenario_name,
            "description": params["description"],
            "return_adj": params["return_adj"],
            "vol_adj": params["vol_adj"],
            "weights": w_opt,
            "expected_return": opt_ret,
            "expected_vol": opt_vol,
            "sharpe": opt_sr,
        })

    return {"tickers": tickers, "scenarios": results}


def display_scenario_optimization(scenario_results, original_weights):
    """
    [NEW — PHASE 3] Display scenario-based optimization comparison.
    """
    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  OPTIMIZACION POR ESCENARIOS MACROECONOMICOS")
    print(f"  (Experimental — muestra como cambian los pesos bajo distintas condiciones)")
    print(SEP)

    tickers = scenario_results["tickers"]
    scenarios = scenario_results["scenarios"]

    # Header
    header = f"  {'Ticker':<8}"
    for s in scenarios:
        header += f" {s['scenario'][:12]:>12}"
    print(f"\n{header}")
    print(f"  {'-'*8}" + f" {'-'*12}" * len(scenarios))

    # Weight rows
    for j, t in enumerate(tickers):
        row = f"  {t:<8}"
        for s in scenarios:
            row += f" {s['weights'][j]*100:>11.1f}%"
        print(row)

    # Metrics rows
    print(f"\n  {'Metrica':<8}" + f" {'-'*12}" * len(scenarios))
    row_ret = f"  {'Rend.':<8}"
    row_vol = f"  {'Vol.':<8}"
    row_sr = f"  {'Sharpe':<8}"
    for s in scenarios:
        row_ret += f" {s['expected_return']*100:>11.2f}%"
        row_vol += f" {s['expected_vol']*100:>11.2f}%"
        row_sr += f" {s['sharpe']:>12.4f}"
    print(row_ret)
    print(row_vol)
    print(row_sr)

    # Stability check: how much do weights change across scenarios?
    all_weights = np.array([s["weights"] for s in scenarios])
    max_change = np.max(np.std(all_weights, axis=0))

    print(f"\n  {'─'*70}")
    print(f"  Interpretacion:")
    if max_change < 0.05:
        print(f"  ✅ Los pesos optimos son bastante estables entre escenarios.")
        print(f"     La asignacion recomendada es robusta a cambios macroeconomicos.")
    elif max_change < 0.12:
        print(f"  🟡 Algunos pesos cambian moderadamente entre escenarios.")
        print(f"     La asignacion base es razonable, pero podria necesitar")
        print(f"     ajustes si las condiciones macroeconomicas cambian.")
    else:
        print(f"  🔴 Los pesos cambian significativamente entre escenarios.")
        print(f"     La asignacion optima depende fuertemente de las")
        print(f"     condiciones macroeconomicas asumidas. Esto sugiere")
        print(f"     que la diversificacion amplia es mas importante que")
        print(f"     la optimizacion precisa.")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def display_results(cfg, valid_tickers, asset_stats,
                    ms_w, ms_ret, ms_vol, ms_sr,
                    mv_w, mv_ret, mv_vol, mv_sr,
                    bench_metrics, benchmark, insights):
    """
    Print structured analysis results in layers.
    Profile/allocation info is NOT repeated here (already shown in summary).
    Now includes a critical analysis section powered by insights.
    """
    SEP = "=" * 75
    monto_rv = cfg['monto_total'] * cfg['pct_variable'] / 100

    print(f"\n{SEP}")
    print("  RESULTADOS DEL ANALISIS")
    print(SEP)

    # ── A. Asset Statistics ───────────────────────────────────────────────
    print(f"\n{'_'*95}")
    print("  A. ESTADISTICAS INDIVIDUALES POR ACTIVO (anualizadas)")
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

    # ── B. Optimal Portfolios ─────────────────────────────────────────────
    print(f"\n{'_'*75}")
    print(f"  B. PORTAFOLIO OPTIMO — Maximo Sharpe")
    print(f"{'_'*75}")
    print(f"  Rendimiento Anual Esperado : {ms_ret*100:>8.2f} %")
    print(f"  Volatilidad Anual Esperada : {ms_vol*100:>8.2f} %")
    print(f"  Ratio de Sharpe            : {ms_sr:>8.4f}")
    print("\n  Pesos de los activos (sobre la porcion de Renta Variable):")
    for t, w in zip(valid_tickers, ms_w):
        if w > 0.001:
            amt = w * monto_rv
            print(f"    {t:<8}: {w*100:>6.2f} %  ->  ${amt:,.2f}")
    print()
    print("  💡 Busca la mejor relacion rendimiento/riesgo.")

    print(f"\n{'_'*75}")
    print(f"  B. PORTAFOLIO OPTIMO — Minima Varianza")
    print(f"{'_'*75}")
    print(f"  Rendimiento Anual Esperado : {mv_ret*100:>8.2f} %")
    print(f"  Volatilidad Anual Esperada : {mv_vol*100:>8.2f} %")
    print(f"  Ratio de Sharpe            : {mv_sr:>8.4f}")
    print("\n  Pesos de los activos (sobre la porcion de Renta Variable):")
    for t, w in zip(valid_tickers, mv_w):
        if w > 0.001:
            amt = w * monto_rv
            print(f"    {t:<8}: {w*100:>6.2f} %  ->  ${amt:,.2f}")
    print()
    if insights["mv_below_rf"]:
        print("  ⚠ CUIDADO: El rendimiento esperado es inferior a la tasa libre de riesgo.")
        print("    Este portafolio minimiza volatilidad absoluta, pero es ineficiente.")
    else:
        print("  💡 Prioriza estabilidad. Ideal para perfiles conservadores.")

    # ── C. Benchmark Comparison ───────────────────────────────────────────
    print(f"\n{'_'*75}")
    print(f"  C. BENCHMARK: {benchmark} (anualizado)")
    print(f"{'_'*75}")
    print(f"  Rendimiento       : {bench_metrics['return']*100:>8.2f} %")
    print(f"  Volatilidad       : {bench_metrics['volatility']*100:>8.2f} %")
    print(f"  Sharpe            : {bench_metrics['sharpe']:>8.4f}")
    print(f"  Retorno acumulado : {bench_metrics['cumulative_return']*100:>8.2f} %")
    print(f"  Max Drawdown      : {bench_metrics['max_drawdown']*100:>8.2f} %")

    # ── D. Critical Analysis ──────────────────────────────────────────────
    print(f"\n{'_'*75}")
    print(f"  D. ANALISIS CRITICO")
    print(f"{'_'*75}")

    # 1. Diversification
    div_class = insights["diversification"]["classification"]
    print(f"  \u25b6 Diversificacion: {div_class.upper()}")
    print(f"    {insights['diversification']['explanation']}")

    # 2. Benchmark tradeoff (Part 4 — full comparison table)
    print(f"\n  \u25b6 Comparacion vs Benchmark ({benchmark}):")
    print(f"    {'Metrica':<24} {'Portafolio':>12} {'Benchmark':>12} {'Resultado':>14}")
    print(f"    {'-'*24} {'-'*12} {'-'*12} {'-'*14}")
    b_met = bench_metrics
    ret_icon = '\u2705' if ms_ret > b_met['return'] else '\u274c'
    vol_icon = '\u2705' if ms_vol < b_met['volatility'] else '\u274c'
    sr_icon  = '\u2705' if ms_sr > b_met['sharpe'] else '\u274c'
    print(f"    {'Rendimiento':<24} {ms_ret*100:>11.2f}% {b_met['return']*100:>11.2f}% {ret_icon:>14}")
    print(f"    {'Volatilidad':<24} {ms_vol*100:>11.2f}% {b_met['volatility']*100:>11.2f}% {vol_icon:>14}")
    print(f"    {'Sharpe':<24} {ms_sr:>12.4f} {b_met['sharpe']:>12.4f} {sr_icon:>14}")
    print(f"\n    {insights['benchmark_tradeoff']['tradeoff_text']}")

    # 3. Profile consistency (Part 2)
    if not insights["profile_consistency"]["aligned"]:
        print("\n  \u25b6 Consistencia Perfil vs Portafolio:")
        for w in insights["profile_consistency"]["warnings"]:
            print(f"    \u26a0 {w}")

    # 4. Objective alignment (Part 3)
    if not insights["objective_alignment"]["aligned"]:
        print("\n  \u25b6 Alineacion Objetivo vs Activos:")
        for w in insights["objective_alignment"]["warnings"]:
            print(f"    \u26a0 {w}")

    # 5. Data quality warnings
    if insights["data_quality"]["warnings"]:
        print("\n  \u25b6 Calidad de datos historicos:")
        for w in insights["data_quality"]["warnings"]:
            print(f"    \u26a0 {w}")

    # 6. Concentration
    has_conc = False
    if insights["concentration_ms"]["is_concentrated"]:
        has_conc = True
        print("\n  \u25b6 Riesgo de Concentracion:")
        for w in insights["concentration_ms"]["warnings"]:
            print(f"    {w}")
    if insights["concentration_mv"]["is_concentrated"]:
        if not has_conc:
            print("\n  \u25b6 Riesgo de Concentracion:")
        for w in insights["concentration_mv"]["warnings"]:
            print(f"    {w}")

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
# SECTION 16B — [NEW — PHASE 1] PORTFOLIO COMPARISON & RISK EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compare_portfolios(cfg, valid_tickers,
                       rec_w, rec_ret, rec_vol, rec_sr, rec_dd,
                       user_w, user_ret, user_vol, user_sr, user_dd,
                       bench_metrics):
    """
    [NEW — PHASE 1] Display a side-by-side comparison between the
    recommended portfolio (based on profile) and the user-selected portfolio.
    Highlights differences in risk, return, and drawdown with assessment.
    """
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  COMPARACION: PORTAFOLIO RECOMENDADO vs. TU SELECCION")
    print(SEP)

    # Table header
    print(f"\n  {'Metrica':<30} {'Recomendado':>14} {'Tu seleccion':>14} {'Diferencia':>12}")
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*12}")

    # Return
    ret_diff = user_ret - rec_ret
    ret_icon = "↑" if ret_diff > 0.001 else ("↓" if ret_diff < -0.001 else "=")
    print(f"  {'Rendimiento anual':<30} {rec_ret*100:>13.2f}% {user_ret*100:>13.2f}% {ret_diff*100:>+10.2f}% {ret_icon}")

    # Volatility
    vol_diff = user_vol - rec_vol
    vol_icon = "⚠" if vol_diff > 0.02 else ("✓" if vol_diff < -0.001 else "=")
    print(f"  {'Volatilidad anual':<30} {rec_vol*100:>13.2f}% {user_vol*100:>13.2f}% {vol_diff*100:>+10.2f}% {vol_icon}")

    # Sharpe
    sr_diff = user_sr - rec_sr
    sr_icon = "✓" if sr_diff > 0.05 else ("⚠" if sr_diff < -0.05 else "=")
    print(f"  {'Ratio de Sharpe':<30} {rec_sr:>14.4f} {user_sr:>14.4f} {sr_diff:>+12.4f} {sr_icon}")

    # Drawdown
    dd_diff = user_dd - rec_dd
    dd_icon = "⚠" if dd_diff < -0.05 else ("✓" if dd_diff > 0.01 else "=")
    print(f"  {'Max Drawdown':<30} {rec_dd*100:>13.2f}% {user_dd*100:>13.2f}% {dd_diff*100:>+10.2f}% {dd_icon}")

    # Risk assessment
    print(f"\n  {'─'*60}")
    print("  EVALUACION DE RIESGO:")

    if vol_diff > 0.05:
        print(f"\n  🔴 Tu portafolio es SIGNIFICATIVAMENTE mas volatil que el")
        print(f"     recomendado (+{vol_diff*100:.1f}% volatilidad adicional).")
        print(f"     Esto incrementa la probabilidad de perdidas grandes.")
    elif vol_diff > 0.02:
        print(f"\n  🟡 Tu portafolio es moderadamente mas riesgoso que el")
        print(f"     recomendado (+{vol_diff*100:.1f}% volatilidad adicional).")
    elif vol_diff < -0.02:
        print(f"\n  ✅ Tu portafolio es menos riesgoso que el recomendado.")
        print(f"     Podrias estar sacrificando rendimiento innecesariamente.")
    else:
        print(f"\n  ✅ Tu portafolio tiene un nivel de riesgo similar al recomendado.")

    # Efficiency comparison
    if sr_diff < -0.1 and vol_diff > 0:
        print(f"\n  ⚠ EFICIENCIA: Tu portafolio toma MAS riesgo pero obtiene")
        print(f"    MENOS rendimiento ajustado por riesgo (Sharpe inferior).")
        print(f"    El riesgo adicional NO esta siendo compensado proporcionalmente.")
    elif sr_diff > 0.05 and vol_diff > 0:
        print(f"\n  ℹ Tu portafolio toma mas riesgo y logra mejor eficiencia")
        print(f"    (Sharpe superior). El riesgo adicional esta siendo compensado,")
        print(f"    pero verifica que puedes tolerar la volatilidad extra.")

    # Weight comparison (top differences)
    if len(valid_tickers) > 0:
        print(f"\n  DIFERENCIAS EN PESOS (top 5):")
        weight_diffs = [(t, user_w[i] - rec_w[i])
                        for i, t in enumerate(valid_tickers)
                        if abs(user_w[i] - rec_w[i]) > 0.01]
        weight_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        for t, diff in weight_diffs[:5]:
            direction = "sobrepondera" if diff > 0 else "subpondera"
            print(f"    {t:<8}: Tu portafolio {direction} {abs(diff)*100:.1f}%")

    print()


def explain_risk_metrics_plain(sharpe, var_results, max_drawdown,
                                monto_total, pct_variable, profile,
                                horizon_years):
    """
    [NEW — PHASE 1] Translate Sharpe, VaR, and Max Drawdown into
    plain-Spanish explanations contextualized to the user.
    """
    monto_rv = monto_total * pct_variable / 100

    SEP = "─" * 65
    print(f"\n  {SEP}")
    print("  INTERPRETACION DE METRICAS DE RIESGO (lenguaje simple)")
    print(f"  {SEP}")

    # ── Sharpe Ratio ─────────────────────────────────────────────────────
    print(f"\n  📊 RATIO DE SHARPE: {sharpe:.4f}")
    print(f"  ¿Que significa?")
    print(f"    Es como la 'eficiencia de combustible' de tu portafolio.")
    print(f"    Mide cuanto rendimiento obtienes por cada unidad de riesgo.")

    if sharpe < 0:
        print(f"\n    🔴 Tu portafolio tiene Sharpe NEGATIVO.")
        print(f"    Esto significa que la tasa libre de riesgo (CETES, bonos)")
        print(f"    ofrece mejor rendimiento con cero riesgo.")
        print(f"    Preguntate: ¿por que no simplemente invertir en CETES?")
    elif sharpe < 0.3:
        print(f"\n    🟡 Baja eficiencia: obtienes poco rendimiento extra por")
        print(f"    el riesgo que estas asumiendo.")
    elif sharpe < 0.7:
        print(f"\n    ✅ Eficiencia aceptable. Tu portafolio esta en el rango")
        print(f"    normal para portafolios diversificados.")
    elif sharpe < 1.0:
        print(f"\n    ✅ Buena eficiencia. Rendimiento solido para el riesgo.")
    else:
        print(f"\n    ⚠ Sharpe muy alto (>{sharpe:.2f}). Esto puede deberse a un")
        print(f"    periodo historico favorable o poca volatilidad. No")
        print(f"    extrapoles este resultado al futuro sin precaucion.")

    # ── VaR ───────────────────────────────────────────────────────────────
    if var_results:
        var_95_annual = var_results.get(0.95, {}).get("annual", 0)
        var_99_1d = var_results.get(0.99, {}).get("1_day", 0)

        print(f"\n  📉 VALUE AT RISK (VaR 95% anual): ${var_95_annual:,.2f}")
        print(f"  ¿Que significa?")
        print(f"    En 19 de cada 20 anos, tu perdida NO excedera ${var_95_annual:,.0f}.")
        print(f"    Pero en 1 de cada 20 anos, PODRIAS perder MAS que eso.")

        pct_loss = var_95_annual / monto_rv * 100 if monto_rv > 0 else 0
        print(f"\n    Sobre tu inversion en renta variable (${monto_rv:,.0f}):")
        print(f"    esto equivale a una perdida potencial de {pct_loss:.1f}%.")

        if pct_loss > 30:
            print(f"\n    🔴 Para un perfil {profile}, una perdida potencial")
            print(f"    del {pct_loss:.0f}% es significativa. Evalua si puedes")
            print(f"    absorber esta perdida sin afectar tu vida diaria.")
        elif pct_loss > 15:
            print(f"\n    🟡 La perdida potencial es moderada ({pct_loss:.0f}%).")
            print(f"    Asegurate de tener reservas de emergencia suficientes.")
        else:
            print(f"\n    ✅ La perdida potencial es baja ({pct_loss:.0f}%).")
            print(f"    Esto es consistente con un portafolio conservador.")

        print(f"\n    ⚠ NOTA: El VaR asume que los rendimientos siguen una")
        print(f"    distribucion normal. En la realidad, los eventos extremos")
        print(f"    ocurren con mas frecuencia de lo que este modelo predice.")

    # ── Max Drawdown ─────────────────────────────────────────────────────
    print(f"\n  📉 MAX DRAWDOWN HISTORICO: {max_drawdown*100:.2f}%")
    print(f"  ¿Que significa?")
    print(f"    Es la peor caida que tu portafolio ha experimentado")
    print(f"    historicamente, de pico a valle.")

    dd_valor = monto_rv * abs(max_drawdown)
    print(f"\n    Si inviertes ${monto_rv:,.0f} en renta variable,")
    print(f"    en el peor momento historico habrias visto tu inversion")
    print(f"    reducirse en ${dd_valor:,.0f}")
    print(f"    (de ${monto_rv:,.0f} a ${monto_rv - dd_valor:,.0f}).")

    if abs(max_drawdown) > 0.40:
        print(f"\n    🔴 Un drawdown mayor al 40% es severo. La mayoria de")
        print(f"    los inversionistas NO pueden sostener esto psicologicamente.")
    elif abs(max_drawdown) > 0.25:
        print(f"\n    🟡 Un drawdown del {abs(max_drawdown)*100:.0f}% es significativo.")
        print(f"    Para un horizonte de {horizon_years} anos, esto podria")
        if horizon_years <= 5:
            print(f"    consumir una porcion importante de tu periodo de inversion.")
        else:
            print(f"    ser recuperable, pero requiere paciencia.")
    else:
        print(f"\n    ✅ Drawdown moderado, consistente con portafolios")
        print(f"    diversificados.")

    print(f"\n  {SEP}")


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
                                    port_max_dd, insights,
                                    future_mc=None):
    """
    Generate a professional advisory-style investment recommendation memo.
    Includes insights generated by the analysis module.
    """
    monto_rv = cfg['monto_total'] * cfg['pct_variable'] / 100
    monto_rf = cfg['monto_total'] * cfg['pct_fixed'] / 100

    div_text = insights["diversification"]["classification"].upper()
    avg_corr = insights["diversification"]["avg_corr"]

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

    # [NEW — PHASE 1] Pillar breakdown in report
    if cfg.get("pillar_levels"):
        print(f"\n  Detalle por pilares:")
        for pillar_name, level_idx in cfg["pillar_levels"].items():
            level_name = _level_index_to_name(level_idx)
            print(f"    {pillar_name:<25}: {level_name}")
        print(f"  Restriccion vinculante  : {cfg.get('binding_pillar', 'N/A')}")
        if cfg.get("max_weight"):
            print(f"  Peso maximo por activo  : {cfg['max_weight']*100:.0f}%")

    # [NEW — PHASE 1] Contradiction alerts in report
    contradictions = cfg.get("contradictions", [])
    if contradictions:
        print(f"\n  ALERTAS DETECTADAS:")
        for c in contradictions:
            print(f"    [{c['severity']}] {c['message'][:80]}..."
                  if len(c['message']) > 80 else f"    [{c['severity']}] {c['message']}")

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
        for w in insights["risk_interpretation"]["warnings"]:
            print(f"  ⚠ {w}")

    # [NEW — PHASE 2] CVaR alongside VaR
    cvar_results = cfg.get("_cvar_results")
    if cvar_results:
        print(f"\n  EXPECTED SHORTFALL (CVaR) — Metodo Historico")
        print(f"  Si el VaR es 'la puerta', el CVaR es 'lo que hay detras'")
        print(f"  {'Confianza':>12} {'VaR (hist.)':>14} {'CVaR':>14} {'Obs. cola':>10}")
        print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*10}")
        for conf in [0.90, 0.95, 0.99]:
            c = cvar_results.get(conf, {})
            print(f"  {conf*100:>11.0f}% "
                  f"{c.get('var_pct', 0)*100:>13.2f}% "
                  f"{c.get('cvar_pct', 0)*100:>13.2f}% "
                  f"{c.get('n_tail_obs', 0):>10d}")

    # 6. Benchmark Comparison
    print(f"\n  6. COMPARACION CON BENCHMARK ({benchmark})")
    print(f"  {'─'*40}")
    print(f"  {'Metrica':<28} {'Portafolio':>12} {'Benchmark':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}")
    print(f"  {'Rendimiento anual':<28} {prim_ret*100:>11.2f}% {bench_metrics['return']*100:>11.2f}%")
    print(f"  {'Volatilidad anual':<28} {prim_vol*100:>11.2f}% {bench_metrics['volatility']*100:>11.2f}%")
    print(f"  {'Sharpe':<28} {prim_sr:>12.4f} {bench_metrics['sharpe']:>12.4f}")
    print(f"  {'Max Drawdown':<28} {port_max_dd*100:>11.2f}% {bench_metrics['max_drawdown']*100:>11.2f}%")

    # [MODIFIED — PHASE 2] Enhanced benchmark interpretation
    ret_diff = prim_ret - bench_metrics["return"]
    vol_diff = prim_vol - bench_metrics["volatility"]
    sr_diff = prim_sr - bench_metrics["sharpe"]

    print(f"\n  Interpretacion:")
    if ret_diff > 0.01 and vol_diff > 0.03 and sr_diff < 0:
        print(f"  ⚠ El portafolio mejora el rendimiento (+{ret_diff*100:.1f}%), pero")
        print(f"    con riesgo significativamente mayor (+{vol_diff*100:.1f}% volatilidad).")
        print(f"    La eficiencia ajustada por riesgo (Sharpe) es INFERIOR al benchmark.")
        print(f"    El rendimiento adicional NO compensa proporcionalmente el riesgo extra.")
    elif ret_diff > 0.01 and vol_diff > 0.02:
        print(f"  ℹ El portafolio mejora el rendimiento (+{ret_diff*100:.1f}%), pero con")
        print(f"    riesgo moderadamente mayor (+{vol_diff*100:.1f}% volatilidad).")
        if sr_diff > 0:
            print(f"    Sin embargo, la eficiencia (Sharpe) mejora, indicando que")
            print(f"    el riesgo adicional esta siendo compensado.")
        else:
            print(f"    El riesgo adicional no esta plenamente recompensado.")
    elif ret_diff > 0 and vol_diff <= 0:
        print(f"  ✅ El portafolio supera al benchmark con igual o menor volatilidad.")
        print(f"    Esta es una situacion favorable de eficiencia.")
    elif ret_diff < -0.01:
        print(f"  ⚠ El portafolio rinde MENOS que el benchmark ({ret_diff*100:+.1f}%).")
        if vol_diff < 0:
            print(f"    Pero con menor riesgo, lo cual puede ser apropiado para tu perfil.")
        else:
            print(f"    Y ademas toma mas riesgo. Esto no es eficiente.")
    else:
        print(f"  ✅ Desempeno similar al benchmark.")

    # Tail risk vs benchmark
    bench_dd = bench_metrics.get("max_drawdown", 0)
    if abs(port_max_dd) > abs(bench_dd) * 1.3:
        print(f"\n  ⚠ Riesgo de cola: el drawdown del portafolio ({port_max_dd*100:.1f}%)")
        print(f"    es significativamente peor que el benchmark ({bench_dd*100:.1f}%).")
        print(f"    La diversificacion puede debilitarse en periodos de crisis.")

    if target_analysis is not None:
        print(f"\n  {'Rendimiento objetivo':<28} {target_analysis['target']*100:>11.2f}%")
        status = "✅ CUMPLE" if target_analysis['meets_target'] else "❌ NO CUMPLE"
        print(f"  {'Estatus vs objetivo':<28} {status:>12}")
        print(f"  {'Probabilidad de alcanzarlo':<28} {target_analysis['probability_pct']:>11.1f}%")

    # 7. Valuation & Rebalancing
    if valuation_signals:
        print(f"\n  7. SENALES DE VALUACION Y REBALANCEO")
        print(f"  {'─'*40}")
        print(f"  NOTA: Estas senales son indicativas y se basan en multiplos")
        print(f"  historicos simplificados. No constituyen un modelo de")
        print(f"  valuacion completo ni consejo de inversion definitivo.\n")
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
        if insights["all_valuation_neutral"]:
            print("\n  ⚠ Todas las senales son neutrales o con datos insuficientes.")
            print("    Las senales de valuacion son indicativas y pueden carecer de")
            print("    precision dependiendo del modelo y los datos disponibles.")
            print("    No deben utilizarse como unico criterio de decision.")

    if rebalancing_suggestions:
        print(f"\n  SUGERENCIAS DE REBALANCEO:")
        for sug in rebalancing_suggestions:
            print(f"    {sug}")

    # 7B. KEY ANALYSIS INSIGHTS (Part 10 — Enhanced)
    print(f"\n  7B. HALLAZGOS CLAVE DEL ANALISIS")
    print(f"  {'─'*40}")
    print(f"  - Diversificacion : {div_text.capitalize()}")
    if insights["concentration_ms"]["is_concentrated"] or insights["concentration_mv"]["is_concentrated"]:
        print(f"  - Concentracion   : Riesgo Detectado")
    else:
        print(f"  - Concentracion   : Riesgo Bajo")
    if insights["data_quality"]["is_small_sample"]:
        print(f"  - Calidad de Datos: Muestra pequena (Baja confiabilidad)")
    elif insights["data_quality"]["horizon_mismatch"]:
        print(f"  - Calidad de Datos: Desfase horizonte vs. datos")
    else:
        print(f"  - Calidad de Datos: Adecuada")
    # Benchmark tradeoff summary
    bt = insights["benchmark_tradeoff"]
    if bt["beats_return"] and bt["beats_vol"]:
        print(f"  - Benchmark       : Portafolio superior en rendimiento y riesgo")
    elif bt["beats_return"]:
        print(f"  - Benchmark       : Mayor rendimiento, pero tambien mayor riesgo")
    elif bt["beats_vol"]:
        print(f"  - Benchmark       : Menor riesgo, pero tambien menor rendimiento")
    else:
        print(f"  - Benchmark       : Benchmark superior en metricas principales")
    # Valuation reliability
    if insights["all_valuation_neutral"]:
        print(f"  - Valuacion       : Senales neutrales (precision limitada)")
    else:
        print(f"  - Valuacion       : Senales diferenciadas disponibles")
    # Profile mismatch
    if not insights["profile_consistency"]["aligned"]:
        sev = insights["profile_consistency"]["severity"]
        sev_label = "Significativo" if sev == "significant" else "Leve"
        print(f"  - Perfil/Riesgo   : Desalineacion detectada ({sev_label})")
    else:
        print(f"  - Perfil/Riesgo   : Alineado con perfil")
    # Objective alignment
    if not insights["objective_alignment"]["aligned"]:
        print(f"  - Objetivo/Activos: Desalineacion detectada")
    else:
        print(f"  - Objetivo/Activos: Alineado")
    # Min variance efficiency
    if insights["mv_below_rf"]:
        print(f"  - Min Varianza    : Rendimiento inferior a tasa libre de riesgo")

    # 8. Professional Conclusion (Part 4 — Enhanced with tradeoff context)
    print(f"\n  8. CONCLUSION PROFESIONAL")
    print(f"  {'─'*40}")

    beats_return = prim_ret > bench_metrics["return"]
    beats_sharpe = prim_sr  > bench_metrics["sharpe"]
    beats_vol    = prim_vol < bench_metrics["volatility"]

    conclusion_parts = []
    conclusion_parts.append(
        f"  Para un inversionista con perfil {cfg['profile']} y horizonte de "
        f"{cfg['horizon_years']} anos ({cfg['horizon_category']}), se recomienda "
        f"una asignacion de {cfg['pct_fixed']:.0f}% renta fija / {cfg['pct_variable']:.0f}% "
        f"renta variable."
    )

    if beats_return and beats_sharpe and beats_vol:
        conclusion_parts.append(
            "  El portafolio optimizado supera al benchmark en rendimiento, "
            "volatilidad y eficiencia ajustada por riesgo. Este es un resultado "
            "muy favorable."
        )
    elif beats_return and beats_sharpe:
        conclusion_parts.append(
            "  El portafolio supera al benchmark en rendimiento y eficiencia "
            "ajustada por riesgo, aunque con mayor volatilidad. El mayor "
            "rendimiento viene acompanado de mayor riesgo."
        )
    elif beats_return and not beats_vol:
        conclusion_parts.append(
            "  El portafolio ofrece mayor rendimiento que el benchmark, pero "
            "tambien exhibe mayor volatilidad y potencialmente drawdowns mas "
            "profundos. El rendimiento adicional no compensa proporcionalmente "
            "el riesgo extra asumido."
        )
    elif beats_sharpe:
        conclusion_parts.append(
            "  Aunque el rendimiento absoluto es menor que el benchmark, la eficiencia "
            "ajustada por riesgo (Sharpe) es superior, adecuado para inversores que "
            "priorizan la relacion riesgo-rendimiento."
        )
    else:
        conclusion_parts.append(
            "  El benchmark supera al portafolio en las metricas principales. "
            "Considere ajustar la seleccion de activos o el horizonte."
        )

    # Profile consistency in conclusion (Part 2)
    if not insights["profile_consistency"]["aligned"]:
        if insights["profile_consistency"]["severity"] == "significant":
            conclusion_parts.append(
                "  ⚠ IMPORTANTE: La asignacion actual es mas agresiva que lo "
                "sugerido para este perfil de inversionista. Los niveles de riesgo "
                "exceden las tolerancias tipicas del perfil detectado."
            )
        else:
            conclusion_parts.append(
                "  Nota: La asignacion actual esta ligeramente fuera del rango "
                "sugerido para este perfil. Monitoree el nivel de riesgo."
            )

    # Objective alignment in conclusion (Part 3)
    if not insights["objective_alignment"]["aligned"]:
        conclusion_parts.append(
            "  Nota: Los activos seleccionados pueden no estar completamente "
            "alineados con el objetivo de inversion declarado. Revise la "
            "seccion de recomendaciones para mayor detalle."
        )

    # Min-variance note in conclusion (Part 8)
    if insights["mv_below_rf"]:
        conclusion_parts.append(
            "  Nota: Aunque el portafolio de Minima Varianza minimiza la volatilidad, "
            "su rendimiento esperado es inferior a la tasa libre de riesgo, lo que "
            "lo hace poco atractivo en terminos de eficiencia."
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

    # 9. Recommendation Bullets
    if insights["recommendations"]:
        print(f"\n  9. RECOMENDACIONES ACCIONABLES")
        print(f"  {'─'*40}")
        for rec in insights["recommendations"]:
            print(f"  • {rec}")

    # 10. Future Scenario Simulation Summary
    if future_mc is not None:
        print(f"\n  10. SIMULACION DE ESCENARIOS FUTUROS")
        print(f"  {'─'*40}")
        # Use total portfolio stats if available, otherwise VI
        fmc_s = future_mc.get("stats_total_portfolio") or \
            future_mc["stats_variable_income"]
        fmc_dd = future_mc["drawdown_stats"]
        lbl = fmc_s["label"]
        print(f"  ({lbl} · {N_FUTURE_SIMULATIONS:,} escenarios · "
              f"{cfg['horizon_years']} anos)")
        print(f"\n  Valor final medio          : ${fmc_s['mean']:>14,.2f}")
        print(f"  Valor final mediano        : ${fmc_s['median']:>14,.2f}")
        print(f"  P5  (pesimista)            : ${fmc_s['p5']:>14,.2f}")
        print(f"  P95 (optimista)            : ${fmc_s['p95']:>14,.2f}")
        print(f"  Probabilidad de perdida    : {fmc_s['prob_loss']:>6.1f} %")
        if fmc_s.get("prob_target") is not None:
            print(f"  Prob. alcanzar objetivo    : {fmc_s['prob_target']:>6.1f} %")
        print(f"  Max drawdown promedio sim. : {fmc_dd['avg_max_dd']*100:>8.2f} %")
        # Brief interpretation
        interp = future_mc.get("interpretation", [])
        if interp:
            print(f"\n  Interpretacion:")
            for line in interp[:3]:  # top 3 insights
                print(f"  -> {line}")

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
    """
    Generate a professional multi-panel dashboard.

    Layout (2 rows x 3 columns):
      Row 1: Efficient Frontier | Correlation Heatmap | Executive Summary
      Row 2: Portfolio Weights  | Returns Distribution (2-col span)

    The normalized price chart is saved as a separate standalone file.
    """

    # ── Color palette ─────────────────────────────────────────────────────
    ACCENT_RED   = "#E63946"
    ACCENT_BLUE  = "#457B9D"
    ACCENT_DARK  = "#1D3557"
    ACCENT_GREEN = "#2A9D8F"
    ACCENT_GOLD  = "#E9C46A"
    BG_COLOR     = "#FAFAFA"
    GRID_ALPHA   = 0.20
    TITLE_SIZE   = 11
    LABEL_SIZE   = 8

    has_fi = fi_recommendation is not None and not fi_recommendation.empty

    # ══════════════════════════════════════════════════════════════════════
    # MAIN DASHBOARD — 2×3 grid
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 14), facecolor=BG_COLOR)
    fig.suptitle(
        f"Dashboard de Analisis  ·  {cfg['start_date']} → {cfg['end_date']}  ·  "
        f"Perfil: {cfg['profile']}  ·  "
        f"Horizonte: {cfg['horizon_years']}a ({cfg['horizon_category']})",
        fontsize=13, fontweight="bold", y=0.99, color=ACCENT_DARK
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.38, wspace=0.32,
        left=0.05, right=0.96, top=0.93, bottom=0.06
    )

    # ── Panel 1 (top-left): Efficient Frontier ────────────────────────────
    ax_ef = fig.add_subplot(gs[0, 0], facecolor=BG_COLOR)
    sc = ax_ef.scatter(
        sim_df["volatility"] * 100, sim_df["return"] * 100,
        c=sim_df["sharpe"], cmap="viridis", alpha=0.30, s=5, zorder=1,
        edgecolors="none"
    )
    plt.colorbar(sc, ax=ax_ef, label="Sharpe", fraction=0.046, pad=0.04)

    ax_ef.scatter(ms_vol*100, ms_ret*100, color=ACCENT_RED, s=180, zorder=5,
                  marker="*", label=f"Max Sharpe ({ms_ret*100:.1f}%)",
                  edgecolors="white", linewidths=0.5)
    ax_ef.scatter(mv_vol*100, mv_ret*100, color=ACCENT_BLUE, s=140, zorder=5,
                  marker="D", label=f"Min Vol ({mv_ret*100:.1f}%)",
                  edgecolors="white", linewidths=0.5)
    if man_ret is not None and man_vol is not None:
        ax_ef.scatter(man_vol*100, man_ret*100, color="magenta", s=160,
                      zorder=6, marker="P", label="Manual",
                      edgecolors="white", linewidths=0.5)
    ax_ef.scatter(bench_metrics["volatility"]*100, bench_metrics["return"]*100,
                  color=ACCENT_DARK, s=140, zorder=5, marker="^",
                  label=benchmark, edgecolors="white", linewidths=0.5)

    for t in valid_tickers:
        ax_ef.scatter(asset_stats.loc[t, "Annual Volatility"]*100,
                      asset_stats.loc[t, "Annual Return"]*100,
                      color=ACCENT_GOLD, s=45, zorder=4, marker="o",
                      alpha=0.85, edgecolors=ACCENT_DARK, linewidths=0.4)
        ax_ef.annotate(t, xy=(asset_stats.loc[t, "Annual Volatility"]*100,
                               asset_stats.loc[t, "Annual Return"]*100),
                       fontsize=6, ha="left", va="bottom",
                       xytext=(3, 3), textcoords="offset points",
                       color=ACCENT_DARK)

    ax_ef.set_title("Frontera Eficiente", fontsize=TITLE_SIZE, fontweight="bold")
    ax_ef.set_xlabel("Volatilidad (%)", fontsize=LABEL_SIZE)
    ax_ef.set_ylabel("Rendimiento (%)", fontsize=LABEL_SIZE)
    ax_ef.legend(fontsize=6.5, loc="upper left", framealpha=0.85,
                 edgecolor="lightgray")
    ax_ef.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax_ef.tick_params(labelsize=7)

    # ── Panel 2 (top-center): Correlation Heatmap ─────────────────────────
    ax_corr = fig.add_subplot(gs[0, 1], facecolor=BG_COLOR)
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-1, vmax=1, ax=ax_corr,
                    cbar_kws={"shrink": 0.75, "aspect": 20},
                    linewidths=0.5, square=True,
                    annot_kws={"fontsize": 7})
    else:
        im = ax_corr.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(corr_matrix.columns)))
        ax_corr.set_yticks(range(len(corr_matrix.columns)))
        ax_corr.set_xticklabels(corr_matrix.columns, rotation=45,
                                ha="right", fontsize=7)
        ax_corr.set_yticklabels(corr_matrix.columns, fontsize=7)
        for i_r in range(len(corr_matrix)):
            for j_c in range(len(corr_matrix)):
                ax_corr.text(j_c, i_r, f"{corr_matrix.values[i_r, j_c]:.2f}",
                             ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax_corr, shrink=0.75)
    ax_corr.set_title("Correlacion", fontsize=TITLE_SIZE, fontweight="bold")
    ax_corr.tick_params(labelsize=7)

    # ── Panel 3 (top-right): Executive Summary ────────────────────────────
    ax_exec = fig.add_subplot(gs[0, 2])
    ax_exec.axis("off")

    prim_ret = ms_ret
    prim_vol = ms_vol
    prim_sr = (ms_ret - cfg["risk_free"]) / ms_vol if ms_vol > 0 else 0

    summary_lines = [
        ("Perfil", cfg["profile"]),
        ("Horizonte", f"{cfg['horizon_years']} años ({cfg['horizon_category']})"),
        ("", ""),
        ("Rend. Portafolio", f"{prim_ret*100:.2f} %"),
        ("Vol. Portafolio", f"{prim_vol*100:.2f} %"),
        ("Sharpe", f"{prim_sr:.4f}"),
        ("", ""),
        ("Rend. Benchmark", f"{bench_metrics['return']*100:.2f} %"),
        ("Vol. Benchmark", f"{bench_metrics['volatility']*100:.2f} %"),
        ("Sharpe Benchmark", f"{bench_metrics['sharpe']:.4f}"),
        ("", ""),
        ("Renta Fija", f"{cfg['pct_fixed']:.0f} %"),
        ("Renta Variable", f"{cfg['pct_variable']:.0f} %"),
        ("Inversion Total", f"${cfg['monto_total']:,.0f}"),
    ]

    y_pos = 0.95
    ax_exec.text(0.5, 1.02, "Resumen Ejecutivo",
                 transform=ax_exec.transAxes, fontsize=TITLE_SIZE,
                 fontweight="bold", ha="center", va="top", color=ACCENT_DARK)

    # Background box
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.03, 0.02), 0.94, 0.96,
                          boxstyle="round,pad=0.02",
                          facecolor="white", edgecolor="#DDDDDD",
                          linewidth=1.0, transform=ax_exec.transAxes,
                          zorder=0)
    ax_exec.add_patch(box)

    for label, value in summary_lines:
        if label == "" and value == "":
            y_pos -= 0.025
            continue
        ax_exec.text(0.10, y_pos, label, transform=ax_exec.transAxes,
                     fontsize=8.5, fontweight="bold", va="top",
                     color=ACCENT_DARK, family="monospace")
        ax_exec.text(0.92, y_pos, value, transform=ax_exec.transAxes,
                     fontsize=8.5, va="top", ha="right",
                     color="#333333", family="monospace")
        y_pos -= 0.065

    # ── Panel 4 (bottom-left): Portfolio Weights ──────────────────────────
    ax_w = fig.add_subplot(gs[1, 0], facecolor=BG_COLOR)
    w_labels = [t for t, w in zip(valid_tickers, ms_w) if w > 0.005]
    w_sizes  = [w for w in ms_w if w > 0.005]

    if w_sizes:
        n_assets = len(w_labels)
        if n_assets <= 5:
            # Pie chart for small number of assets
            colors_pie = plt.cm.Set2(np.linspace(0.1, 0.9, n_assets))
            explode = [0.03] * n_assets
            wedges, texts, autotexts = ax_w.pie(
                w_sizes, labels=w_labels, autopct="%1.1f%%",
                explode=explode, startangle=90, colors=colors_pie,
                textprops={"fontsize": 8}
            )
            for at in autotexts:
                at.set_fontsize(7.5)
                at.set_fontweight("bold")
            ax_w.set_title("Pesos — Max Sharpe", fontsize=TITLE_SIZE,
                           fontweight="bold")
        else:
            # Horizontal bar for many assets
            sorted_pairs = sorted(zip(w_labels, w_sizes),
                                  key=lambda x: x[1], reverse=True)
            s_labels, s_sizes = zip(*sorted_pairs)
            y_positions = range(len(s_labels))
            colors_bar = plt.cm.Set2(np.linspace(0.1, 0.9, len(s_labels)))

            bars = ax_w.barh(y_positions, [s*100 for s in s_sizes],
                             color=colors_bar, edgecolor="white",
                             linewidth=0.5, height=0.6)
            ax_w.set_yticks(y_positions)
            ax_w.set_yticklabels(s_labels, fontsize=LABEL_SIZE)
            ax_w.set_xlabel("Peso (%)", fontsize=LABEL_SIZE)
            ax_w.set_title("Pesos — Max Sharpe", fontsize=TITLE_SIZE,
                           fontweight="bold")
            ax_w.invert_yaxis()
            ax_w.grid(axis="x", alpha=GRID_ALPHA, linestyle="--")
            ax_w.tick_params(labelsize=7)

            # Add value labels on bars
            for bar, s in zip(bars, s_sizes):
                ax_w.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                          f"{s*100:.1f}%", va="center", fontsize=7,
                          color=ACCENT_DARK)
    else:
        ax_w.text(0.5, 0.5, "Sin pesos > 0.5%", ha="center", va="center",
                  fontsize=10)
        ax_w.set_title("Pesos — Max Sharpe", fontsize=TITLE_SIZE,
                       fontweight="bold")

    # ── Panel 5+6 (bottom-center + bottom-right): Returns Distribution ─────
    # Spans two columns for a larger, clearer visualization
    ax_hist = fig.add_subplot(gs[1, 1:], facecolor=BG_COLOR)

    # Calculate portfolio returns from weights and asset returns
    weights_arr = np.array(ms_w)
    portfolio_returns = (returns * weights_arr).sum(axis=1)

    # Benchmark returns from bench_prices
    bench_returns_series = bench_prices.pct_change().dropna()
    # Align both series to common dates
    common_idx = portfolio_returns.index.intersection(bench_returns_series.index)
    port_rets = portfolio_returns.loc[common_idx]
    bench_rets = bench_returns_series.iloc[:, 0].loc[common_idx] \
        if hasattr(bench_returns_series, 'columns') \
        else bench_returns_series.loc[common_idx]

    # Determine bin count (Sturges' rule, capped)
    n_obs = len(port_rets)
    n_bins = min(int(1 + 3.322 * np.log10(max(n_obs, 1))), 60)
    n_bins = max(n_bins, 15)

    # Plot overlaid histograms
    ax_hist.hist(port_rets * 100, bins=n_bins, alpha=0.55,
                 color=ACCENT_GREEN, edgecolor="white", linewidth=0.4,
                 label="Portafolio", density=True)
    ax_hist.hist(bench_rets * 100, bins=n_bins, alpha=0.40,
                 color=ACCENT_BLUE, edgecolor="white", linewidth=0.4,
                 label=f"{benchmark}", density=True)

    # KDE overlay (if scipy available)
    try:
        from scipy.stats import gaussian_kde
        port_kde = gaussian_kde(port_rets * 100)
        bench_kde = gaussian_kde(bench_rets * 100)
        x_range = np.linspace(
            min(port_rets.min(), bench_rets.min()) * 100 - 1,
            max(port_rets.max(), bench_rets.max()) * 100 + 1,
            300
        )
        ax_hist.plot(x_range, port_kde(x_range), color=ACCENT_GREEN,
                     linewidth=1.8, alpha=0.9)
        ax_hist.plot(x_range, bench_kde(x_range), color=ACCENT_BLUE,
                     linewidth=1.8, alpha=0.9, linestyle="--")
    except ImportError:
        pass  # Skip KDE if scipy not available

    # Mean return vertical lines
    port_mean = port_rets.mean() * 100
    bench_mean = bench_rets.mean() * 100
    ax_hist.axvline(port_mean, color=ACCENT_GREEN, linewidth=1.5,
                    linestyle="--", alpha=0.9,
                    label=f"Media Port. ({port_mean:.3f}%)")
    ax_hist.axvline(bench_mean, color=ACCENT_BLUE, linewidth=1.5,
                    linestyle=":", alpha=0.9,
                    label=f"Media Bench. ({bench_mean:.3f}%)")

    # Periodicity label
    period_map = {"diaria": "diarios", "semanal": "semanales",
                  "mensual": "mensuales"}
    freq_label = period_map.get(cfg.get("periodicity", "").lower(),
                                "periodicos")

    ax_hist.set_title(f"Distribucion de Rendimientos ({freq_label})",
                      fontsize=TITLE_SIZE, fontweight="bold")
    ax_hist.set_xlabel("Rendimiento (%)", fontsize=LABEL_SIZE)
    ax_hist.set_ylabel("Densidad", fontsize=LABEL_SIZE)
    ax_hist.legend(fontsize=7, loc="upper right", framealpha=0.85,
                   edgecolor="lightgray")
    ax_hist.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")
    ax_hist.tick_params(labelsize=7)

    # Risk interpretation annotation
    port_std = port_rets.std() * 100
    skew_val = port_rets.skew()
    skew_text = "negativa" if skew_val < -0.3 else (
                "positiva" if skew_val > 0.3 else "aprox. simetrica")
    ax_hist.text(0.02, 0.95,
                 f"Vol. {freq_label}: {port_std:.3f}%  |  "
                 f"Asimetria: {skew_text} ({skew_val:.2f})\n"
                 f"Mayor dispersion = mayor riesgo",
                 transform=ax_hist.transAxes, fontsize=7,
                 va="top", ha="left", color=ACCENT_DARK, alpha=0.8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#DDDDDD", alpha=0.85))

    # ── Save and display ──────────────────────────────────────────────────
    chart_path = os.path.join(OUTPUT_DIR, "portfolio_dashboard.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    print(f"  [OK] Dashboard guardado -> {chart_path}")
    plt.show()
    plt.close()

    # ── Standalone: Normalized Price Chart (saved separately) ─────────────
    _plot_normalized_prices(prices, bench_prices, benchmark, valid_tickers, cfg)


def _plot_normalized_prices(prices, bench_prices, benchmark, valid_tickers, cfg):
    """Save normalized price chart as a separate standalone file."""
    fig2, ax = plt.subplots(figsize=(14, 5), facecolor="#FAFAFA")
    combined = prices.copy()
    bench_aligned = bench_prices.reindex(prices.index, method="ffill")
    combined[benchmark] = bench_aligned
    normalized = combined / combined.iloc[0] * 100
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_tickers)))

    for i, t in enumerate(valid_tickers):
        if t in normalized.columns:
            ax.plot(normalized.index, normalized[t], label=t,
                    color=colors[i], linewidth=1.4, alpha=0.85)
    if benchmark in normalized.columns:
        ax.plot(normalized.index, normalized[benchmark],
                label=f"{benchmark} (bench)", color="black",
                linewidth=2.0, linestyle="--", alpha=0.85)

    ax.set_title(
        f"Precio Normalizado (Base=100)  ·  "
        f"{cfg['start_date']} → {cfg['end_date']}",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Precio Indexado", fontsize=9)
    ax.legend(loc="upper left", fontsize=7,
              ncol=min(6, len(valid_tickers)+1), framealpha=0.85)
    ax.grid(True, alpha=0.20, linestyle="--")
    ax.axhline(100, color="gray", linewidth=0.6, linestyle=":")
    ax.tick_params(labelsize=8)

    norm_path = os.path.join(OUTPUT_DIR, "normalized_prices.png")
    plt.savefig(norm_path, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA", edgecolor="none")
    print(f"  [OK] Precios normalizados -> {norm_path}")
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 — FLOW CONTROL FUNCTIONS & MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def show_welcome_screen():
    """Display a brief welcome message explaining what the tool does."""
    W = "=" * 65
    print(f"\n{W}")
    print("  HERRAMIENTA DE ANALISIS Y OPTIMIZACION DE PORTAFOLIOS")
    print(W)
    print("""
  Bienvenido. Este programa te guiara paso a paso para:

    1. Determinar tu perfil de inversionista
    2. Construir un portafolio optimo (Markowitz)
    3. Analizar riesgo y rendimiento de tus activos
    4. Comparar contra un benchmark de mercado
    5. Evaluar renta fija y sugerencias de rebalanceo
    6. Generar un reporte profesional de recomendacion

  El proceso esta disenado como un asesor financiero guiado.
  Puedes explorar conceptos educativos antes de comenzar.
""")
    print(W)


def show_main_menu():
    """
    Display the top-level main menu (shown ONCE per session).
    Returns the user's choice: 'analysis', 'education', or 'exit'.
    """
    while True:
        print(f"\n{'=' * 65}")
        print("  MENU PRINCIPAL")
        print(f"{'=' * 65}")
        print("""
    1. Iniciar analisis de portafolio
    2. Explorar conceptos financieros y ejemplos
    3. Salir
""")
        choice = input("  Tu eleccion (1/2/3): ").strip()
        if choice == "1":
            return "analysis"
        elif choice == "2":
            return "education"
        elif choice == "3":
            return "exit"
        else:
            print("  [X] Opcion invalida. Elige 1, 2 o 3.")


def run_education_module():
    """
    Standalone educational submenu. After the user finishes exploring,
    control returns to the main menu (not to the analysis flow).
    """
    while True:
        print(f"\n{'=' * 65}")
        print("  MODULO EDUCATIVO")
        print(f"{'=' * 65}")
        print("""
    1. Ver explicacion de conceptos clave
       (renta fija/variable, Sharpe, varianza, benchmark)

    2. Ver empresas de ejemplo por sector y ticker

    3. Volver al menu principal <<<
""")
        choice = input("  Tu eleccion (1/2/3): ").strip()
        if choice == "1":
            explain_financial_concepts()
        elif choice == "2":
            show_example_tickers_by_sector()
        elif choice == "3":
            print("\n  Regresando al menu principal...\n")
            break
        else:
            print("  [X] Opcion invalida. Elige 1, 2 o 3.")


def show_profile_result(profile, answers):
    """
    [MODIFIED — PHASE 1] Display the 4-pillar questionnaire results,
    contradiction warnings, and binding constraint explanation.
    Allow manual adjustments.
    Returns (profile, pct_fixed, pct_variable, wants_fi).
    """
    horizon_cat = answers["horizon_category"]

    print(f"\n{'=' * 65}")
    print("  RESULTADO DEL CUESTIONARIO")
    print(f"{'=' * 65}")

    # ── [NEW] 4-Pillar Breakdown ─────────────────────────────────────────
    if "pillar_tolerance_level" in answers:
        synthesis = synthesize_final_profile(answers)
        profile = synthesis["final_profile"]

        print(f"\n  ANALISIS POR PILARES:")
        print(f"  {'─'*55}")

        pillar_display = [
            ("Tolerancia psicologica", answers.get("pillar_tolerance_name", "N/A"),
             answers.get("pillar_tolerance_level", 0)),
            ("Capacidad financiera",   answers.get("pillar_capacity_name", "N/A"),
             answers.get("pillar_capacity_level", 0)),
            ("Necesidades de liquidez", answers.get("pillar_liquidity_name", "N/A"),
             answers.get("pillar_liquidity_level", 0)),
            ("Objetivo de inversion",  answers.get("pillar_objective_name", "N/A"),
             answers.get("pillar_objective_level", 0)),
        ]

        for pillar_name, level_name, level_idx in pillar_display:
            bar = "█" * (level_idx + 1) + "░" * (4 - level_idx)
            print(f"    {pillar_name:<25} {bar}  {level_name}")

        binding = synthesis["binding_pillar"]
        print(f"\n  {'─'*55}")
        print(f"  Perfil final: {profile}")
        print(f"  Restriccion vinculante: {binding}")

        # Explanation
        if synthesis["final_level"] < answers.get("pillar_tolerance_level", 0):
            print(f"\n  ℹ Tu tolerancia psicologica sugiere un perfil mas agresivo,")
            print(f"    pero tu {binding.lower()} limita el riesgo aceptable.")
            print(f"    Esto protege tu patrimonio en escenarios adversos.")

        # ── [NEW] Contradiction Warnings ─────────────────────────────────
        contradictions = synthesis["contradictions"]
        if contradictions:
            print(f"\n  {'═'*55}")
            print(f"  ⚠ ALERTAS DE CONSISTENCIA ({len(contradictions)} detectadas)")
            print(f"  {'═'*55}")

            severity_icon = {
                "CRITICO": "🔴",
                "ADVERTENCIA": "🟡",
                "NOTA": "🔵",
            }

            for i, c in enumerate(contradictions, 1):
                icon = severity_icon.get(c["severity"], "⚠")
                print(f"\n  {icon} [{c['severity']}] Alerta {i}:")
                # Word-wrap message
                msg_lines = _wrap_text(c["message"], width=55, indent=5)
                for line in msg_lines:
                    print(line)
                rec_lines = _wrap_text(f"→ {c['recommendation']}", width=55, indent=5)
                for line in rec_lines:
                    print(line)

            print(f"\n  {'─'*55}")
            print("  Estas alertas son informativas. Puedes ajustar tu perfil")
            print("  manualmente si consideras que las restricciones no aplican.")
    else:
        # Legacy display
        print(f"\n  Perfil determinado : {profile}")
        print(f"  Puntuacion         : {answers['raw_score']}")

    print(f"  Horizonte          : {answers['horizon_years']} anos ({horizon_cat})")

    # Short profile explanation
    profile_desc = {
        "Conservador":
            "  Priorizas seguridad y preservacion de capital.",
        "Moderadamente Conservador":
            "  Buscas seguridad con algo de crecimiento moderado.",
        "Moderado":
            "  Buscas equilibrio entre crecimiento y proteccion.",
        "Moderadamente Agresivo":
            "  Priorizas crecimiento aceptando mayor volatilidad.",
        "Agresivo":
            "  Buscas maximizar rendimiento a largo plazo.",
    }
    print(f"\n{profile_desc.get(profile, '')}")

    # Allow override
    override = prompt("\n  Deseas ajustar el perfil manualmente? (si/no): ",
                       valid_options=["si", "no"])
    if override.lower() == "si":
        profile = prompt("  Selecciona perfil: ",
                          valid_options=list(PROFILE_SETTINGS.keys()))

    # ── FI/RV split ──────────────────────────────────────────────────────
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

    # ── FI diversification flag (actual recommendation happens later) ────
    wants_fi = prompt("\n  Deseas incluir recomendacion de diversificacion de renta fija? (si/no): ",
                       valid_options=["si", "no"])

    return profile, pct_fixed, pct_variable, wants_fi.lower() == "si"


def _wrap_text(text, width=55, indent=5):
    """[NEW — PHASE 1] Simple word-wrap for console output."""
    prefix = " " * indent
    words = text.split()
    lines = []
    current_line = prefix
    for word in words:
        if len(current_line) + len(word) + 1 > width + indent:
            lines.append(current_line)
            current_line = prefix + word
        else:
            current_line += (" " if current_line.strip() else "") + word
    if current_line.strip():
        lines.append(current_line)
    return lines


def show_summary(cfg):
    """Display ALL configuration inputs on a single screen."""
    S = "=" * 65
    print(f"\n{S}")
    print("  RESUMEN DE CONFIGURACION")
    print(S)
    print(f"  Perfil            : {cfg['profile']}")
    print(f"  Horizonte         : {cfg['horizon_years']} anos ({cfg['horizon_category']})")
    print(f"  Renta Fija        : {cfg['pct_fixed']:.0f} %")
    print(f"  Renta Variable    : {cfg['pct_variable']:.0f} %")
    print(f"  Tickers           : {', '.join(cfg['tickers'])}")
    print(f"  Benchmark         : {cfg['benchmark']}")
    print(f"  Periodicidad      : {cfg['periodicity']}")
    print(f"  Rango fechas      : {cfg['start_date']} a {cfg['end_date']}")
    print(f"  Inversion total   : ${cfg['monto_total']:,.2f}")
    monto_rv = cfg['monto_total'] * cfg['pct_variable'] / 100
    monto_fi = cfg['monto_total'] * cfg['pct_fixed'] / 100
    print(f"  Monto en RV       : ${monto_rv:,.2f}")
    print(f"  Monto en RF       : ${monto_fi:,.2f}")
    print(f"  Tasa libre riesgo : {cfg['risk_free']*100:.2f} %")
    if cfg.get("target_return") is not None:
        print(f"  Rendimiento obj.  : {cfg['target_return']*100:.2f} %")
    print(S)


def show_final_options(cfg, valid_tickers, returns, bench_returns,
                       asset_stats, corr_matrix, cov_matrix,
                       ms_w, ms_ret, ms_vol, mv_w, mv_ret, mv_vol,
                       fi_recommendation,
                       prices, bench_prices, bench_metrics, benchmark,
                       sim_df, man_ret_init=None, man_vol_init=None,
                       future_mc=None):
    """
    End-of-run menu: manual portfolio, export (CSV/graphs/both), restart, exit.
    Returns ('restart' or 'exit', man_ret, man_vol).
    """
    man_ret, man_vol = man_ret_init, man_vol_init

    while True:
        print(f"\n{'=' * 65}")
        print("  OPCIONES FINALES")
        print(f"{'=' * 65}")
        print("""
    1. Probar portafolio manual (pesos personalizados)
    2. Exportar solo CSV
    3. Mostrar solo graficas
    4. Exportar todo (CSV + graficas)
    5. Reiniciar con nueva configuracion
    6. Salir
""")
        choice = input("  Tu eleccion (1-6): ").strip()

        if choice == "1":
            manual_w = get_manual_weights(valid_tickers)
            man_ret, man_vol, man_sr = display_manual_portfolio(
                cfg, valid_tickers, manual_w, returns, bench_returns, asset_stats
            )

        elif choice == "2":
            print("\n  Guardando archivos CSV...")
            save_outputs(corr_matrix, cov_matrix, asset_stats,
                         valid_tickers, ms_w, mv_w, fi_recommendation)
            print("  [OK] Archivos CSV exportados correctamente.")

        elif choice == "3":
            print("\n  Generando graficas...")
            plot_results(
                prices, returns, bench_prices, bench_metrics, benchmark,
                sim_df, valid_tickers, corr_matrix,
                ms_w, ms_ret, ms_vol,
                mv_w, mv_ret, mv_vol,
                man_ret, man_vol,
                asset_stats, cfg, fi_recommendation
            )
            if future_mc is not None:
                print("  Generando graficas de simulacion Monte Carlo...")
                plot_monte_carlo_paths(future_mc, cfg)
                plot_monte_carlo_terminal_distribution(future_mc, cfg)
                plot_monte_carlo_percentile_bands(future_mc, cfg)

        elif choice == "4":
            print("\n  Guardando archivos CSV...")
            save_outputs(corr_matrix, cov_matrix, asset_stats,
                         valid_tickers, ms_w, mv_w, fi_recommendation)
            print("  Generando graficas...")
            plot_results(
                prices, returns, bench_prices, bench_metrics, benchmark,
                sim_df, valid_tickers, corr_matrix,
                ms_w, ms_ret, ms_vol,
                mv_w, mv_ret, mv_vol,
                man_ret, man_vol,
                asset_stats, cfg, fi_recommendation
            )
            if future_mc is not None:
                print("  Generando graficas de simulacion Monte Carlo...")
                plot_monte_carlo_paths(future_mc, cfg)
                plot_monte_carlo_terminal_distribution(future_mc, cfg)
                plot_monte_carlo_percentile_bands(future_mc, cfg)
            print("  [OK] Todo exportado correctamente.")

        elif choice == "5":
            return "restart", man_ret, man_vol

        elif choice == "6":
            return "exit", man_ret, man_vol

        else:
            print("  [X] Opcion invalida. Elige 1, 2, 3, 4, 5 o 6.")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS (Part 11 — modular main)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_pipeline(cfg, pct_fixed, pct_variable, wants_fi,
                          profile, horizon_cat):
    """
    Execute the full data-processing and analysis pipeline.
    Returns all computed objects needed for display and export.
    """
    print(f"\n{'=' * 65}")
    print("  PROCESANDO DATOS...")
    print(f"{'=' * 65}\n")

    # Download data
    prices, bench_prices, valid_tickers, benchmark = download_asset_data(
        cfg["tickers"], cfg["benchmark"],
        cfg["start_date"], cfg["end_date"], cfg["interval"]
    )
    cfg["benchmark"] = benchmark

    # Calculate returns
    print("\n  Calculando rendimientos...")
    returns       = calculate_returns(prices)
    bench_returns = bench_prices.pct_change().dropna()

    # Part 5 — Inline data quality warning (right after download)
    n_obs = len(returns)
    if n_obs < 30:
        print(f"\n  \u26a0 ADVERTENCIA: La muestra tiene solo {n_obs} observaciones.")
        print("    El analisis se basa en un conjunto de datos limitado,")
        print("    lo que puede reducir la confiabilidad de las estimaciones.\n")
    horizon_years = cfg.get("horizon_years", 5)
    if not returns.empty:
        data_span_years = (returns.index[-1] - returns.index[0]).days / 365.25
        if (horizon_years > 5 and data_span_years < 2) or \
           (horizon_years > 10 and data_span_years < 5):
            print(f"  \u26a0 ADVERTENCIA: Horizonte de inversion de {horizon_years} anos")
            print(f"    pero solo hay {data_span_years:.1f} anos de datos historicos.")
            print("    Una inversion a largo plazo se evalua con una muestra corta.\n")

    # Statistics
    asset_stats = calculate_summary_statistics(
        returns, bench_returns, cfg["annualize"], cfg["risk_free"]
    )

    # Matrices
    print("  Calculando matrices de correlacion y covarianza...")
    corr_matrix = calculate_correlation_matrix(returns)
    cov_matrix  = calculate_covariance_matrix(returns, cfg["annualize"])
    display_matrix(corr_matrix, "MATRIZ DE CORRELACION")
    display_matrix(cov_matrix,  "MATRIZ DE COVARIANZA (anualizada)")

    # ── [NEW — PHASE 2] Shrunk covariance ────────────────────────────────
    print("  Calculando covarianza Ledoit-Wolf (shrinkage)...")
    cov_shrunk, shrinkage_intensity = calculate_shrunk_covariance(
        returns, cfg["annualize"]
    )
    if shrinkage_intensity > 0:
        print(f"    Intensidad de shrinkage: {shrinkage_intensity:.4f}")
        if shrinkage_intensity > 0.5:
            print(f"    ℹ Shrinkage alto — los datos tienen poca informacion")
            print(f"      y el estimador aplica mucha regularizacion.")
    cfg["shrinkage_intensity"] = shrinkage_intensity

    # Benchmark metrics
    print("  Calculando metricas del benchmark...")
    bench_metrics = calculate_benchmark_metrics(
        bench_returns, cfg["annualize"], cfg["risk_free"]
    )

    # ── [MODIFIED — PHASE 1] Position-size constraint from profile ─────
    max_weight = MAX_WEIGHT_BY_PROFILE.get(cfg.get("profile", "Moderado"), 0.30)
    cfg["max_weight"] = max_weight

    # Optimization
    print("  Optimizando portafolio (Maximo Sharpe)...")
    ms_w, ms_ret, ms_vol, ms_sr, ms_beta, ms_alpha, ms_treynor = \
        optimize_portfolio(returns, bench_returns, asset_stats,
                           cfg["annualize"], cfg["risk_free"], target="sharpe",
                           max_weight=max_weight, cov_override=cov_shrunk)

    print("  Optimizando portafolio (Minima Varianza)...")
    mv_w, mv_ret, mv_vol, mv_sr, mv_beta, mv_alpha, mv_treynor = \
        optimize_portfolio(returns, bench_returns, asset_stats,
                           cfg["annualize"], cfg["risk_free"], target="minvol",
                           max_weight=max_weight, cov_override=cov_shrunk)

    # Monte Carlo simulation
    print(f"  Simulando {N_SIMULATIONS} portafolios aleatorios...")
    sim_df = simulate_portfolios(returns, cfg["annualize"], cfg["risk_free"])

    # VaR
    print("  Calculando Value at Risk (VaR)...")
    monto_rv = cfg['monto_total'] * pct_variable / 100
    var_results = calculate_var_multiple(ms_ret, ms_vol, monto_rv)

    # ── [NEW — PHASE 2] Historical CVaR ──────────────────────────────────
    print("  Calculando CVaR / Expected Shortfall (historico)...")
    settings = PROFILE_SETTINGS.get(cfg['profile'], PROFILE_SETTINGS["Moderado"])
    if settings["opt_target"] == "minvol":
        prim_w_for_cvar = mv_w
    else:
        prim_w_for_cvar = ms_w
    portfolio_returns = (returns * prim_w_for_cvar).sum(axis=1)
    cvar_results = calculate_cvar_multiple(portfolio_returns)
    cfg["_cvar_results"] = cvar_results

    # ── [NEW — PHASE 2] Stress Testing ───────────────────────────────────
    print("  Ejecutando pruebas de estres...")
    stress_results = run_stress_tests(
        cfg["monto_total"], pct_variable, pct_fixed
    )

    # Valuation signals
    print("  Obteniendo senales de valuacion...")
    valuation_signals = estimate_valuation_signals(valid_tickers)

    # Portfolio max drawdown
    port_max_dd = calculate_portfolio_drawdown(ms_w, returns)

    # Rebalancing suggestions
    rebalancing_suggestions = generate_rebalancing_suggestions(
        valid_tickers, ms_w, valuation_signals, ms_vol, profile, horizon_cat
    )

    # Target return analysis
    target_analysis = None
    if cfg.get("target_return") is not None:
        target_analysis = check_target_return(
            cfg["target_return"], ms_ret, ms_vol, cfg["risk_free"]
        )

    # Fixed-income recommendation
    fi_recommendation = None
    if wants_fi:
        fi_recommendation = recommend_fixed_income_mix(
            profile, horizon_cat, pct_fixed, cfg["monto_total"]
        )
        print(f"\n{'\u2500'*65}")
        print("  RECOMENDACION DE RENTA FIJA")
        print(f"{'\u2500'*65}")
        monto_fi = cfg['monto_total'] * pct_fixed / 100
        print(f"  Monto asignado a renta fija: ${monto_fi:,.2f}\n")
        for _, row in fi_recommendation.iterrows():
            print(f"    {row['Instrumento']:<45} {row['Peso (%)']:>5.1f}%  ${row['Monto ($)']:>12,.2f}")
        print()

    # Generate analysis insights
    print()
    insights = generate_insights(
        corr_matrix, valid_tickers, returns, cfg,
        ms_w, ms_ret, ms_vol, ms_sr,
        mv_w, mv_ret, mv_vol, mv_sr,
        bench_metrics, port_max_dd,
        var_results, valuation_signals,
        fi_recommendation
    )

    # ── Future Monte Carlo Scenario Simulation ────────────────────────────
    # Determine the recommended portfolio (profile-adjusted)
    settings = PROFILE_SETTINGS.get(cfg['profile'], PROFILE_SETTINGS["Moderado"])
    if settings["opt_target"] == "minvol":
        prim_w = mv_w
        prim_label = "Minima Varianza"
    else:
        prim_w = ms_w
        prim_label = "Maximo Sharpe"

    future_mc = run_future_monte_carlo(
        cfg, returns, prim_w, prim_label, n_obs=len(returns)
    )

    # ── [NEW — PHASE 3] Bootstrap weight sensitivity analysis ────────────
    print(f"  Ejecutando analisis de sensibilidad ({N_BOOTSTRAP_SAMPLES} muestras)...")
    opt_target = settings["opt_target"] if settings["opt_target"] in ("sharpe", "minvol") \
        else "sharpe"
    sensitivity = bootstrap_optimal_weights(
        returns, bench_returns, asset_stats,
        cfg["annualize"], cfg["risk_free"],
        target=opt_target, max_weight=max_weight,
        cov_override=cov_shrunk,
        n_samples=N_BOOTSTRAP_SAMPLES
    )
    if sensitivity:
        print(f"    {sensitivity['n_successful']}/{sensitivity['n_attempted']} muestras exitosas")
    else:
        print("    [!] Analisis de sensibilidad no pudo completarse.")

    # ── [NEW — PHASE 3] Scenario-based optimization ─────────────────────
    print("  Ejecutando optimizacion por escenarios macroeconomicos...")
    scenario_results = scenario_based_optimization(
        returns, bench_returns, asset_stats,
        cfg["annualize"], cfg["risk_free"],
        target=opt_target, max_weight=max_weight,
        cov_override=cov_shrunk
    )
    print(f"    {len(scenario_results['scenarios'])} escenarios evaluados")

    return {
        "prices": prices, "bench_prices": bench_prices,
        "valid_tickers": valid_tickers, "benchmark": benchmark,
        "returns": returns, "bench_returns": bench_returns,
        "asset_stats": asset_stats,
        "corr_matrix": corr_matrix, "cov_matrix": cov_matrix,
        "cov_shrunk": cov_shrunk,  # [NEW — PHASE 2]
        "shrinkage_intensity": shrinkage_intensity,  # [NEW — PHASE 2]
        "bench_metrics": bench_metrics,
        "ms_w": ms_w, "ms_ret": ms_ret, "ms_vol": ms_vol, "ms_sr": ms_sr,
        "mv_w": mv_w, "mv_ret": mv_ret, "mv_vol": mv_vol, "mv_sr": mv_sr,
        "sim_df": sim_df, "var_results": var_results,
        "cvar_results": cvar_results,  # [NEW — PHASE 2]
        "stress_results": stress_results,  # [NEW — PHASE 2]
        "valuation_signals": valuation_signals,
        "port_max_dd": port_max_dd,
        "rebalancing_suggestions": rebalancing_suggestions,
        "target_analysis": target_analysis,
        "fi_recommendation": fi_recommendation,
        "insights": insights,
        "future_monte_carlo": future_mc,
        "sensitivity": sensitivity,  # [NEW — PHASE 3]
        "scenario_results": scenario_results,  # [NEW — PHASE 3]
    }


def run_reporting_pipeline(cfg, analysis):
    """[MODIFIED — PHASE 2] Display results, risk explanations, stress tests, and risk summary."""
    display_results(
        cfg, analysis["valid_tickers"], analysis["asset_stats"],
        analysis["ms_w"], analysis["ms_ret"], analysis["ms_vol"], analysis["ms_sr"],
        analysis["mv_w"], analysis["mv_ret"], analysis["mv_vol"], analysis["mv_sr"],
        analysis["bench_metrics"], analysis["benchmark"], analysis["insights"]
    )
    generate_recommendation_report(
        cfg, analysis["valid_tickers"], analysis["corr_matrix"],
        analysis["ms_w"], analysis["ms_ret"], analysis["ms_vol"], analysis["ms_sr"],
        analysis["mv_w"], analysis["mv_ret"], analysis["mv_vol"], analysis["mv_sr"],
        analysis["bench_metrics"], analysis["benchmark"],
        analysis["var_results"], analysis["valuation_signals"],
        analysis["rebalancing_suggestions"],
        analysis["fi_recommendation"], analysis["target_analysis"],
        analysis["port_max_dd"], analysis["insights"],
        future_mc=analysis.get("future_monte_carlo")
    )

    # ── [NEW — PHASE 1] Plain-language risk metric explanations ──────────
    settings = PROFILE_SETTINGS.get(cfg['profile'], PROFILE_SETTINGS["Moderado"])
    if settings["opt_target"] == "minvol":
        prim_sr = analysis["mv_sr"]
    else:
        prim_sr = analysis["ms_sr"]

    explain_risk_metrics_plain(
        sharpe=prim_sr,
        var_results=analysis["var_results"],
        max_drawdown=analysis["port_max_dd"],
        monto_total=cfg["monto_total"],
        pct_variable=cfg["pct_variable"],
        profile=cfg["profile"],
        horizon_years=cfg.get("horizon_years", 5),
    )

    # ── [NEW — PHASE 1] Max Sharpe vs Min Variance comparison ────────────
    compare_portfolios(
        cfg=cfg,
        valid_tickers=analysis["valid_tickers"],
        rec_w=analysis["ms_w"],
        rec_ret=analysis["ms_ret"],
        rec_vol=analysis["ms_vol"],
        rec_sr=analysis["ms_sr"],
        rec_dd=analysis["port_max_dd"],
        user_w=analysis["mv_w"],
        user_ret=analysis["mv_ret"],
        user_vol=analysis["mv_vol"],
        user_sr=analysis["mv_sr"],
        user_dd=calculate_portfolio_drawdown(analysis["mv_w"], analysis["returns"]),
        bench_metrics=analysis["bench_metrics"],
    )

    # ── [NEW — PHASE 2] Stress test display ──────────────────────────────
    stress_results = analysis.get("stress_results", [])
    if stress_results:
        display_stress_tests(stress_results, cfg["monto_total"])

    # ── [NEW — PHASE 2] Integrated risk summary ─────────────────────────
    cvar_results = analysis.get("cvar_results", {})
    if cvar_results:
        risk_summary = generate_risk_summary(
            cfg=cfg,
            sharpe=prim_sr,
            var_results=analysis["var_results"],
            cvar_results=cvar_results,
            max_drawdown=analysis["port_max_dd"],
            stress_results=stress_results,
            bench_metrics=analysis["bench_metrics"],
        )
        # Store for downstream access
        analysis["risk_summary"] = risk_summary

    # Display Monte Carlo future simulation results
    # (fat-tail and rebalancing comparisons are shown inside this function)
    future_mc = analysis.get("future_monte_carlo")
    if future_mc is not None:
        display_monte_carlo_simulation_results(future_mc, cfg)

    # ── [NEW — PHASE 3] Weight sensitivity display ──────────────────────
    sensitivity = analysis.get("sensitivity")
    if sensitivity:
        # Determine primary weights for comparison
        if settings["opt_target"] == "minvol":
            prim_w = analysis["mv_w"]
        else:
            prim_w = analysis["ms_w"]
        display_weight_sensitivity(sensitivity, prim_w)

    # ── [NEW — PHASE 3] Scenario optimization display ───────────────────
    scenario_results = analysis.get("scenario_results")
    if scenario_results:
        if settings["opt_target"] == "minvol":
            prim_w = analysis["mv_w"]
        else:
            prim_w = analysis["ms_w"]
        display_scenario_optimization(scenario_results, prim_w)


def handle_final_actions(cfg, analysis):
    """Run the final options menu and return 'restart' or 'exit'."""
    a = analysis
    action, man_ret, man_vol = show_final_options(
        cfg, a["valid_tickers"], a["returns"], a["bench_returns"],
        a["asset_stats"], a["corr_matrix"], a["cov_matrix"],
        a["ms_w"], a["ms_ret"], a["ms_vol"],
        a["mv_w"], a["mv_ret"], a["mv_vol"],
        a["fi_recommendation"],
        a["prices"], a["bench_prices"], a["bench_metrics"], a["benchmark"],
        a["sim_df"],
        future_mc=a.get("future_monte_carlo")
    )
    return action


def main():
    """
    Main program flow — designed as a guided financial advisor:
      1. Welcome → 2. Menu → 3. (Education) → 4. Questionnaire →
      5. Profile → 6. Config → 7. Summary → 8. Analysis → 9. Results →
      10. Recommendation → 11. Final options
    """

    # Phase 1: Welcome
    show_welcome_screen()

    while True:  # Main loop (allows restart)

        # Phase 2: Main Menu
        menu_choice = show_main_menu()

        if menu_choice == "exit":
            print("\n  Hasta luego. Gracias por usar la herramienta.\n")
            return

        if menu_choice == "education":
            run_education_module()
            continue

        # Phase 3: Investor Questionnaire
        answers = get_investor_questionnaire()
        profile = determine_investor_profile(answers)
        horizon_cat = answers["horizon_category"]

        # Phase 4: Profile Result + Allocation
        profile, pct_fixed, pct_variable, wants_fi = show_profile_result(
            profile, answers
        )

        # Phase 5: Investment Configuration
        market_cfg = configure_investment(profile)

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

        # ── [NEW — PHASE 1] Store pillar profiling data if available ─────
        if "pillar_tolerance_level" in answers:
            synthesis = synthesize_final_profile(answers)
            cfg["pillar_levels"] = synthesis["pillar_levels"]
            cfg["binding_pillar"] = synthesis["binding_pillar"]
            cfg["contradictions"] = synthesis["contradictions"]
            cfg["pillar_tolerance_name"] = answers.get("pillar_tolerance_name")
            cfg["pillar_capacity_name"] = answers.get("pillar_capacity_name")
            cfg["pillar_liquidity_name"] = answers.get("pillar_liquidity_name")
            cfg["pillar_objective_name"] = answers.get("pillar_objective_name")

        # Phase 6: Summary Before Execution
        show_summary(cfg)

        confirm = prompt("\n  Proceder con esta configuracion? (si/no): ",
                         valid_options=["si", "no"])
        if confirm.lower() == "no":
            print("  Regresando al menu principal...\n")
            continue

        # Phase 7: Analysis Pipeline
        analysis = run_analysis_pipeline(
            cfg, pct_fixed, pct_variable, wants_fi, profile, horizon_cat
        )

        # Phase 8-9: Display Results & Recommendation Report
        run_reporting_pipeline(cfg, analysis)

        # Phase 10: Final Options
        action = handle_final_actions(cfg, analysis)

        if action == "restart":
            print("\n  Reiniciando...\n")
            continue
        else:
            print("\n  Hasta luego. Gracias por usar la herramienta.\n")
            return


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()