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
# SECTION 4 — INVESTOR QUESTIONNAIRE & PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def get_investor_questionnaire():
    """
    Ask 10 structured questions to understand the investor.
    Questions are grouped into 4 logical blocks for clarity.
    Returns a dict of raw answers and a computed profile score.
    """
    print("\n" + "=" * 65)
    print("  CUESTIONARIO DEL INVERSIONISTA")
    print("=" * 65)
    print("  Responde las siguientes preguntas para determinar tu perfil.")
    print("  Las preguntas estan organizadas en 4 bloques tematicos.\n")

    answers = {}
    score = 0  # Higher score → more aggressive

    # ── BLOQUE 1: Objetivos y Horizonte ───────────────────────────────────
    print(f"  {'─'*60}")
    print("  BLOQUE 1 de 4 — OBJETIVOS Y HORIZONTE")
    print(f"  {'─'*60}")

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

    # Q3 — Withdrawal timing (was Q8 — moved here for logical grouping)
    print("\n  [3/10] Cuando planeas comenzar a retirar fondos?")
    print("    1. Ya estoy retirando / muy pronto")
    print("    2. Dentro de 1-5 anos")
    print("    3. En mas de 5 anos")
    q8 = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["withdrawal_timing"] = q8
    score += (q8 - 1) * 2  # 0, 2, 4

    # Q4 — Liquidity needs (was Q7 — moved here for logical grouping)
    print("\n  [4/10] Necesidades de liquidez")
    print("    1. Puedo necesitar el dinero en cualquier momento")
    print("    2. Puedo necesitarlo dentro de 1 ano")
    print("    3. No necesito el dinero en los proximos 3+ anos")
    q7 = prompt_int("    Tu respuesta (1-3): ", 1, 3)
    answers["liquidity"] = q7
    score += (q7 - 1) * 3  # 0, 3, 6

    # ── BLOQUE 2: Experiencia ─────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  BLOQUE 2 de 4 — EXPERIENCIA")
    print(f"  {'─'*60}")

    # Q5 — Experience level (was Q3)
    print("\n  [5/10] Experiencia previa en inversiones")
    print("    1. Ninguna")
    print("    2. Basica (cuentas de ahorro, depositos)")
    print("    3. Intermedia (fondos, bonos, algunas acciones)")
    print("    4. Avanzada (acciones, derivados, trading activo)")
    q3 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["experience"] = q3
    score += (q3 - 1) * 2  # 0, 2, 4, 6

    exp_labels = {1: "Ninguna", 2: "Basica", 3: "Intermedia", 4: "Avanzada"}
    answers["experience_label"] = exp_labels[q3]

    # Q6 — Instruments used before (was Q4)
    print("\n  [6/10] Instrumentos que has utilizado antes (puedes elegir varios)")
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

    # ── BLOQUE 3: Tolerancia al Riesgo ────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  BLOQUE 3 de 4 — TOLERANCIA AL RIESGO")
    print(f"  {'─'*60}")

    # Q7 — Reaction to 10% drop (was Q5)
    print("\n  [7/10] Si tu portafolio pierde 10% en un mes, tu reaccion seria:")
    print("    1. Vender todo inmediatamente")
    print("    2. Vender una parte para reducir riesgo")
    print("    3. Mantener y esperar la recuperacion")
    print("    4. Comprar mas aprovechando precios bajos")
    q5 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["reaction_10pct"] = q5
    score += (q5 - 1) * 3  # 0, 3, 6, 9

    # Q8 — Reaction to 20% drop (was Q6)
    print("\n  [8/10] Si tu portafolio pierde 20% en tres meses, tu reaccion seria:")
    print("    1. Vender todo inmediatamente")
    print("    2. Vender una parte para reducir riesgo")
    print("    3. Mantener y esperar")
    print("    4. Comprar mas")
    q6 = prompt_int("    Tu respuesta (1-4): ", 1, 4)
    answers["reaction_20pct"] = q6
    score += (q6 - 1) * 3  # 0, 3, 6, 9

    # Q9 — Comfort with volatility
    print("\n  [9/10] Que tan comodo te sientes con la volatilidad?")
    print("    1 = Muy incomodo (prefiero estabilidad total)")
    print("    5 = Muy comodo (acepto fluctuaciones fuertes)")
    q9 = prompt_int("    Tu respuesta (1-5): ", 1, 5)
    answers["volatility_comfort"] = q9
    score += (q9 - 1) * 2  # 0, 2, 4, 6, 8

    # ── BLOQUE 4: Preferencia de Asignacion ───────────────────────────────
    print(f"\n  {'─'*60}")
    print("  BLOQUE 4 de 4 — PREFERENCIA DE ASIGNACION")
    print(f"  {'─'*60}")

    # Brief education before the allocation question
    explain_fixed_vs_variable_rate()

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
# SECTION 17 — PROFESSIONAL RECOMMENDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendation_report(cfg, valid_tickers, corr_matrix,
                                    ms_w, ms_ret, ms_vol, ms_sr,
                                    mv_w, mv_ret, mv_vol, mv_sr,
                                    bench_metrics, benchmark,
                                    var_results, valuation_signals,
                                    rebalancing_suggestions,
                                    fi_recommendation, target_analysis,
                                    port_max_dd, insights):
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
    Display the questionnaire results and allow manual adjustments.
    Returns (profile, pct_fixed, pct_variable, wants_fi).
    """
    horizon_cat = answers["horizon_category"]

    print(f"\n{'=' * 65}")
    print("  RESULTADO DEL CUESTIONARIO")
    print(f"{'=' * 65}")
    print(f"\n  Perfil determinado : {profile}")
    print(f"  Horizonte          : {answers['horizon_years']} anos ({horizon_cat})")
    print(f"  Puntuacion         : {answers['raw_score']}")

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
                       sim_df, man_ret_init=None, man_vol_init=None):
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

    # Benchmark metrics
    print("  Calculando metricas del benchmark...")
    bench_metrics = calculate_benchmark_metrics(
        bench_returns, cfg["annualize"], cfg["risk_free"]
    )

    # Optimization
    print("  Optimizando portafolio (Maximo Sharpe)...")
    ms_w, ms_ret, ms_vol, ms_sr, ms_beta, ms_alpha, ms_treynor = \
        optimize_portfolio(returns, bench_returns, asset_stats,
                           cfg["annualize"], cfg["risk_free"], target="sharpe")

    print("  Optimizando portafolio (Minima Varianza)...")
    mv_w, mv_ret, mv_vol, mv_sr, mv_beta, mv_alpha, mv_treynor = \
        optimize_portfolio(returns, bench_returns, asset_stats,
                           cfg["annualize"], cfg["risk_free"], target="minvol")

    # Monte Carlo simulation
    print(f"  Simulando {N_SIMULATIONS} portafolios aleatorios...")
    sim_df = simulate_portfolios(returns, cfg["annualize"], cfg["risk_free"])

    # VaR
    print("  Calculando Value at Risk (VaR)...")
    monto_rv = cfg['monto_total'] * pct_variable / 100
    var_results = calculate_var_multiple(ms_ret, ms_vol, monto_rv)

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

    return {
        "prices": prices, "bench_prices": bench_prices,
        "valid_tickers": valid_tickers, "benchmark": benchmark,
        "returns": returns, "bench_returns": bench_returns,
        "asset_stats": asset_stats,
        "corr_matrix": corr_matrix, "cov_matrix": cov_matrix,
        "bench_metrics": bench_metrics,
        "ms_w": ms_w, "ms_ret": ms_ret, "ms_vol": ms_vol, "ms_sr": ms_sr,
        "mv_w": mv_w, "mv_ret": mv_ret, "mv_vol": mv_vol, "mv_sr": mv_sr,
        "sim_df": sim_df, "var_results": var_results,
        "valuation_signals": valuation_signals,
        "port_max_dd": port_max_dd,
        "rebalancing_suggestions": rebalancing_suggestions,
        "target_analysis": target_analysis,
        "fi_recommendation": fi_recommendation,
        "insights": insights,
    }


def run_reporting_pipeline(cfg, analysis):
    """Display results and generate recommendation report."""
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
        analysis["port_max_dd"], analysis["insights"]
    )


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
        a["sim_df"]
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