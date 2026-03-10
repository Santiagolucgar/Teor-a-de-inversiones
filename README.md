# Markowitz Portfolio Optimizer

A Python script that optimizes an investment portfolio based on Modern Portfolio Theory (Markowitz), using historical data directly from Yahoo Finance.

## Features

- **Automated Data Retrieval:** Fetches historical adjusted close prices for selected stocks and a benchmark from Yahoo Finance.
- **Investor Profiling:** Automatically determines your investor profile (Conservative, Moderate, Aggressive) based on your desired proportion of variable versus fixed income.
- **Markowitz Optimization:** Uses `scipy.optimize` to find two key portfolios:
  - **Maximum Sharpe Ratio Portfolio:** The portfolio with the highest expected return per unit of risk.
  - **Minimum Variance Portfolio:** The portfolio with the lowest overall volatility.
- **Monte Carlo Simulation:** Simulates thousands of random portfolios to generate the Efficient Frontier.
- **Visual Dashboard:** Generates a 4-panel chart (`portfolio_analysis_charts.png`) containing:
  - Normalized price history of the selected assets and benchmark.
  - Risk-Return scatter plot representing the Efficient Frontier.
  - Pie chart representing the optimal asset allocation weights (Maximum Sharpe).

## Requirements

Ensure you have Python 3 installed. The required libraries are listed in `requirements.txt`.

## Installation

Es altamente recomendable usar un entorno virtual (virtual environment) para evitar problemas con librerías globales y asegurar que `pip` funcione correctamente.

1. Abre tu terminal (PowerShell o Git Bash) en la carpeta del proyecto.
2. Crea un entorno virtual ejecutando:
   ```bash
   python -m venv venv
   ```
   *(Si `python` no funciona, intenta usar `py -m venv venv` o `python3 -m venv venv`)*

3. Activa el entorno virtual:
   - **En PowerShell (Windows):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
     *(Nota: Si te da un error de permisos de ejecución, corre este comando primero como administrador: `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`, luego intenta activar de nuevo).*
   - **En Git Bash / Git CMD (Windows):**
     ```bash
     source venv/Scripts/activate
     ```
   - **En macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

4. Una vez que tu entorno esté activado (verás un `(venv)` al inicio de tu línea de comandos), instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script from your terminal or command prompt:

```bash
python portafolio_optimizer.py
```

The script will interactively guide you through the following setup:
1. **Stock Tickers**: E.g., `AAPL MSFT KO WMT SPY`
2. **Benchmark**: E.g., `SPY` (default)
3. **Data Periodicity**: `daily`, `weekly`, or `monthly`
4. **Investment Horizon**: `6 months`, `1 year`, `3 years`, or `5 years`
5. **Asset Allocation**: The percentage of Fixed vs. Variable income you wish to hold.
6. **Risk-Free Rate**: The baseline annual interest rate (e.g., `4.5%`).

Upon completion, an analysis snapshot will be printed in the console, and a dashboard image (`portfolio_analysis_charts.png`) will be saved in the directory where the script was run.