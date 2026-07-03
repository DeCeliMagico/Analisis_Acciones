"""Descarga datos de mercado de referencia: SPY (S&P 500 ETF) y VIX.

Se guardan en data/market_data/ separados del Bronze de acciones,
para que cargar_bronze() no los trate como tickers a entrenar.
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


MARKET_SYMBOLS = {
    "SPY": "SPY",    # S&P 500 ETF — mercado general
    "VIX": "^VIX",  # Índice de volatilidad implícita
    "XLK": "XLK",   # Sector Technology
    "XLC": "XLC",   # Sector Communication Services
    "XLY": "XLY",   # Sector Consumer Discretionary
    "XLF": "XLF",   # Sector Financials
}


def descargar_symbol(yahoo_ticker: str) -> pd.DataFrame | None:
    period1 = int(datetime(1990, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int(time.time())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
        f"?interval=1d&period1={period1}&period2={period2}"
    )
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    if response.status_code != 200:
        print(f"  Error HTTP {response.status_code} para {yahoo_ticker}")
        return None

    data = response.json()
    try:
        result = data["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "ts_event_utc": pd.to_datetime(result["timestamp"], unit="s", utc=True),
            "open":   quote["open"],
            "high":   quote["high"],
            "low":    quote["low"],
            "close":  quote["close"],
            "volume": quote["volume"],
        })
        df = df.dropna(subset=["close"]).sort_values("ts_event_utc").reset_index(drop=True)
        return df
    except (KeyError, IndexError, TypeError) as e:
        print(f"  Error al parsear respuesta de {yahoo_ticker}: {e}")
        return None


def guardar_market_data() -> None:
    project_root = Path(__file__).resolve().parents[2]
    market_dir = project_root / "data" / "market_data"
    market_dir.mkdir(parents=True, exist_ok=True)

    for nombre, yahoo_ticker in MARKET_SYMBOLS.items():
        print(f"Descargando {nombre} ({yahoo_ticker})...")
        df = descargar_symbol(yahoo_ticker)
        if df is None:
            print(f"  ❌ Fallo al descargar {nombre}")
            continue
        output_path = market_dir / f"{nombre}_1d.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  ✅ {nombre}: {len(df)} filas → {output_path}")
        time.sleep(0.5)


if __name__ == "__main__":
    guardar_market_data()
