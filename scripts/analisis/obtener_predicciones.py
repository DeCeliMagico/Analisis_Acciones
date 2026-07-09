"""Genera el ranking de predicciones actual para todos los modelos entrenados.

Para cada ticker con modelo disponible, coge la última fila del Silver
(el día más reciente con datos) y pide al modelo su P(sube).
Muestra TODOS los tickers ordenados de mayor a menor probabilidad.

El flujo normal es:
  1. Ejecutar backtest_clasificacion.py -> te da el comando con los buenos tickers.
  2. Copiar ese comando aqui para ver las predicciones solo de esos tickers.

Uso:
    python obtener_predicciones.py
    python obtener_predicciones.py --umbral 0.52
    python obtener_predicciones.py --umbral 0.52 --tickers MU,AMD,FCX,KLAC,AMAT,NVDA,CSCO,BAC,LRCX
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from autogluon.tabular import TabularPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELOS_DIR = PROJECT_ROOT / "modelos" / "clasificacion"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "clasificacion"

# Exactamente las mismas features que se usaron para entrenar los modelos.
# El orden y los nombres deben coincidir con los del script de entrenamiento.
FEATURES = [
    "ret_1d", "ret_3d", "ret_10d", "ret_20d", "ret_60d",
    "gap_prop", "range_norm", "price_vs_ma20", "bb_position", "dist_52w_high",
    "volatility_5d", "volatility_20d",
    "rsi", "macd", "volume", "obv_ratio",
    "spy_ret_1d", "spy_ret_5d", "spy_ret_20d",
    "ret_rel_spy_1d", "ret_rel_spy_5d",
    "vix_level", "vix_change_1d", "vix_change_5d",
]


def cargar_silver() -> pd.DataFrame:
    # Carga el archivo Silver mas reciente (el nombre incluye timestamp)
    archivos = list(SILVER_DIR.glob("clasificacion_5d_*.parquet"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay Silver en {SILVER_DIR}\n"
            "Ejecuta: python scripts/procesamiento/procesado_clasificacion_5d.py"
        )
    return pd.read_parquet(sorted(archivos)[-1])


def cargar_modelos() -> dict[str, TabularPredictor]:
    """Carga todos los modelos Clf_AI disponibles.

    El nombre de cada carpeta tiene el formato: Clf_AI_Ticker_NVDA_2026-07-06_...
    Se extrae el ticker de la 4a parte del nombre (indice 3).
    Si hay varios modelos para el mismo ticker (reentrenamientos), se queda
    con el primero que encuentra al ordenar alfabeticamente.
    """
    modelos: dict[str, TabularPredictor] = {}
    for d in sorted(MODELOS_DIR.glob("Clf_AI_Ticker_*")):
        parts = d.name.split("_")
        if len(parts) >= 4:
            ticker = parts[3]
            if ticker not in modelos:
                modelos[ticker] = TabularPredictor.load(str(d))
    return modelos


def obtener_ultima_fila(df_silver: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Devuelve la fila mas reciente del ticker en el Silver.

    Esta fila contiene las features calculadas con todos los datos disponibles
    hasta ese dia: retornos, volatilidad, RSI, MACD, SPY, VIX, etc.
    Es lo que se le pasa al modelo para que prediga el movimiento futuro.
    """
    df_t = df_silver[df_silver["symbol"] == ticker]
    if df_t.empty:
        return None
    df_t = df_t.sort_values("ts_event_utc")
    return df_t.iloc[-1]


def calcular_predicciones(
    df_silver: pd.DataFrame,
    modelos: dict[str, TabularPredictor],
) -> list[dict]:
    """Pide a cada modelo su P(sube) para el dia mas reciente disponible.

    predict_proba devuelve probabilidades para cada clase:
      columna 0 -> P(no sube)
      columna 1 -> P(sube)  <- esto es lo que usamos para ordenar
    """
    resultados = []
    for ticker, modelo in modelos.items():
        fila = obtener_ultima_fila(df_silver, ticker)
        if fila is None:
            print(f"  [{ticker}] sin datos en Silver, omitido.")
            continue

        features_faltantes = [f for f in FEATURES if f not in fila.index]
        if features_faltantes:
            print(f"  [{ticker}] faltan features {features_faltantes}, omitido.")
            continue

        # El modelo espera un DataFrame (aunque sea de una sola fila), no una Series
        fila_df = fila[FEATURES].to_frame().T.reset_index(drop=True)
        proba_df = modelo.predict_proba(fila_df)

        # AutoGluon puede devolver DataFrame o array segun la version; ambos se manejan igual
        if hasattr(proba_df, "columns"):
            prob_up = float(proba_df[1].iloc[0])
        else:
            prob_up = float(proba_df[0, 1])

        fecha = str(fila["ts_event_utc"])[:10]
        resultados.append({
            "ticker": ticker,
            "fecha": fecha,
            "prob_sube": prob_up,
        })

    # Ordenar de mayor a menor probabilidad: los mejores candidatos van arriba
    resultados.sort(key=lambda x: x["prob_sube"], reverse=True)
    return resultados


def imprimir_ranking(resultados: list[dict], umbral: float) -> None:
    ahora = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print()
    print("=" * 55)
    print(f"  RANKING DE PREDICCIONES -- {ahora}")
    print(f"  umbral referencia P(sube) > {umbral:.2f} | horizonte 5d")
    print("=" * 55)
    print(f"  {'#':<4} {'Ticker':<8} {'Fecha':<12} {'P(sube)':>8}")
    print("  " + "-" * 38)

    for i, r in enumerate(resultados, 1):
        print(f"  {i:<4} {r['ticker']:<8} {r['fecha']:<12} {r['prob_sube']:>7.1%}")

    print("=" * 55)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranking de predicciones actuales")
    parser.add_argument(
        "--umbral", type=float, default=0.52,
        help="Umbral de referencia P(sube) que se muestra en la cabecera (default: 0.52). "
             "No filtra el output: siempre se muestran todos los tickers.",
    )
    parser.add_argument(
        "--tickers", type=str, default="",
        help="Tickers separados por coma. Si no se especifica, usa todos los modelos disponibles.",
    )
    args = parser.parse_args()

    print("Cargando Silver...")
    df_silver = cargar_silver()

    print("Cargando modelos...")
    modelos = cargar_modelos()

    # Si se pasan tickers concretos (normalmente los que dio backtest_clasificacion), filtrar
    if args.tickers:
        filtro = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        modelos = {t: m for t, m in modelos.items() if t in filtro}

    print(f"  {len(modelos)} modelos cargados: {sorted(modelos.keys())}")

    print("Calculando predicciones...")
    resultados = calcular_predicciones(df_silver, modelos)

    imprimir_ranking(resultados, umbral=args.umbral)


if __name__ == "__main__":
    main()
