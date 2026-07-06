"""Backtest portfolio top-N: cada 5 días selecciona los mejores tickers por confianza.

Estrategia (igual que en producción real):
- Cada 5 días se corren todos los modelos disponibles.
- Cada modelo da P(sube) para su ticker.
- Se compran los top-N tickers con mayor P(sube) (solo si supera umbral_min).
- Capital distribuido equitativamente entre los N seleccionados.
- A los 5 días se cierra todo y se repite.

Uso:
    python backtest_portfolio_topn.py
    python backtest_portfolio_topn.py --top 3 --umbral 0.55
    python backtest_portfolio_topn.py --top 5 --umbral 0.52
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELOS_DIR = PROJECT_ROOT / "modelos" / "clasificacion"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "clasificacion"

HORIZON = 5
TRAIN_PCT = 0.70
VALID_PCT = 0.15
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
    archivos = list(SILVER_DIR.glob("clasificacion_5d_*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No hay Silver en {SILVER_DIR}")
    return pd.read_parquet(sorted(archivos)[-1])


def cargar_modelos() -> dict[str, TabularPredictor]:
    """Carga todos los modelos Clf_AI disponibles."""
    modelos = {}
    for d in sorted(MODELOS_DIR.glob("Clf_AI_Ticker_*")):
        parts = d.name.split("_")
        if len(parts) >= 4:
            ticker = parts[3]
            if ticker not in modelos:
                modelos[ticker] = TabularPredictor.load(str(d))
    return modelos


def get_test_set(df_ticker: pd.DataFrame) -> pd.DataFrame:
    df_ticker = df_ticker.sort_values("ts_event_utc").reset_index(drop=True)
    n = len(df_ticker)
    inicio = int(n * (TRAIN_PCT + VALID_PCT))
    return df_ticker.iloc[inicio:].copy()


def calcular_score(prob_up: float, vol_5d: float) -> float:
    """Score = P(sube). Ordenamos por confianza del modelo, sin más."""
    return prob_up


def simular_portfolio(
    df_silver: pd.DataFrame,
    modelos: dict[str, TabularPredictor],
    top_n: int,
    umbral_min: float,
    capital_inicial: float,
    comision_pct: float,
) -> tuple[list[dict], dict]:
    """Simula la estrategia portfolio top-N."""

    tickers = list(modelos.keys())

    # Preparar test sets por ticker
    test_sets: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df_t = df_silver[df_silver["symbol"] == ticker]
        if df_t.empty:
            continue
        df_test = get_test_set(df_t)
        if len(df_test) > HORIZON:
            test_sets[ticker] = df_test.reset_index(drop=True)

    if not test_sets:
        raise ValueError("No hay tickers con datos de test disponibles.")

    # Determinar fechas comunes (intersección)
    fechas_por_ticker = {
        t: set(df["ts_event_utc"].astype(str).tolist())
        for t, df in test_sets.items()
    }
    todas_fechas = sorted(set.union(*fechas_por_ticker.values()))

    # Pre-computar probabilidades por ticker
    print(f"  Calculando probabilidades para {len(test_sets)} tickers...")
    proba_por_ticker: dict[str, pd.Series] = {}
    vol_por_ticker: dict[str, pd.Series] = {}
    for ticker, df_test in test_sets.items():
        proba_df = modelos[ticker].predict_proba(df_test[FEATURES])
        proba_por_ticker[ticker] = pd.Series(
            proba_df[1].values if hasattr(proba_df, "columns") else proba_df[:, 1],
            index=df_test["ts_event_utc"].astype(str).tolist(),
        )
        vol_por_ticker[ticker] = pd.Series(
            df_test["volatility_5d"].values,
            index=df_test["ts_event_utc"].astype(str).tolist(),
        )

    # Simular: cada HORIZON días, escoger top-N
    capital = capital_inicial
    trades_log: list[dict] = []
    peak = capital
    max_dd = 0.0

    i = 0
    while i < len(todas_fechas) - HORIZON:
        fecha_entrada = todas_fechas[i]
        fecha_salida = todas_fechas[i + HORIZON] if i + HORIZON < len(todas_fechas) else None
        if fecha_salida is None:
            break

        # Calcular score para cada ticker en esta fecha
        candidatos = []
        for ticker in test_sets:
            if fecha_entrada not in proba_por_ticker[ticker]:
                continue
            if fecha_salida not in proba_por_ticker[ticker]:
                continue

            prob = float(proba_por_ticker[ticker][fecha_entrada])
            if prob <= umbral_min:
                continue

            vol = float(vol_por_ticker[ticker].get(fecha_entrada, np.nan))
            score = calcular_score(prob, vol)
            if score <= 0:
                continue

            candidatos.append({
                "ticker": ticker,
                "prob_up": prob,
                "vol_5d": vol,
                "score": score,
            })

        if not candidatos:
            i += HORIZON
            continue

        # Ordenar por score y tomar top-N
        candidatos.sort(key=lambda x: x["score"], reverse=True)
        seleccionados = candidatos[:top_n]

        capital_por_pos = capital / len(seleccionados)
        retorno_periodo = 0.0
        detalles_pos = []

        for pos in seleccionados:
            ticker = pos["ticker"]
            df_test = test_sets[ticker]

            idx_entrada = df_test.index[
                df_test["ts_event_utc"].astype(str) == fecha_entrada
            ]
            idx_salida = df_test.index[
                df_test["ts_event_utc"].astype(str) == fecha_salida
            ]
            if len(idx_entrada) == 0 or len(idx_salida) == 0:
                continue

            # Retorno acumulado de los HORIZON días siguientes
            i_e = int(idx_entrada[0])
            i_s = int(idx_salida[0])
            ret_1d_vals = df_test["ret_1d"].values
            if i_s > i_e:
                ret_real = np.prod(1 + ret_1d_vals[i_e + 1:i_s + 1]) - 1
            else:
                ret_real = 0.0

            ret_log = np.log(1 + ret_real) if ret_real > -1 else -10.0
            ret_neto = ret_log - 2 * comision_pct
            ret_pct = float(np.expm1(ret_neto))
            pnl = capital_por_pos * ret_pct

            retorno_periodo += pnl
            detalles_pos.append({
                "ticker": ticker,
                "prob_up": pos["prob_up"],
                "vol_5d": pos["vol_5d"],
                "score": pos["score"],
                "ret_pct": ret_pct * 100,
                "pnl": pnl,
            })

        capital += retorno_periodo
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        if detalles_pos:
            trades_log.append({
                "fecha_entrada": fecha_entrada,
                "fecha_salida": fecha_salida,
                "n_posiciones": len(detalles_pos),
                "capital_antes": capital - retorno_periodo,
                "capital_despues": capital,
                "retorno_periodo_pct": retorno_periodo / (capital - retorno_periodo) * 100
                if (capital - retorno_periodo) > 0 else 0.0,
                "posiciones": detalles_pos,
            })

        i += HORIZON

    resumen = {
        "capital_inicial": capital_inicial,
        "capital_final": capital,
        "retorno_total_pct": (capital / capital_inicial - 1) * 100,
        "num_periodos": len(trades_log),
        "max_drawdown_pct": max_dd * 100,
        "tickers_usados": sorted(set(
            p["ticker"] for t in trades_log for p in t["posiciones"]
        )),
    }
    return trades_log, resumen


def imprimir_resumen(trades_log: list[dict], resumen: dict, top_n: int, umbral: float) -> None:
    print("\n" + "=" * 70)
    print("BACKTEST PORTFOLIO TOP-N (out-of-sample)")
    print("=" * 70)
    print(f"Selección: top-{top_n} tickers por mayor P(sube) ese día")
    print(f"Umbral mínimo P(sube): {umbral:.2f}")
    print(f"Capital inicial: ${resumen['capital_inicial']:,.0f}")
    print()
    print(f"  Capital final:     ${resumen['capital_final']:>10,.0f}")
    print(f"  Retorno total:     {resumen['retorno_total_pct']:>+.2f}%")
    print(f"  Períodos (5d):     {resumen['num_periodos']}")
    print(f"  Max Drawdown:      {resumen['max_drawdown_pct']:.1f}%")
    print(f"  Tickers operados:  {', '.join(resumen['tickers_usados'])}")

    if not trades_log:
        return

    rets = [t["retorno_periodo_pct"] for t in trades_log]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    print()
    print(f"  Win rate períodos: {len(wins)/len(rets):.1%} ({len(wins)}/{len(rets)})")
    print(f"  Avg retorno/período: {np.mean(rets):+.2f}%")
    print(f"  Avg win:  {np.mean(wins):+.2f}%" if wins else "  Avg win:  —")
    print(f"  Avg loss: {np.mean(losses):+.2f}%" if losses else "  Avg loss: —")

    print("\n  Últimos 5 períodos:")
    print(f"  {'Entrada':<12} {'Salida':<12} {'Posiciones':<30} {'Ret%':>7}")
    print("  " + "-" * 65)
    for t in trades_log[-5:]:
        tickers_str = ",".join(p["ticker"] for p in t["posiciones"])
        print(
            f"  {t['fecha_entrada'][:10]:<12} {t['fecha_salida'][:10]:<12} "
            f"{tickers_str:<30} {t['retorno_periodo_pct']:>+6.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest portfolio top-N")
    parser.add_argument("--top", type=int, default=3, help="Número de posiciones por período (default: 3)")
    parser.add_argument("--umbral", type=float, default=0.52, help="P(sube) mínimo para considerar un ticker (default: 0.52)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--comision", type=float, default=0.001)
    parser.add_argument(
        "--tickers", type=str, default="",
        help="Tickers separados por coma. Si no se especifica, usa todos los modelos disponibles."
    )
    args = parser.parse_args()

    print("Cargando Silver...")
    df_silver = cargar_silver()

    print("Cargando modelos...")
    modelos = cargar_modelos()

    if args.tickers:
        filtro = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        modelos = {t: m for t, m in modelos.items() if t in filtro}

    print(f"  Modelos disponibles: {sorted(modelos.keys())}")

    print(f"\nSimulando portfolio top-{args.top} (umbral={args.umbral})...")
    trades_log, resumen = simular_portfolio(
        df_silver, modelos,
        top_n=args.top,
        umbral_min=args.umbral,
        capital_inicial=args.capital,
        comision_pct=args.comision,
    )

    imprimir_resumen(trades_log, resumen, args.top, args.umbral)


if __name__ == "__main__":
    main()
