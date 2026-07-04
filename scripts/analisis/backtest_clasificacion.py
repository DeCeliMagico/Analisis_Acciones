"""Backtest para modelos de clasificacion binaria (horizonte 5d).

Estrategia:
- El modelo devuelve P(sube) = probabilidad de que el precio suba en 5 días.
- Solo se opera cuando la confianza supera el umbral:
    long  si P(sube) > umbral_largo   (ej: 0.55)
    short si P(sube) < umbral_corto   (ej: 0.45) → solo en modo long_short
- Esto implementa simultáneamente los pasos B (clasificación) y C (umbral).

Uso:
    python backtest_clasificacion.py --modo long_only
    python backtest_clasificacion.py --modo long_short --umbral 0.55
    python backtest_clasificacion.py --modo long_only --tickers NVDA,CSCO,GOOG,KLAC,MSFT
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELOS_DIR = PROJECT_ROOT / "modelos" / "clasificacion"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "clasificacion"
EVAL_DIR = PROJECT_ROOT / "evaluaciones"

HORIZON_DAYS = 5
TRAIN_PCT = 0.7
VALID_PCT = 0.15


@dataclass
class BacktestConfig:
	capital_inicial: float = 10_000.0
	comision_pct: float = 0.001
	modo: str = "long_only"
	tickers: list[str] | None = None
	umbral_largo: float = 0.5    # P(sube) > umbral_largo → long
	umbral_corto: float = 0.5    # P(sube) < umbral_corto → short (1 - umbral_largo)
	fecha_inicio_test: str | None = None
	fecha_fin_test: str | None = None


@dataclass
class TradeResult:
	ticker: str
	fecha_entrada: str
	fecha_salida: str
	direccion: str
	prob_up: float
	retorno_log: float
	retorno_pct: float
	pnl: float
	capital_despues: float


def cargar_silver() -> pd.DataFrame:
	archivos = list(SILVER_DIR.glob("clasificacion_5d_*.parquet"))
	if not archivos:
		raise FileNotFoundError(
			f"No hay Silver de clasificacion_5d en {SILVER_DIR}\n"
			"Ejecuta: python scripts/procesamiento/procesado_clasificacion_5d.py"
		)
	return pd.read_parquet(sorted(archivos)[-1])


def detectar_tickers_modelos() -> list[str]:
	tickers = set()
	for d in MODELOS_DIR.glob("Clf_AI_Ticker_*"):
		parts = d.name.split("_")
		if len(parts) >= 4:
			tickers.add(parts[3])
	return sorted(tickers)


def cargar_modelo(ticker: str) -> TabularPredictor:
	dirs = sorted(MODELOS_DIR.glob(f"Clf_AI_Ticker_{ticker}_*"))
	if not dirs:
		raise FileNotFoundError(f"No hay modelo de clasificacion para {ticker}")
	return TabularPredictor.load(str(dirs[-1]))


def split_test(df_ticker: pd.DataFrame) -> pd.DataFrame:
	df_ticker = df_ticker.sort_values("ts_event_utc").reset_index(drop=True)
	n = len(df_ticker)
	n_train = int(n * TRAIN_PCT)
	n_valid = int(n * VALID_PCT)
	return df_ticker.iloc[n_train + n_valid :].copy()


def split_test_subperiodo(
	df_ticker: pd.DataFrame,
	fecha_inicio: str | None = None,
	fecha_fin: str | None = None,
) -> pd.DataFrame:
	df_test = split_test(df_ticker)
	if fecha_inicio is None or fecha_fin is None:
		return df_test
	df_test = df_test.sort_values("ts_event_utc")
	return df_test[
		(df_test["ts_event_utc"] >= fecha_inicio) & (df_test["ts_event_utc"] <= fecha_fin)
	].copy()


def log_a_pct(ret_log: float) -> float:
	return float(np.expm1(ret_log))


def simular_ticker(
	ticker: str,
	df_test: pd.DataFrame,
	predictor: TabularPredictor,
	config: BacktestConfig,
	capital_inicial: float,
) -> tuple[list[TradeResult], float, dict]:
	"""Simula operaciones usando probabilidades de clasificación."""

	capital = capital_inicial
	df = df_test.sort_values("ts_event_utc").reset_index(drop=True)

	# Obtener P(sube) para cada fila
	proba_df = predictor.predict_proba(df)
	if hasattr(proba_df, "columns"):
		proba_up = proba_df[1].values
	else:
		proba_up = proba_df[:, 1]

	fechas = pd.to_datetime(df["ts_event_utc"]).values
	targets = df["target_updown_t5"].values  # 1=sube, 0=baja

	# Necesitamos el retorno real para calcular P&L: lo derivamos del target_updown_t5
	# y de ret_1d acumulado. Pero el Silver de clasificación no guarda ret_log_5d.
	# Aproximación: usamos el signo del target + volatilidad como retorno base.
	# Mejor: calcular desde ret_1d los 5 días siguientes cuando estén disponibles.
	# Usamos los campos disponibles: target_updown_t5 y ret_1d rolling
	ret_1d_vals = df["ret_1d"].values if "ret_1d" in df.columns else np.zeros(len(df))

	trades: list[TradeResult] = []
	i = 0
	wins = 0
	peak = capital
	max_dd = 0.0
	trades_saltados = 0

	while i < len(df):
		prob = float(proba_up[i])

		# Filtro de confianza
		if config.modo == "long_only":
			if prob <= config.umbral_largo:
				i += 1
				trades_saltados += 1
				continue
			direccion = "long"
		else:
			if prob > config.umbral_largo:
				direccion = "long"
			elif prob < config.umbral_corto:
				direccion = "short"
			else:
				i += 1
				trades_saltados += 1
				continue

		j = i + HORIZON_DAYS
		if j >= len(df):
			break

		# Calcular retorno real acumulado de los 5 días siguientes
		ret_real_5d = np.prod(1 + ret_1d_vals[i + 1 : j + 1]) - 1
		ret_log_bruto = np.log(1 + ret_real_5d) if ret_real_5d > -1 else -10.0

		if direccion == "short":
			ret_log_bruto = -ret_log_bruto

		costo = 2 * config.comision_pct
		ret_log_neto = ret_log_bruto - costo
		ret_pct_neto = log_a_pct(ret_log_neto)
		pnl = capital * ret_pct_neto
		capital += pnl

		if ret_log_neto > 0:
			wins += 1

		peak = max(peak, capital)
		dd = (peak - capital) / peak if peak > 0 else 0.0
		max_dd = max(max_dd, dd)

		trades.append(
			TradeResult(
				ticker=ticker,
				fecha_entrada=str(fechas[i]),
				fecha_salida=str(fechas[j]),
				direccion=direccion,
				prob_up=prob,
				retorno_log=ret_log_neto,
				retorno_pct=ret_pct_neto,
				pnl=pnl,
				capital_despues=capital,
			)
		)

		i = j + 1

	n_trades = len(trades)
	resumen = {
		"num_trades": n_trades,
		"trades_saltados_umbral": trades_saltados,
		"win_rate": wins / n_trades if n_trades else 0.0,
		"capital_inicial": capital_inicial,
		"capital_final": capital,
		"retorno_total_pct": (capital / capital_inicial - 1) * 100,
		"max_drawdown_pct": max_dd * 100,
		"avg_prob_up_operada": float(np.mean([t.prob_up for t in trades])) if trades else 0.0,
	}

	return trades, capital, resumen


def buy_and_hold_benchmark(df_test: pd.DataFrame, capital: float, comision_pct: float) -> dict:
	if len(df_test) < 2:
		return {"retorno_total_pct": 0.0, "capital_final": capital}
	df = df_test.sort_values("ts_event_utc").reset_index(drop=True)
	if "ret_1d" in df.columns:
		ret_acumulado = np.prod(1 + df["ret_1d"].values) - 1
		ret_log = np.log(1 + ret_acumulado) if ret_acumulado > -1 else np.log(1e-10)
	else:
		return {"retorno_total_pct": 0.0, "capital_final": capital}
	ret_log_neto = ret_log - 2 * comision_pct
	ret_pct = log_a_pct(ret_log_neto)
	capital_final = capital * (1 + ret_pct)
	return {"retorno_total_pct": (capital_final / capital - 1) * 100, "capital_final": capital_final}


def ejecutar_backtest(config: BacktestConfig) -> dict:
	df = cargar_silver()
	tickers = config.tickers or detectar_tickers_modelos()
	if not tickers:
		raise FileNotFoundError("No hay modelos de clasificación en modelos/Clf_AI_*")

	resultados: dict = {
		"config": asdict(config),
		"fecha_ejecucion": datetime.now().strftime("%d-%m-%y_%H%M%S"),
		"horizonte_dias": HORIZON_DAYS,
		"periodo": "test (15% final temporal, out-of-sample)",
		"periodo_custom": (
			f"{config.fecha_inicio_test} a {config.fecha_fin_test}"
			if config.fecha_inicio_test
			else "Default (últimos 15%)"
		),
		"tickers": {},
		"portfolio": {},
	}

	capital_por_ticker = config.capital_inicial / len(tickers)
	capital_total = 0.0
	all_trades: list[TradeResult] = []

	for ticker in tickers:
		df_t = df[df["symbol"] == ticker]
		if df_t.empty:
			print(f"  AVISO: {ticker} no encontrado en Silver de clasificacion")
			continue

		if config.fecha_inicio_test and config.fecha_fin_test:
			df_test = split_test_subperiodo(df_t, config.fecha_inicio_test, config.fecha_fin_test)
		else:
			df_test = split_test(df_t)

		if df_test.empty:
			continue

		try:
			predictor = cargar_modelo(ticker)
		except FileNotFoundError as e:
			print(f"  AVISO: {e}")
			continue

		trades, cap_final, resumen = simular_ticker(
			ticker, df_test, predictor, config, capital_por_ticker
		)
		bh = buy_and_hold_benchmark(df_test, capital_por_ticker, config.comision_pct)

		resultados["tickers"][ticker] = {
			**resumen,
			"buy_hold_retorno_pct": bh["retorno_total_pct"],
			"buy_hold_capital_final": bh["capital_final"],
			"periodo_test_inicio": str(df_test["ts_event_utc"].min()),
			"periodo_test_fin": str(df_test["ts_event_utc"].max()),
			"filas_test": len(df_test),
		}
		capital_total += cap_final
		all_trades.extend(trades)

	resultados["portfolio"] = {
		"capital_inicial_total": config.capital_inicial,
		"capital_final_total": capital_total,
		"retorno_total_pct": (capital_total / config.capital_inicial - 1) * 100,
		"num_trades_total": len(all_trades),
		"tickers_operados": len(resultados["tickers"]),
	}

	return resultados


def imprimir_resumen(resultados: dict) -> None:
	print("=" * 100)
	print("BACKTEST CLASIFICACION (periodo TEST, out-of-sample)")
	print("=" * 100)
	cfg = resultados["config"]
	umbral_str = f"largo>{cfg['umbral_largo']:.2f} | corto<{cfg['umbral_corto']:.2f}"
	print(
		f"Modo: {cfg['modo']} | Capital: ${cfg['capital_inicial']:,.0f} | "
		f"Comision: {cfg['comision_pct']*100:.2f}% | Umbral: {umbral_str}"
	)
	print(f"Horizonte: {resultados['horizonte_dias']} días | {resultados['periodo']}")
	print(f"Período custom: {resultados['periodo_custom']}")
	print()

	print(f"{'Ticker':<8} {'Fecha Inicio':<32} {'Fecha Fin':<32} {'Filas':<8}")
	print("-" * 80)
	for ticker, m in resultados["tickers"].items():
		print(f"{ticker:<8} {m['periodo_test_inicio']:<32} {m['periodo_test_fin']:<32} {m['filas_test']:<8}")
	print()

	print(
		f"{'Ticker':<8} {'Trades':>7} {'Saltados':>9} {'Win%':>7} "
		f"{'Ret.Modelo':>12} {'Ret.B&H':>12} {'MaxDD':>8} {'Capital':>12}"
	)
	print("-" * 100)
	for ticker, m in resultados["tickers"].items():
		print(
			f"{ticker:<8} {m['num_trades']:>7} {m['trades_saltados_umbral']:>9} "
			f"{m['win_rate']:>6.1%} {m['retorno_total_pct']:>11.2f}% "
			f"{m['buy_hold_retorno_pct']:>11.2f}% {m['max_drawdown_pct']:>7.1f}% "
			f"${m['capital_final']:>10,.0f}"
		)

	p = resultados["portfolio"]
	print("-" * 100)
	print(
		f"{'TOTAL':<8} {p['num_trades_total']:>7} {'':>9} {'':>7} "
		f"{p['retorno_total_pct']:>11.2f}% {'':>12} {'':>8} "
		f"${p['capital_final_total']:>10,.0f}"
	)
	print()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Backtest con modelos de clasificacion binaria (5d)"
	)
	parser.add_argument(
		"--modo",
		choices=["long_only", "long_short"],
		default="long_only",
	)
	parser.add_argument("--capital", type=float, default=10_000.0)
	parser.add_argument("--comision", type=float, default=0.001)
	parser.add_argument(
		"--tickers",
		type=str,
		default="",
		help="Tickers separados por coma. Default: todos los modelos Clf_AI_*",
	)
	parser.add_argument(
		"--umbral",
		type=float,
		default=0.5,
		help=(
			"Umbral de confianza. Solo opera cuando P(sube) > umbral (long) "
			"o P(sube) < 1-umbral (short). Default: 0.5 (sin filtro)"
		),
	)
	parser.add_argument("--fecha-inicio", type=str, default=None)
	parser.add_argument("--fecha-fin", type=str, default=None)
	args = parser.parse_args()

	tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
	config = BacktestConfig(
		capital_inicial=args.capital,
		comision_pct=args.comision,
		modo=args.modo,
		tickers=tickers,
		umbral_largo=args.umbral,
		umbral_corto=1.0 - args.umbral,
		fecha_inicio_test=args.fecha_inicio,
		fecha_fin_test=args.fecha_fin,
	)

	resultados = ejecutar_backtest(config)
	imprimir_resumen(resultados)


if __name__ == "__main__":
	main()
