"""Backtest de modelos por ticker sobre datos historicos (periodo test).

MODIFICACIÓN: Ahora permite analizar sub-períodos DENTRO del test (15% final)
sin contaminar el entrenamiento.

Simula paper trading usando la senal direccional del modelo:
- pred > 0 -> largo (comprar)
- pred < 0 -> corto (solo si modo long_short) o quedarse en cash (modo long_only)

Cada operacion mantiene posicion 5 dias (horizonte del target).
El backtest usa SOLO el split de test (15% final temporal): datos que el modelo
no vio durante el entrenamiento.

CAMBIO: buy_and_hold_benchmark ahora usa target_ret_log_t5 en lugar de close
(close ya no está en el Silver tras el filtrado).
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
MODELOS_DIR = PROJECT_ROOT / "modelos" / "regresion"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "regresion"
EVAL_DIR = PROJECT_ROOT / "evaluaciones"

HORIZON_DAYS = 5
TRAIN_PCT = 0.7
VALID_PCT = 0.15


@dataclass
class BacktestConfig:
	capital_inicial: float = 10_000.0
	comision_pct: float = 0.001  # 0.1% por lado (entrada + salida)
	modo: str = "long_only"  # long_only | long_short
	tickers: list[str] | None = None  # None = detectar de modelos/
	min_pred_abs: float = 0.0  # umbral minimo |pred| para operar
	fecha_inicio_test: str | None = None  # NUEVO: fecha inicio para sub-período
	fecha_fin_test: str | None = None  # NUEVO: fecha fin para sub-período


@dataclass
class TradeResult:
	ticker: str
	fecha_entrada: str
	fecha_salida: str
	direccion: str
	retorno_log: float
	retorno_pct: float
	pnl: float
	capital_despues: float


def cargar_silver() -> pd.DataFrame:
	archivos = list(SILVER_DIR.glob("regresion_5d_*.parquet"))
	if not archivos:
		raise FileNotFoundError(f"No hay Silver de regresion en {SILVER_DIR}")
	return pd.read_parquet(sorted(archivos)[-1])


def detectar_tickers_modelos() -> list[str]:
	tickers = set()
	for d in MODELOS_DIR.glob("Market_AI_Ticker_*"):
		parts = d.name.split("_")
		if len(parts) >= 4:
			tickers.add(parts[3])
	return sorted(tickers)


def cargar_modelo(ticker: str) -> TabularPredictor:
	dirs = sorted(MODELOS_DIR.glob(f"Market_AI_Ticker_{ticker}_*"))
	if not dirs:
		raise FileNotFoundError(f"No hay modelo para {ticker}")
	return TabularPredictor.load(str(dirs[-1]))


def split_test(df_ticker: pd.DataFrame) -> pd.DataFrame:
	"""Retorna el 15% final temporal (sin modificaciones)."""
	df_ticker = df_ticker.sort_values("ts_event_utc").reset_index(drop=True)
	n = len(df_ticker)
	n_train = int(n * TRAIN_PCT)
	n_valid = int(n * VALID_PCT)
	return df_ticker.iloc[n_train + n_valid :].copy()


def split_test_subperiodo(df_ticker: pd.DataFrame, 
                          fecha_inicio: str | None = None, 
                          fecha_fin: str | None = None) -> pd.DataFrame:
	"""Retorna un sub-período DENTRO del 15% test.
	
	IMPORTANTE: Solo debe usarse para dividir el 15% test en sub-períodos.
	No usa datos de train/valid.
	"""
	df_test = split_test(df_ticker)
	
	if fecha_inicio is None or fecha_fin is None:
		return df_test
	
	df_test = df_test.sort_values("ts_event_utc")
	df_filtrado = df_test[
		(df_test["ts_event_utc"] >= fecha_inicio) & 
		(df_test["ts_event_utc"] <= fecha_fin)
	].copy()
	
	return df_filtrado


def log_a_pct(ret_log: float) -> float:
	return float(np.expm1(ret_log))


def simular_ticker(
	ticker: str,
	df_test: pd.DataFrame,
	predictor: TabularPredictor,
	config: BacktestConfig,
	capital_inicial: float,
) -> tuple[list[TradeResult], float, dict]:
	"""Simula operaciones sin solapamiento (max 1 posicion abierta)."""

	capital = capital_inicial
	df = df_test.sort_values("ts_event_utc").reset_index(drop=True)
	preds = predictor.predict(df).values
	fechas = pd.to_datetime(df["ts_event_utc"]).values
	targets = df["target_ret_log_t5"].values

	trades: list[TradeResult] = []
	i = 0
	wins = 0
	peak = capital
	max_dd = 0.0

	while i < len(df):
		pred = float(preds[i])
		if abs(pred) < config.min_pred_abs:
			i += 1
			continue

		if config.modo == "long_only":
			if pred <= 0:
				i += 1
				continue
			direccion = "long"
			ret_log_bruto = float(targets[i])
		else:
			direccion = "long" if pred > 0 else "short"
			ret_log_bruto = float(targets[i]) if direccion == "long" else float(-targets[i])

		j = i + HORIZON_DAYS
		if j >= len(df):
			break

		# Comision round-trip sobre notional
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
		"win_rate": wins / n_trades if n_trades else 0.0,
		"capital_inicial": capital_inicial,
		"capital_final": capital,
		"retorno_total_pct": (capital / capital_inicial - 1) * 100 if capital_inicial else 0.0,
		"max_drawdown_pct": max_dd * 100,
		"avg_ret_trade_pct": float(np.mean([t.retorno_pct for t in trades]) * 100) if trades else 0.0,
	}

	return trades, capital, resumen


def buy_and_hold_benchmark(df_test: pd.DataFrame, capital: float, comision_pct: float) -> dict:
	"""Compra al inicio del test y vende al final (referencia pasiva).
	
	CAMBIO: Usa target_ret_log_t5 en lugar de close (close no está en Silver filtrado).
	Calcula el retorno acumulado usando los retornos diarios y luego aplica comisión.
	"""
	if len(df_test) < 2:
		return {"retorno_total_pct": 0.0, "capital_final": capital}

	df = df_test.sort_values("ts_event_utc").reset_index(drop=True)
	
	# Calcular retorno acumulado usando target_ret_log_t5
	# El target es el retorno esperado en 5 días desde cada punto
	# Para buy_and_hold, tomamos el acumulado de los retornos
	
	# Opción: usar el primer y último target_ret_log_t5 como aproximación
	# Pero más preciso: sumar los ret_1d acumulados
	
	# Si tenemos ret_1d, el retorno acumulado es:
	if "ret_1d" in df.columns:
		# Retorno acumulado = producto de (1 + ret_1d)
		ret_acumulado = np.prod(1 + df["ret_1d"].values) - 1
		ret_log = np.log(1 + ret_acumulado) if ret_acumulado > -1 else np.log(1e-10)
	else:
		# Fallback: usar target_ret_log_t5 del primer y último día
		# Esto es una aproximación
		ret_log_primero = float(df.iloc[0]["target_ret_log_t5"])
		ret_log_ultimo = float(df.iloc[-1]["target_ret_log_t5"])
		ret_log = (ret_log_primero + ret_log_ultimo) / 2  # Aproximación cruda
	
	# Descontar comisión (round-trip)
	ret_log_neto = ret_log - 2 * comision_pct
	ret_pct = log_a_pct(ret_log_neto)
	capital_final = capital * (1 + ret_pct)
	
	return {
		"retorno_total_pct": (capital_final / capital - 1) * 100,
		"capital_final": capital_final,
	}


def ejecutar_backtest(config: BacktestConfig) -> dict:
	df = cargar_silver()
	tickers = config.tickers or detectar_tickers_modelos()
	if not tickers:
		raise FileNotFoundError("No hay modelos en modelos/")

	resultados: dict = {
		"config": asdict(config),
		"fecha_ejecucion": datetime.now().strftime("%d-%m-%y_%H%M%S"),
		"horizonte_dias": HORIZON_DAYS,
		"periodo": "test (15% final temporal, out-of-sample)",
		"periodo_custom": f"{config.fecha_inicio_test} a {config.fecha_fin_test}" if config.fecha_inicio_test else "Default (últimos 15%)",
		"tickers": {},
		"portfolio": {},
	}

	capital_por_ticker = config.capital_inicial / len(tickers)
	capital_total = 0.0
	all_trades: list[TradeResult] = []

	for ticker in tickers:
		df_t = df[df["symbol"] == ticker]
		if df_t.empty:
			continue

		# CAMBIO AQUÍ: usar sub-período si se especifica
		if config.fecha_inicio_test and config.fecha_fin_test:
			df_test = split_test_subperiodo(df_t, config.fecha_inicio_test, config.fecha_fin_test)
		else:
			df_test = split_test(df_t)
		
		if df_test.empty:
			continue

		predictor = cargar_modelo(ticker)
		trades, cap_final, resumen = simular_ticker(
			ticker, df_test, predictor, config, capital_por_ticker
		)
		bh = buy_and_hold_benchmark(df_test, capital_por_ticker, config.comision_pct)

		fecha_ini = str(df_test["ts_event_utc"].min())
		fecha_fin = str(df_test["ts_event_utc"].max())

		resultados["tickers"][ticker] = {
			**resumen,
			"buy_hold_retorno_pct": bh["retorno_total_pct"],
			"buy_hold_capital_final": bh["capital_final"],
			"periodo_test_inicio": fecha_ini,
			"periodo_test_fin": fecha_fin,
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
	print("=" * 90)
	print("BACKTEST PAPER TRADING (periodo TEST, out-of-sample)")
	print("=" * 90)
	cfg = resultados["config"]
	umbral_str = f" | Umbral |pred|>{cfg['min_pred_abs']:.4f}" if cfg['min_pred_abs'] > 0 else ""
	print(f"Modo: {cfg['modo']} | Capital inicial: ${cfg['capital_inicial']:,.0f} | Comision: {cfg['comision_pct']*100:.2f}% por lado{umbral_str}")
	print(f"Horizonte: {resultados['horizonte_dias']} dias | {resultados['periodo']}")
	print(f"Período custom: {resultados['periodo_custom']}")
	print()

	# Mostrar fechas de test por ticker
	print(f"{'Ticker':<8} {'Fecha Inicio':<12} {'Fecha Fin':<12} {'Filas':<8}")
	print("-" * 40)
	for ticker, m in resultados["tickers"].items():
		print(f"{ticker:<8} {m['periodo_test_inicio']:<12} {m['periodo_test_fin']:<12} {m['filas_test']:<8}")
	print()

	print(f"{'Ticker':<8} {'Trades':>7} {'Win%':>7} {'Ret.Modelo':>12} {'Ret.B&H':>12} {'MaxDD':>8} {'Capital':>12}")
	print("-" * 90)

	for ticker, m in resultados["tickers"].items():
		print(
			f"{ticker:<8} {m['num_trades']:>7} {m['win_rate']:>6.1%} "
			f"{m['retorno_total_pct']:>11.2f}% {m['buy_hold_retorno_pct']:>11.2f}% "
			f"{m['max_drawdown_pct']:>7.1f}% ${m['capital_final']:>10,.0f}"
		)

	p = resultados["portfolio"]
	print("-" * 90)
	print(
		f"{'TOTAL':<8} {p['num_trades_total']:>7} {'':>7} "
		f"{p['retorno_total_pct']:>11.2f}% {'':>12} {'':>8} ${p['capital_final_total']:>10,.0f}"
	)
	print()


def main() -> None:
	parser = argparse.ArgumentParser(description="Backtest paper trading con modelos por ticker")
	parser.add_argument(
		"--modo",
		choices=["long_only", "long_short"],
		default="long_only",
		help="long_only: solo compra si pred>0. long_short: compra o vende en corto",
	)
	parser.add_argument("--capital", type=float, default=10_000.0, help="Capital inicial USD")
	parser.add_argument("--comision", type=float, default=0.001, help="Comision por operacion (0.001 = 0.1%%)")
	parser.add_argument(
		"--tickers",
		type=str,
		default="",
		help="Tickers separados por coma (default: todos los modelos). Ej: NVDA,MSFT,CSCO",
	)
	# NUEVOS ARGUMENTOS
	parser.add_argument(
		"--fecha-inicio",
		type=str,
		default=None,
		help="Fecha inicio test custom (YYYY-MM-DD). Si no se especifica, usa últimos 15%",
	)
	parser.add_argument(
		"--fecha-fin",
		type=str,
		default=None,
		help="Fecha fin test custom (YYYY-MM-DD)",
	)
	parser.add_argument(
		"--umbral",
		type=float,
		default=0.0,
		help="Umbral mínimo |pred| para operar (ej: 0.005). 0 = sin filtro",
	)
	args = parser.parse_args()

	tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
	config = BacktestConfig(
		capital_inicial=args.capital,
		comision_pct=args.comision,
		modo=args.modo,
		tickers=tickers,
		min_pred_abs=args.umbral,
		fecha_inicio_test=args.fecha_inicio,
		fecha_fin_test=args.fecha_fin,
	)

	resultados = ejecutar_backtest(config)
	imprimir_resumen(resultados)


if __name__ == "__main__":
	main()