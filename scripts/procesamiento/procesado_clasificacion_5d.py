"""Procesado para clasificacion binaria (horizonte 5 dias).

Objetivo: crear dataset Silver para predecir si el precio SUBE o BAJA
en los próximos 5 días de mercado (igual que la regresión, pero binario).

Target: target_updown_t5 = 1 si ret_log_5d > 0, else 0.

Features: idénticos a procesado_regresion.py (Grupo 0 + Grupo 1 + Grupo 2).
- Grupo 0: ret_1d, ret_3d, gap_prop, range_norm, price_vs_ma20, bb_position,
           dist_52w_high, volatility_5d, volatility_20d, rsi, macd, volume, obv_ratio
- Grupo 1: ret_10d, ret_20d, ret_60d
- Grupo 2: spy_ret_1d, spy_ret_5d, spy_ret_20d, vix_level, vix_change_1d,
           vix_change_5d, ret_rel_spy_1d, ret_rel_spy_5d
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from lectura_parquets import cargar_bronze


COLS_REQUERIDAS = {
	"symbol",
	"ts_event_utc",
	"open",
	"high",
	"low",
	"close",
	"volume",
}

COLS_TRAIN = [
	"ret_1d",
	"ret_3d",
	"ret_10d",
	"ret_20d",
	"ret_60d",
	"gap_prop",
	"range_norm",
	"price_vs_ma20",
	"bb_position",
	"dist_52w_high",
	"volatility_5d",
	"volatility_20d",
	"rsi",
	"macd",
	"volume",
	"obv_ratio",
	"target_updown_t5",
]


def cargar_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Carga SPY y VIX desde data/market_data/."""
	project_root = Path(__file__).resolve().parents[2]
	market_dir = project_root / "data" / "market_data"

	spy_path = market_dir / "SPY_1d.parquet"
	vix_path = market_dir / "VIX_1d.parquet"

	if not spy_path.exists() or not vix_path.exists():
		raise FileNotFoundError(
			f"Faltan archivos en {market_dir}. "
			"Ejecuta primero: python scripts/ingesta/ingesta_market_data.py"
		)

	df_spy = pd.read_parquet(spy_path).sort_values("ts_event_utc").reset_index(drop=True)
	df_vix = pd.read_parquet(vix_path).sort_values("ts_event_utc").reset_index(drop=True)

	df_spy["fecha"] = pd.to_datetime(df_spy["ts_event_utc"]).dt.date
	df_vix["fecha"] = pd.to_datetime(df_vix["ts_event_utc"]).dt.date

	return df_spy, df_vix


def crear_features_clasificacion_5d(df: pd.DataFrame) -> pd.DataFrame:
	"""Crea features (idénticos a regresión) y target binario a 5 días."""

	missing = sorted(COLS_REQUERIDAS - set(df.columns))
	if missing:
		raise ValueError(f"Faltan columnas obligatorias en Bronze: {missing}")

	df = df.sort_values(["symbol", "ts_event_utc"]).reset_index(drop=True)

	gb_close = df.groupby("symbol")["close"]

	df["close_prev"] = gb_close.shift(1)

	df["ret_1d"] = (df["close"] / df["close_prev"]) - 1
	df["ret_3d"] = (df["close"] / gb_close.shift(3)) - 1
	df["gap_prop"] = (df["open"] / df["close_prev"]) - 1

	df["ma_20"] = gb_close.transform(lambda s: s.rolling(20).mean()).shift(1)
	df["price_vs_ma20"] = df["close_prev"] / df["ma_20"]
	df["range_norm"] = (df["high"] - df["low"]) / df["ma_20"]

	gb_ret_1d = df.groupby("symbol")["ret_1d"]
	df["volatility_5d"] = gb_ret_1d.transform(lambda s: s.rolling(5).std().shift(1))
	df["volatility_20d"] = gb_ret_1d.transform(lambda s: s.rolling(20).std().shift(1))

	def calcular_rsi(prices, period=14):
		deltas = prices.diff()
		seed = deltas[: period + 1]
		up = seed[seed >= 0].sum() / period
		down = -seed[seed < 0].sum() / period
		rs = up / down if down != 0 else 0
		rsi = np.zeros_like(prices)
		rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
		for i in range(period, len(prices)):
			delta = deltas.iloc[i]
			if delta > 0:
				up = (up * (period - 1) + delta) / period
				down = down * (period - 1) / period
			else:
				up = up * (period - 1) / period
				down = (down * (period - 1) - delta) / period
			rs = up / down if down != 0 else 0
			rsi[i] = 100.0 - 100.0 / (1.0 + rs)
		return pd.Series(rsi, index=prices.index)

	df["rsi"] = gb_close.transform(lambda s: calcular_rsi(s, period=14))

	def calcular_macd(prices):
		return prices.ewm(span=12).mean() - prices.ewm(span=26).mean()

	df["macd"] = gb_close.transform(calcular_macd)

	# Grupo 1: momentum largo plazo
	df["ret_10d"] = (df["close"] / gb_close.shift(10)) - 1
	df["ret_20d"] = (df["close"] / gb_close.shift(20)) - 1
	df["ret_60d"] = (df["close"] / gb_close.shift(60)) - 1

	# Bandas de Bollinger
	bb_std = gb_close.transform(lambda s: s.rolling(20).std().shift(1))
	bb_upper = df["ma_20"] + 2 * bb_std
	bb_lower = df["ma_20"] - 2 * bb_std
	bb_width = bb_upper - bb_lower
	df["bb_position"] = np.where(
		bb_width > 0,
		(df["close_prev"] - bb_lower) / bb_width,
		0.5,
	)
	df["bb_position"] = df["bb_position"].clip(0, 1)

	# Distancia al máximo de 52 semanas
	high_52w = gb_close.transform(lambda s: s.rolling(252).max().shift(1))
	df["dist_52w_high"] = (df["close_prev"] / high_52w) - 1

	# OBV ratio
	_direction = np.sign(df["ret_1d"].fillna(0))
	df["_obv_delta"] = df["volume"] * _direction
	df["_obv"] = df.groupby("symbol")["_obv_delta"].cumsum()
	_obv_ma20 = df.groupby("symbol")["_obv"].transform(lambda s: s.rolling(20).mean())
	df["_obv_ratio"] = df["_obv"] / _obv_ma20.replace(0, np.nan)
	df["obv_ratio"] = df.groupby("symbol")["_obv_ratio"].transform(lambda s: s.shift(1))
	df = df.drop(columns=["_obv_delta", "_obv", "_obv_ratio"])

	# Grupo 2: contexto de mercado (SPY + VIX)
	df_spy, df_vix = cargar_market_data()

	spy_close = df_spy["close"].values
	df_spy["spy_ret_1d"] = (spy_close / df_spy["close"].shift(1).values) - 1
	df_spy["spy_ret_5d"] = (spy_close / df_spy["close"].shift(5).values) - 1
	df_spy["spy_ret_20d"] = (spy_close / df_spy["close"].shift(20).values) - 1

	vix_close = df_vix["close"].values
	df_vix["vix_level"] = vix_close
	df_vix["vix_change_1d"] = (vix_close / df_vix["close"].shift(1).values) - 1
	df_vix["vix_change_5d"] = (vix_close / df_vix["close"].shift(5).values) - 1

	spy_join = df_spy[["fecha", "spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]].copy()
	vix_join = df_vix[["fecha", "vix_level", "vix_change_1d", "vix_change_5d"]].copy()

	df["fecha"] = pd.to_datetime(df["ts_event_utc"]).dt.date
	df = df.merge(spy_join, on="fecha", how="left")
	df = df.merge(vix_join, on="fecha", how="left")
	df[["spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]] = df[
		["spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]
	].ffill()
	df[["vix_level", "vix_change_1d", "vix_change_5d"]] = df[
		["vix_level", "vix_change_1d", "vix_change_5d"]
	].ffill()

	ret_5d_ticker = (df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5))) - 1
	df["ret_rel_spy_1d"] = df["ret_1d"] - df["spy_ret_1d"]
	df["ret_rel_spy_5d"] = ret_5d_ticker - df["spy_ret_5d"]

	df = df.drop(columns=["fecha"])

	# Target binario: 1 si el precio sube en 5 días, 0 si baja o igual
	next_close = df.groupby("symbol")["close"].shift(-5)
	ret_log_5d = np.log(next_close / df["close"])
	df["target_updown_t5"] = (ret_log_5d > 0).astype("Int64")

	df = df.replace([np.inf, -np.inf], np.nan)

	return df


def preparar_para_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
	"""Filtra columnas eliminando data leakage (open/high/low/close de hoy)."""
	cols_permitidas = [
		"symbol",
		"ts_event_utc",
		"ret_1d",
		"ret_3d",
		"ret_10d",
		"ret_20d",
		"ret_60d",
		"gap_prop",
		"range_norm",
		"price_vs_ma20",
		"bb_position",
		"dist_52w_high",
		"volatility_5d",
		"volatility_20d",
		"rsi",
		"macd",
		"volume",
		"obv_ratio",
		"spy_ret_1d",
		"spy_ret_5d",
		"spy_ret_20d",
		"ret_rel_spy_1d",
		"ret_rel_spy_5d",
		"vix_level",
		"vix_change_1d",
		"vix_change_5d",
		"target_updown_t5",
	]
	return df[cols_permitidas].copy()


def limpiar_para_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
	"""Elimina nulos de columnas core para entrenar."""
	return df.dropna(subset=COLS_TRAIN).copy()


def guardar_silver(df: pd.DataFrame) -> None:
	"""Guarda dataset Silver de clasificación 5d."""
	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver" / "clasificacion"
	silver_dir.mkdir(parents=True, exist_ok=True)

	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")
	output_path = silver_dir / f"clasificacion_5d_{fecha_ejecucion}.parquet"
	df.to_parquet(output_path, index=False)
	print(f"Dataset Silver guardado: {output_path}")
	print(f"Filas finales: {len(df)}")
	print(f"Columnas: {list(df.columns)}")
	dist = df["target_updown_t5"].value_counts().to_dict()
	total = len(df)
	print(f"Distribución target: {dist} → up={dist.get(1,0)/total:.1%} | down={dist.get(0,0)/total:.1%}")


def main() -> None:
	df = cargar_bronze()
	print(f"Filas Bronze: {len(df)} | Símbolos: {df['symbol'].nunique()}")

	df_proc = crear_features_clasificacion_5d(df)

	print("\n[FILTRACIÓN] Eliminando columnas con data leakage...")
	print(f"  Antes: {len(df_proc.columns)} columnas")
	df_proc = preparar_para_entrenamiento(df_proc)
	print(f"  Después: {len(df_proc.columns)} columnas")

	df_train = limpiar_para_entrenamiento(df_proc)
	guardar_silver(df_train)


if __name__ == "__main__":
	main()
