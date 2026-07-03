"""Procesado para regresion (retorno logaritmico semanal).

Objetivo: crear dataset Silver para predecir retorno logaritmico 5 dias adelante.
Target: retorno log puro sin umbral (regresion continua), horizonte: +5 dias.

CAMBIOS IMPORTANTES:
- ma_20: sintaxis más clara con shift(1)
- price_vs_ma20: usa close_prev en lugar de close (sin data leakage)
- preparar_para_entrenamiento(): filtra explícitamente para evitar open/high/low
- Grupo 1: ret_10d, ret_20d, ret_60d, bb_position, dist_52w_high, obv_ratio
- Grupo 2: features de contexto de mercado (SPY, VIX) desde data/market_data/
- Grupo 3 (sector ETFs) probado y descartado: no añade valor sobre SPY/VIX
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
	"target_ret_log_t5",
	# SPY, VIX y sector NO están aquí: AutoGluon maneja NaN en features.
	# Incluirlos forzaría dropna y recortaría el histórico de tickers con
	# datos anteriores al lanzamiento de XLC (junio 2018).
]


def cargar_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Carga SPY y VIX desde data/market_data/.

	Devuelve (df_spy, df_vix) con columna 'fecha' (date) para el join.
	Lanza FileNotFoundError si no existen — ejecutar ingesta_market_data.py primero.
	"""
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



def crear_features_regresion(df: pd.DataFrame) -> pd.DataFrame:
	"""Crea features y target de regresion (retorno logaritmico).

	Nota: aqui usamos solo calculos por simbolo y orden temporal.
	"""

	missing = sorted(COLS_REQUERIDAS - set(df.columns))
	if missing:
		raise ValueError(f"Faltan columnas obligatorias en Bronze: {missing}")

	# Ordenar para que shift y rolling funcionen bien.
	df = df.sort_values(["symbol", "ts_event_utc"]).reset_index(drop=True)

	# Group by simbolo(close) para calculos posteriores.
	gb_close = df.groupby("symbol")["close"]

	# Columna de cierre previo (ayer) por simbolo.
	df["close_prev"] = gb_close.shift(1)

	# Retorno en proporción.
	df["ret_1d"] = (df['close'] / df['close_prev'] ) - 1
	df["ret_3d"] = (df['close'] / gb_close.shift(3)) - 1

	# Gap en proporción.
	df["gap_prop"] = (df['open'] / df['close_prev'] - 1)
	

	# Medias moviles y ratios de tendencia.
	# CAMBIO: sintaxis más clara - shift se aplica DESPUÉS del rolling
	df["ma_20"] = gb_close.transform(lambda s: s.rolling(20).mean()).shift(1)
	
	# CAMBIO: usar close_prev en lugar de close para evitar data leakage de HOY
	df["price_vs_ma20"] = df["close_prev"] / df["ma_20"]

	# Rango en proporcion (está OK, el rango de HOY es información válida al cierre)
	df["range_norm"] = (df["high"] - df["low"]) / df["ma_20"]

	# Group by simbolo de ret_1d para volatilidad.
	gb_ret_1d = df.groupby("symbol")["ret_1d"]

	# Volatilidad (std de retornos) con shift para no incluir retorno de HOY
	df["volatility_5d"] = gb_ret_1d.transform(lambda s: s.rolling(5).std().shift(1))
	df["volatility_20d"] = gb_ret_1d.transform(lambda s: s.rolling(20).std().shift(1))

	# RSI (Relative Strength Index) - mide momentum
	def calcular_rsi(prices, period=14):
		"""Calcula RSI basado en cambios de precio."""
		deltas = prices.diff()
		seed = deltas[:period+1]
		up = seed[seed >= 0].sum() / period
		down = -seed[seed < 0].sum() / period
		rs = up / down if down != 0 else 0
		rsi = np.zeros_like(prices)
		rsi[:period] = 100. - 100. / (1. + rs)
		
		for i in range(period, len(prices)):
			delta = deltas.iloc[i]
			if delta > 0:
				up = (up * (period - 1) + delta) / period
				down = down * (period - 1) / period
			else:
				up = up * (period - 1) / period
				down = (down * (period - 1) - delta) / period
			
			rs = up / down if down != 0 else 0
			rsi[i] = 100. - 100. / (1. + rs)
		
		return pd.Series(rsi, index=prices.index)
	
	df["rsi"] = gb_close.transform(lambda s: calcular_rsi(s, period=14))
	
	# MACD (Moving Average Convergence Divergence)
	def calcular_macd(prices):
		"""Calcula MACD (diferencia entre EMA12 y EMA26)."""
		ema12 = prices.ewm(span=12).mean()
		ema26 = prices.ewm(span=26).mean()
		macd = ema12 - ema26
		return macd
	
	df["macd"] = gb_close.transform(lambda s: calcular_macd(s))

	# Group by simbolo de volume para volumen relativo.
	gb_volume = df.groupby("symbol")["volume"]

	df["vol_ma_20"] = gb_volume.transform(lambda s: s.rolling(20).mean())

	# Momentum de largo plazo — consistente con ret_1d y ret_3d (usa close de hoy)
	df["ret_10d"] = (df["close"] / gb_close.shift(10)) - 1
	df["ret_20d"] = (df["close"] / gb_close.shift(20)) - 1
	df["ret_60d"] = (df["close"] / gb_close.shift(60)) - 1

	# Posición dentro de las Bandas de Bollinger (20 días)
	# ma_20 ya está calculada con shift(1); usamos la misma ventana para la std
	bb_std = gb_close.transform(lambda s: s.rolling(20).std().shift(1))
	bb_upper = df["ma_20"] + 2 * bb_std
	bb_lower = df["ma_20"] - 2 * bb_std
	bb_width = bb_upper - bb_lower
	# Posición 0 = en la banda inferior, 1 = en la banda superior
	# Usamos close_prev para ser consistentes con price_vs_ma20
	df["bb_position"] = np.where(
		bb_width > 0,
		(df["close_prev"] - bb_lower) / bb_width,
		0.5,
	)
	df["bb_position"] = df["bb_position"].clip(0, 1)

	# Distancia al máximo de 52 semanas (≤ 0: cuánto falta para el máximo)
	# rolling(252) con shift(1): nunca incluye el día actual
	high_52w = gb_close.transform(lambda s: s.rolling(252).max().shift(1))
	df["dist_52w_high"] = (df["close_prev"] / high_52w) - 1

	# OBV ratio (On Balance Volume / media móvil 20 días)
	# OBV acumula volumen con signo según dirección del precio.
	# shift(1) al final: el feature en T usa OBV acumulado hasta T-1.
	_direction = np.sign(df["ret_1d"].fillna(0))
	df["_obv_delta"] = df["volume"] * _direction
	df["_obv"] = df.groupby("symbol")["_obv_delta"].cumsum()
	_obv_ma20 = df.groupby("symbol")["_obv"].transform(lambda s: s.rolling(20).mean())
	df["_obv_ratio"] = df["_obv"] / _obv_ma20.replace(0, np.nan)
	df["obv_ratio"] = df.groupby("symbol")["_obv_ratio"].transform(lambda s: s.shift(1))
	df = df.drop(columns=["_obv_delta", "_obv", "_obv_ratio"])

	# --- Grupo 2: contexto de mercado (SPY + VIX) ---
	df_spy, df_vix = cargar_market_data()

	# Calcular features de SPY (retornos de mercado)
	spy_close = df_spy["close"].values
	spy_close_prev = df_spy["close"].shift(1).values
	spy_close_5 = df_spy["close"].shift(5).values
	spy_close_20 = df_spy["close"].shift(20).values
	df_spy["spy_ret_1d"]  = (spy_close / spy_close_prev) - 1
	df_spy["spy_ret_5d"]  = (spy_close / spy_close_5) - 1
	df_spy["spy_ret_20d"] = (spy_close / spy_close_20) - 1

	# Calcular features de VIX
	vix_close = df_vix["close"].values
	vix_close_prev = df_vix["close"].shift(1).values
	vix_close_5 = df_vix["close"].shift(5).values
	df_vix["vix_level"]     = vix_close
	df_vix["vix_change_1d"] = (vix_close / vix_close_prev) - 1
	df_vix["vix_change_5d"] = (vix_close / vix_close_5) - 1

	# Preparar tablas de join (solo fecha + features calculados)
	spy_join = df_spy[["fecha", "spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]].copy()
	vix_join = df_vix[["fecha", "vix_level", "vix_change_1d", "vix_change_5d"]].copy()

	# Columna fecha para todos los joins (se elimina al final)
	df["fecha"] = pd.to_datetime(df["ts_event_utc"]).dt.date

	# Left join por fecha — forward-fill para días donde no hay dato de SPY/VIX
	df = df.merge(spy_join, on="fecha", how="left")
	df = df.merge(vix_join, on="fecha", how="left")
	df[["spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]] = (
		df[["spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]].ffill()
	)
	df[["vix_level", "vix_change_1d", "vix_change_5d"]] = (
		df[["vix_level", "vix_change_1d", "vix_change_5d"]].ffill()
	)

	# Retorno relativo al mercado: cuánto superó (o no) al SPY
	# ret_5d_ticker mira hacia atrás (t-5 → t), sin leakage
	ret_5d_ticker = (df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5))) - 1
	df["ret_rel_spy_1d"] = df["ret_1d"] - df["spy_ret_1d"]
	df["ret_rel_spy_5d"] = ret_5d_ticker - df["spy_ret_5d"]

	# Eliminar columna auxiliar usada solo para el join
	df = df.drop(columns=["fecha"])

	# Target de regresion: retorno logaritmico a 5 dias
	next_close = df.groupby("symbol")["close"].shift(-5)
	df["target_ret_log_t5"] = np.log(next_close / df["close"])

	# Evita que divisiones por cero pasen al modelo como +/-inf.
	df = df.replace([np.inf, -np.inf], np.nan)

	return df


def preparar_para_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
	"""Filtra columnas ANTES de entrenar para evitar data leakage.
	
	Elimina:
	- open, high, low: información de HOY (problema en MSFT, NVDA, ORCL)
	- close: información de HOY
	- vol_ma_20: no está en COLS_TRAIN
	- ma_20: variable intermedia, no necesaria en modelo final
	
	Mantiene:
	- Features correctos (sin data leakage)
	- Target
	"""
	
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
		"vol_ma_20",
		"obv_ratio",
		"spy_ret_1d",
		"spy_ret_5d",
		"spy_ret_20d",
		"ret_rel_spy_1d",
		"ret_rel_spy_5d",
		"vix_level",
		"vix_change_1d",
		"vix_change_5d",
		"target_ret_log_t5",
	]
	
	return df[cols_permitidas].copy()


def limpiar_para_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
	"""Elimina nulos de columnas necesarias para entrenar regresion."""
	return df.dropna(subset=COLS_TRAIN).copy()


def guardar_silver(df: pd.DataFrame) -> None:
	"""Guarda dataset procesado para regresion."""

	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver"
	silver_dir.mkdir(parents=True, exist_ok=True)

	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")
	output_path = silver_dir / f"regresion_5d_{fecha_ejecucion}.parquet"
	df.to_parquet(output_path, index=False)
	print(f"Dataset Silver guardado: {output_path}")
	print(f"Filas finales: {len(df)}")
	print(f"Columnas guardadas: {list(df.columns)}")


def main() -> None:
	# Carga Bronze
	df = cargar_bronze()
	print(f"Filas Bronze: {len(df)}")
	print(f"Columnas Bronze: {list(df.columns)}")
	print(f"Simbolos Bronze: {df['symbol'].nunique()}")

	# Crea features + target de regresion.
	df_proc = crear_features_regresion(df)
	
	# NUEVO: Filtra columnas para evitar data leakage
	print("\n[FILTRACIÓN] Eliminando columnas con data leakage...")
	print(f"  Antes: {len(df_proc.columns)} columnas")
	df_proc = preparar_para_entrenamiento(df_proc)
	print(f"  Después: {len(df_proc.columns)} columnas")
	print(f"  Columnas finales: {list(df_proc.columns)}")

	# Limpia para entrenamiento
	df_train = limpiar_para_entrenamiento(df_proc)

	# Muestra estadisticas del target
	print(f"\nEstadisticas del target (retorno log 5 dias):")
	print(f"Media: {df_train['target_ret_log_t5'].mean():.6f}")
	print(f"Std: {df_train['target_ret_log_t5'].std():.6f}")
	print(f"Min: {df_train['target_ret_log_t5'].min():.6f}")
	print(f"Max: {df_train['target_ret_log_t5'].max():.6f}")

	# Guarda Silver
	guardar_silver(df_train)


if __name__ == "__main__":
	main()