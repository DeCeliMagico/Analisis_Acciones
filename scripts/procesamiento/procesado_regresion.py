"""Procesado para regresion (retorno logaritmico semanal).

Objetivo: crear dataset Silver para predecir retorno logaritmico 5 dias adelante.
Target: retorno log puro sin umbral (regresion continua), horizonte: +5 dias.
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
	"gap_prop",
	"range_norm",
	"price_vs_ma20",
	"volatility_5d",
	"target_ret_log_t5",
]


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
	df["ma_20"] = gb_close.transform(lambda s: s.rolling(20).mean().shift(1))
	df["price_vs_ma20"] = df["close"] / df["ma_20"]

	# Rango en proporcion
	df["range_norm"] = (df["high"] - df["low"]) / df["ma_20"]

	# Group by simbolo de ret_1d para volatilidad.
	gb_ret_1d = df.groupby("symbol")["ret_1d"]

	# Volatilidad (std de retornos) y volumen relativo.
	df["volatility_5d"] = gb_ret_1d.transform(lambda s: s.rolling(5).std())

	# Group by simbolo de volume para volumen relativo.
	gb_volume = df.groupby("symbol")["volume"]

	df["vol_ma_20"] = gb_volume.transform(lambda s: s.rolling(20).mean())

	# Target de regresion: retorno logaritmico a 5 dias
	next_close = df.groupby("symbol")["close"].shift(-5)
	df["target_ret_log_t5"] = np.log(next_close / df["close"])

	# Evita que divisiones por cero pasen al modelo como +/-inf.
	df = df.replace([np.inf, -np.inf], np.nan)

	return df


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


def main() -> None:
	# Carga Bronze
	df = cargar_bronze()
	print(f"Filas Bronze: {len(df)}")
	print(f"Columnas Bronze: {list(df.columns)}")
	print(f"Simbolos Bronze: {df['symbol'].nunique()}")

	# Crea features + target de regresion.
	df_proc = crear_features_regresion(df)

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
