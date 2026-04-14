"""Procesado para clasificacion (plantilla guiada).

Objetivo: crear dataset Silver para predecir si manana sube (1) o no sube (0).
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
	"volatility_20d",
	"target_updown_t1",
]


def crear_features_clasificacion(df: pd.DataFrame) -> pd.DataFrame:
	"""Crea features y target de clasificacion.

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
	df["volatility_20d"] = gb_ret_1d.transform(lambda s: s.rolling(20).std())

	# Group by simbolo de volume para volumen relativo.
	gb_volume = df.groupby("symbol")["volume"]

	df["vol_ma_20"] = gb_volume.transform(lambda s: s.rolling(20).mean())

	# Target de clasificacion

	# Retorno futuro
	next_ret = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1

	# Volatilidad histórica (ruido típico del activo)
	volatilidad = df.groupby("symbol")["ret_1d"].transform(lambda x: x.rolling(50).std())

	# Threshold dinámico (ajuste simple y efectivo)
	threshold = volatilidad * 0.1

	# Target final
	df["target_updown_t1"] = (next_ret > threshold).where(next_ret.notna(), pd.NA).astype("Int64")

	# Evita que divisiones por cero pasen al modelo como +/-inf.
	df = df.replace([np.inf, -np.inf], np.nan)

	return df


def limpiar_para_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
	"""Elimina nulos de columnas necesarias para entrenar clasificacion."""
	return df.dropna(subset=COLS_TRAIN).copy()


def guardar_silver(df: pd.DataFrame) -> None:
	"""Guarda dataset procesado para clasificacion."""

	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver"
	silver_dir.mkdir(parents=True, exist_ok=True)

	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")
	output_path = silver_dir / f"clasificacion_1d_{fecha_ejecucion}.parquet"
	df.to_parquet(output_path, index=False)
	print(f"Dataset Silver guardado: {output_path}")
	print(f"Filas finales: {len(df)}")


def main() -> None:
	# Carga Bronze
	df = cargar_bronze()
	print(f"Filas Bronze: {len(df)}")
	print(f"Columnas Bronze: {list(df.columns)}")
	print(f"Simbolos Bronze: {df['symbol'].nunique()}")

	# Crea features + target de clasificacion.
	df_proc = crear_features_clasificacion(df)

	# Limpia para entrenamiento
	df_train = limpiar_para_entrenamiento(df_proc)

	# Guarda Silver
	guardar_silver(df_train)


if __name__ == "__main__":
	main()

