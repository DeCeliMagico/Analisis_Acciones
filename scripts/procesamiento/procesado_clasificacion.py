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
	"ret_5d",
	"ret_10d",
	"gap_prop",
	"range_prop",
	"ma_ratio",
	"price_vs_ma20",
	"volatility_10d",
	"volatility_20d",
	"vol_ratio",
	"vol_change_1d",
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
	df = df.sort_values(["symbol", "ts_event_utc"]).copy()

	# Group by simbolo(close) para calculos posteriores.
	gb_close = df.groupby("symbol")["close"]

	# Columna de cierre previo (ayer) por simbolo.
	df["close_prev"] = gb_close.shift(1)

	# Retorno diario en proporción.
	df["ret_1d"] = (df['close'] / df['close_prev'] ) - 1

	# Retornos de horizonte mayor (tendencia corta).
	df["ret_5d"] = gb_close.pct_change(5)
	df["ret_10d"] = gb_close.pct_change(10)

	# Gap y rango en proporción.
	df["gap_prop"] = (df['open'] / df['close_prev'] - 1)
	df["range_prop"] = (df['high'] - df['low']) / df['close']

	# Medias moviles y ratios de tendencia.
	df["ma_5"] = gb_close.transform(lambda s: s.rolling(5).mean())
	df["ma_20"] = gb_close.transform(lambda s: s.rolling(20).mean())
	df["ma_50"] = gb_close.transform(lambda s: s.rolling(50).mean())
	df["ma_ratio"] = df["ma_5"] / df["ma_20"]
	df["price_vs_ma20"] = df["close"] / df["ma_20"]

	# Volatilidad (std de retornos) y volumen relativo.

	# Group by simbolo de ret_1d para volatilidad.
	gb_ret_1d = df.groupby("symbol")["ret_1d"]
	
	df["volatility_10d"] = gb_ret_1d.transform(lambda s: s.rolling(10).std())
	df["volatility_20d"] = gb_ret_1d.transform(lambda s: s.rolling(20).std())

	# Group by simbolo de volume para volumen relativo.
	gb_volume = df.groupby("symbol")["volume"]

	df["vol_ma_20"] = gb_volume.transform(lambda s: s.rolling(20).mean())
	df["vol_ratio"] = df["volume"] / df["vol_ma_20"]
	df["vol_change_1d"] = (df["volume"] / gb_volume.shift(1)) - 1

	# Target de clasificacion para manana (t+1).
	next_ret_1d = df.groupby("symbol")["ret_1d"].shift(-1)
	df["target_updown_t1"] = (next_ret_1d > 0).where(next_ret_1d.notna(), pd.NA).astype("Int64")

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

