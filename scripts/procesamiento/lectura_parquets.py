"""Plantilla simple para empezar el procesamiento de parquet.

Idea: ir completando paso a paso sin agobio.
"""

from pathlib import Path
import pandas as pd


def cargar_bronze() -> pd.DataFrame:
	"""Carga todos los parquet de data/bronze y los une en un solo DataFrame."""

	# PASO 1: localizar carpeta del proyecto y carpeta bronze
	project_root = Path(__file__).resolve().parents[2]
	bronze_dir = project_root / "data" / "bronze"

	# PASO 2: buscar archivos parquet
	parquet_files = sorted(bronze_dir.glob("*.parquet"))
	print(f"Parquet encontrados: {len(parquet_files)}")

	if not parquet_files:
		raise FileNotFoundError(f"No se encontraron parquet en: {bronze_dir}")

	# PASO 3: unir todos los parquet en un unico dataframe logico
	df = pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)
	return df


def chequeo_basico(df: pd.DataFrame) -> None:
	"""Muestra un resumen rapido para confirmar que cargaste bien los datos."""

	print("\n=== CHEQUEO BASICO ===")
	print(f"Filas totales: {len(df)}")
	print(f"Columnas: {list(df.columns)}")

	# TODO A (facil): descomenta esta linea cuando veas que existe la columna symbol.
	print(f"Simbolos unicos: {df['symbol'].nunique()}")

	print("Primeras filas:")
	print(df.head(3))


if __name__ == "__main__":
	df =cargar_bronze()
	chequeo_basico(df)
	resumen = (
    df.groupby("symbol")
      .agg(
          filas=("symbol", "size"),
          fecha_min=("ts_event_utc", "min"),
          fecha_max=("ts_event_utc", "max"),
      )
      .sort_values("filas", ascending=False)
    )
	print("\n=== COBERTURA POR SIMBOLO ===")
	print(resumen.head(10))
	print("\nMedia de filas por simbolo:", round(resumen["filas"].mean(), 2))

