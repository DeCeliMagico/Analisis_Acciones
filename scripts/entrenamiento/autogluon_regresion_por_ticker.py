"""Entrenamiento de regresion por ticker (top 10 por volumen).

Objetivo: entrenar modelos individuales por ticker en lugar de un modelo global.
Esto permite capturar dinámicas específicas de cada acción.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import json
from datetime import datetime


# Cargar datos Silver de regresion
def cargar_silver() -> pd.DataFrame:
	"""Carga el parquet de regresion generado en la fase anterior.
	
	Retorna:
		DataFrame con features, ticker y target.
	"""
	
	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver"
	archivos = list(silver_dir.glob("regresion_5d_*.parquet"))
	if not archivos:
		raise FileNotFoundError(f"No se encontraron archivos Silver de regresion (5d) en: {silver_dir}")
	archivo_mas_reciente = sorted(archivos)[-1]
	df = pd.read_parquet(archivo_mas_reciente)
	
	return df


# Seleccionar top tickers por volumen
def seleccionar_top_tickers(df: pd.DataFrame, n: int = 10) -> list:
	"""Selecciona los top N tickers por volumen promedio.
	
	Parametros:
		df: dataset con columna 'symbol'
		n: cantidad de tickers a seleccionar
	
	Retorna:
		lista de n tickers con mayor volumen promedio
	"""
	
	vol_promedio = df.groupby("symbol")["volume"].mean().sort_values(ascending=False)
	top_tickers = vol_promedio.head(n).index.tolist()
	
	print(f"Top {n} tickers por volumen promedio:")
	for ticker in top_tickers:
		print(f"  {ticker}: {vol_promedio[ticker]:,.0f}")
	
	return top_tickers


# Split temporal por ticker
def hacer_split_temporal_ticker(df_ticker: pd.DataFrame, train_pct: float = 0.7, valid_pct: float = 0.15):
	"""Divide dataset de un ticker en train/valid/test respetando orden temporal.
	
	Parametros:
		df_ticker: dataset de un ticker ordenado por fecha
		train_pct: porcentaje para entrenamiento
		valid_pct: porcentaje para validacion
	
	Retorna:
		(df_train, df_valid, df_test)
	"""
	
	df_ticker = df_ticker.sort_values("ts_event_utc").reset_index(drop=True)
	
	n = len(df_ticker)
	n_train = int(n * train_pct)
	n_valid = int(n * valid_pct)
	
	df_train = df_ticker[:n_train]
	df_valid = df_ticker[n_train:n_train+n_valid]
	df_test = df_ticker[n_train+n_valid:]
	
	return df_train, df_valid, df_test


# Entrenar AutoGluon para un ticker
def entrenar_modelo_ticker(df_train: pd.DataFrame, df_valid: pd.DataFrame, ticker: str, time_limit: int = 120) -> TabularPredictor:
	"""Entrena AutoGluon para un ticker específico.
	
	Parametros:
		df_train: datos de entrenamiento
		df_valid: datos de validacion
		ticker: nombre del ticker (para nombrar el modelo)
		time_limit: segundos máximo de entrenamiento
	
	Retorna:
		predictor entrenado
	"""
	
	predictor = TabularPredictor(
	    label="target_ret_log_t5",
	    problem_type="regression",
		eval_metric="rmse",
		verbosity=0
	)
    
	predictor.fit(
	    train_data=df_train,
	    tuning_data=df_valid,
	    time_limit=time_limit,
		excluded_model_types=["RF","XT"],
		num_gpus=0
    )
	
	return predictor


# Evaluar modelo para un ticker
def evaluar_ticker(y_true, predictor, df_test: pd.DataFrame) -> dict:
	"""Calcula metricas para un ticker.
	
	Retorna:
		dict con RMSE, MAE, R2, MAPE, accuracy_direccion, correlation, relative_rmse, error_magnitud
	"""
	
	y_pred = predictor.predict(df_test)
	
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	mae = mean_absolute_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	
	# Accuracy de dirección: % de veces que acierta si sube o baja
	dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
	
	# Correlation: mide cómo de relacionadas están las predicciones con los valores reales
	correlation = np.corrcoef(y_true, y_pred)[0, 1]
	
	# Relative RMSE: RMSE normalizado por la desviación estándar del target
	y_std = np.std(y_true)
	relative_rmse = rmse / y_std if y_std > 0 else np.nan
	
	# Error promedio en magnitud cuando acierta la dirección
	dir_correcta = np.sign(y_pred) == np.sign(y_true)
	if dir_correcta.sum() > 0:
		error_magnitud_cuando_acierta = np.mean(np.abs(y_true[dir_correcta] - y_pred[dir_correcta]))
	else:
		error_magnitud_cuando_acierta = np.nan
	
	mask = y_true != 0
	if mask.sum() > 0:
		mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
	else:
		mape = np.nan

	return {
	    "rmse": rmse,
	    "mae": mae,
	    "r2": r2,
	    "mape": mape,
	    "accuracy_direccion": dir_acc,
	    "correlation": correlation,
	    "relative_rmse": relative_rmse,
	    "error_magnitud_cuando_acierta": error_magnitud_cuando_acierta,
	}


# MAIN
def main() -> None:
	"""Entrena modelos por ticker."""
	
	print("=" * 70)
	print("ENTRENAMIENTO DE REGRESION POR TICKER (TOP 10 POR VOLUMEN)")
	print("=" * 70)
	
	# 1) Cargar datos
	print("\n[1/3] Cargando Silver...")
	df = cargar_silver()
	print(f"Datos cargados: {len(df)} filas")
	
	# 2) Seleccionar top tickers
	print("\n[2/3] Seleccionando top 10 tickers...")
	top_tickers = seleccionar_top_tickers(df, n=10)
	
	# 3) Entrenar modelo por ticker
	print("\n[3/3] Entrenando modelos por ticker...")
	
	project_root = Path(__file__).resolve().parents[2]
	modelos_dir = project_root / "modelos"
	evaluaciones_dir = project_root / "evaluaciones"
	modelos_dir.mkdir(parents=True, exist_ok=True)
	evaluaciones_dir.mkdir(parents=True, exist_ok=True)
	
	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")
	resultados_todos = {}
	
	for ticker in top_tickers:
		print(f"\n--- Ticker: {ticker} ---")
		
		# Datos del ticker
		df_ticker = df[df["symbol"] == ticker].copy()
		print(f"  Filas: {len(df_ticker)}")
		
		# Split temporal
		df_train, df_valid, df_test = hacer_split_temporal_ticker(df_ticker)
		print(f"  Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")
		
		# Entrenar
		try:
			predictor = entrenar_modelo_ticker(df_train, df_valid, ticker, time_limit=120)
			
			# Evaluar
			metricas = evaluar_ticker(df_test["target_ret_log_t5"], predictor, df_test)
			print(f"  Dir.Acc: {metricas['accuracy_direccion']:.2%} | Corr: {metricas['correlation']:.4f} | Rel.RMSE: {metricas['relative_rmse']:.4f}")
			
			# Guardar modelo
			model_path = modelos_dir / f"Market_AI_Ticker_{ticker}_{fecha_ejecucion}"
			predictor.save(str(model_path))
			
			# Registro de resultados
			resultados_todos[ticker] = metricas
			
		except Exception as e:
			print(f"  ❌ Error en {ticker}: {str(e)}")
			resultados_todos[ticker] = {"error": str(e)}
	
	# Guardar resultados consolidados
	evaluaciones_path = evaluaciones_dir / f"metricas_regresion_por_ticker_{fecha_ejecucion}.json"
	with open(evaluaciones_path, "w") as f:
		json.dump(resultados_todos, f, indent=2)
	print(f"\n\nResultados guardados: {evaluaciones_path}")
	
	# Resumen
	print("\n" + "=" * 100)
	print("RESUMEN DE RESULTADOS")
	print("=" * 100)
	print(f"{'Ticker':<10} {'Dir.Acc':<12} {'Correlation':<15} {'Rel.RMSE':<12} {'Error.Mag':<12}")
	print("-" * 100)
	for ticker, metricas in resultados_todos.items():
		if "error" in metricas:
			print(f"{ticker:<10} {'ERROR':<12}")
		else:
			dir_acc_str = f"{metricas['accuracy_direccion']:.2%}"
			corr_str = f"{metricas['correlation']:.4f}"
			rel_rmse_str = f"{metricas['relative_rmse']:.4f}"
			error_mag_str = f"{metricas['error_magnitud_cuando_acierta']:.6f}"
			print(f"{ticker:<10} {dir_acc_str:<12} {corr_str:<15} {rel_rmse_str:<12} {error_mag_str:<12}")


if __name__ == "__main__":
	main()
