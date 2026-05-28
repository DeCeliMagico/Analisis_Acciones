"""Entrenamiento de regresion con AutoGluon.

Objetivo: entrenar modelo baseline para predecir retorno logaritmico del dia siguiente.
Metodo: AutoML con AutoGluon en modo regresion.
"""

from pathlib import Path
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
from datetime import datetime


# Cargar datos Silver de regresion
def cargar_silver() -> pd.DataFrame:
	"""Carga el parquet de regresion generado en la fase anterior.
	
	Retorna:
		DataFrame con features y target.
	"""
	
	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver"
	archivos = list(silver_dir.glob("regresion_5d_*.parquet"))
	if not archivos:
		raise FileNotFoundError(f"No se encontraron archivos Silver de regresion (5d) en: {silver_dir}")
	archivo_mas_reciente = sorted(archivos)[-1]
	df = pd.read_parquet(archivo_mas_reciente)
	
	return df


# Split temporal de train, validation, test
def hacer_split_temporal(df: pd.DataFrame, train_pct: float = 0.7, valid_pct: float = 0.15):
	"""Divide dataset en train/valid/test respetando el orden temporal.
	
	IMPORTANTE: no baraja los datos (shuffle=False => sin fuga de informacion futura).
	
	Parametros:
		df: dataset completo ordenado por tiempo.
		train_pct: porcentaje para entrenamiento (default 70%).
		valid_pct: porcentaje para validacion (default 15%).
		remainder: automaticamente va a test (15%).
	
	Retorna:
		(df_train, df_valid, df_test)
	"""

	if not 0 < train_pct < 1:
		raise ValueError("train_pct debe estar entre 0 y 1")
	if not 0 < valid_pct < 1:
		raise ValueError("valid_pct debe estar entre 0 y 1")

	test_pct = 1 - train_pct - valid_pct
	if test_pct <= 0:
		raise ValueError("La suma train_pct + valid_pct debe ser menor que 1")

	# Para panel multi-activo, ordenamos globalmente por fecha para un corte temporal limpio.
	if "ts_event_utc" in df.columns:
		df = df.sort_values(["ts_event_utc", "symbol"]).reset_index(drop=True)
	
	# Split 1: train vs (valid + test)
	df_train, df_rest = train_test_split(
		df,
		test_size=(valid_pct + test_pct),
		shuffle=False  # CRITICO: respeta orden temporal
	)
	
	# Split 2: del resto, valid vs test
	test_ratio = test_pct / (valid_pct + test_pct)
	df_valid, df_test = train_test_split(
		df_rest,
		test_size=test_ratio,
		shuffle=False  # CRITICO: respeta orden temporal
	)
	
	return df_train, df_valid, df_test


# Entrenar con AutoGluon
def entrenar_autogluon(df_train: pd.DataFrame, df_valid: pd.DataFrame, time_limit: int = 3600):
	"""Entrena AutoGluon en modo regresion.
	
	Parametros:
		df_train: datos de entrenamiento (70% del total).
		df_valid: datos de validacion (15% del total).
		time_limit: segundos maximo de entrenamiento (default 1 hora).
			- Mas tiempo => mejor tuning pero mas lento.
			- Para prototipado: 300-600 seg.
			- Para produccion: 1800-3600 seg.
	
	Retorna:
		predictor: modelo entrenado listo para usar.
	"""
	
	predictor = TabularPredictor(
	    label="target_ret_log_t5",
	    problem_type="regression",
		eval_metric="rmse",  # Root Mean Squared Error
		verbosity=0
	)
    
	predictor.fit(
	    train_data=df_train,
	    tuning_data=df_valid,
	    time_limit=time_limit,
		excluded_model_types=["RF","XT"], # Excluimos modelos innecesarios para ahorrar RAM 
		num_gpus=0
    )
	
	return predictor


# Evaluar metricas
def evaluar_modelo(y_true, predictor, df_test: pd.DataFrame):
	"""Calcula metricas de regresion.
	
	Metricas:
	- RMSE: raiz del error cuadratico medio (en mismas unidades que target).
	- MAE: error absoluto medio (robusto a outliers).
	- R2: coeficiente de determinacion (0-1, siendo 1 perfecto).
	- MAPE: error porcentual absoluto medio (cuando el target es muy pequeño).
	- Accuracy de dirección: % de veces que acierta si sube o baja.
	- Correlation: correlacion entre predicciones y valores reales (-1 a 1).
	- Relative RMSE: RMSE normalizado por volatilidad del target.
	
	Retorna:
		dict con todas las metricas.
	"""
	y_pred = predictor.predict(df_test)
	
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	mae = mean_absolute_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	
	# Accuracy de dirección
	dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
	
	# Correlation
	correlation = np.corrcoef(y_true, y_pred)[0, 1]
	
	# Relative RMSE
	y_std = np.std(y_true)
	relative_rmse = rmse / y_std if y_std > 0 else np.nan
	
	# MAPE: evita division por cero si hay valores cerca de 0
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
	}


# Guardar modelo y resultados
def guardar_modelo_y_resultados(predictor, metricas: dict, df_train: pd.DataFrame, df_test: pd.DataFrame):
	"""Guarda el modelo entrenado y las metricas en un archivo.
	
	Archivos generados:
	 - models/autogluon_regresion_*/: modelo completo (directorio).
	 - results/metricas_regresion.json: metricas de validacion.
	"""
	
	project_root = Path(__file__).resolve().parents[2]
	
	modelos_dir = project_root / "modelos"
	evaluaciones_dir = project_root / "evaluaciones"
	modelos_dir.mkdir(parents=True, exist_ok=True)
	evaluaciones_dir.mkdir(parents=True, exist_ok=True)
	
	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")

	model_path = modelos_dir / f"Market_AI_Regresion_5d_{fecha_ejecucion}"
	predictor.save(str(model_path))
	print(f"Modelo guardado: {model_path}")
	
	results = {
	    "metricas": metricas,
	    "filas_train": len(df_train),
	    "filas_test": len(df_test),
	}

	evaluaciones_path = evaluaciones_dir / "metricas_regresion_5d.json"
	with open(evaluaciones_path, "w") as f:
		json.dump(results, f, indent=2)
		print(f"Metricas guardadas: {evaluaciones_path}")


# Ver ranking de modelos entrenados
def mostrar_leaderboard(predictor):
	"""Muestra el ranking de todos los modelos que AutoGluon entrenó.
	
	Te permite ver:
	- Que algoritmo fue mejor (XGBoost, LightGBM, CatBoost, etc).
	- Metricas de cada uno.
	- Cual es el ensemble final.
	"""

	leaderboard = predictor.leaderboard()
	print(leaderboard)



# MAIN: Orquestar todo
def main() -> None:
	"""Ejecuta el pipeline completo de entrenamiento de regresion."""
	
	print("=" * 70)
	print("ENTRENAMIENTO DE REGRESION (RETORNO LOG) CON AUTOGLUON")
	print("=" * 70)
	
	# 1) Cargar Silver
	print("\n[1/5] Cargando Silver...")
	df = cargar_silver()
	print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
	print(f"Estadisticas del target (5 dias):")
	print(f"  Media: {df['target_ret_log_t5'].mean():.6f}")
	print(f"  Std: {df['target_ret_log_t5'].std():.6f}")
	print(f"  Min: {df['target_ret_log_t5'].min():.6f}")
	print(f"  Max: {df['target_ret_log_t5'].max():.6f}")
	
	# 2) Split temporal
	print("\n[2/5] Haciendo split temporal (70% / 15% / 15%)...")
	df_train, df_valid, df_test = hacer_split_temporal(df)
	print(f"Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")
	
	# 3) Crear y entrenar predictor
	print("\n[3/5] Creando y entrenando predictor de AutoGluon...")
	print("Nota: este paso puede tardar dependiendo del time_limit que elijas.")
	predictor = entrenar_autogluon(df_train, df_valid, time_limit=300)  # Para prototipado, 5 minutos
	
	# 4) Evaluar
	print("\n[4/5] Evaluando metricas en test...")
	metricas = evaluar_modelo(df_test["target_ret_log_t5"], predictor, df_test)
	print(f"R2: {metricas['r2']:.4f}")
	print(f"RMSE: {metricas['rmse']:.6f}")
	print(f"MAE: {metricas['mae']:.6f}")
	print(f"Accuracy de Dirección: {metricas['accuracy_direccion']:.2%}")
	print(f"Correlation: {metricas['correlation']:.4f}")
	print(f"Relative RMSE: {metricas['relative_rmse']:.4f}")
	if not np.isnan(metricas['mape']):
		print(f"MAPE: {metricas['mape']:.2f}%")
	else:
		print(f"MAPE: N/A (target cercano a cero)")
	
	# 5) Guardar
	print("\n[5/5] Guardando modelo y resultados...")
	guardar_modelo_y_resultados(predictor, metricas, df_train, df_test)
	
	# 6) Leaderboard
	print("\n[BONUS] Ranking de modelos entrenados:")
	mostrar_leaderboard(predictor)
	
	print("\n" + "=" * 70)
	print("ENTRENAMIENTO TERMINADO")
	print("=" * 70)


if __name__ == "__main__":
	main()
