"""Entrenamiento de clasificacion binaria con AutoGluon.

Objetivo: entrenar modelo baseline para predecir si manana sube (1) o no sube (0).
Metodo: AutoML con AutoGluon (prueba multiples algoritmos y elige el mejor).
"""

from pathlib import Path
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import json
from datetime import datetime


# Cargar datos Silver de clasificacion
def cargar_silver() -> pd.DataFrame:
	"""Carga el parquet de clasificacion generado en la fase anterior.
	
	Retorna:
		DataFrame con features y target.
	"""
	
	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver"
	archivos = list(silver_dir.glob("clasificacion_1d_*.parquet"))
	if not archivos:
		raise FileNotFoundError(f"No se encontraron archivos Silver en: {silver_dir}")
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
	"""Entrena AutoGluon sobre datos de entrenamiento.
	
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
	    label="target_updown_t1",
	    problem_type="binary",
		eval_metric="roc_auc",  # ----------------------------------------
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
	"""Calcula metricas de clasificacion.
	
	Metricas:
	- Accuracy: % de predicciones correctas (simple pero sesga con desbalance).
	- ROC-AUC: area bajo curva ROC (mejor para desbalance).
	- Precision: de los que predijimos 1, cuantos eran realmente 1.
	- Recall: de los que eran realmente 1, cuantos predijimos.
	- F1: balance entre precision y recall.
	
	Retorna:
		dict con todas las metricas.
	"""
	y_pred = predictor.predict(df_test)
	y_proba = predictor.predict_proba(df_test)
	proba_clase_1 = y_proba[1] if hasattr(y_proba, "columns") else y_proba[:, 1]

	acc = accuracy_score(y_true, y_pred)
	auc = roc_auc_score(y_true, proba_clase_1)
	prec = precision_score(y_true, y_pred, zero_division=0)
	recall = recall_score(y_true, y_pred, zero_division=0)
	f1 = f1_score(y_true, y_pred)

	return {
	    "accuracy": acc,
	    "auc": auc,
	    "precision": prec,
	    "recall": recall,
	    "f1": f1,
	}


# Guardar modelo y resultados
def guardar_modelo_y_resultados(predictor, metricas: dict, df_train: pd.DataFrame, df_test: pd.DataFrame):
	"""Guarda el modelo entrenado y las metricas en un archivo.
	
	Archivos generados:
	 - models/autogluon_baseline/: modelo completo (directorio).
	 - results/metricas_clasificacion.json: metricas de validacion.
	"""
	
	project_root = Path(__file__).resolve().parents[2]
	
	modelos_dir = project_root / "modelos"
	evaluaciones_dir = project_root / "evaluaciones"
	modelos_dir.mkdir(parents=True, exist_ok=True)
	evaluaciones_dir.mkdir(parents=True, exist_ok=True)
	
	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")

	model_path = modelos_dir / f"Market_AI_{fecha_ejecucion}"
	predictor.save(str(model_path))
	print(f"Modelo guardado: {model_path}")
	
	results = {
	    "metricas": metricas,
	    "filas_train": len(df_train),
	    "filas_test": len(df_test),
	}

	evaluaciones_path = evaluaciones_dir / "metricas_clasificacion.json"
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
	"""Ejecuta el pipeline completo de entrenamiento."""
	
	print("=" * 70)
	print("ENTRENAMIENTO DE CLASIFICACION BINARIA CON AUTOGLUON")
	print("=" * 70)
	
	# 1) Cargar Silver
	print("\n[1/5] Cargando Silver...")
	df = cargar_silver()
	print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
	print(f"Distribucion target: {df['target_updown_t1'].value_counts().to_dict()}")
	
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
	metricas = evaluar_modelo(df_test["target_updown_t1"], predictor, df_test)
	print(f"Accuracy: {metricas['accuracy']:.4f}")
	print(f"ROC-AUC: {metricas['auc']:.4f}")
	print(f"Precision: {metricas['precision']:.4f}")
	print(f"Recall: {metricas['recall']:.4f}")
	print(f"F1: {metricas['f1']:.4f}")
	
	# 5) Guardar
	print("\n[5/5] Guardando modelo y resultados...")
	guardar_modelo_y_resultados(predictor, metricas, df_train, df_test)
	
	# 7) Leaderboard
	print("\n[BONUS] Ranking de modelos entrenados:")
	mostrar_leaderboard(predictor)
	
	print("\n" + "=" * 70)
	print("ENTRENAMIENTO TERMINADO")
	print("=" * 70)


if __name__ == "__main__":
	main()
