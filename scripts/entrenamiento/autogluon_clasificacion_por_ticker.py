"""Entrenamiento de clasificacion binaria por ticker (horizonte 5 dias).

Objetivo: entrenar un modelo por ticker que prediga si el precio sube (1) o baja (0)
en los próximos 5 días. AutoGluon está mejor calibrado para clasificación que para
regresión, por lo que esta aproximación busca mejor señal direccional.

Diferencias clave respecto a la regresión:
- Target binario: target_updown_t5 (0/1)
- Métrica de optimización: roc_auc
- Salida: probabilidad P(sube) vía predict_proba → umbral de confianza aplicable
- Mismas features: Grupo 0 + Grupo 1 + Grupo 2 (SPY/VIX)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
	accuracy_score,
	roc_auc_score,
	precision_score,
	recall_score,
	f1_score,
)
import json
from datetime import datetime

LABEL = "target_updown_t5"

# Features explícitas: excluye ts_event_utc (leakage de fecha) y symbol (constante por ticker)
# Idéntico al conjunto de regresión — Grupo 0 + Grupo 1 + Grupo 2
FEATURES = [
	"ret_1d", "ret_3d", "ret_10d", "ret_20d", "ret_60d",
	"gap_prop", "range_norm", "price_vs_ma20", "bb_position", "dist_52w_high",
	"volatility_5d", "volatility_20d",
	"rsi", "macd", "volume", "obv_ratio",
	"spy_ret_1d", "spy_ret_5d", "spy_ret_20d",
	"ret_rel_spy_1d", "ret_rel_spy_5d",
	"vix_level", "vix_change_1d", "vix_change_5d",
]


def cargar_silver() -> pd.DataFrame:
	"""Carga el parquet de clasificación 5d más reciente."""
	project_root = Path(__file__).resolve().parents[2]
	silver_dir = project_root / "data" / "silver" / "clasificacion"
	archivos = list(silver_dir.glob("clasificacion_5d_*.parquet"))
	if not archivos:
		raise FileNotFoundError(
			f"No se encontraron archivos Silver de clasificacion_5d en: {silver_dir}\n"
			"Ejecuta primero: python scripts/procesamiento/procesado_clasificacion_5d.py"
		)
	return pd.read_parquet(sorted(archivos)[-1])


def seleccionar_top_tickers(
	df: pd.DataFrame,
	n: int = 25,
	min_filas: int = 6_000,
	excluir: list | None = None,
	solo: list | None = None,
) -> list:
	"""Selecciona tickers por volumen, filtrando los que tienen pocos datos históricos.

	min_filas: tickers con menos filas se descartan — modelos con poco histórico
	generalizan peor y el set de test queda demasiado corto.
	"""
	if solo:
		tickers_disponibles = df["symbol"].unique().tolist()
		return [t for t in solo if t in tickers_disponibles]

	if excluir is None:
		excluir = []

	conteo = df.groupby("symbol").size()
	tickers_suficientes = conteo[conteo >= min_filas].index.tolist()

	vol_promedio = df.groupby("symbol")["volume"].mean().sort_values(ascending=False)
	top = [t for t in vol_promedio.index if t in tickers_suficientes and t not in excluir][:n]

	descartados = conteo[conteo < min_filas].shape[0]
	print(f"Tickers descartados por pocos datos (<{min_filas} filas): {descartados}")
	print(f"Top {len(top)} tickers seleccionados (volumen + datos suficientes):")
	for ticker in top:
		print(f"  {ticker}: vol={vol_promedio[ticker]:,.0f} | filas={conteo[ticker]:,}")

	return top


def hacer_split_temporal_ticker(
	df_ticker: pd.DataFrame,
	train_pct: float = 0.7,
	valid_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split temporal por ticker (70 / 15 / 15%)."""
	df_ticker = df_ticker.sort_values("ts_event_utc").reset_index(drop=True)
	n = len(df_ticker)
	n_train = int(n * train_pct)
	n_valid = int(n * valid_pct)
	return (
		df_ticker[:n_train],
		df_ticker[n_train : n_train + n_valid],
		df_ticker[n_train + n_valid :],
	)


def preparar_split(df: pd.DataFrame) -> pd.DataFrame:
	"""Selecciona solo features + target, sin ts_event_utc ni symbol."""
	df = df.sort_values("ts_event_utc").reset_index(drop=True)
	return df[FEATURES + [LABEL]].copy()


def añadir_pesos_clase(df: pd.DataFrame) -> pd.DataFrame:
	"""Añade columna sample_weight con pesos inversamente proporcionales a la frecuencia.

	Con target desbalanceado (70% clase 0, 30% clase 1), el modelo tiende a predecir
	siempre clase 0 (nunca sube >2%). Los pesos inversos hacen que equivocarse en
	clase 1 cueste 2.3x más — forzando al modelo a aprender a detectar subidas grandes.
	Los pesos se calculan por ticker (no globalmente) para respetar sus distribuciones.
	"""
	df = df.copy()
	counts = df["target_updown_t5"].value_counts()
	n_total = len(df)
	n_clases = len(counts)
	# Peso = n_total / (n_clases * count_clase) — fórmula estándar sklearn
	peso_por_clase = {cls: n_total / (n_clases * cnt) for cls, cnt in counts.items()}
	df["sample_weight"] = df["target_updown_t5"].map(peso_por_clase)
	return df


def entrenar_modelo_ticker(
	df_train: pd.DataFrame,
	df_valid: pd.DataFrame,
	ticker: str,
	time_limit: int = 180,
	modelo_path: str | None = None,
) -> TabularPredictor:
	"""Entrena AutoGluon clasificación binaria para un ticker."""
	predictor = TabularPredictor(
		label="target_updown_t5",
		problem_type="binary",
		eval_metric="roc_auc",
		sample_weight="sample_weight",  # pesos de clase inversos
		verbosity=0,
		path=modelo_path,
	)
	predictor.fit(
		train_data=df_train,
		tuning_data=df_valid,
		time_limit=time_limit,
		excluded_model_types=["RF", "XT"],
		num_gpus=0,
	)
	return predictor


def evaluar_ticker(
	y_true: pd.Series,
	predictor: TabularPredictor,
	df_test: pd.DataFrame,
) -> dict:
	"""Calcula métricas de clasificación para un ticker."""
	y_pred = predictor.predict(df_test)
	y_proba_df = predictor.predict_proba(df_test)

	# La columna 1 es P(sube)
	if hasattr(y_proba_df, "columns"):
		proba_up = y_proba_df[1].values
	else:
		proba_up = y_proba_df[:, 1]

	y_true_arr = y_true.values
	y_pred_arr = y_pred.values

	acc = accuracy_score(y_true_arr, y_pred_arr)
	auc = roc_auc_score(y_true_arr, proba_up)
	prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
	rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
	f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

	# Distribución de probabilidades predichas (útil para calibrar umbral)
	proba_mean = float(np.mean(proba_up))
	proba_std = float(np.std(proba_up))
	# % de predicciones con alta confianza (>55% o <45%)
	high_conf = float(np.mean((proba_up > 0.55) | (proba_up < 0.45)))

	return {
		"accuracy": acc,
		"roc_auc": auc,
		"precision": prec,
		"recall": rec,
		"f1": f1,
		"proba_up_mean": proba_mean,
		"proba_up_std": proba_std,
		"pct_alta_confianza_55": high_conf,
	}


def main() -> None:
	print("=" * 70)
	print("ENTRENAMIENTO CLASIFICACION BINARIA POR TICKER (horizonte 5d)")
	print("=" * 70)

	print("\n[1/3] Cargando Silver clasificacion_5d...")
	df = cargar_silver()
	print(f"Datos cargados: {len(df)} filas | Símbolos: {df['symbol'].nunique()}")
	dist = df["target_updown_t5"].value_counts().to_dict()
	print(f"Distribución global: up={dist.get(1,0)} | down={dist.get(0,0)}")

	print("\n[2/3] Seleccionando tickers...")
	top_tickers = seleccionar_top_tickers(
		df,
		n=25,
		min_filas=6_000,
		excluir=["AMZN", "GOOGL", "AAPL", "INTC", "META", "QCOM"],
	)

	print("\n[3/3] Entrenando modelos por ticker...")
	project_root = Path(__file__).resolve().parents[2]
	modelos_dir = project_root / "modelos" / "clasificacion"
	evaluaciones_dir = project_root / "evaluaciones"
	modelos_dir.mkdir(parents=True, exist_ok=True)
	evaluaciones_dir.mkdir(parents=True, exist_ok=True)

	fecha_ejecucion = datetime.now().strftime("%d-%m-%y_%H%M%S")
	resultados_todos = {}

	for ticker in top_tickers:
		print(f"\n--- Ticker: {ticker} ---")
		df_ticker = df[df["symbol"] == ticker].copy()
		print(f"  Filas: {len(df_ticker)}")

		dist_t = df_ticker["target_updown_t5"].value_counts().to_dict()
		print(f"  Dist target: up={dist_t.get(1,0)} | down={dist_t.get(0,0)}")

		df_train, df_valid, df_test = hacer_split_temporal_ticker(df_ticker)
		print(f"  Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")

		# Seleccionar solo features + target (sin ts_event_utc ni symbol)
		df_train_f = preparar_split(df_train)
		df_valid_f = preparar_split(df_valid)
		df_test_f = preparar_split(df_test)

		# Pesos inversamente proporcionales a la frecuencia de cada clase
		df_train_w = añadir_pesos_clase(df_train_f)
		df_valid_w = añadir_pesos_clase(df_valid_f)

		try:
			model_path = modelos_dir / f"Clf_AI_Ticker_{ticker}_{fecha_ejecucion}"
			predictor = entrenar_modelo_ticker(
				df_train_w, df_valid_w, ticker, time_limit=180, modelo_path=str(model_path)
			)

			metricas = evaluar_ticker(df_test_f[LABEL], predictor, df_test_f)
			print(
				f"  Acc: {metricas['accuracy']:.2%} | AUC: {metricas['roc_auc']:.4f} | "
				f"Alta conf(55%): {metricas['pct_alta_confianza_55']:.1%}"
			)
			resultados_todos[ticker] = metricas

		except Exception as e:
			print(f"  ERROR en {ticker}: {e}")
			resultados_todos[ticker] = {"error": str(e)}

	# Guardar resultados
	evaluaciones_path = (
		evaluaciones_dir / f"metricas_clasificacion_por_ticker_{fecha_ejecucion}.json"
	)
	with open(evaluaciones_path, "w") as f:
		json.dump(resultados_todos, f, indent=2)
	print(f"\n\nResultados guardados: {evaluaciones_path}")

	# Resumen
	print("\n" + "=" * 90)
	print("RESUMEN")
	print("=" * 90)
	print(
		f"{'Ticker':<10} {'Accuracy':<12} {'ROC-AUC':<12} {'Precision':<12} "
		f"{'Recall':<10} {'Alta_Conf%':<12}"
	)
	print("-" * 90)
	for ticker, m in resultados_todos.items():
		if "error" in m:
			print(f"{ticker:<10} {'ERROR'}")
		else:
			print(
				f"{ticker:<10} {m['accuracy']:<12.2%} {m['roc_auc']:<12.4f} "
				f"{m['precision']:<12.2%} {m['recall']:<10.2%} "
				f"{m['pct_alta_confianza_55']:<12.1%}"
			)


if __name__ == "__main__":
	main()
