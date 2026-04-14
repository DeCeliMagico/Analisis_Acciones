# Entrenamiento: Clasificación Binaria con AutoGluon

## Resumen

Fase 3 del proyecto: entrenar un modelo que predice si mañana la acción sube (1) o no sube (0).

- **Dataset**: Silver (1.321M filas, 12 features + target).
- **Algoritmo**: AutoML con AutoGluon (selecciona automáticamente el mejor modelo).
- **Validación**: split temporal 70% train / 15% valid / 15% test (sin shuffle).
- **Tiempo**: ~1 hora en máquina media (ajustable).
- **Output**: modelo guardado + métricas JSON + leaderboard.

---

## Ejecución

### Paso 1: Instalar dependencias

```bash
pip install -r requirements.txt
```

Verifica que incluye:
- `autogluon[tabular]>=1.0.0`
- `scikit-learn>=1.3.0`

### Paso 2: Ejecutar entrenamiento

```bash
cd scripts/entrenamiento
python autogluon_clasificacion.py
```

O desde la raíz:

```bash
python scripts/entrenamiento/autogluon_clasificacion.py
```

### Paso 3: Esperar a que termine

Terminal mostará:
```
========================================
ENTRENAMIENTO DE CLASIFICACION BINARIA CON AUTOGLUON
========================================

[1/5] Cargando Silver...
Datos cargados: 1320873 filas, 15 columnas
Distribucion target: {0: 660000, 1: 660873}

[2/5] Haciendo split temporal (70% / 15% / 15%)...
Train: 924611 | Valid: 198131 | Test: 198131

[3/5] Creando y entrenando predictor de AutoGluon...
Nota: este paso puede tardar dependiendo del time_limit que elijas.
[Training in progress...]

[4/5] Evaluando metricas en test...
Accuracy: 0.5234
ROC-AUC: 0.5812
Precision: 0.5189
Recall: 0.4956
F1: 0.5070

[5/5] Guardando modelo y resultados...
Modelo guardado: modelos/Market_AI_14-04-26_123456
Metricas guardadas: evaluaciones/metricas_clasificacion.json

[BONUS] Ranking de modelos entrenados:
                  Model       Score
WeightedEnsemble_  0.5812
XGBoost            0.5801
LightGBM           0.5795
CatBoost           0.5612
...
```

---

## Archivos generados

```
modelos/
  └─ Market_AI_DD-MM-YY_HHMMSS/
     ├─ metadata.json
     ├─ auxiliary/
     └─ models/
        ├─ WeightedEnsemble_L3
        ├─ XGBoost_...
        ├─ LightGBM_...
        └─ CatBoost_...

evaluaciones/
  └─ metricas_clasificacion.json
     {
       "metricas": {
         "accuracy": 0.5234,
         "auc": 0.5812,
         "precision": 0.5189,
         "recall": 0.4956,
         "f1": 0.5070
       },
       "filas_train": 924611,
       "filas_test": 198131
     }
```

---

## Interpretación de métricas

| Métrica | Rango | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.0 - 1.0 | % de predicciones correctas. Sesga con target desbalanceado. |
| **AUC** | 0.5 - 1.0 | Capacidad de ranking. 0.5 = azar, >0.55 = aprende, >0.65 = útil. |
| **Precision** | 0.0 - 1.0 | De los que predijimos sube, cuántos realmente subieron. |
| **Recall** | 0.0 - 1.0 | De los que realmente subieron, cuántos predijimos. |
| **F1** | 0.0 - 1.0 | Balance entre precision y recall. |

**Regla de oro**: en mercados financieros, esperar AUC > 0.55 es realista; >0.60 es buena señal.

---

## Customización

### Cambiar time_limit

En `scripts/entrenamiento/autogluon_clasificacion.py`, línea ~203:

```python
# Para prototipado rápido (5 min):
predictor = entrenar_autogluon(df_train, df_valid, time_limit=300)

# Para producción (1 hora):
predictor = entrenar_autogluon(df_train, df_valid, time_limit=3600)
```

Más tiempo = mejores modelos, pero más espera.

### Cambiar proporciones de split

Línea ~197:

```python
df_train, df_valid, df_test = hacer_split_temporal(df, train_pct=0.7, valid_pct=0.15)
# También: train_pct=0.8, valid_pct=0.1 (80/10/10)
```

---

## Troubleshooting

### Error: "No se encontraron archivos Silver"

Verifica que existe `data/silver/clasificacion_1d_*.parquet`.

```powershell
ls data/silver/
```

Si está vacío, ejecuta primero:
```bash
python scripts/procesamiento/procesado_clasificacion.py
```

### Error: "Import autogluon could not be resolved"

Dependencias no instaladas. Ejecuta:
```bash
pip install -r requirements.txt
```

### Error: "Memory overflow"

Si la máquina es lenta, reduce `time_limit` a 300-600 segundos o reduce dataset en desarrollo (muestra primeras N filas).

---

## Próximos pasos

1. Evaluar si AUC > 0.55. Si no, revisar features o datos.
2. Analizar feature importance (qué features importan más).
3. Entrenar modelo de regresión sobre retornos.
4. Comparar: ¿cuál modelo es más estable?
5. Decidir: ¿usar clasificación, regresión o ensemble?

Ver [docs/clasificacion_binaria.md](clasificacion_binaria.md) para conceptos.
