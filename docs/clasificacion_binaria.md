# Clasificación Binaria con AutoGluon

## Control de ejecución

**Estado**: 🟡 Listo para entrenar (Abril 2026)

- Script: `scripts/entrenamiento/autogluon_clasificacion.py`
- Dataset: `data/silver/clasificacion_1d_*.parquet` (1.321M filas, 12 features)
- Instrucciones: [docs/Entrenamiento.md](Entrenamiento.md)

**Comando de ejecución**:
```bash
python scripts/entrenamiento/autogluon_clasificacion.py
```

---

## Objetivo

Entrenar un modelo que predice si mañana la acción sube (1) o no sube (0).

## Por qué clasificación binaria

1. **Pregunta clara**: "¿sube o no?"
2. **Reducción de riesgo**: primer paso antes de estimar magnitud (regresión).
3. **Métrica sencilla**: accuracy, AUC, F1.
4. **Producción ágil**: despliega rápido con decisión binaria.

## Metodología: AutoGluon

### Qué es AutoGluon

Framework de AutoML que automáticamente:
1. Prueba ~10 algoritmos en paralelo (XGBoost, LightGBM, CatBoost, redes, etc.).
2. Busca mejores hiperparámetros para cada uno.
3. Valida con datos de validación (no de entrenamiento).
4. Combina los mejores modelos (ensemble).
5. Devuelve predictor final + ranking.

### Por qué AutoGluon

- **Sin decisiones manuales**: el framework decide qué modelo es mejor.
- **Rápido**: entrena múltiples modelos en paralelo.
- **Robusto**: validación temporal evita fugas de información.
- **Interpretable**: te muestra qué modelo ganó y por qué.

### Flujo de AutoGluon

```
Datos Silver (clasificación)
    ↓
Split temporal: train (70%) / valid (15%) / test (15%)
    ↓
AutoGluon entrena en paralelo:
  - XGBoost
  - LightGBM
  - CatBoost
  - Neural Network
  - SVM
  - ...
    ↓
Valida cada modelo con validation data
    ↓
Ensemble: combina predicciones de mejores modelos
    ↓
Modelo final + ranking
    ↓
Evalúa en test (sin haber visto antes)
```

## Split temporal (crítico)

### Qué es

Dividir los datos respetando el orden temporal:

```
|------ 70% train ------|--- 15% valid ---|---- 15% test ----|
(pasado: 1000 días)   (futuro cercano)  (futuro lejano)
```

### Por qué temporal, no aleatorio

1. **Sin fuga de información**: train no mezcla con datos futuros.
2. **Simula producción real**: cuando predices hoy, solo tienes datos del pasado.
3. **Evita overfitting temporal**: el modelo no "memoriza" patrones locales.

### Error común

Barajar los datos (split aleatorio):
- ❌ MALO: mezcla pasado y futuro.
- ❌ MALO: el modelo "ve" el futuro en entrenamiento.
- ❌ MALO: métricas optimistas (no funciona en producción).

## Features disponibles

12 columnas de clasificación Silver:

| Tipo | Columnas |
|------|----------|
| Retornos | ret_1d, ret_5d, ret_10d |
| Tendencia | ma_ratio, price_vs_ma20 |
| Volatilidad | volatility_10d, volatility_20d |
| Gap/Rango | gap_prop, range_prop |
| Volumen | vol_ratio, vol_change_1d |

**Target**: target_updown_t1 (1 si sube, 0 si no sube)

## Métricas de evaluación

### Accuracy

% de predicciones correctas.

$$
\text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
$$

**Cuidado**: sesga si el target está desbalanceado (más 1 que 0 o vice versa).

### ROC-AUC

Mide la capacidad de ranking del modelo (independiente del umbral 0.5).

- 0.5 = puro azar.
- 1.0 = perfecto.
- >0.7 = bueno.

**Mejor que accuracy** cuando el target es desbalanceado.

### Precision

De los que predijimos sube (1), cuántos realmente suben.

$$
\text{Precision} = \frac{\text{TP}}{\text{TP + FP}}
$$

**Usa cuando**: queremos reducir falsos positivos (no quieres invertir en aumento donde no sube).

### Recall (Sensibilidad)

De los que realmente suben, cuántos predijimos.

$$
\text{Recall} = \frac{\text{TP}}{\text{TP + FN}}
$$

**Usa cuando**: no queremos perder oportunidades de ganancia.

### F1

Balance entre precision y recall.

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision + Recall}}
$$

**Usa cuando**: quieres equilibrio entre no perder ganancias y no invertir en falsas señales.

## Interpretación de resultados

Ejemplo de salida:

```
Accuracy: 0.5234
ROC-AUC: 0.5812
Precision: 0.5189
Recall: 0.4956
F1: 0.5070
```

### ¿Es bueno o malo?

1. **Peor que 0.5 en AUC**: el modelo no aprende (azar puro o peor).
2. **0.5-0.55 AUC**: model muy débil, revisar features.
3. **0.55-0.65 AUC**: modelo inicial razonable, base para mejorar.
4. **0.65-0.75 AUC**: modelo bueno, algo útil en producción.
5. **>0.75 AUC**: modelo sólido, confianza alta.

En mercados financieros:
- Esperar >0.55 AUC es realista (mercados son difíciles).
- >0.60 AUC ya es buena señal.
- >0.65+ puede ser productivo.

## Leaderboard de modelos

AutoGluon muestra ranking:

```
Model                Score
WeightedEnsemble_... 0.5812
XGBoost              0.5801
LightGBM             0.5795
CatBoost             0.5612
...
```

**Interpretación**:
- El ensemble (ponderado) es mejor que cada modelo individual.
- Si XGBoost > WeightedEnsemble: revisar si hay overfitting.
- Si muchos modelos tienen score similar: dataset es difícil.

## Uso en producción

1. **Guardar modelo**: AutoGluon guarda en `models/autogluon_baseline/`.
2. **Cargar y predecir**: 
   ```python
   from autogluon.tabular import TabularPredictor
   predictor = TabularPredictor.load("models/autogluon_baseline")
   pred = predictor.predict(new_data)
   ```
3. **Reciclar**: cada mes/trimestre re-entrenar con datos nuevos.

## Feature importance

Tras entrenar, puedes ver qué features importan más:

```python
importance = predictor.feature_importance(test_data)
```

**Uso**: elimina features poco relevantes en futuras versiones.

## Próximas mejoras

1. **Fase 2**: entrenar modelo de regresión para magnitud de retorno.
2. **Fase 3**: ensemble de clasificación + regresión para ranking final.
3. **Fase 4**: validación in-sample vs out-of-sample con datos reales.
4. **Fase 5**: backtesting y simulación financiera.

## Estado actual (checkpoint)

- ✅ Dataset Silver de clasificación generado (1.32M filas).
- 🟡 Entrenamiento AutoGluon en progreso (guía en script).
- ⏳ Evaluación y ajustes pendientes.
