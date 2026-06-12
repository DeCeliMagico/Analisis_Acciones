# Próximos Pasos para Mejorar Modelos de Regresión

## Objetivo
Aumentar el rendimiento predictivo del modelo de regresión (actualmente con accuracy direccional ~52-60%, muy cercano al aleatorio) mediante:
1. **Enriquecimiento de features** (enfoque en volumen)
2. **Validación más robusta** (walk-forward en lugar de split temporal simple)

---

## Estado Actual
- **Modelos:** 10 tickers entrenados con AutoGluon (NVDA, NFLX, GOOG, TSLA, MSFT, CSCO, BAC, ORCL, F, AVGO)
- **Target:** Retorno logarítmico crudo a 5 días (`target_ret_log_t5`)
- **Features actuales:** 12 features (retornos, MAs, RSI, MACD, volatilidad)
- **Validación actual:** Split temporal 70/15/15 (sin walk-forward)
- **Problema:** R² negativo, accuracy ~50-60% (casi aleatorio)

---

## Paso 1: Enriquecimiento de Features (Volumen)

### 1.1 Análisis del Código Actual
**Archivo:** `scripts/procesamiento/procesado_regresion.py`

Actualmente, el volumen se ingesta del Bronze pero **NO se generan features derivadas** de él más allá de lo básico.

### 1.2 Features de Volumen a Agregar

Implementar las siguientes variables en `procesado_regresion.py` dentro de `crear_features_regresion()`:

| Feature | Fórmula | Interpretación |
|---------|---------|---|
| `volume_ma20` | `volume.rolling(20).mean()` | Volumen promedio de 20 días |
| `volume_ratio` | `volume / volume_ma20` | ¿Es el volumen anómalo hoy? (>1 = alto, <1 = bajo) |
| `obv` | On-Balance Volume | Acumulación/distribución (volumen con signo de precio) |
| `vwap` | Volume-Weighted Average Price | Precio promedio ponderado por volumen (últimos 20 días) |
| `volume_trend_5d` | `volume.rolling(5).mean()` | Tendencia reciente de volumen |

### 1.3 Implementación Detallada

**En `scripts/procesamiento/procesado_regresion.py`:**

1. Agregar las 5 features descritas arriba en la función `crear_features_regresion()` después de los features de volatilidad existentes
2. Actualizar `COLS_TRAIN` para incluir las nuevas columnas
3. Validar que no haya NaNs (usar `.fillna(method='bfill')` si es necesario para los primeros 20 valores)
4. Regenerar archivos Silver con `python scripts/procesamiento/procesado_regresion.py`

**Detalles técnicos:**
- Usar `groupby('symbol')` para que cada ticker calcule sus propios ratios
- El volumen debe estar en datos Bronze (validar que existe la columna `volume`)
- OBV requiere sumar volumen con signo de cambio de precio

### 1.4 Validación
Después de regenerar Silver, verificar:
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/silver/regresion_5d_*.parquet'); print(df.columns); print(df[['volume_ratio', 'obv', 'vwap']].describe())"
```

---

## Paso 2: Implementar Walk-Forward Validation

### 2.1 ¿Por qué cambiar de validación?

**Problema actual:** Split 70/15/15 asume que los patrones de 2018-2022 siguen siendo válidos en 2026 (no es realista en mercados).

**Solución:** Walk-Forward Validation (rolling window) evalúa el modelo en múltiples períodos consecutivos:
- Entrenar: Mes 1-12 → Predecir Mes 13
- Entrenar: Mes 2-13 → Predecir Mes 14
- Promediar resultados

Esto refleja mejor cómo el modelo se comportaría en producción.

### 2.2 Cambios en `autogluon_regresion_por_ticker.py`

**Ubicación:** `scripts/entrenamiento/autogluon_regresion_por_ticker.py`

Reemplazar la función `hacer_split_temporal_ticker()` por una nueva que implemente walk-forward:

```python
def hacer_splits_walkforward(df_ticker: pd.DataFrame, window_size_months: int = 6, stride_months: int = 1):
    """
    Divide dataset en múltiples folds usando walk-forward validation.
    
    Cada fold:
    - Training: window_size_months meses
    - Test: stride_months meses (típicamente 1 mes)
    
    Retorna lista de (df_train, df_test) tuples.
    """
```

### 2.3 Integración en el Flujo de Entrenamiento

**Cambio conceptual:** En lugar de entrenar 1 modelo global (70% train), entrenar 5-10 modelos con diferentes ventanas temporales.

**Pseudo-código:**
```
Para cada ticker:
    Obtener datos ordenados por fecha
    Para cada fold en walk_forward_splits(datos, window_size=6meses):
        - Entrenar AutoGluon con df_train
        - Evaluar en df_test
        - Guardar métricas del fold
    Promediar métricas de todos los folds
    Guardar en JSON
```

### 2.4 Archivos a Modificar

1. **`scripts/entrenamiento/autogluon_regresion_por_ticker.py`:**
   - Agregar función `hacer_splits_walkforward()`
   - Modificar loop de tickers para iterar sobre folds
   - Acumular métricas de cada fold antes de guardar

2. **`evaluaciones/metricas_regresion_por_ticker_*.json`:**
   - Estructura nuevo formato:
   ```json
   {
     "MSFT": {
       "folds": [
         {"fold": 1, "rmse": 0.035, "accuracy_direccion": 0.58, ...},
         {"fold": 2, "rmse": 0.038, "accuracy_direccion": 0.55, ...},
         ...
       ],
       "promedio": {"rmse": 0.036, "accuracy_direccion": 0.56, ...},
       "std": {"rmse": 0.002, "accuracy_direccion": 0.02, ...}
     }
   }
   ```

### 2.5 Parámetros Sugeridos

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `window_size_months` | 6 | Histórico suficiente para entrenar, pero no viejo |
| `stride_months` | 1 | Evaluación mensual (más rigurosa) |
| `num_folds` | ~8-10 | Depende del tamaño total de datos (años de histórico) |

---

## Paso 3: Entrenamiento y Evaluación

### 3.1 Workflow

```bash
# 1. Regenerar Silver con features de volumen
python scripts/procesamiento/procesado_regresion.py

# 2. Entrenar modelos con walk-forward validation
python scripts/entrenamiento/autogluon_regresion_por_ticker.py

# 3. Analizar resultados
python -c "import json; data = json.load(open('evaluaciones/metricas_regresion_por_ticker_*.json')); print(json.dumps(data, indent=2))"
```

### 3.2 Métricas a Monitorear

Después de implementar los cambios, esperar mejora en:
- **Accuracy direccional:** De ~52-60% → Objetivo 55-62% (incluso +1-2% es útil)
- **R²:** Menos negativo o más cercano a 0 (significa menos fuga)
- **Correlación:** Más cercana a 0.1-0.2 (señal débil pero presente)
- **Varianza entre folds:** Baja (modelo estable en diferentes períodos)

---

## Paso 4: Análisis y Debugging (Iterativo)

Si las métricas no mejoran significativamente después de los Pasos 1-2:

1. **Analizar qué features importan:**
   ```bash
   python scripts/analisis/analizar_feature_importance.py
   ```
   Buscar si las nuevas features de volumen tienen importancia >0.05

2. **Revisar distribución de datos:**
   - ¿Hay desbalance extremo en el target? (ej: 90% suben, 10% bajan)
   - ¿Hay outliers extremos en features?

3. **Considerar ajustes en AutoGluon:**
   - Aumentar `time_limit` en entrenamientos
   - Ajustar hiperparámetros específicos (learning rate, max_depth)

---

## Roadmap Detallado

| Fase | Tarea | Archivo | Tiempo Estimado |
|------|-------|---------|---|
| **Fase 1** | Agregar 5 features de volumen | `procesado_regresion.py` | 2-3 horas |
| **Fase 1** | Regenerar Silver | Terminal | 30 min |
| **Fase 2** | Implementar `hacer_splits_walkforward()` | `autogluon_regresion_por_ticker.py` | 3-4 horas |
| **Fase 2** | Adaptar loop de entrenamiento para múltiples folds | `autogluon_regresion_por_ticker.py` | 2-3 horas |
| **Fase 3** | Ejecutar entrenamiento completo (10 tickers × 8 folds) | Terminal | 8-12 horas |
| **Fase 4** | Análisis de resultados y debugging | Análisis manual | 1-2 horas |

---

## Verificación de Progreso

✅ Completado cuando:
1. Silver contiene las 5 nuevas columnas de volumen (sin NaNs)
2. `autogluon_regresion_por_ticker.py` ejecuta walk-forward sin errores
3. Métricas JSON incluye "folds" y "promedio" con desviación estándar
4. Accuracy direccional ha aumentado respecto a los modelos antiguos (02-06-26)

---

## Notas Finales

- **Mantener versión antigua:** Guardar modelos actuales como baseline antes de reentrenar
- **Documentar cambios:** Actualizar timestamps en archivos de salida
- **Testing incremental:** Ejecutar primero con 1 ticker antes de hacer todos los 10
- **Backup:** Los datos Silver regenerados sobrescribirán los antiguos; considerar backup si es crítico

---

**Última actualización:** Junio 12, 2026  
**Próxima revisión:** Después de completar Paso 2
