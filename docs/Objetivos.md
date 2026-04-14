# Objetivos del proyecto (fase actual)

## Objetivo principal

Construir un pipeline util y simple para predecir movimientos diarios de acciones usando datos de Yahoo.

## Enfoque de modelado

Trabajaremos en dos etapas:

1. Clasificacion binaria (primero)
	- Objetivo: predecir si manana sube o no sube.
	- Target: `target_updown_t1`.
2. Regresion de retorno (despues)
	- Objetivo: predecir cuanto sube o baja manana.
	- Target: `target_ret_t1`.

## Horizonte temporal

Fase actual:

1. Datos diarios (1d)
2. Prediccion a 1 dia (t+1)

Mas adelante se podra probar t+5 o t+10.

## Prioridades del proyecto

1. Datos utiles y limpios antes que complejidad
2. Features explicables y con sentido
3. Evaluacion correcta con corte temporal
4. Comparar clasificacion vs regresion

## Resultado esperado

1. Modelo de clasificacion que de una senal diaria (sube/no sube)
2. Modelo de regresion que estime la magnitud del retorno
3. Ranking de acciones: entre las que se espera que suban, priorizar las de mayor retorno estimado

## Estado actual (Abril 2026)

🟡 **Fase 3 - Clasificación**: 
- Script `scripts/entrenamiento/autogluon_clasificacion.py` **listo para ejecutar**.
- AutoGluon entrena automáticamente 15+ modelos (XGBoost, LightGBM, CatBoost, NN, etc.).
- Output: modelo + leaderboard + métricas (Accuracy, AUC, Precision, Recall, F1).

**Ejecución**:
```bash
python scripts/entrenamiento/autogluon_clasificacion.py
```

## Plan de ejecucion

**Corto plazo (próximos días)**:
1. ✅ Entrenar baseline de clasificación (en marcha).
2. Evaluar AUC y métricas en test.
3. Documentar feature importance (cuáles features importan más).

**Mediano plazo**:
4. Entrenar baseline de regresión sobre retorno.
5. Comparar estabilidad: ¿cuál modelo es más confiable?
6. Decidir: ¿solo clasificación o ensemble?

## Features disponibles (12 en Silver)

**Retornos**: ret_1d, ret_5d, ret_10d  
**Tendencia**: ma_ratio, price_vs_ma20  
**Volatilidad**: volatility_10d, volatility_20d  
**Gap/Rango**: gap_prop, range_prop  
**Volumen**: vol_ratio, vol_change_1d  

Siguientes iteraciones: agregar features técnicas adicionales si es necesario.

## Métricas de éxito

- AUC > 0.55: modelo aprende algo útil.
- AUC > 0.60: señal confiable para trading.
- AUC > 0.65: potencial de retorno real.
3. Mantener solo lo que mejore resultados en validacion temporal.

## Regla de escalado

1. Escalado opcional en Fase 1.
2. Escalado recomendado si se usan features absolutas con modelos lineales.
3. Ajustar scaler solo con train para evitar fuga de informacion.

## Checkpoint actual

1. Ingesta diaria completada y validada.
2. Dataset Bronze consolidado (139 simbolos, 1,324,059 filas).
3. Dataset Silver de clasificacion completado (1,320,873 filas).
4. Target de clasificacion operativo: `target_updown_t1`.
5. Siguiente hito: entrenamiento baseline con split temporal.
