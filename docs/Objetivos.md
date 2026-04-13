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

## Plan corto de ejecucion

1. Entrenar baseline de clasificacion
2. Evaluar y ajustar features
3. Entrenar baseline de regresion sobre retorno
4. Comparar estabilidad y utilidad real de ambos modelos

## Plan de features para entrenamiento

1. Fase 1: empezar con features relativas (retornos, porcentajes, ratios, volatilidad).
2. Fase 2: probar incluir features absolutas (OHLCV y rangos absolutos) como experimento.
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
