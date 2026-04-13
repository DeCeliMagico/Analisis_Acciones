# Procesamiento de datos (fase actual)

## Decision tecnica

Con el volumen actual de datos pandas basta.

Usaremos:

1. pandas para transformaciones
2. pyarrow/parquet para lectura y escritura

No hace falta procesamiento distribuido en esta fase.

## Objetivo del procesamiento

Pasar de Bronze (parquet por simbolo) a un dataset limpio y util para:

1. Analisis
2. Modelo de subida/bajada (clasificacion)
3. Modelo de regresion de retorno (magnitud)

## Limpieza (simple y suficiente por ahora)

Como los datos de Yahoo ya vienen razonablemente bien, la limpieza sera minima:

1. Eliminar nulos en columnas clave: ts_event_utc, open, high, low, close, volume
2. Ordenar por symbol + ts_event_utc
3. Eliminar duplicados por symbol + ts_event_utc
4. Validaciones basicas de negocio:
	1. high >= low
	2. volume >= 0

## Metricas/columnas recomendadas (con sentido)

Estas son las mas utiles para empezar, sin meter ruido.

## Nota rapida de nombres

`pct` significa porcentaje (percentage).

Ejemplos:

1. `ret_1d_pct`: retorno diario en %
2. `gap_pct`: gap en %
3. `range_pct`: rango de la vela en %

## 1) Retornos

1. `ret_1d` = cuanto subio o bajo hoy respecto al cierre de ayer (en %)
	- Para que sirve: ver la rentabilidad diaria de forma clara.
2. `ret_5d` = cuanto subio o bajo respecto a hace 5 dias (en %)
	- Para que sirve: ver tendencia corta sin tanto ruido diario.
3. `ret_10d` = cambio en % frente a hace 10 dias
	- Para que sirve: ver una tendencia un poco mas estable.

## 2) Gap de apertura

1. `close_prev` = precio de cierre del dia anterior
	- Para que sirve: comparar hoy contra ayer.
2. `gap` = open - close_prev
	- Para que sirve: ver si el precio abrio con salto.
3. `gap_pct` = (open / close_prev) - 1
	- Para que sirve: medir ese salto en porcentaje.

## 3) Rango y volatilidad simple

1. `range_hl` = high - low
	- Para que sirve: cuanto se movio el precio dentro del dia.
2. `range_pct` = (high - low) / close
	- Para que sirve: comparar movimientos entre acciones caras y baratas.
3. `volatility_10d` = desviacion estandar de ret_1d en ventana 10
	- Para que sirve: ver si ultimamente se mueve mucho o poco.
4. `volatility_20d` = lo mismo pero en 20 dias
	- Para que sirve: version mas estable del riesgo reciente.

## 4) Tendencia (medias moviles)

1. `ma_5` = media del precio de cierre de los ultimos 5 dias
2. `ma_20` = media del cierre de los ultimos 20 dias
3. `ma_50` = media del cierre de los ultimos 50 dias
4. `ma_ratio` = ma_5 / ma_20
	- Para que sirve: ver si la tendencia corta va por encima o por debajo de la media.
5. `price_vs_ma20` = close / ma_20
	- Para que sirve: saber si el precio actual esta por encima o por debajo de su media reciente.

## 5) Volumen

1. `vol_ma_20` = media movil de volumen a 20 dias
2. `vol_ratio` = volume / vol_ma_20
	- Para que sirve: detectar dias con actividad anormal.
3. `vol_change_1d` = cambio porcentual del volumen frente al dia anterior
	- Para que sirve: ver si hoy se negocio mucho mas o mucho menos que ayer.

## 6) Etiquetas objetivo (targets)

Para regresion:

1. `target_ret_t1` = retorno de manana
	- Para que sirve: estimar cuanto puede subir o bajar.

Para clasificacion:

1. `target_updown_t1` = 1 si target_ret_t1 > 0, 0 en caso contrario
	- Para que sirve: predecir direccion (sube o no sube).

## Que NO meter de momento (ruido o complejidad extra)

1. Decenas de indicadores tecnicos avanzados de golpe
2. Features intradia (hasta consolidar diario)
3. Variables macro/fundamentales externas en esta fase

Primero: base pequena, limpia y explicable.

## Estrategia por fases para el modelo

## Fase 1 (recomendada para empezar)

Usar sobre todo features relativas/normalizadas por construccion:

1. retornos (`ret_*`)
2. porcentajes (`*_pct`)
3. ratios (`ma_ratio`, `vol_ratio`, `price_vs_ma20`)
4. volatilidad (`volatility_*`)

Ventaja: comparables entre acciones con precios muy distintos.

## Fase 2 (experimento opcional)

Anadir features absolutas de precio/volumen:

1. `open`, `high`, `low`, `close`, `volume`
2. `gap`, `range_hl`

Estas pueden ayudar, pero tambien meter ruido por escala.

## Escalado (normalizacion)

1. En Fase 1, muchas features ya son relativas; escalado no siempre es critico.
2. En Fase 2, para modelos lineales/logisticos, conviene escalar.
3. Si se escala, el scaler se ajusta SOLO con train y luego se aplica a valid/test.

## Esquema de salida recomendado

Dataset Silver/Features (minimo):

1. ts_event_utc
2. symbol
3. open, high, low, close, volume
4. ret_1d, ret_5d, ret_10d
5. close_prev, gap, gap_pct
6. range_hl, range_pct, volatility_10d, volatility_20d
7. ma_5, ma_20, ma_50, ma_ratio, price_vs_ma20
8. vol_ma_20, vol_ratio, vol_change_1d
9. target_ret_t1, target_updown_t1

## Plan de implementacion (simple)

1. Leer todos los parquet de Bronze como dataset logico
2. Calcular features por simbolo ordenado por fecha
3. Eliminar filas iniciales sin ventana suficiente (por rolling)
4. Guardar resultado en `data/silver/features_1d.parquet`

## Estado actual

Ingesta terminada y corregida en granularidad diaria.
Lectura de Bronze validada (139 simbolos, 1,324,059 filas).
Pipeline Silver de clasificacion completado y ejecutado correctamente:

1. Features calculadas por simbolo (retornos, tendencia, volatilidad, volumen).
2. Target `target_updown_t1` generado por simbolo con control de nulos en bordes.
3. Limpieza final aplicada para entrenamiento.
4. Dataset Silver generado: 1,320,873 filas finales.

## Siguiente paso

Entrenamiento baseline de clasificacion binaria:

1. Cargar parquet Silver.
2. Definir X/y.
3. Hacer split temporal train/valid/test.
4. Entrenar y medir metricas base.
