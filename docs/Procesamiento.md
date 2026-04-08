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
Lectura de Bronze ya validada (139 simbolos, ~1.32M filas).
Siguiente fase: implementar features y targets con pandas para clasificacion y regresion de retorno a t+1.
