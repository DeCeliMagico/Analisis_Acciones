## Grupo 1 — Momentum largo plazo + indicadores adicionales (sin datos externos)
Añadir ret_10d, ret_20d, ret_60d, posición en Bandas de Bollinger, proximidad al máximo de 52 semanas y OBV.
Todo se calcula desde los datos Bronze existentes, sin tocar la ingesta.
**ESTADO: Implementado. Resultados mixtos — KLAC y TSLA empeoraron, NFLX y MSFT mejoraron.**

## Grupo 2 — SPY y VIX como contexto de mercado ✅ MEJOR CONFIGURACIÓN ACTUAL
Descargar SPY (mercado general) y ^VIX con la ingesta existente y añadir features relativos:
retorno del ticker vs SPY en 1d/5d/20d, nivel VIX y su cambio en 1d/5d.
**ESTADO: Implementado. Mejor configuración global (CSCO 57.8%, GOOG 57.1%, MSFT 56.0%).**

## Grupo 3 — Retorno relativo al sector ETF
Asignar a cada ticker su ETF de sector (XLK tech, XLF financials, XLY consumo...) y calcular
el retorno relativo ticker vs sector en 1d/5d/20d, complementando al Grupo 2.
**ESTADO: Probado y descartado. XLC solo existe desde 2018, recortaba histórico de GOOG/NFLX.**

---

## Próximos pasos (a explorar)

## A — Selección de tickers por señal real
Filtrar el portfolio para operar solo con los tickers donde el modelo tiene señal consistente
(F, BAC, GOOG, NFLX) e ignorar los que no responden al modelo (TSLA, ORCL, KLAC para regresión).

## B — Cambio a clasificación binaria
En lugar de predecir el retorno exacto (regresión), predecir solo si sube o baja (clasificación).
AutoGluon está más optimizado para clasificación y puede calibrar mejor la señal direccional.

## C — Umbral de entrada por confianza
Solo operar cuando el modelo predice con alta magnitud (|pred| > umbral).
Menos operaciones pero de mayor calidad — evitar trades con señal débil que añaden ruido.
