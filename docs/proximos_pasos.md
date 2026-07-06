---

## Estado actual (06-07-2026)

Pipeline de clasificación funcionando con los siguientes scripts:
1. `procesado_clasificacion_5d.py` → Silver con target_updown_t5 (sube >1% en 5d)
2. `autogluon_clasificacion_por_ticker.py` → 25 tickers, 180s, tuning_data temporal
3. `backtest_clasificacion.py` → evaluación por ticker (EV/trade, win%, avg win/loss)
4. `backtest_portfolio_topn.py` → portfolio top-N por P(sube), solo tickers con EV>0

**Mejor resultado conseguido:** $55,490 (+454.90%) con 9 tickers top (180s, 25 modelos entrenados)
**Tickers buenos confirmados:** MU, AMD, FCX, KLAC, AMAT, NVDA, CSCO, BAC, LRCX
**Comando portfolio:** `python backtest_portfolio_topn.py --top 3 --umbral 0.52 --tickers MU,AMD,FCX,KLAC,AMAT,NVDA,CSCO,BAC,LRCX`

---

## Próximos pasos

### A — Mejorar señal con nuevas features
Las features actuales (técnicas de precio/volumen) tienen techo bajo.
Para mejorar los AUC de forma real:
- **Earnings proximity**: días hasta/desde el siguiente earnings report (señal fuerte)
- **Relative strength vs sector**: retorno del ticker vs ETF de su sector (XLK, XLF...)
- **Winsorización del target**: recortar outliers extremos para que el modelo no persiga movimientos irrepetibles

### B — Aumentar tiempo de entrenamiento (pendiente de solución técnica)
El setup actual (tuning_data + sin preset) no escala en tiempo: AutoGluon termina en ~60-100s
por ticker independientemente del time_limit fijado.
Intentar aumentar a 480s con medium_quality sin tuning_data empeoró los resultados (leakage temporal).
**Posible solución:** usar `TimeSeriesSplit` de sklearn para el k-fold interno, o explorar
AutoGluon `TabularPredictor` con `auto_stack=True` + `num_bag_fold=0`.

### C — Ampliar universo de tickers
Con el pipeline actual se pueden entrenar más tickers fácilmente.
Evaluar otros sectores (healthcare, energy, consumer) que no estén en el top-25 actual.
