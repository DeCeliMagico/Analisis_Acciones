---

## Próximos pasos

### E — Evaluar impacto de las features nuevas (Grupo 3)
Comparar el backtest del run con features estacionarias + best_quality
contra el baseline ($23,803 long_only, $28,122 long_short).
Métricas clave: dirección accuracy, correlación, capital final.


### G — Nuevas features a explorar
- **Earnings proximity**: días hasta/desde el siguiente earnings report
- **Winsorización del target**: recortar outliers extremos del retorno para que
  el modelo no persiga movimientos irrepetibles
