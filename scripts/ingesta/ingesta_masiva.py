from ingesta_individual import guardar_datos_accion
import time

# Lista fija de simbolos para ingesta masiva (aprox. 130)
SIMBOLOS = [
	"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM",
	"V", "MA", "UNH", "HD", "PG", "XOM", "JNJ", "LLY", "AVGO", "COST",
	"ABBV", "MRK", "PEP", "KO", "ADBE", "CRM", "WMT", "BAC", "MCD", "TMO",
	"ACN", "NFLX", "LIN", "AMD", "ORCL", "CVX", "ABT", "DHR", "WFC", "CSCO",
	"VZ", "CMCSA", "TXN", "QCOM", "INTC", "MS", "PM", "AMGN", "RTX", "NEE",
	"HON", "UPS", "LOW", "IBM", "UNP", "INTU", "SPGI", "CAT", "GS", "BLK",
	"AMAT", "DE", "ISRG", "GILD", "GE", "SYK", "ELV", "BKNG", "LRCX", "T",
	"BA", "MDT", "TJX", "ADP", "SCHW", "C", "PLD", "MMC", "ADI", "CB",
	"VRTX", "MDLZ", "MO", "SO", "ZTS", "CI", "PGR", "NOW", "DUK", "BSX",
	"AXP", "PANW", "AMT", "REGN", "NKE", "MU", "AON", "EOG", "PYPL", "USB",
	"CL", "PNC", "EW", "MPC", "CVS", "CSX", "ITW", "BDX", "NSC", "FCX",
	"GM", "F", "SLB", "TGT", "APD", "EQIX", "SHW", "ICE", "MAR", "COP",
	"ROP", "KLAC", "SNPS", "CDNS", "AEP", "HUM", "ORLY", "MMM", "EMR", "SBUX",
	"PH", "NOC", "GD", "FDX", "TRV", "AIG", "CME", "MET", "KMB", "PSA"
]




def ejecutar_ingesta_masiva() -> None:
	"""Orquesta la ingesta de todos los simbolos."""

	simbolos = SIMBOLOS
	total = len(simbolos)
	exitos = 0
	fallos = 0
	simbolos_fallidos = []
	contador = 1

	for simbolo in simbolos:
		print(f"[{contador}/{total}] Guardando datos de {simbolo}...")
		contador += 1
		try:
			guardar_datos_accion(simbolo)
			exitos += 1
		except Exception as e:
			print(f"Error al procesar {simbolo}: {e}")
			fallos += 1
			simbolos_fallidos.append(simbolo)

		time.sleep(0.5)  # Pausa corta entre procesos
	print("Total de simbolos:", total)
	print("Exitos:", exitos)
	print("Fallos:", fallos)
	if simbolos_fallidos:
		print("Simbolos fallidos:", simbolos_fallidos)


if __name__ == "__main__":
    ejecutar_ingesta_masiva()
