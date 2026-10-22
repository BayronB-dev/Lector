"""
Evaluación del lector: mide qué tan bien funciona en cada condición.
[ÉTICA: equidad] El principio se vuelve medible comparando el desempeño entre
condiciones (poca luz, borroso, torcido, letra pequeña...) y mostrando dónde falla.

Qué mide:
    CER  = porcentaje de caracteres mal leídos (0 = perfecto)
    WER  = porcentaje de palabras mal leídas (y WER de voz, sin contar la puntuación)
    Aviso = si el sistema advirtió que no estaba seguro (y si hizo bien en advertir)
    Voz   = si se pudo generar el audio de lo leído

Uso (desde esta carpeta):
    python evaluar.py                                                        # imágenes sintéticas
    python evaluar.py --carpeta fotos_reales --salida resultados/fotos_reales   # fotos reales

Salidas (por defecto en resultados/sinteticas):
    resultados_por_imagen.csv, resumen_por_condicion.csv, resumen_por_documento.csv,
    cer_por_condicion.png (si está instalado matplotlib)
"""

import argparse
import csv
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from statistics import mean

import jiwer

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "aplicacion"))   # el lector está en la carpeta "aplicacion"
import lector  # noqa: E402

# Se considera "lectura mala" si más del 10 % de los caracteres salen mal, o si más del 15 % de las
# palabras salen mal (WER "de voz"). La segunda condición importa porque el CER puede verse bajo
# aunque una de cada cuatro palabras esté mal, y eso para quien escucha es una lectura mala.
CER_LECTURA_MALA = 0.10
WER_VOZ_LECTURA_MALA = 0.15


def normalizar(texto):
    """Deja el texto en una sola línea con espacios simples, para comparar justo."""
    return re.sub(r"\s+", " ", texto).strip()


def solo_palabras(texto):
    """Minúsculas y sin signos de puntuación: lo que importa para quien escucha, porque la voz no pronuncia las comas."""
    texto = re.sub(r"[^\w\s/]", " ", texto.lower().replace("_", " "))
    return re.sub(r"\s+", " ", texto).strip()


def evaluar_imagen(ruta_imagen, referencia):
    imagen = lector.cargar_imagen(ruta_imagen)
    calidad = lector.medir_calidad(imagen)

    inicio = time.time()
    texto, confianza, lineas = lector.reconocer_texto(imagen)
    t_ocr = time.time() - inicio

    aviso, leer = lector.decidir_lectura(texto, confianza, calidad, lineas)

    pred = normalizar(texto)
    ref = normalizar(referencia)
    cer = jiwer.cer(ref, pred) if pred else 1.0
    wer = jiwer.wer(ref, pred) if pred else 1.0
    wer_voz = jiwer.wer(solo_palabras(ref), solo_palabras(pred)) if pred else 1.0

    # Se genera el audio de lo que se leería (aviso + texto) y se comprueba que exista.
    inicio = time.time()
    voz_ok, duracion = False, 0.0
    try:
        a_decir = lector.mensaje_hablado(texto, aviso, leer)
        with tempfile.TemporaryDirectory() as carpeta:   # el audio se borra solo
            duracion = lector.sintetizar(a_decir, os.path.join(carpeta, "a.wav"))
        voz_ok = duracion > 0
    except Exception as error:
        print("   error de voz:", error)
    t_voz = time.time() - inicio

    return {
        "cer": cer, "wer": wer, "wer_voz": wer_voz, "confianza": confianza,
        "palabras_conocidas": lector.proporcion_palabras_conocidas(texto),
        "lineas_leidas": lineas["leidas"], "lineas_detectadas": lineas["detectadas"],
        "brillo": calidad["brillo"], "nitidez": calidad["nitidez"],
        "aviso": aviso is not None, "leido": leer, "voz_ok": voz_ok, "duracion_audio": duracion,
        "t_ocr": t_ocr, "t_voz": t_voz, "texto": pred,
    }


def resumir(filas, clave):
    grupos = defaultdict(list)
    for f in filas:
        grupos[f[clave]].append(f)
    resumen = []
    for nombre, lista in grupos.items():
        resumen.append({
            clave: nombre,
            "n": len(lista),
            "cer_prom": mean(f["cer"] for f in lista),
            "wer_prom": mean(f["wer"] for f in lista),
            "wer_voz_prom": mean(f["wer_voz"] for f in lista),
            "confianza_prom": mean(f["confianza"] for f in lista),
            "con_aviso": sum(f["aviso"] for f in lista),
            "no_leidas": sum(not f["leido"] for f in lista),
        })
    return sorted(resumen, key=lambda r: r["cer_prom"])


def guardar_csv(ruta, filas, columnas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()})


def grafico(resumen, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    nombres = [r["condicion"] for r in resumen]
    valores = [r["cer_prom"] * 100 for r in resumen]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(nombres, valores, color="#555555")
    for i, v in enumerate(valores):
        ax.text(v + 0.5, i, f"{v:.1f} %", va="center", fontsize=9)
    ax.set_xlabel("Caracteres mal leídos (CER promedio, %)")
    ax.set_xlim(0, max(valores) * 1.2 + 1)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carpeta", default=os.path.join(AQUI, "muestras"))
    parser.add_argument("--salida", default=os.path.join(AQUI, "resultados", "sinteticas"))
    args = parser.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    with open(os.path.join(args.carpeta, "metadata.csv"), encoding="utf-8") as f:
        metadata = list(csv.DictReader(f))

    print(f"Evaluando {len(metadata)} imágenes (la primera tarda más porque carga los modelos)...")
    filas = []
    for i, fila in enumerate(metadata, 1):
        with open(os.path.join(args.carpeta, fila["referencia"]), encoding="utf-8") as f:
            referencia = f.read()
        r = evaluar_imagen(os.path.join(args.carpeta, fila["archivo"]), referencia)
        r.update({"archivo": fila["archivo"], "documento": fila["documento"], "condicion": fila["condicion"]})
        filas.append(r)
        print(f"[{i:2d}/{len(metadata)}] {fila['archivo']:<45} CER={r['cer']:.3f} conf={r['confianza']:.2f} aviso={'sí' if r['aviso'] else 'no'}")

    columnas = ["archivo", "documento", "condicion", "cer", "wer", "wer_voz", "confianza", "palabras_conocidas",
                "lineas_leidas", "lineas_detectadas", "brillo", "nitidez", "aviso", "leido", "voz_ok", "duracion_audio", "t_ocr", "t_voz", "texto"]
    guardar_csv(os.path.join(args.salida, "resultados_por_imagen.csv"), filas, columnas)

    por_condicion = resumir(filas, "condicion")
    por_documento = resumir(filas, "documento")
    cols_resumen = ["n", "cer_prom", "wer_prom", "wer_voz_prom", "confianza_prom", "con_aviso", "no_leidas"]
    guardar_csv(os.path.join(args.salida, "resumen_por_condicion.csv"), por_condicion, ["condicion"] + cols_resumen)
    guardar_csv(os.path.join(args.salida, "resumen_por_documento.csv"), por_documento, ["documento"] + cols_resumen)
    grafico(por_condicion, os.path.join(args.salida, "cer_por_condicion.png"))

    print("\n=== Resultados por condición (ordenados de mejor a peor) ===")
    print(f"{'condición':<19}{'n':>3}{'CER':>8}{'WER':>8}{'WER voz':>9}{'conf.':>8}{'avisos':>8}{'no leídas':>11}")
    for r in por_condicion:
        print(f"{r['condicion']:<19}{r['n']:>3}{r['cer_prom']*100:>7.1f}%{r['wer_prom']*100:>7.1f}%{r['wer_voz_prom']*100:>8.1f}%{r['confianza_prom']:>8.2f}{r['con_aviso']:>8}{r['no_leidas']:>11}")
    print("\n=== Resultados por tipo de documento ===")
    for r in por_documento:
        print(f"{r['documento']:<19}{r['n']:>3}{r['cer_prom']*100:>7.1f}%{r['wer_prom']*100:>7.1f}%{r['wer_voz_prom']*100:>8.1f}%{r['confianza_prom']:>8.2f}{r['con_aviso']:>8}{r['no_leidas']:>11}")

    # Brecha entre la mejor y la peor condición. [ÉTICA: equidad]
    brecha = por_condicion[-1]["cer_prom"] - por_condicion[0]["cer_prom"]
    print(f"\nBrecha de equidad (peor condición - mejor condición, en CER): {brecha*100:.1f} puntos "
          f"({por_condicion[-1]['condicion']} vs {por_condicion[0]['condicion']})")

    # ¿El aviso funciona? Se compara con las lecturas realmente malas.
    es_mala = lambda f: f["cer"] > CER_LECTURA_MALA or f["wer_voz"] > WER_VOZ_LECTURA_MALA
    malas = [f for f in filas if es_mala(f)]
    buenas = [f for f in filas if not es_mala(f)]
    con_aviso = sum(f["aviso"] for f in malas)
    no_leidas = sum(not f["leido"] for f in malas)
    sin_aviso = len(malas) - con_aviso
    print(f"\n=== Aviso (avisa si confianza < {lector.UMBRAL_CONFIANZA}, palabras conocidas < {lector.UMBRAL_PALABRAS_CONOCIDAS:.0%} "
          f"o faltan {lector.LINEAS_FALTANTES_AVISO}+ líneas; no lee si confianza < {lector.UMBRAL_NO_LEER}; lectura mala = CER > {CER_LECTURA_MALA:.0%} o WER de voz > {WER_VOZ_LECTURA_MALA:.0%}) ===")
    print(f"Lecturas malas: {len(malas)} -> con aviso: {con_aviso} (no leídas en voz alta: {no_leidas}), "
          f"SIN aviso (omisiones peligrosas): {sin_aviso}")
    print(f"Lecturas buenas: {len(buenas)} -> con aviso (falsas alarmas): {sum(f['aviso'] for f in buenas)}, "
          f"no leídas: {sum(not f['leido'] for f in buenas)}")

    print(f"\nAudio generado correctamente: {sum(f['voz_ok'] for f in filas)} de {len(filas)}")
    print(f"Tiempo promedio por imagen: OCR {mean(f['t_ocr'] for f in filas):.1f} s | voz {mean(f['t_voz'] for f in filas):.1f} s")


if __name__ == "__main__":
    main()
