"""
Vuelve a medir el posproceso (orden de líneas, corrección del texto, aviso) a partir de la salida cruda
del OCR guardada por guardar_ocr_crudo.py, sin correr los modelos. Usa las mismas métricas que
evaluar.py. Sirve para comprobar en segundos que un cambio no empeora ninguna imagen; la cifra final
siempre se confirma con evaluar.py.

Uso (desde esta carpeta):
    python reprocesar.py crudo_sinteticas.json crudo_libros.json crudo_etiquetas.json
"""

import json
import os
import sys

import jiwer

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("LECTOR_DIR", os.path.join(AQUI, "..", "aplicacion")))   # otra versión para comparar
import lector  # noqa: E402
from evaluar import CER_LECTURA_MALA, WER_VOZ_LECTURA_MALA, normalizar, solo_palabras  # noqa: E402

CARPETAS = {"crudo_sinteticas.json": "muestras", "crudo_libros.json": "fotos_reales",
            "crudo_etiquetas.json": "fotos_reales_etiquetas"}


def reprocesar(d, carpeta):
    resultados = lector.descartar_dudosos([tuple(r) for r in d["resultados"]])
    lineas_leidas = lector.ordenar_en_lineas(resultados)
    detectadas = len(lector.ordenar_en_lineas([(caja, "", 1.0) for caja in d["cajas_detector"]]))
    lineas = {"leidas": len(lineas_leidas), "detectadas": detectadas}
    try:
        en_orden = lector.ordenar_en_lineas(resultados, columnas=True)
    except TypeError:   # versión anterior del lector, sin lectura por columnas
        en_orden = lineas_leidas
    texto = lector.corregir_texto("\n".join(en_orden)) if resultados else ""
    total = sum(len(r[1]) for r in resultados)
    confianza = sum(r[2] * len(r[1]) for r in resultados) / max(total, 1) if resultados else 0.0
    aviso, leer = lector.decidir_lectura(texto, confianza, d["calidad"], lineas)
    with open(os.path.join(AQUI, carpeta, d["referencia"]), encoding="utf-8") as f:
        ref = normalizar(f.read())
    pred = normalizar(texto)
    return {
        "archivo": d["archivo"], "cer": jiwer.cer(ref, pred) if pred else 1.0,
        "wer_voz": jiwer.wer(solo_palabras(ref), solo_palabras(pred)) if pred else 1.0,
        "aviso": aviso is not None, "leido": leer, "texto": pred,
        "voz": lector.normalizar_para_voz(texto),
    }


def main():
    for nombre in sys.argv[1:]:
        with open(nombre, encoding="utf-8") as f:
            datos = json.load(f)
        filas = [reprocesar(d, CARPETAS[os.path.basename(nombre)]) for d in datos]
        salida = nombre.replace(".json", "_reprocesado.json")
        with open(salida, "w", encoding="utf-8") as f:
            json.dump(filas, f, ensure_ascii=False, indent=1)
        malas = [r for r in filas if r["cer"] > CER_LECTURA_MALA or r["wer_voz"] > WER_VOZ_LECTURA_MALA]
        print(f"{nombre}: {len(filas)} imágenes | WER de voz promedio {100 * sum(r['wer_voz'] for r in filas) / len(filas):.2f} % | "
              f"malas {len(malas)} (sin aviso: {sum(not r['aviso'] for r in malas)})")


if __name__ == "__main__":
    main()
