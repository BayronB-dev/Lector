"""
Guarda la salida cruda del OCR (cajas, textos y confianzas) y las cajas del detector de líneas de cada
imagen de prueba. Sirve para probar cambios del posproceso (orden de líneas, corrección del texto)
sin volver a correr los modelos, que es lo que más tarda.

Uso (desde esta carpeta):
    python guardar_ocr_crudo.py --carpeta muestras --salida crudo_sinteticas.json
"""

import argparse
import csv
import json
import os
import sys

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "aplicacion"))
import lector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carpeta", required=True)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()

    carpeta = os.path.join(AQUI, args.carpeta)
    datos = []
    with open(os.path.join(carpeta, "metadata.csv"), encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            imagen = lector.cargar_imagen(os.path.join(carpeta, fila["archivo"]))
            calidad = lector.medir_calidad(imagen)
            preparada = lector.mejorar_si_hay_poca_luz(lector.poner_texto_oscuro(imagen))
            if preparada.ndim == 2:
                preparada = cv2.cvtColor(preparada, cv2.COLOR_GRAY2BGR)
            r = lector.cargar_ocr().predict(preparada)[0]
            resultados = [(caja.tolist(), texto, float(conf))
                          for caja, texto, conf in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]) if texto.strip()]
            horizontales, libres = lector.cargar_detector().detect(preparada)
            cajas = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, x1, y0, y1 in horizontales[0]]
            cajas += [[list(p) for p in caja] for caja in libres[0]]
            datos.append({**fila, "calidad": calidad, "resultados": resultados,
                          "cajas_detector": json.loads(json.dumps(cajas, default=float))})
            print(fila["archivo"], len(resultados), "fragmentos", flush=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
