"""
Genera el conjunto de prueba: documentos de ejemplo (con datos inventados) que se
dibujan como imagen y luego se deforman para parecerse a fotos tomadas con un celular.

Cada documento se produce en 12 condiciones, y por cada imagen se guarda su
transcripción de referencia (lo que DEBERÍA leer el OCR). No contiene datos reales de nadie.

Uso:
    python generar_muestras.py              # crea la carpeta "muestras"
    python generar_muestras.py --salida otra_carpeta

Resultado:
    muestras/metadata.csv          archivo, documento, condicion, referencia
    muestras/imagenes/*.jpg        las imágenes
    muestras/referencias/*.txt     el texto correcto de cada documento

Para agregar fotos reales (tomadas con el celular), se deja el mismo formato:
una fila en metadata.csv por foto y su .txt con el texto correcto.
"""

import argparse
import csv
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANCHO = 900
PAPEL = (244, 241, 234)
TINTA = (25, 25, 25)

# Documentos de ejemplo (todo inventado). Cada elemento de la lista es un párrafo/línea.
DOCUMENTOS = {
    "factura": [
        "FACTURA DE VENTA No. 004521",
        "Droguería San José",
        "NIT 900.123.456-7",
        "Fecha: 12/03/2026",
        "Acetaminofén 500 mg x 20 tabletas: $8.500",
        "Jarabe para la tos 120 ml: $14.900",
        "Total a pagar: $23.400",
        "Gracias por su compra.",
    ],
    "receta": [
        "Receta médica (ejemplo)",
        "Paciente: Ana María Rojas",
        "Medicamento: Ibuprofeno 400 mg",
        "Indicación: tomar una tableta cada 8 horas durante 5 días, después de las comidas.",
        "Fecha: 05/10/2026",
        "Si el dolor continúa, consulte a su médico.",
    ],
    "etiqueta": [
        "Leche entera",
        "Contenido neto: 900 ml",
        "Consumir preferiblemente antes de: 15/08/2027",
        "Conservar en un lugar fresco y seco.",
        "Una vez abierto, manténgase refrigerado y consúmase en 3 días.",
    ],
    "aviso": [
        "AVISO IMPORTANTE",
        "El servicio de agua será suspendido el sábado 14 de noviembre, de 8:00 a. m. a 2:00 p. m., en el barrio El Prado.",
        "Por favor almacene agua con anticipación.",
        "Disculpe las molestias.",
    ],
    "autorizacion": [
        "Autorización de servicios",
        "EPS Salud Vida",
        "Número de autorización: 7788-2026",
        "Servicio autorizado: consulta de medicina general",
        "Debe presentarse con su documento de identidad.",
        "Válida hasta el 30/11/2026",
    ],
    "carta": [
        "Señor usuario:",
        "Le informamos que su cita fue programada para el martes 3 de noviembre a las 9:30 a. m. en la sede de Medellín.",
        "Si no puede asistir, comuníquese al 604 555 0123 con dos días de anticipación.",
        "Cordialmente, Atención al Usuario",
    ],
}

CONDICIONES = ["normal", "poca_luz", "desenfoque", "rotacion", "sombra", "letra_pequena", "cursiva",
               "desenfoque_fuerte", "luz_tenue_ruido", "perspectiva"]
# Condiciones más difíciles, agregadas cuando el OCR nuevo (PaddleOCR) leyó bien todas las anteriores: sirven
# para encontrar dónde falla y para comprobar que el aviso funciona. Usan otra semilla, para que las 60
# imágenes anteriores sigan siendo exactamente las mismas.
CONDICIONES_DIFICILES = ["desenfoque_extremo", "letra_diminuta"]

# Fuentes: se prueban rutas de Linux y de Windows; se usa la primera que exista.
FUENTES = {
    "normal": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
               "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"],
    "negrita": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf"],
    "cursiva": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
                "C:/Windows/Fonts/timesi.ttf", "/Library/Fonts/Times New Roman Italic.ttf"],
}


def buscar_fuente(estilo, tamano):
    for ruta in FUENTES[estilo]:
        if os.path.exists(ruta):
            return ImageFont.truetype(ruta, tamano)
    raise FileNotFoundError(f"No se encontró una fuente para el estilo '{estilo}'. Edite FUENTES.")


def dividir_en_lineas(texto, fuente, ancho_max):
    """Parte un párrafo en varias líneas para que quepa en el ancho de la hoja."""
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = (actual + " " + palabra).strip()
        if fuente.getlength(prueba) <= ancho_max:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


def dibujar_documento(parrafos, estilo="normal", tamano=30):
    """Dibuja el documento sobre un fondo de papel. Devuelve un arreglo RGB."""
    fuente = buscar_fuente(estilo, tamano)
    fuente_titulo = buscar_fuente("negrita" if estilo == "normal" else estilo, int(tamano * 1.15))
    margen = 60
    interlineado = int(tamano * 1.55)

    filas = []   # (texto, fuente)
    for i, parrafo in enumerate(parrafos):
        f = fuente_titulo if i == 0 else fuente
        for linea in dividir_en_lineas(parrafo, f, ANCHO - 2 * margen):
            filas.append((linea, f))

    alto = max(2 * margen + interlineado * len(filas) + 40, 500)
    imagen = Image.new("RGB", (ANCHO, alto), PAPEL)
    lapiz = ImageDraw.Draw(imagen)
    y = margen
    for linea, f in filas:
        lapiz.text((margen, y), linea, font=f, fill=TINTA)
        y += interlineado
    return np.array(imagen)


# --- Condiciones que imitan fotos reales -----------------------------------
def aplicar_condicion(img, condicion, rng):
    img = img.astype(np.float32)
    ruido = 4   # ruido leve de sensor en todas las condiciones (como una cámara real)
    if condicion == "poca_luz":
        img = img * 0.35
    elif condicion == "desenfoque":
        img = cv2.GaussianBlur(img, (0, 0), 1.5)
    elif condicion == "desenfoque_fuerte":
        img = cv2.GaussianBlur(img, (0, 0), 3.0)
    elif condicion == "desenfoque_extremo":
        img = cv2.GaussianBlur(img, (0, 0), 5.0)
    elif condicion == "luz_tenue_ruido":
        # Muy poca luz: la cámara sube la sensibilidad y aparece mucho ruido.
        img = img * 0.20
        ruido = 14
    elif condicion == "perspectiva":
        # El celular queda inclinado: la parte de arriba se ve más angosta que la de abajo.
        alto, ancho = img.shape[:2]
        origen = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
        destino = np.float32([[0.14 * ancho, 0.03 * alto], [0.86 * ancho, 0.03 * alto],
                              [ancho, alto], [0, alto]])
        matriz = cv2.getPerspectiveTransform(origen, destino)
        img = cv2.warpPerspective(img, matriz, (ancho, alto), borderValue=PAPEL)
    elif condicion == "sombra":
        degradado = np.linspace(1.0, 0.45, img.shape[1], dtype=np.float32)[None, :, None]
        img = img * degradado
    elif condicion == "rotacion":
        alto, ancho = img.shape[:2]
        angulo = 12
        matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, 1.0)
        cos, sen = abs(matriz[0, 0]), abs(matriz[0, 1])
        nuevo_ancho, nuevo_alto = int(alto * sen + ancho * cos), int(alto * cos + ancho * sen)
        matriz[0, 2] += nuevo_ancho / 2 - ancho / 2
        matriz[1, 2] += nuevo_alto / 2 - alto / 2
        img = cv2.warpAffine(img, matriz, (nuevo_ancho, nuevo_alto), borderValue=PAPEL)
    img = img + rng.normal(0, ruido, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salida", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "muestras"))
    args = parser.parse_args()

    os.makedirs(os.path.join(args.salida, "imagenes"), exist_ok=True)
    os.makedirs(os.path.join(args.salida, "referencias"), exist_ok=True)
    rng = np.random.default_rng(2026)   # semilla fija: siempre salen las mismas imágenes

    rng_dificiles = np.random.default_rng(2027)

    filas = []
    for condiciones, generador in [(CONDICIONES, rng), (CONDICIONES_DIFICILES, rng_dificiles)]:
        for nombre, parrafos in DOCUMENTOS.items():
            referencia = " ".join(parrafos)
            ruta_ref = os.path.join("referencias", f"{nombre}.txt")
            with open(os.path.join(args.salida, ruta_ref), "w", encoding="utf-8") as f:
                f.write(referencia)

            for condicion in condiciones:
                if condicion == "letra_pequena":
                    base = dibujar_documento(parrafos, "normal", tamano=17)
                elif condicion == "letra_diminuta":
                    base = dibujar_documento(parrafos, "normal", tamano=11)
                elif condicion == "cursiva":
                    base = dibujar_documento(parrafos, "cursiva", tamano=30)
                else:
                    base = dibujar_documento(parrafos, "normal", tamano=30)
                final = aplicar_condicion(base, condicion, generador)

                archivo = f"{nombre}_{condicion}.jpg"
                ruta_img = os.path.join(args.salida, "imagenes", archivo)
                cv2.imwrite(ruta_img, cv2.cvtColor(final, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                filas.append([os.path.join("imagenes", archivo), nombre, condicion, ruta_ref])

    with open(os.path.join(args.salida, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["archivo", "documento", "condicion", "referencia"])
        escritor.writerows(filas)
    print(f"Listo: {len(filas)} imágenes en '{args.salida}'.")


if __name__ == "__main__":
    main()
