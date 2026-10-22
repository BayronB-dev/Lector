"""
Lector de documentos para personas con discapacidad visual (prototipo / MVP).

Flujo del sistema:
    captura de imagen -> OCR (PaddleOCR, con aplanado de la página) -> texto ordenado
    -> verificación de que no falten líneas (detector de EasyOCR) -> aviso
    -> texto a voz (Piper) -> reproducción -> borrado de archivos temporales

Se usan SOLO modelos preentrenados (no se entrena nada):
    - PaddleOCR 3.7 (UVDoc + PP-OCRv6: aplana la página, detecta y reconoce el texto en español)
    - Detector CRAFT de EasyOCR 1.7.2 -> solo cuenta las líneas de texto, para comprobar que no falte ninguna
    - Piper (modelo VITS, voz es_MX-claude-high) -> convierte el texto en voz

Principio ético priorizado: equidad y no discriminación (ver informe, sección 5).
Los puntos del código donde se aplica el principio están marcados con "[ÉTICA]".

Uso:
    python lector.py                      # modo cámara (Enter = fotografiar)
    python lector.py --imagen foto.jpg    # lee una imagen que ya existe
    python lector.py --imagen foto.jpg --sin-audio   # solo muestra el texto
"""

import argparse
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave

import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
VOZ_POR_DEFECTO = os.path.join(CARPETA, "voces", "es_MX-claude-high.onnx")

# Confianza del OCR (0 a 1): por debajo de UMBRAL_CONFIANZA el sistema avisa que no está
# seguro; por debajo de UMBRAL_NO_LEER ni siquiera lee el texto. PaddleOCR da 0,98 o más en todas las
# lecturas buenas de prueba y entre 0,59 y 0,77 en las malas (desenfoque extremo); los valores se
# fijaron mirando esas mismas imágenes, así que no están validados con fotos nuevas.
UMBRAL_CONFIANZA = 0.90
UMBRAL_NO_LEER = 0.80
# Brillo promedio mínimo (0 a 255) y nitidez mínima para considerar buena la foto.
BRILLO_MINIMO = 90
NITIDEZ_MINIMA = 150
# Proporción mínima de palabras (de 3 letras o más) que deben existir en el diccionario. El OCR puede
# estar "seguro" de una lectura con errores; en cambio, cuando salen muchas palabras que no existen,
# casi siempre es porque leyó mal. Valor fijado mirando las imágenes de prueba (ver informe).
UMBRAL_PALABRAS_CONOCIDAS = 0.90

# Si el detector encuentra al menos esta cantidad de líneas más de las que leyó el OCR, se avisa que
# puede faltar texto (en una página curva, PaddleOCR a veces se salta una línea sin dar señal).
LINEAS_FALTANTES_AVISO = 2

_ocr = None        # PaddleOCR: lee el texto (se carga una sola vez)
_detector = None   # detector de EasyOCR: cuenta las líneas (se carga una sola vez)
_voz = None        # la voz (se carga una sola vez)


# ---------------------------------------------------------------------------
# 1. Imagen
# ---------------------------------------------------------------------------
LADO_MAXIMO = 2000   # las fotos de celular son enormes (4000 px o más); se reducen para que el OCR no tarde


def decodificar_imagen(datos):
    """
    Convierte los bytes de una imagen (JPG, PNG...) en un arreglo BGR.
    Si es muy grande (foto de celular) se reduce a un lado máximo de 2000 px.
    OpenCV ya corrige la orientación guardada por el celular (EXIF).
    """
    import cv2
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError("No se pudo leer la imagen. Si es una foto .heic, expórtela como .jpg.")
    lado = max(imagen.shape[:2])
    if lado > LADO_MAXIMO:
        factor = LADO_MAXIMO / lado
        imagen = cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return imagen


def cargar_imagen(ruta):
    """Lee una imagen del disco (funciona con tildes en la ruta)."""
    try:
        return decodificar_imagen(np.fromfile(ruta, dtype=np.uint8))
    except ValueError as error:
        raise ValueError(f"{error} ({ruta})")


def medir_nitidez(imagen):
    """Varianza del Laplaciano: mientras más alta, más nítida es la imagen."""
    import cv2
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def poner_texto_oscuro(imagen):
    """
    El OCR fue entrenado con texto oscuro sobre fondo claro. Si la imagen es al revés (por ejemplo,
    una foto de una pantalla en modo oscuro), se invierten los colores.

    Cómo se decide: se compara cada punto con el promedio de sus vecinos. En texto oscuro sobre papel
    claro, lo más extremo son puntos mucho MÁS OSCUROS que su entorno (las letras); en texto claro
    sobre fondo oscuro, son puntos mucho MÁS CLAROS. Se mira cuál de los dos extremos es mayor, así
    que una sombra o una mesa oscura alrededor del papel no confunden la decisión.
    """
    import cv2
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    diferencia = gris - cv2.blur(gris, (31, 31))
    mas_oscuro = -np.percentile(diferencia, 1)
    mas_claro = np.percentile(diferencia, 99)
    if mas_claro > 0 and mas_oscuro / mas_claro < 0.8:
        return 255 - imagen
    return imagen


def mejorar_si_hay_poca_luz(imagen):
    """
    En fotos oscuras (brillo promedio menor a BRILLO_MINIMO) el texto queda con poco contraste.
    Se aplica CLAHE, que realza el contraste por zonas pequeñas en vez de en toda la imagen.
    En 16 imágenes oscuras (de prueba y una real) bajó el error de palabras de 4,4 % a 1,2 %;
    en fotos con buena luz no se usa, porque ahí no aporta.
    """
    import cv2
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if gris.mean() >= BRILLO_MINIMO:
        return imagen
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)


def medir_calidad(imagen):
    """Devuelve el brillo promedio y la nitidez de la imagen."""
    import cv2
    gris = cv2.cvtColor(poner_texto_oscuro(imagen), cv2.COLOR_BGR2GRAY)
    return {"brillo": float(gris.mean()), "nitidez": medir_nitidez(imagen)}


def capturar_desde_camara(indice=0, cuadros=8):
    """
    Toma varios cuadros de la cámara y se queda con el más nítido.
    Los primeros cuadros se descartan porque la cámara aún está ajustando la exposición.
    La imagen vive solo en memoria: NUNCA se guarda en disco. [ÉTICA: privacidad]
    """
    import cv2
    camara = cv2.VideoCapture(indice)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")
    mejor, mejor_nitidez = None, -1.0
    try:
        for i in range(cuadros):
            ok, cuadro = camara.read()
            if not ok or i < 3:
                continue
            nitidez = medir_nitidez(cuadro)
            if nitidez > mejor_nitidez:
                mejor, mejor_nitidez = cuadro, nitidez
    finally:
        camara.release()
    if mejor is None:
        raise RuntimeError("La cámara no entregó ninguna imagen.")
    return mejor


# ---------------------------------------------------------------------------
# 2. OCR (reconocimiento de texto)
# ---------------------------------------------------------------------------
def cargar_ocr():
    """
    PaddleOCR con tres modelos: UVDoc "aplana" la foto (corrige la curva de una página de libro),
    PP-OCRv6 detecta las líneas de texto y PP-OCRv6 reconoce cada línea.
    Se usa PaddlePaddle 3.2.0: en la 3.3 la aceleración para procesador (oneDNN) falla.
    """
    global _ocr
    if _ocr is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")   # no revisar internet al arrancar
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(lang="es", use_doc_unwarping=True,
                         use_doc_orientation_classify=False, use_textline_orientation=False)
    return _ocr


def cargar_detector():
    """Solo el detector de EasyOCR (sin reconocedor): encuentra las zonas con texto."""
    global _detector
    if _detector is None:
        import easyocr
        # gpu=True usa la tarjeta gráfica si hay una compatible (NVIDIA con CUDA); si no, el procesador.
        _detector = easyocr.Reader(["es"], gpu=True, recognizer=False, verbose=False)
    return _detector


def contar_lineas_detectadas(imagen):
    """
    Cuántas líneas de texto hay en la imagen según el detector de EasyOCR. Este detector marca palabras
    o grupos cortos, así que no junta dos líneas en una; con ordenar_en_lineas se agrupan en líneas.
    """
    horizontales, libres = cargar_detector().detect(imagen)
    cajas = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, x1, y0, y1 in horizontales[0]]
    cajas += [[list(p) for p in caja] for caja in libres[0]]
    return len(ordenar_en_lineas([(caja, "", 1.0) for caja in cajas]))


def ordenar_en_lineas(resultados):
    """
    El OCR devuelve fragmentos de texto sin un orden de lectura garantizado.
    Aquí se agrupan por línea y se leen de arriba abajo y de izquierda a derecha.

    Si el documento está torcido, las líneas quedan inclinadas y mezclarían las
    palabras. Por eso primero se estima la inclinación (con el borde superior de
    cada caja) y se "endereza" la posición de los fragmentos antes de ordenarlos.
    Si además la hoja está curva, la inclinación cambia a lo largo de la línea; eso
    se resuelve armando cada línea fragmento por fragmento (paso 3).
    """
    if not resultados:
        return []

    # 1. Inclinación del texto: mediana del ángulo del borde superior de cada caja.
    #    Solo cuentan las cajas inclinadas (dy distinto de 0): las cajas rectas no
    #    informan nada sobre la inclinación y falsearían la mediana.
    angulos = []
    for caja, _texto, _conf in resultados:
        dx, dy = caja[1][0] - caja[0][0], caja[1][1] - caja[0][1]
        if dx > 0 and abs(dy) >= 1:
            angulos.append(math.atan2(dy, dx))
    #    Si solo hay unas pocas cajas inclinadas, no es evidencia suficiente de que el
    #    documento esté torcido (podría ser una sola caja rara), y se asume 0 grados.
    hay_evidencia = len(angulos) >= max(5, 0.4 * len(resultados))
    angulo = float(np.median(angulos)) if hay_evidencia else 0.0
    cos_a, sen_a = math.cos(angulo), math.sin(angulo)

    def enderezar(punto):
        return punto[0] * cos_a + punto[1] * sen_a, -punto[0] * sen_a + punto[1] * cos_a

    # 2. Bordes de cada fragmento con el documento "enderezado".
    fragmentos = []
    for caja, texto, _conf in resultados:
        sup_izq, sup_der, inf_der, inf_izq = (enderezar(p) for p in caja)
        fragmentos.append({
            "izq": min(sup_izq[0], inf_izq[0]), "der": max(sup_der[0], inf_der[0]),
            "izq_arriba": sup_izq[1], "izq_abajo": inf_izq[1],    # alto del fragmento en su borde izquierdo
            "der_arriba": sup_der[1], "der_abajo": inf_der[1],    # y en su borde derecho
            "texto": texto,
        })
    alto_linea = float(np.median([f["izq_abajo"] - f["izq_arriba"] for f in fragmentos]))

    # 3. Se arman las líneas de izquierda a derecha: cada fragmento se pega a la línea cuyo último
    #    fragmento termina a su misma altura. Así se siguen las líneas aunque la hoja esté curva
    #    (una página de libro), porque solo se compara cada fragmento con su vecino inmediato.
    #    El OCR suele dar cajas rectas aunque el texto esté inclinado; en una caja larga, el inicio del
    #    texto queda arriba o abajo de la caja. Por eso también se compara una franja del alto de
    #    una línea anclada arriba o abajo, y se toma la coincidencia mayor.
    def coincidencia(a_arriba, a_abajo, b_arriba, b_abajo):
        return (min(a_abajo, b_abajo) - max(a_arriba, b_arriba)) / max(min(a_abajo - a_arriba, b_abajo - b_arriba), 1)

    fragmentos.sort(key=lambda f: f["izq"])
    lineas = []
    for f in fragmentos:
        alto_f = max(f["izq_abajo"] - f["izq_arriba"], 1)
        mejor, mejor_coincidencia = None, 0.4
        for linea in lineas:
            ultimo = linea[-1]
            if f["izq"] < ultimo["der"] - 0.5 * alto_f:   # se monta encima del último: no lo continúa
                continue
            c = max(coincidencia(ultimo["der_arriba"], ultimo["der_abajo"], f["izq_arriba"], f["izq_abajo"]),
                    # texto que baja hacia la derecha
                    coincidencia(ultimo["der_abajo"] - alto_linea, ultimo["der_abajo"], f["izq_arriba"], f["izq_arriba"] + alto_linea),
                    # texto que sube hacia la derecha
                    coincidencia(ultimo["der_arriba"], ultimo["der_arriba"] + alto_linea, f["izq_abajo"] - alto_linea, f["izq_abajo"]))
            if c > mejor_coincidencia:
                mejor, mejor_coincidencia = linea, c
        if mejor is None:
            lineas.append([f])
        else:
            mejor.append(f)

    # 4. Las líneas se leen de arriba abajo, según la altura de su primer fragmento.
    lineas.sort(key=lambda l: (l[0]["izq_arriba"] + l[0]["izq_abajo"]) / 2)
    return [" ".join(f["texto"] for f in l) for l in lineas]


DICCIONARIO = os.path.join(CARPETA, "diccionario_es.txt")
_palabras = None   # {palabra: frecuencia}
_por_base = None   # {palabra sin tildes: la forma más frecuente, con tilde}

# Palabras de 2 letras que se aceptan al separar (nunca al final de la palabra pegada).
PARTICULAS = {"de", "la", "el", "en", "un", "es", "se", "lo", "su", "me", "mi", "al", "ya", "no", "si", "le", "te"}


def sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def cargar_diccionario():
    """
    Lista de unas 79 000 palabras del español con su frecuencia de uso (escala "zipf": 7 = muy común,
    3 = poco común). Se creó una sola vez con la librería wordfreq. Si el archivo falta, el corrector
    simplemente no actúa y el sistema funciona igual.
    """
    global _palabras, _por_base
    if _palabras is None:
        _palabras, _por_base = {}, {}
        if os.path.exists(DICCIONARIO):
            with open(DICCIONARIO, encoding="utf-8") as archivo:
                for linea in archivo:
                    palabra, frecuencia = linea.rstrip("\n").split("\t")
                    _palabras[palabra] = float(frecuencia)
            for palabra, frecuencia in _palabras.items():
                base = sin_tildes(palabra)
                if base not in _por_base or frecuencia > _palabras[_por_base[base]]:
                    _por_base[base] = palabra
    return _palabras, _por_base


def igualar_mayusculas(original, nueva):
    if original.isupper() and len(original) > 1:
        return nueva.upper()
    if original[0].isupper():
        return nueva[0].upper() + nueva[1:]
    return nueva


def separar_palabras_pegadas(token):
    """
    Si el OCR juntó varias palabras ("confotosreales"), busca la forma de partirlas en palabras
    reales del diccionario (la de menor "costo": palabras comunes y pocas piezas).
    Reglas conservadoras, para no dañar palabras correctas que no estén en el diccionario
    (por ejemplo "almacene"): todas las partes deben ser palabras comunes, las de 3 letras
    deben ser muy comunes, y no se aceptan muchas piezas cortas (en promedio, de al menos 2,8 letras).
    Si no hay una separación clara, devuelve None y la palabra queda como está.
    """
    palabras, por_base = cargar_diccionario()
    base = sin_tildes(token.lower())
    n = len(base)

    def parte_valida(trozo, es_primera, es_ultima):
        if trozo in ("y", "o", "a"):   # una letra suelta solo vale en medio ("verdad y no"), nunca al principio ni al final
            return trozo if not (es_primera or es_ultima) else None
        palabra = por_base.get(trozo)
        if palabra is None:
            return None
        if len(trozo) == 2:
            # Las partículas ("de", "la"...) no valen al final ("consúmase" no es "consuma se"),
            # salvo en palabras largas pegadas ("unacámarade").
            return palabra if (trozo in PARTICULAS and (not es_ultima or n >= 10)) else None
        if len(trozo) < 3:
            return None
        minimo = 5.0 if len(trozo) == 3 else 3.5
        return palabra if palabras[palabra] >= minimo else None

    SIN_CAMINO = 1e9
    costo = [SIN_CAMINO] * (n + 1)
    costo[0] = 0.0
    anterior = [None] * (n + 1)
    piezas = [0] * (n + 1)
    for fin in range(1, n + 1):
        for inicio in range(max(0, fin - 20), fin):
            if costo[inicio] >= SIN_CAMINO or piezas[inicio] + 1 > 12:
                continue
            palabra = parte_valida(base[inicio:fin], inicio == 0, fin == n)
            if palabra is None:
                continue
            nuevo = costo[inicio] + (8.0 - palabras.get(palabra, 5.0)) + 2.5   # 2.5 = castigo por cada pieza
            if nuevo < costo[fin]:
                costo[fin], anterior[fin], piezas[fin] = nuevo, (inicio, palabra), piezas[inicio] + 1
    if costo[n] >= SIN_CAMINO or piezas[n] < 2 or n / piezas[n] < 2.8:
        return None
    partes, pos = [], n
    while pos > 0:
        pos, palabra = anterior[pos]
        partes.append(palabra)
    return partes[::-1]


# Letras que el OCR confunde por su forma parecida (vistas en las fotos de libros): "quc" por "que",
# "pocsía" por "poesía", "Ilorar" por "llorar", "Jíderes" por "líderes", "nornal" por "normal".
CONFUSIONES = [("c", "e"), ("e", "c"), ("I", "l"), ("J", "l"), ("rn", "m")]


def corregir_letras_confundidas(token):
    """
    Prueba cambiar una sola letra confundida. Solo se acepta si el resultado es una palabra común
    (frecuencia 3 o más); si no hay ninguna, devuelve None y la palabra queda como estaba.
    """
    palabras, por_base = cargar_diccionario()
    mejor, mejor_frecuencia = None, 3.0
    for mal, bien in CONFUSIONES:
        inicio = token.find(mal)
        while inicio != -1:
            nueva = token[:inicio] + bien + token[inicio + len(mal):]
            forma = por_base.get(sin_tildes(nueva.lower()))
            if forma and palabras[forma] >= mejor_frecuencia:
                mejor, mejor_frecuencia = igualar_mayusculas(nueva, forma), palabras[forma]
            inicio = token.find(mal, inicio + 1)
    return mejor


def corregir_palabra(token):
    palabras, por_base = cargar_diccionario()
    if len(token) < 3 or not palabras:
        return token
    bajo = token.lower()
    conocida = bajo in palabras
    mejor_forma = por_base.get(sin_tildes(bajo))   # la misma palabra, con la tilde que le corresponde

    if conocida:
        # Ya es una palabra: solo se le pone la tilde si la forma con tilde es mucho más usada
        # ("camara" y "cámara" son ambas válidas, pero "cámara" es 10 veces más común).
        if mejor_forma and mejor_forma != bajo and palabras[mejor_forma] - palabras[bajo] >= 1.0:
            return igualar_mayusculas(token, mejor_forma)
        return token
    if mejor_forma:   # no existe tal cual, pero sí con tilde ("Medellin" -> "Medellín")
        return igualar_mayusculas(token, mejor_forma)
    corregida = corregir_letras_confundidas(token)
    if corregida:
        return corregida
    # El OCR a veces lee la barra "/" como una "l" minúscula ("técnicalherramienta" por "técnica/herramienta").
    for i in range(4, len(bajo) - 4):
        if bajo[i] == "l":
            izquierda, derecha = por_base.get(sin_tildes(bajo[:i])), por_base.get(sin_tildes(bajo[i + 1:]))
            if izquierda and derecha and palabras[izquierda] >= 3.5 and palabras[derecha] >= 3.5:
                return igualar_mayusculas(token, izquierda) + "/" + derecha
    if len(token) >= 8:
        partes = separar_palabras_pegadas(token)
        if partes:
            partes[0] = igualar_mayusculas(token, partes[0])
            return " ".join(partes)
    return token


def cambiar_digitos(token):
    palabras, por_base = cargar_diccionario()
    nueva = token.replace("0", "o").replace("1", "l")
    return nueva if por_base.get(sin_tildes(nueva.lower())) else token


def corregir_texto(texto):
    """
    Arregla fallos típicos del OCR que afectan la voz: palabras pegadas, tildes perdidas y símbolos
    confundidos. No toca palabras que no están en el diccionario (nombres, medicinas).
    """
    # Palabra partida con guion al final de la línea: "mise-" + "rables" -> "miserables"
    texto = re.sub(r"([A-Za-zÁÉÍÓÚÑáéíóúñ]) ?[-–_=] *\n *(?=[a-záéíóúñ])", r"\1", texto)
    texto = re.sub(r"[ \t]+([:;,.!?)])", r"\1", texto)        # "médico ." -> "médico."
    texto = re.sub(r"\(\s+", "(", texto)
    texto = re.sub(r"(?<=\d)y(?=\s)", " y", texto)             # "1y" -> "1 y"
    texto = re.sub(r"(?<=\d)O", "0", texto)                      # "2O" -> "20" (solo pegada a un número)
    texto = re.sub(r"(?<=\d)S\b", "5", texto)                  # "1S" -> "15"
    # Cero o uno dentro de una palabra: "l0" -> "lo", "1o" -> "lo" (solo si así queda una palabra conocida)
    texto = re.sub(r"\b(?=[A-Za-záéíóúñ]*[01])(?=[01]*[A-Za-záéíóúñ])[A-Za-záéíóúñ01]{2,}\b",
                   lambda m: cambiar_digitos(m.group(0)), texto)
    texto = re.sub(r"(?<=[a-záéíóúñ]) 0 (?=[a-záéíóúñ])", " o ", texto)   # "dos 0 tres" -> "dos o tres"
    return re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", lambda m: corregir_palabra(m.group(0)), texto)


def descartar_dudosos(resultados):
    """
    Quita los caracteres sueltos con confianza muy baja (menos de 0,3). En las fotos de libros eran letras
    o signos mal leídos (la palabra "a" y el signo "¿" salieron como "2" y "3"): es mejor no leerlos
    que decir en voz alta un número que no está en el texto.
    """
    return [r for r in resultados if not (len(r[1].strip()) <= 1 and r[2] < 0.3)]


def reconocer_texto(imagen):
    """
    Devuelve (texto, confianza, lineas):
        confianza -> promedio (de 0 a 1) de lo seguro que está el modelo de cada línea, ponderado por sus letras
        lineas    -> {"leidas": líneas que leyó el OCR, "detectadas": líneas que encontró el detector}
    """
    import cv2
    preparada = mejorar_si_hay_poca_luz(poner_texto_oscuro(imagen))
    if preparada.ndim == 2:   # el realce de contraste devuelve la imagen en gris; PaddleOCR espera color
        preparada = cv2.cvtColor(preparada, cv2.COLOR_GRAY2BGR)

    r = cargar_ocr().predict(preparada)[0]
    resultados = [(caja.tolist(), texto, float(conf))
                  for caja, texto, conf in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]) if texto.strip()]
    resultados = descartar_dudosos(resultados)
    lineas_leidas = ordenar_en_lineas(resultados)
    # [ÉTICA: transparencia] Verificación cruzada con otro modelo: si el detector ve más líneas de las que
    # se leyeron, puede faltar texto, y una persona que no ve la hoja no tendría cómo darse cuenta.
    lineas = {"leidas": len(lineas_leidas), "detectadas": contar_lineas_detectadas(preparada)}
    if not resultados:
        return "", 0.0, lineas
    texto = corregir_texto("\n".join(lineas_leidas))
    total_letras = sum(len(r[1]) for r in resultados)
    confianza = sum(r[2] * len(r[1]) for r in resultados) / max(total_letras, 1)
    return texto, float(confianza), lineas


# ---------------------------------------------------------------------------
# 3. Aviso de confianza  [ÉTICA: transparencia sobre las limitaciones]
# ---------------------------------------------------------------------------
def proporcion_palabras_conocidas(texto):
    """Fracción de las palabras de 3 letras o más que existen en el diccionario (1.0 si no se puede calcular)."""
    palabras, por_base = cargar_diccionario()
    tokens = [t for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", texto) if len(t) >= 3]
    if not palabras or len(tokens) < 5:   # con tan pocas palabras la proporción no es confiable
        return 1.0
    conocidas = sum(1 for t in tokens if t.lower() in palabras or sin_tildes(t.lower()) in por_base)
    return conocidas / len(tokens)


def decidir_lectura(texto, confianza, calidad, lineas=None):
    """
    Decide qué hacer con lo que el OCR reconoció. Devuelve (aviso, leer):
        aviso -> mensaje que se dice ANTES de leer (o None si la lectura parece confiable)
        leer  -> si se debe leer el texto en voz alta

    Se usan tres señales: la confianza del OCR, la proporción de palabras que existen en el diccionario
    y si el detector encontró más líneas de las que se leyeron.
        todo bien             -> se lee sin aviso
        faltan líneas         -> se avisa que puede faltar texto y se lee lo que hay
        alguna dudosa         -> se avisa que no está seguro y se lee (la persona decide si confía)
        confianza muy baja    -> NO se lee, porque leer en voz alta datos casi seguramente errados
                                 (cifras, dosis) sería peor que no leer nada
    Así la persona no asume que el resultado es igual de bueno en todas las condiciones.
    """
    if not texto.strip():
        return ("No encontré texto en la imagen. Acerca el documento a la cámara "
                "y verifica que haya buena luz.", False)
    faltan_lineas = bool(lineas) and lineas["detectadas"] - lineas["leidas"] >= LINEAS_FALTANTES_AVISO
    dudosa = confianza < UMBRAL_CONFIANZA or proporcion_palabras_conocidas(texto) < UMBRAL_PALABRAS_CONOCIDAS
    if not dudosa and not faltan_lineas:
        return None, True
    if not dudosa:
        return ("Atención: puede que me haya saltado parte del texto. Pon el documento más plano y que llene "
                "la foto, e intenta de nuevo. Esto es lo que logré leer."), True

    # Sugerencia concreta según la causa más probable.
    if calidad["brillo"] < BRILLO_MINIMO:
        pista = " Hay poca luz; acerca una lámpara e intenta de nuevo."
    elif calidad["nitidez"] < NITIDEZ_MINIMA:
        pista = " La imagen está borrosa; sostén firme el celular e intenta de nuevo."
    else:
        pista = " Intenta de nuevo con el documento más recto y más cerca."

    if confianza < UMBRAL_NO_LEER:
        return ("No logré leer este texto con claridad y prefiero no leerlo para no darte datos errados."
                + pista), False
    tambien = " También puede faltar parte del texto." if faltan_lineas else ""
    return ("Atención: no estoy seguro de haber leído bien este texto." + tambien + pista +
            " Si es un dato importante, como dinero o medicinas, pide a otra persona que lo confirme."), True


# ---------------------------------------------------------------------------
# 4. Texto a voz
# ---------------------------------------------------------------------------
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def normalizar_para_voz(texto):
    """
    El modelo de voz lee "$23.400" como "dólar veintitrés punto cuatrocientos" y
    "12/03/2026" como "doce barra cero tres...". Aquí se reescriben los formatos
    colombianos más comunes para que se lean como los diría una persona.
    """
    # Fechas dd/mm/aaaa -> "12 de marzo de 2026"
    def fecha(m):
        dia, mes, anio = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mes <= 12 and 1 <= dia <= 31:
            return f"{dia} de {MESES[mes - 1]} de {anio}"
        return m.group(0)
    texto = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", fecha, texto)

    # NIT 900.123.456-7 -> "NIT 900 123 456, dígito de verificación 7"
    texto = re.sub(r"NIT\s*(\d{1,3})\.(\d{3})\.(\d{3})-(\d)",
                   r"NIT \1 \2 \3, dígito de verificación \4", texto)

    # Dinero: $23.400 -> "23400 pesos"
    texto = re.sub(r"\$\s?(\d{1,3}(?:\.\d{3})*)(?:,\d+)?",
                   lambda m: m.group(1).replace(".", "") + " pesos", texto)

    # Horas: "8:00 a. m." -> "8 de la mañana"
    def hora(m):
        h, minutos, jornada = int(m.group(1)), m.group(2), m.group(3).lower().replace(" ", "")
        if jornada == "a.m.":
            momento = "de la mañana"
        else:
            momento = "de la noche" if 6 <= h < 12 else "de la tarde"
        partes = f"{h}" if minutos == "00" else f"{h} y {int(minutos)}"
        return f"{partes} {momento}"
    texto = re.sub(r"\b(\d{1,2}):(\d{2})\s*([ap]\.\s?m\.)", hora, texto, flags=re.IGNORECASE)

    # Miles con punto (1.250.000 -> 1250000), número y unidades comunes
    texto = re.sub(r"(?<![\d.])\d{1,3}(?:\.\d{3})+(?![\d.])", lambda m: m.group(0).replace(".", ""), texto)
    texto = re.sub(r"\bNo\.\s*(?=\d)", "número ", texto)
    texto = re.sub(r"(?<=\d)\s?mg\b", " miligramos", texto)
    texto = re.sub(r"(?<=\d)\s?ml\b", " mililitros", texto)
    texto = re.sub(r"\s+x\s+(?=\d)", " por ", texto)
    texto = re.sub(r"(?<=[a-záéíóúñ]{4})/(?=[a-záéíóúñ]{4})", " o ", texto)   # técnica/herramienta -> "técnica o herramienta"
    return texto


def cargar_voz(ruta=None):
    global _voz
    if _voz is None:
        ruta = ruta or os.environ.get("LECTOR_VOZ") or VOZ_POR_DEFECTO
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró la voz en: {ruta}\n"
                "Descárgala con:  python -m piper.download_voices es_MX-claude-high --download-dir voces")
        from piper import PiperVoice
        _voz = PiperVoice.load(ruta)
    return _voz


def sintetizar(texto, ruta_wav):
    """Convierte el texto en un archivo de audio WAV. Devuelve la duración en segundos."""
    voz = cargar_voz()
    with wave.open(ruta_wav, "wb") as archivo:
        voz.synthesize_wav(normalizar_para_voz(texto), archivo)
    with wave.open(ruta_wav, "rb") as archivo:
        return archivo.getnframes() / archivo.getframerate()


def reproducir(ruta_wav):
    """Reproduce un WAV con lo que tenga el sistema operativo. Devuelve True si pudo."""
    sistema = platform.system()
    try:
        if sistema == "Windows":
            import winsound
            winsound.PlaySound(ruta_wav, winsound.SND_FILENAME)
        elif sistema == "Darwin":
            subprocess.run(["afplay", ruta_wav], check=True)
        else:
            comandos = (["aplay", "-q"], ["paplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"])
            for comando in comandos:
                if shutil.which(comando[0]):
                    subprocess.run(comando + [ruta_wav], check=True)
                    break
            else:
                raise RuntimeError("no hay un reproductor de audio instalado")
        return True
    except Exception as error:
        print(f"[aviso] No se pudo reproducir el audio: {error}")
        return False


def decir(texto, reproducir_audio=True, ruta_guardar=None):
    """
    Dice un texto en voz alta. El audio se crea en una carpeta temporal que se
    borra al terminar, salvo que se pida guardarlo. [ÉTICA: privacidad]
    """
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "audio.wav")
        sintetizar(texto, ruta)
        if ruta_guardar:
            shutil.copyfile(ruta, ruta_guardar)
        if reproducir_audio:
            reproducir(ruta)


# ---------------------------------------------------------------------------
# 5. Flujo completo
# ---------------------------------------------------------------------------
def mensaje_hablado(texto, aviso, leer, avisos="breves"):
    """
    Arma lo que se dirá en voz alta. [ÉTICA: transparencia] Una persona que no ve la pantalla no puede
    enterarse del aviso si solo aparece escrito, por eso por defecto se dice una versión breve.

    avisos = "breves"   -> una frase corta antes del texto (valor por defecto)
             "completos" -> el aviso entero, con sus sugerencias
             "ninguno"   -> no se dice el aviso cuando igual se va a leer el texto (el aviso sigue en pantalla).
                            Si no se lee el texto, siempre se dice una frase breve: callar del todo
                            dejaría a la persona sin saber qué pasó.
    """
    if not aviso:
        return "El texto dice: " + texto.replace("\n", ". ") if texto.strip() else ""

    if leer:      # confianza media o posible texto faltante: se lee de todos modos
        breve = ("Atención: puede faltar parte del texto." if aviso.startswith("Atención: puede que me haya saltado")
                 else "Atención: lectura poco confiable.")
        previo = {"completos": aviso, "breves": breve, "ninguno": ""}[avisos]
        return (previo + " El texto dice: " + texto.replace("\n", ". ")).strip()

    # No se lee el texto: siempre se dice algo.
    if avisos == "completos":
        return aviso
    if not texto.strip():
        return "No encontré texto en la imagen."
    return "No logré leer este texto con claridad. Intenta de nuevo."


def analizar_imagen(imagen, avisos="breves"):
    """OCR + decisión de aviso para una imagen (arreglo BGR). Lo usan el modo consola y el servidor."""
    calidad = medir_calidad(imagen)
    texto, confianza, lineas = reconocer_texto(imagen)
    aviso, leer = decidir_lectura(texto, confianza, calidad, lineas)
    return {"texto": texto, "confianza": confianza, "aviso": aviso, "leer": leer, "calidad": calidad,
            "lineas": lineas, "mensaje": mensaje_hablado(texto, aviso, leer, avisos)}


def audio_en_bytes(texto):
    """Devuelve el audio (WAV) de un texto sin dejar archivos en disco. [ÉTICA: privacidad]"""
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "audio.wav")
        sintetizar(texto, ruta)
        with open(ruta, "rb") as archivo:
            return archivo.read()


def leer_imagen(imagen, reproducir_audio=True, ruta_guardar=None):
    """Hace todo el flujo con una imagen (arreglo BGR). Devuelve un resumen."""
    resultado = analizar_imagen(imagen)
    texto, confianza, aviso = resultado["texto"], resultado["confianza"], resultado["aviso"]
    leer, calidad = resultado["leer"], resultado["calidad"]

    print(f"\nConfianza: {confianza:.2f} | Brillo: {calidad['brillo']:.0f} | Nitidez: {calidad['nitidez']:.0f}")
    if aviso:
        print("AVISO:", aviso)
    print("TEXTO RECONOCIDO:\n" + (texto if texto else "(vacío)"))
    if texto and not leer:
        print("(El texto no se lee en voz alta por tener confianza muy baja.)")

    if resultado["mensaje"]:
        decir(resultado["mensaje"], reproducir_audio, ruta_guardar)
    return resultado


def modo_camara(args):
    decir("Lector listo. Presiona Enter para fotografiar el documento. "
          "Escribe r y Enter para repetir la lectura. Escribe q y Enter para salir.",
          not args.sin_audio)
    ultimo_texto = ""
    while True:
        tecla = input("Enter = fotografiar | r = repetir | q = salir: ").strip().lower()
        if tecla == "q":
            decir("Hasta luego.", not args.sin_audio)
            break
        if tecla == "r":
            if ultimo_texto:
                decir(ultimo_texto, not args.sin_audio)
            else:
                decir("Todavía no hay nada para repetir.", not args.sin_audio)
            continue
        try:
            decir("Fotografiando. Mantén el documento quieto.", not args.sin_audio)
            imagen = capturar_desde_camara(args.camara)
            resultado = leer_imagen(imagen, not args.sin_audio)
            ultimo_texto = resultado["texto"]
        except Exception as error:   # [ÉTICA: manejo de errores comprensible]
            print("Error:", error)
            decir("Ocurrió un problema con la cámara. Revisa que esté conectada e intenta de nuevo.",
                  not args.sin_audio)


def main():
    parser = argparse.ArgumentParser(description="Lector de documentos con voz para personas con discapacidad visual.")
    parser.add_argument("--imagen", help="ruta de una imagen para leer (si no se da, se usa la cámara)")
    parser.add_argument("--camara", type=int, default=0, help="número de la cámara (por defecto 0)")
    parser.add_argument("--sin-audio", action="store_true", help="no reproducir audio, solo mostrar el texto")
    parser.add_argument("--guardar-wav", help="guardar el audio de la lectura en este archivo (opcional)")
    args = parser.parse_args()

    if args.imagen:
        leer_imagen(cargar_imagen(args.imagen), not args.sin_audio, args.guardar_wav)
    else:
        modo_camara(args)


if __name__ == "__main__":
    sys.exit(main())
