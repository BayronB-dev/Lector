"""
Servidor web del lector. El servidor hace el trabajo pesado (OCR y voz) y el celular solo aporta
la cámara y el altavoz:

    celular (navegador)  --foto-->  servidor (este programa)  --texto y audio-->  celular

Se puede usar de dos formas:
    - En internet (versión principal): corre en un servidor (VPS) detrás de Caddy, que pone el HTTPS.
      Cualquiera abre el enlace desde su navegador, sin instalar nada. Ver Dockerfile y docker-compose.yml.
    - En un computador propio, sin internet: el celular y el computador en la misma red Wi-Fi.

Uso local:
    python servidor.py
    (luego abrir en el celular la dirección que aparece en pantalla)

[ÉTICA: privacidad] La foto se procesa en memoria y nunca se guarda en disco; el audio tampoco.
No se registra quién usa el servicio ni qué fotos envía.
"""

import argparse
import base64
import os
import socket
import threading
import time
from collections import defaultdict, deque

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

import lector

app = Flask(__name__, static_folder="web", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # fotos de hasta 20 MB
candado = threading.Lock()   # el OCR usa mucho procesador: se atiende una foto a la vez

# Detrás de Caddy, la dirección real del usuario llega en la cabecera X-Forwarded-For.
# Solo se confía en ella si se pide (LECTOR_DETRAS_DE_PROXY=1); si no, cualquiera podría falsificarla.
if os.environ.get("LECTOR_DETRAS_DE_PROXY") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

# Protección contra abuso: el servidor está abierto en internet y cada foto ocupa el procesador
# varios segundos. Sin límites, unas pocas personas (o un programa) podrían dejarlo sin servicio
# para todos los demás.
FOTOS_POR_MINUTO = 8   # por dirección IP; una persona leyendo documentos no llega a tanto
EN_ESPERA_MAXIMO = 4   # fotos que pueden esperar turno; si hay más, se pide intentar luego
ESPERA_MAXIMA = 60     # segundos que una foto puede esperar su turno
_envios = defaultdict(deque)   # IP -> momentos de sus últimos envíos (solo en memoria; se olvidan al minuto)
_contador = threading.Lock()   # protege _envios y _en_espera, que comparten varios hilos
_en_espera = 0


def demasiados_envios(ip):
    """True si esta IP ya envió FOTOS_POR_MINUTO fotos en el último minuto."""
    ahora = time.monotonic()
    with _contador:
        for clave in list(_envios):   # se olvidan las IP sin actividad reciente
            while _envios[clave] and ahora - _envios[clave][0] > 60:
                _envios[clave].popleft()
            if not _envios[clave]:
                del _envios[clave]
        if len(_envios[ip]) >= FOTOS_POR_MINUTO:
            return True
        _envios[ip].append(ahora)
        return False


def esperar_turno():
    """Espera a que el OCR quede libre. Devuelve False si hay demasiada fila o si la espera es muy larga."""
    global _en_espera
    with _contador:
        if _en_espera >= EN_ESPERA_MAXIMO:
            return False
        _en_espera += 1
    try:
        return candado.acquire(timeout=ESPERA_MAXIMA)
    finally:
        with _contador:
            _en_espera -= 1


@app.errorhandler(413)
def foto_muy_grande(_error):
    return jsonify(error="La foto es demasiado grande (máximo 20 MB)."), 413


@app.after_request
def sin_cache(respuesta):
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


@app.get("/")
def inicio():
    return send_from_directory("web", "index.html")


@app.post("/leer")
def leer():
    archivo = request.files.get("foto")
    if archivo is None:
        return jsonify(error="No llegó ninguna foto. Intenta de nuevo."), 400
    try:
        imagen = lector.decodificar_imagen(np.frombuffer(archivo.read(), dtype=np.uint8))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    # Qué dice la voz cuando la lectura es dudosa: "breves" (por defecto), "completos" o "ninguno".
    avisos = request.form.get("avisos", "breves")
    if avisos not in ("breves", "completos", "ninguno"):
        avisos = "breves"

    if demasiados_envios(request.remote_addr):
        return jsonify(error="Enviaste muchas fotos seguidas. Espera un minuto."), 429
    if not esperar_turno():
        return jsonify(error="El servidor está ocupado. Espera unos segundos."), 503
    try:
        resultado = lector.analizar_imagen(imagen, avisos)
        audio = b""
        if resultado["mensaje"].strip():
            try:
                audio = lector.audio_en_bytes(resultado["mensaje"])
            except Exception as error:   # si falla la voz, igual se muestra el texto
                print("[aviso] No se pudo generar el audio:", error)
    finally:
        candado.release()

    # El nivel resume la decisión del sistema (que usa la confianza y también las palabras conocidas).
    if not resultado["leer"]:
        nivel = "baja"
    elif resultado["aviso"]:
        nivel = "media"
    else:
        nivel = "alta"

    return jsonify(
        texto=resultado["texto"],
        aviso=resultado["aviso"],
        leido=resultado["leer"],
        confianza=round(resultado["confianza"], 2),
        nivel=nivel,
        audio=base64.b64encode(audio).decode("ascii"),
    )


def ip_local():
    """Dirección del computador dentro de la red Wi-Fi."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no envía nada; solo averigua qué red se usaría
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(description="Servidor web del lector de documentos.")
    parser.add_argument("--puerto", type=int, default=8000)
    args = parser.parse_args()

    print("Cargando los modelos (la primera vez tarda un poco)...", flush=True)
    lector.cargar_ocr()
    lector.cargar_detector()
    lector.cargar_voz()

    if os.environ.get("LECTOR_DETRAS_DE_PROXY") == "1":
        print(f"Lector listo en el puerto {args.puerto} (detrás del proxy HTTPS).", flush=True)
    else:
        print("\n" + "=" * 60)
        print("Lector listo. En el celular (misma red Wi-Fi) abre:")
        print(f"    http://{ip_local()}:{args.puerto}")
        print("En este computador también sirve:  http://localhost:%d" % args.puerto)
        print("Aviso: cualquier dispositivo de esa red podría abrir la página. Úsalo en una red de confianza.")
        print("=" * 60 + "\n")

    # waitress es un servidor web pensado para uso real (el de Flask es solo para desarrollo).
    # Funciona igual en Windows y en Linux.
    from waitress import serve
    serve(app, host="0.0.0.0", port=args.puerto, threads=8)


if __name__ == "__main__":
    main()
