# Imagen del lector para el servidor (VPS). El servidor no tiene tarjeta gráfica, así que se instala
# PyTorch solo para CPU (lo usa el detector de EasyOCR; es mucho más liviano que la versión con CUDA).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True MPLCONFIGDIR=/tmp

# Librerías del sistema que necesita PaddleOCR (OpenCV completo y OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios: si alguien encontrara una falla en el servidor, no tendría acceso de administrador.
RUN useradd --create-home lector
WORKDIR /app
RUN chown lector /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

USER lector

# Los modelos se descargan al construir la imagen, no cuando llega el primer usuario.
# PaddleOCR los guarda en /home/lector/.paddlex, EasyOCR en /home/lector/.EasyOCR y la voz queda en /app/voces.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='es', use_doc_unwarping=True, use_doc_orientation_classify=False, use_textline_orientation=False)"
RUN python -c "import easyocr; easyocr.Reader(['es'], gpu=False, recognizer=False, verbose=False)"
RUN mkdir voces && python -m piper.download_voices es_MX-claude-high --download-dir voces

COPY --chown=lector lector.py servidor.py diccionario_es.txt ./
COPY --chown=lector web ./web

EXPOSE 8000
CMD ["python", "servidor.py", "--puerto", "8000"]
