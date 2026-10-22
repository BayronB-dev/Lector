# Lector de documentos con voz

Aplicación web que ayuda a una persona con discapacidad visual a leer textos impresos (facturas, recetas, etiquetas,
páginas de libros): se toma una foto con el celular y la aplicación lee el texto en voz alta.

**Para usarla: https://lector.bami.lol** (desde el navegador del celular, sin instalar nada).

Proyecto académico del curso de Inteligencia Artificial (Universidad Simón Bolívar), por Bayron Barrios y Samuel Hernández.

## Cómo se usa

1. Abrir el enlace en el celular y tocar el botón grande **"Fotografiar documento"**.
2. Tomar la foto con el documento recto, completo y con buena luz.
3. Escuchar la lectura. Se puede repetir, pausar, cambiar la velocidad y agrandar la letra.

Si la foto no se pudo leer bien, la aplicación lo **dice en voz alta antes de leer** (por ejemplo, "puede faltar parte del texto")
y sugiere qué hacer. Si casi no se entiende, prefiere no leer el texto para no dar datos errados.

La foto viaja cifrada, se procesa y se descarta: no se guarda ninguna foto ni se registra quién usa la aplicación.

## Qué hay en este repositorio

| Carpeta | Contenido |
|---|---|
| `aplicacion/` | La aplicación: el programa que lee la foto y genera la voz, la página web y la configuración del servidor |
| `evaluacion/` | Cómo se midió que funciona: 72 imágenes de prueba con documentos inventados, 3 fotos reales de libros y los resultados |

## Resultados

Se midió el porcentaje de palabras mal leídas (sin contar la puntuación, que la voz no pronuncia):

| Prueba | Palabras mal leídas |
|---|---|
| 60 imágenes de prueba en 10 condiciones (poca luz, foto borrosa, girada, letra pequeña...) | 0,6 % |
| 3 fotos reales de páginas de libro | 3,9 % |
| Foto extremadamente borrosa (6 imágenes) | 73 %: la aplicación avisa y no lee el texto |

Antes se usaba otro reconocedor de texto (EasyOCR), con el que las mismas 60 imágenes daban 10,5 % y las fotos de libro 8,4 %.

## Límites

- Se probó con pocas fotos reales; los ajustes se hicieron con las mismas imágenes con que se mide.
- Lee texto impreso en español, en una columna. No lee escritura a mano ni tablas complejas.
- Necesita internet; la foto se procesa en un servidor del equipo.
- No se ha probado con personas con discapacidad visual.

## Para instalarlo en un computador propio

Requiere Python 3.11 o 3.12. Desde la carpeta `aplicacion`:

```
pip install -r requirements.txt
python -m piper.download_voices es_MX-claude-high --download-dir voces
python servidor.py
```

Luego se abre en el celular la dirección que muestra el programa (celular y computador en la misma red Wi-Fi).
Para montarlo en un servidor con Docker: `docker compose up -d --build` desde la misma carpeta.

Para repetir la medición, desde la carpeta `evaluacion`: `python evaluar.py` (imágenes de prueba) y
`python evaluar.py --carpeta fotos_reales --salida resultados/fotos_reales` (fotos reales).

Modelos usados, todos ya entrenados y de licencia abierta: PaddleOCR (reconocer el texto), detector de EasyOCR (comprobar que
no falten líneas) y Piper con la voz es_MX-claude-high (voz). El nombre "claude" es el de esa voz, creada por terceros.
