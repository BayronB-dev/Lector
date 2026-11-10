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
| `evaluacion/` | Cómo se midió que funciona: 72 imágenes de prueba con documentos inventados, 3 fotos reales de libros, 5 fotos reales de etiquetas y los resultados |

## Resultados

Se midió el porcentaje de palabras mal leídas (sin contar la puntuación, que la voz no pronuncia):

| Prueba | Palabras mal leídas |
|---|---|
| 60 imágenes de prueba en 10 condiciones (poca luz, foto borrosa, girada, letra pequeña...) | 0,6 % |
| 3 fotos reales de páginas de libro | 3,9 % |
| Foto extremadamente borrosa (6 imágenes) | 73 %: la aplicación avisa y no lee el texto |
| 5 fotos reales de etiquetas, tomadas después de fijar los ajustes | 16,1 % en la primera medición (las 2 lecturas malas se avisaron); 4,1 % tras corregir las fallas que mostraron (lectura a dos columnas, ceros leídos como "o" y unidades) |

Antes se usaba otro reconocedor de texto (EasyOCR), con el que las mismas 60 imágenes daban 10,5 % y las fotos de libro 8,4 %.

## Dónde se aplica el principio ético

El proyecto prioriza el principio de **equidad y no discriminación** (Marco Ético para la IA en Colombia). En el código, cada punto
donde se aplica está marcado con un comentario `[ÉTICA: aspecto]`:

| Aspecto | Dónde | Qué hace |
|---|---|---|
| Equidad | [lector.py:109](aplicacion/lector.py#L109) | Invierte los colores si el texto es claro sobre fondo oscuro |
| Equidad | [lector.py:131](aplicacion/lector.py#L131) | Realza el contraste en fotos con poca luz |
| Equidad | [lector.py:218](aplicacion/lector.py#L218) | Ordena las líneas aunque la hoja esté torcida o curva |
| Equidad | [lector.py:312](aplicacion/lector.py#L312) | Lee los textos a dos columnas columna por columna |
| Equidad | [index.html:240](aplicacion/web/index.html#L240) | Reduce la foto en el celular para que funcione con datos móviles lentos |
| Equidad | [servidor.py:43](aplicacion/servidor.py#L43) | Limita las fotos por minuto para que nadie deje sin servicio a los demás |
| Equidad | [evaluar.py:3](evaluacion/evaluar.py#L3), [evaluar.py:180](evaluacion/evaluar.py#L180) | Mide el error por condición y la brecha entre la mejor y la peor |
| Transparencia | [lector.py:559](aplicacion/lector.py#L559) | Verificación cruzada: otro modelo cuenta las líneas para detectar texto faltante |
| Transparencia | [lector.py:571](aplicacion/lector.py#L571) | Decide cuándo avisar que la lectura es dudosa |
| Transparencia | [lector.py:746](aplicacion/lector.py#L746), [index.html:195](aplicacion/web/index.html#L195) | El aviso se dice en voz alta, no solo en pantalla |
| Seguridad | [lector.py:535](aplicacion/lector.py#L535) | Descarta caracteres sueltos que casi seguro están mal |
| Seguridad | [lector.py:617](aplicacion/lector.py#L617) | Con confianza muy baja no lee el texto, para no dar datos errados |
| Privacidad | [servidor.py:16](aplicacion/servidor.py#L16), [lector.py:155](aplicacion/lector.py#L155), [lector.py:730](aplicacion/lector.py#L730), [lector.py:782](aplicacion/lector.py#L782) | La foto y el audio solo existen en memoria |
| Privacidad | [docker-compose.yml:14](aplicacion/docker-compose.yml#L14) | El servidor tiene el disco en solo lectura: no puede guardar fotos |
| Privacidad | [index.html:112](aplicacion/web/index.html#L112) | La página le explica a la persona qué pasa con su foto |
| Accesibilidad | [index.html:142](aplicacion/web/index.html#L142) | Botón grande abajo, al alcance del pulgar |
| Accesibilidad | [lector.py:634](aplicacion/lector.py#L634) | Fechas, dinero, NIT y unidades (gramos, mililitros, microgramos) se leen como los diría una persona |
| Accesibilidad | [lector.py:829](aplicacion/lector.py#L829) | Los errores se dicen en voz alta |

## Límites

- Se probó con pocas fotos reales (8); los ajustes se hicieron con las imágenes de prueba y las fotos de libros, y solo las 5 etiquetas son una evaluación independiente.
- Lee texto impreso en español, en una o dos columnas. No lee escritura a mano ni tablas complejas.
- Un número suelto mal leído no activa el aviso (se corrigen casos conocidos, como un 0 leído como la letra o antes de una unidad).
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
`python evaluar.py --carpeta fotos_reales --salida resultados/fotos_reales` (fotos de libros) y
`python evaluar.py --carpeta fotos_reales_etiquetas --salida resultados/fotos_reales_etiquetas` (fotos de etiquetas).
La primera medición de las etiquetas, antes de las correcciones, está en `resultados/fotos_reales_etiquetas_antes_del_ajuste`.
`guardar_ocr_crudo.py` y `reprocesar.py` permiten probar cambios del ordenamiento y la corrección del texto sin volver a correr los modelos.

Modelos usados, todos ya entrenados y de licencia abierta: PaddleOCR (reconocer el texto), detector de EasyOCR (comprobar que
no falten líneas) y Piper con la voz es_MX-claude-high (voz). El nombre "claude" es el de esa voz, creada por terceros.
