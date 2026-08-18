# Proyecto_DifuminarRostros
Sistema de detección y desenfoque de rostros desarrollado en Python con OpenCV. El proyecto permite procesar imágenes, videos y flujo en tiempo real desde la cámara para ocultar rostros y proteger la privacidad visual de las personas.

## Descripción general

Este repositorio contiene varias implementaciones relacionadas con la detección facial y el desenfoque automático. La idea principal es identificar rostros en un frame o imagen y aplicar un efecto de desenfoque sobre esa zona, dejando el resto del contenido intacto.

El proyecto está orientado a escenarios donde se requiere anonimizar personas en contenido multimedia, como grabaciones de video, material fotográfico o transmisión en vivo desde webcam.

## Funcionalidades principales

- Detección automática de rostros usando Haar Cascades de OpenCV.
- Procesamiento de imágenes individuales.
- Procesamiento de videos almacenados.
- Captura en tiempo real desde webcam.
- Opción de forzar orientación horizontal o vertical.
- Guardado automático de resultados en carpetas de salida.
- Visualización de la imagen o video procesado en ventana local.
- Medición básica de información del video y estadísticas del procesamiento.

## Estructura del repositorio

- `Proyecto.py`: script principal con interfaz de menú para procesar video, webcam e imágenes.
- `Difuminado_Caras_Imagenes.py`: procesamiento masivo de imágenes dentro de un dataset.
- `Difuminado_Caras_Video.py`: procesamiento de video y webcam con detección de rostros y estadísticas.
- `Prueba.py`: archivo de pruebas o versión experimental del sistema.

## Requisitos

- Python 3.9 o superior
- OpenCV (`cv2`)
- NumPy

Instalación recomendada:

```bash
pip install opencv-python numpy
```

## Cómo ejecutar

### 1. Script principal

```bash
python Proyecto.py
```

Se mostrará un menú con estas opciones:

- Procesar video grabado
- Grabar desde webcam
- Procesar imagen
- Salir

### 2. Procesamiento masivo de imágenes

```bash
python Difuminado_Caras_Imagenes.py
```

Este script recorre un directorio de imágenes, detecta rostros y guarda una copia con rostros difuminados en una carpeta de salida.

### 3. Procesamiento de video

```bash
python Difuminado_Caras_Video.py
```

Permite elegir entre:

- Procesar un archivo de video
- Usar cámara web
- Salir

## Carpetas de salida

Los scripts crean automáticamente carpetas para guardar los resultados, por ejemplo:

- `processed_videos/`
- `processed_images/`
- `processed_video_frames/`
- `processed_realtime_frames/`

La ubicación real puede variar según el script que se ejecute.

## Observaciones

- La detección facial depende del modelo Haar Cascade incluido con OpenCV.
- El nivel de desenfoque puede ajustarse en la configuración de cada script.
- El rendimiento puede variar según la resolución del video, la calidad de la cámara y la potencia del equipo.
- Es recomendable usar iluminación adecuada para mejorar la detección de rostros.

## Uso 

1. Ejecuta el archivo principal con `python Proyecto.py`.
2. Selecciona el tipo de entrada: imagen, video o webcam.
3. Indica la ruta del archivo o permite usar la cámara.
4. El programa procesa los rostros y guarda una versión desenfocada.

## Licencia

Este proyecto se comparte con fines educativos y de investigación. Si se reutiliza en otro contexto, se recomienda respetar la propiedad intelectual de las librerías utilizadas y del código original del proyecto.

## Autor

Proyecto desarrollado como ejercicio de visión artificial y privacidad digital con OpenCV.
