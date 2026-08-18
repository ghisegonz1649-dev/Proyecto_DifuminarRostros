# -*- coding: utf-8 -*-
"""
SISTEMA DE DETECCIÓN Y DESENFOQUE DE ROSTROS CON GUARDADO DE VIDEO E IMÁGENES
-----------------------------------------------------------------------------
Funcionalidades:
1. Procesamiento de videos grabados con guardado del resultado
2. Procesamiento en tiempo real desde webcam con grabación
3. Procesamiento de imágenes individuales
4. Detección automática de FPS y metadatos del video
5. Opción de forzar orientación horizontal/vertical
6. Rango de detección configurable (0.3-6 metros)
7. Desenfoque más agresivo para mejor privacidad

INSTRUCCIONES:
- Ejecutar y seleccionar una opción del menú
- Para videos: se procesará y guardará un nuevo video
- Para webcam: se grabará hasta presionar 'Q'
- Para imágenes: se procesará y guardará una nueva imagen
- Los archivos se guardan en formato MP4 para video y JPG para imágenes
"""

import cv2
import os
from datetime import datetime
import time
import numpy as np



# ================= CONFIGURACIÓN GLOBAL =================
MIN_DETECTION_DISTANCE = 0.3  
MAX_DETECTION_DISTANCE = 6    
BLUR_AGGRESSIVENESS = 0.2     


OUTPUT_FOLDER = "processed_videos"
IMAGE_FOLDER = "processed_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)




# ================= CONFIGURACIÓN DE DETECCIÓN =================
PRECISION_CONFIG = {
    'scale_factor': 1.07,
    'min_neighbors': 8,
    'min_size': (30, 30),
    'max_size': (600, 600),
    'flags': cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
}


try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("No se pudo cargar el clasificador Haar Cascade")
    print("✅ Detector de rostros cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el detector: {e}")
    exit(1)



# ================= FUNCIONES DE ORIENTACIÓN =================
def detect_orientation(frame):
    height, width = frame.shape[:2]
    return 'horizontal' if width > height else 'vertical'

def auto_rotate_if_needed(frame, force_orientation=None):
    current_orientation = detect_orientation(frame)
    
    if force_orientation == 'vertical' and current_orientation == 'horizontal':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 'vertical'
    elif force_orientation == 'horizontal' and current_orientation == 'vertical':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), 'horizontal'
    return frame, current_orientation



# ================= FUNCIONES DE DETECCIÓN Y DESENFOQUE =================
def detect_faces(gray_frame):
    return face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=PRECISION_CONFIG['scale_factor'],
        minNeighbors=PRECISION_CONFIG['min_neighbors'],
        minSize=PRECISION_CONFIG['min_size'],
        maxSize=PRECISION_CONFIG['max_size'],
        flags=PRECISION_CONFIG['flags']
    )

def estimate_distance(face_width):
    FACE_WIDTH_CM = 15  
    FOCAL_LENGTH = 600  
    if face_width > 0:
        distance_cm = (FACE_WIDTH_CM * FOCAL_LENGTH) / face_width
        return max(0.3, min(6.0, distance_cm / 100))  
    return 0

def apply_aggressive_blur(face_region, w, h):
    
    kernel_size = max(5, int(min(w, h) * BLUR_AGGRESSIVENESS))
    
    
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
  
    blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
    
   
    return cv2.medianBlur(blurred, 15)

def process_frame(frame, force_orientation=None):
   
    processed_frame, orientation = auto_rotate_if_needed(frame, force_orientation)
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = detect_faces(gray)
    
   
    for (x, y, w, h) in faces:
        distance = estimate_distance(w)
        if MIN_DETECTION_DISTANCE <= distance <= MAX_DETECTION_DISTANCE:
            # Dibujar rectángulo y etiqueta
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(processed_frame, f"{distance:.1f}m", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
           
            face_region = processed_frame[y:y+h, x:x+w]
            processed_frame[y:y+h, x:x+w] = apply_aggressive_blur(face_region, w, h)
    
    return processed_frame, orientation, processed_frame.shape[:2]  




# ================= FUNCIONES DE IMAGEN =================
def process_image(input_path, force_orientation=None):
  
    img = cv2.imread(input_path)
    if img is None:
        print("❌ No se pudo leer la imagen")
        return
    
    
    start_time = time.time()
    processed_img, orientation, _ = process_frame(img, force_orientation)
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(IMAGE_FOLDER, f"{name}_processed_{timestamp}.jpg")
    
    cv2.imwrite(output_path, processed_img)
    
   
    print(f"\n✅ Imagen procesada guardada en: {output_path}")
    print(f"Tiempo de procesamiento: {time.time() - start_time:.2f} segundos")
    

    cv2.imshow("Imagen procesada (Presiona cualquier tecla para cerrar)", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# ================= FUNCIONES DE VIDEO =================
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
    return info

def process_video(input_path, force_orientation=None):
  
    video_info = get_video_info(input_path)
    if not video_info:
        print("❌ No se pudo leer el video")
        return
    
    print("\n📊 Información del video:")
    print(f"- Resolución: {video_info['width']}x{video_info['height']}")
    print(f"- FPS: {video_info['fps']:.2f}")
    print(f"- Duración: {video_info['duration']:.2f} segundos")
    print(f"- Total de frames: {video_info['total_frames']}")
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
   
    cap = cv2.VideoCapture(input_path)
    ret, first_frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el primer frame del video")
        cap.release()
        return
    
   
    processed_frame, orientation, dims = process_frame(first_frame, force_orientation)
    out_width, out_height = dims[1], dims[0]  # (width, height)
    
  
    out = cv2.VideoWriter(output_path, fourcc, video_info['fps'], (out_width, out_height))
    if not out.isOpened():
        print(f"❌ No se pudo crear el archivo de video: {output_path}")
        cap.release()
        return
    

    out.write(processed_frame)
    
    
    frame_count = 1
    start_time = time.time()
    
    print("\n⏳ Procesando video... (Presiona 'Q' para cancelar)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, _, _ = process_frame(frame, force_orientation)
        out.write(processed_frame)
        
       
        frame_count += 1
        if frame_count % int(video_info['fps'] * 5) == 0:
            elapsed = time.time() - start_time
            percent = min(100, (frame_count/video_info['total_frames'])*100)
            remaining = (video_info['duration'] - elapsed) if elapsed < video_info['duration'] else 0
            print(f"Progreso: {percent:.1f}% | Tiempo restante: {remaining:.1f}s")
        
      
        cv2.imshow("Procesando video (Presiona Q para cancelar)", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Video procesado guardado en: {output_path}")
    print(f"Tiempo de procesamiento: {time.time() - start_time:.2f} segundos")
    print(f"Dimensiones del video: {out_width}x{out_height}")

def process_webcam(force_orientation=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara")
        return
    
   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n⏳ Midiendo FPS real... (espere 2 segundos)")
    start = time.time()
    frames = 0
    while time.time() - start < 2:
        ret, frame = cap.read()
        if ret:
            frames += 1
    real_fps = max(5, frames / 2)  # Mínimo 5 FPS
    print(f"✅ FPS real medido: {real_fps:.1f}")

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_FOLDER, f"webcam_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
   
    ret, first_frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar frame de la cámara")
        cap.release()
        return
    
   
    processed_frame, orientation, dims = process_frame(first_frame, force_orientation)
    out_width, out_height = dims[1], dims[0]  # (width, height)

    out = cv2.VideoWriter(output_path, fourcc, real_fps, (out_width, out_height))
    if not out.isOpened():
        print(f"❌ No se pudo crear el archivo de video: {output_path}")
        cap.release()
        return
    

    out.write(processed_frame)
    
    print("\n🔴 Grabando... Presiona 'Q' para detener")
    start_time = time.time()
    frame_count = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, _, _ = process_frame(frame, force_orientation)
        out.write(processed_frame)
        cv2.imshow("Webcam - Presiona Q para detener", processed_frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    duration = time.time() - start_time
    print(f"\n✅ Grabación guardada en: {output_path}")
    print(f"- Duración: {duration:.2f} segundos")
    print(f"- Frames capturados: {frame_count}")
    print(f"- Dimensiones: {out_width}x{out_height}")


# ================= INTERFAZ DE USUARIO =================
def get_orientation_preference():
    print("\n🔄 Configuración de orientación:")
    print("1. Automática (detectar según el video/imagen)")
    print("2. Forzar horizontal")
    print("3. Forzar vertical")
    
    while True:
        choice = input("Seleccione (1-3): ").strip()
        if choice == "1":
            return None
        elif choice == "2":
            return "horizontal"
        elif choice == "3":
            return "vertical"
        print("❌ Opción no válida. Intente nuevamente.")

def main():
    print("\n" + "="*60)
    print(" SISTEMA DE DETECCIÓN DE ROSTROS Y GUARDADO DE VIDEO/IMÁGENES ")
    print("="*60)
    print(f"🔹 Rango de detección: {MIN_DETECTION_DISTANCE}-{MAX_DETECTION_DISTANCE} metros")
    print(f"🔹 Intensidad de desenfoque: {BLUR_AGGRESSIVENESS} (más bajo = más fuerte)")
    print(f"📂 Videos de salida: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"🖼️ Imágenes de salida: {os.path.abspath(IMAGE_FOLDER)}")
    
    while True:
        print("\nOpciones principales:")
        print("1. Procesar video grabado")
        print("2. Grabar desde webcam")
        print("3. Procesar imagen")
        print("4. Salir")
        
        choice = input("Seleccione una opción (1-4): ").strip()
        
        if choice == "1":
            video_path = input("📁 Ingrese la ruta del video: ").strip()
            if not os.path.exists(video_path):
                print("❌ El archivo no existe")
                continue
                
            orientation = get_orientation_preference()
            process_video(video_path, orientation)
            
        elif choice == "2":
            orientation = get_orientation_preference()
            process_webcam(orientation)
            
        elif choice == "3":
            img_path = input("🖼️ Ingrese la ruta de la imagen: ").strip()
            if not os.path.exists(img_path):
                print("❌ El archivo no existe")
                continue
                
            orientation = get_orientation_preference()
            process_image(img_path, orientation)
            
        elif choice == "4":
            print("👋 Saliendo del programa...")
            break
            
        else:
            print("❌ Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()