import cv2
import os
from datetime import datetime
import time
import numpy as np

MIN_DETECTION_DISTANCE = 0.3 
MAX_DETECTION_DISTANCE = 6  

VIDEO_OUTPUT_FOLDER = "processed_video_frames"
REALTIME_OUTPUT_FOLDER = "processed_realtime_frames"
IMAGE_OUTPUT_FOLDER = "processed_images"

os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REALTIME_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

PRECISION_CONFIG = {
    'scale_factor': 1.07,       
    'min_neighbors': 8,         
    'min_size': (30, 30),      
    'max_size': (600, 600),     
    'skip_frames': 0,           
    'target_fps': 30            
}

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        raise Exception("No se pudo cargar el clasificador Haar Cascade")
    
    print("Detector de rostros OpenCV cargado con configuración de alta precisión")
    print(f"Rango de detección: {MIN_DETECTION_DISTANCE}-{MAX_DETECTION_DISTANCE} metros")
    
except Exception as e:
    print(f"Error al cargar el detector: {e}")
    print("Asegúrate de tener OpenCV instalado correctamente")
    exit(1)

def detect_orientation(frame):
    height, width = frame.shape[:2]
    
    if width > height:
        return 'horizontal'
    elif height > width:
        return 'vertical'
    else:
        return 'square' 

def get_optimal_window_size(frame, orientation):
    height, width = frame.shape[:2]
    
    if orientation == 'horizontal':
        window_width = min(1200, width)
        window_height = int(window_width * height / width)
        if window_height > 800:
            window_height = 800
            window_width = int(window_height * width / height)
    elif orientation == 'vertical':
        window_height = min(800, height)
        window_width = int(window_height * width / height)
        if window_width > 600:
            window_width = 600
            window_height = int(window_width * height / width)
    else:  
        size = min(700, min(width, height))
        window_width = window_height = size
    
    return window_width, window_height

def auto_rotate_if_needed(frame, force_orientation=None):
    current_orientation = detect_orientation(frame)
    
    if force_orientation is None:
        return frame, current_orientation
    
    if force_orientation == 'vertical' and current_orientation == 'horizontal':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 'vertical'
    elif force_orientation == 'horizontal' and current_orientation == 'vertical':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), 'horizontal'
    return frame, current_orientation

def detect_faces_precision(gray_frame):
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=PRECISION_CONFIG['scale_factor'],
        minNeighbors=PRECISION_CONFIG['min_neighbors'],
        minSize=PRECISION_CONFIG['min_size'],
        maxSize=PRECISION_CONFIG['max_size'],
        flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
    )
    
    filtered_faces = []
    for (x, y, w, h) in faces:
        face_size = (w + h) / 2
        if 30 <= face_size <= 600:  
            filtered_faces.append((x, y, w, h))  
    return filtered_faces

def estimate_distance(face_width):
    FACE_WIDTH_CM = 15  
    FOCAL_LENGTH = 600  
    
    if face_width > 0:
        distance_cm = (FACE_WIDTH_CM * FOCAL_LENGTH) / face_width
        distance_m = distance_cm / 100
        return max(0.3, min(6.0, distance_m))  
    return 0

def process_frame_adaptive(frame, output_folder=None, frame_count=0, mode="adaptive", force_orientation=None):
    processed_frame, final_orientation = auto_rotate_if_needed(frame, force_orientation)
    
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_precision(gray)
   
    for (x, y, w, h) in faces:
        distance = estimate_distance(w)
        if MIN_DETECTION_DISTANCE <= distance <= MAX_DETECTION_DISTANCE:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"Face {distance:.1f}m"
            cv2.putText(processed_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            face_region = processed_frame[y:y+h, x:x+w]
            blur_intensity = max(60, int(min(w, h) // 8))
            if blur_intensity % 2 == 0:  
                blur_intensity += 1
                
            blurred_face = cv2.GaussianBlur(face_region, (blur_intensity, blur_intensity), 0)
            processed_frame[y:y+h, x:x+w] = blurred_face
    
    orientation_text = f"Auto ({final_orientation.title()})" if force_orientation is None else f"Forzada ({final_orientation.title()})"
    info_text = f"{mode.title()} Mode - {orientation_text} | Range: {MIN_DETECTION_DISTANCE}-{MAX_DETECTION_DISTANCE}m | Faces: {len(faces)}"
    cv2.putText(processed_frame, info_text, (10, processed_frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    valid_faces = [f for f in faces if MIN_DETECTION_DISTANCE <= estimate_distance(f[2]) <= MAX_DETECTION_DISTANCE]
    if output_folder and len(valid_faces) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(output_folder, f"{mode}_{final_orientation}_{timestamp}.jpg")
        cv2.imwrite(filename, processed_frame)
        print(f"Archivo guardado: {filename} - Rostros válidos: {len(valid_faces)} - Orientación: {final_orientation}")
    
    return processed_frame, final_orientation

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Archivo no encontrado - {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error al leer la imagen: {image_path}")
        return
    
    original_orientation = detect_orientation(frame)
    print(f"Procesando imagen con orientación detectada: {original_orientation}")
    print(f"Dimensiones originales: {frame.shape[1]}x{frame.shape[0]} (WxH)")
    
    processed_frame, final_orientation = process_frame_adaptive(frame, IMAGE_OUTPUT_FOLDER, 0, "image")
    
    window_name = f'Imagen Procesada ({final_orientation.title()}) - Presione cualquier tecla'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    window_width, window_height = get_optimal_window_size(processed_frame, final_orientation)
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.imshow(window_name, processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_video_orientation_preference():
    print("\n--- CONFIGURACIÓN DE ORIENTACIÓN PARA VIDEO ---")
    print("1. Automática (mantener orientación original del video)")
    print("2. Forzar horizontal (rotar a formato horizontal)")
    print("3. Forzar vertical (rotar a formato vertical)")
    
    while True:
        choice = input("Seleccione orientación (1-3): ").strip()
        
        if choice == "1":
            return None, "Automática"
        elif choice == "2":
            return "horizontal", "Forzada Horizontal"
        elif choice == "3":
            return "vertical", "Forzada Vertical"
        else:
            print("Opción no válida. Intente nuevamente (1-3).")

def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer el primer frame del video")
        cap.release()
        return
    
    original_orientation = detect_orientation(first_frame)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nInfo del video: {video_path}")
    print(f"Orientación original detectada: {original_orientation}")
    print(f"Dimensiones originales: {first_frame.shape[1]}x{first_frame.shape[0]} (WxH)")
    print(f"FPS original: {original_fps:.1f} | Total frames: {total_frames}")
    
    force_orientation, orientation_description = get_video_orientation_preference()
    
    print(f"\nConfiguración seleccionada: {orientation_description}")
    print(f"Modo: Alta precisión, rango {MIN_DETECTION_DISTANCE}-{MAX_DETECTION_DISTANCE}m")
    print(f"Guardando frames en: {VIDEO_OUTPUT_FOLDER}")
    print("Presione Q para salir...")
    
    reference_frame, final_orientation = auto_rotate_if_needed(first_frame, force_orientation)
    
    window_name = f'Video - {orientation_description} ({final_orientation.title()}) - Presione Q para salir'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    window_width, window_height = get_optimal_window_size(reference_frame, final_orientation)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    delay = max(1, int(1000 / PRECISION_CONFIG['target_fps']))
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, current_orientation = process_frame_adaptive(
            frame, VIDEO_OUTPUT_FOLDER, frame_count, "video", force_orientation
        )
        
        cv2.imshow(window_name, processed_frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        
        if frame_count % 50 == 0:
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"Progreso: {progress:.1f}% | FPS procesamiento: {current_fps:.1f} | Orientación: {current_orientation}")
        
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"\nProcesamiento completado en {elapsed_time:.2f} segundos")
    print(f"Frames procesados: {frame_count}")
    print(f"Orientación utilizada: {orientation_description}")

def process_realtime_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer frame de la cámara")
        cap.release()
        return
    
    camera_orientation = detect_orientation(first_frame)
    
    print(f"\nProcesamiento en tiempo real - Webcam")
    print(f"Orientación de cámara detectada: {camera_orientation}")
    print(f"Dimensiones de cámara: {first_frame.shape[1]}x{first_frame.shape[0]} (WxH)")
    print(f"Rango de detección: {MIN_DETECTION_DISTANCE}-{MAX_DETECTION_DISTANCE} metros")
    print(f"Guardando frames en: {REALTIME_OUTPUT_FOLDER}")
    print("Nota: En tiempo real solo se usa orientación automática")
    print("Presione Q para salir")
    
    window_name = f'Webcam Tiempo Real ({camera_orientation.title()}) - Presione Q para salir'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    window_width, window_height = get_optimal_window_size(first_frame, camera_orientation)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, current_orientation = process_frame_adaptive(
            frame, REALTIME_OUTPUT_FOLDER, frame_count, "realtime"
        )
        
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        else:
            fps = 0
        
        if fps > 0:
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(processed_frame, f"Orientacion: {current_orientation.title()}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("\n" + "="*80)
    print("SISTEMA DE DETECCIÓN Y DESENFOQUE DE ROSTROS")
    print("="*80)
    print("Imágenes: Orientación automática")
    print("Videos: Orientación configurable (automática, horizontal, vertical)")
    print("Tiempo real: Orientación automática")
    
    while True:
        print("\nOpciones disponibles:")
        print("1. Procesar imagen (orientación automática)")
        print("2. Procesar video grabado (orientación configurable)")
        print("3. Procesar video en tiempo real - webcam (orientación automática)")
        print("4. Salir")
        
        choice = input("Seleccione una opción (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Ruta de la imagen: ").strip().replace('"', '')
            process_image(image_path)
        elif choice == "2":
            video_path = input("Ruta del video: ").strip().replace('"', '')
            process_video_file(video_path)
        elif choice == "3":
            process_realtime_video()
        elif choice == "4":
            print("Saliendo del programa")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    print("Iniciando sistema de detección de rostros")
    print("Detección automática de orientación")
    print("Ventanas adaptativas según contenido")
    print(f"Configuración: Rango {MIN_DETECTION_DISTANCE}-{MAX_DETECTION_DISTANCE}m, Máxima precisión")
    print("Videos: Orientación configurable | Imágenes y tiempo real: Orientación automática")
    main()