import cv2
import os
import time

# Configuración
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 30
ASPECT_RATIO_RANGE = (0.75, 1.3)
MIN_FACE_SIZE = (50, 50)
MAX_FACE_SIZE = (300, 300)

def procesar_video(input_path, output_path=None, mostrar_preview=True):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(input_path if input_path != '0' else 0)
    
    if not cap.isOpened():
        print("Error al abrir el video/cámara")
        return

    # Configurar escritura de video de salida si se especifica
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    stats = {
        'total_frames': 0,
        'frames_con_rostros': 0,
        'rostros_reales': 0,
        'falsos_positivos': 0,
        'tiempo_procesamiento': 0
    }

    print("🔍 Procesando video... (Presione 'q' para detener)")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats['total_frames'] += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # Detección de rostros
        faces = face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=MIN_FACE_SIZE,
            maxSize=MAX_FACE_SIZE
        )

        valid_faces, falsos = filtrar_rostros(faces)
        stats['falsos_positivos'] += falsos

        if valid_faces:
            stats['frames_con_rostros'] += 1
            stats['rostros_reales'] += len(valid_faces)
            frame = desenfocar_rostros(frame, valid_faces)

        # Mostrar métricas en el frame
        porcentaje_deteccion = (stats['frames_con_rostros'] / stats['total_frames']) * 100
        total_detecciones = stats['rostros_reales'] + stats['falsos_positivos']
        precision = (stats['rostros_reales'] / total_detecciones) * 100 if total_detecciones > 0 else 0
        aprendizaje = (porcentaje_deteccion + precision) / 2

        cv2.putText(frame, f"Rostros: {len(valid_faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Precision: {precision:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Aprendizaje: {aprendizaje:.1f}%", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if mostrar_preview:
            cv2.imshow('Video - Deteccion de Rostros', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out:
            out.write(frame)

    stats['tiempo_procesamiento'] = time.time() - start_time
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    mostrar_reporte_video(stats)

def filtrar_rostros(faces):
    valid = []
    falsos = 0
    for (x, y, w, h) in faces:
        aspect_ratio = w / float(h)
        area = w * h
        if (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1] and
            1000 <= area <= 50000):
            valid.append((x, y, w, h))
        else:
            falsos += 1
    return valid, falsos

def desenfocar_rostros(img, faces):
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], BLUR_KERNEL, BLUR_SIGMA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def mostrar_reporte_video(stats):
    print("\n" + "="*50)
    print("📊 REPORTE FINAL - PROCESAMIENTO DE VIDEO")
    print("="*50)
    
    porcentaje_deteccion = (stats['frames_con_rostros'] / stats['total_frames']) * 100
    total_detecciones = stats['rostros_reales'] + stats['falsos_positivos']
    precision = (stats['rostros_reales'] / total_detecciones) * 100 if total_detecciones > 0 else 0
    aprendizaje = (porcentaje_deteccion + precision) / 2

    print(f"• Frames procesados: {stats['total_frames']}")
    print(f"• Frames con rostros: {stats['frames_con_rostros']} ({porcentaje_deteccion:.2f}%)")
    print(f"• Rostros detectados: {stats['rostros_reales']}")
    print(f"• Falsos positivos: {stats['falsos_positivos']}")
    print(f"• Tiempo total: {stats['tiempo_procesamiento']:.2f} segundos")
    print(f"• Velocidad: {stats['total_frames']/stats['tiempo_procesamiento']:.2f} FPS")
    print("="*50)
    print(f"🌟 PORCENTAJE DE APRENDIZAJE: {aprendizaje:.2f}%")
    print("="*50)

def main():
    while True:
        print("\n🎥 MENÚ PRINCIPAL - DETECCIÓN DE ROSTROS EN VIDEO")
        print("1. Procesar archivo de video")
        print("2. Usar cámara web")
        print("3. Salir")
        opcion = input("Seleccione una opción (1-3): ").strip()

        if opcion == "1":
            input_path = input("Ruta del video (ej: video.mp4): ").strip()
            output_path = input("Ruta de salida (dejar vacío para no guardar): ").strip()
            output_path = output_path if output_path else None
            procesar_video(input_path, output_path)
        elif opcion == "2":
            print("🔴 Usando cámara web (Presione 'q' para detener)")
            procesar_video('0', None)  # '0' para cámara web
        elif opcion == "3":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("⚠️ Opción no válida")

if __name__ == "__main__":
    main()