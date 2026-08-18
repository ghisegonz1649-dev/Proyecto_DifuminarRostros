import cv2
import os

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 30
ASPECT_RATIO_RANGE = (0.75, 1.3)

def procesar_imagenes(dataset_dir, output_dir, mostrar_preview=True):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    stats = {
        'total_imagenes': 0,
        'imagenes_con_rostros': 0,
        'imagenes_sin_rostros': 0,
        'rostros_reales': 0,
        'falsos_positivos': 0
    }

    # Crear carpetas de salida
    salida_rostros_difuminados = os.path.join(output_dir, "Rostros_Difuminados")
    salida_rostros_sindifuminar = os.path.join(output_dir, "Rostros_SinDifuminar")
    os.makedirs(salida_rostros_difuminados, exist_ok=True)
    os.makedirs(salida_rostros_sindifuminar, exist_ok=True)

    print("Procesando imágenes...")

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            
            stats['total_imagenes'] += 1
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, dataset_dir)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Imagen invalida o corrupta: {input_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = detectar_rostros(gray, face_cascade)
            valid_faces, falsos = filtrar_rostros(faces)
            stats['falsos_positivos'] += falsos

            if valid_faces:
                stats['imagenes_con_rostros'] += 1
                stats['rostros_reales'] += len(valid_faces)

                img = desenfocar_rostros(img, valid_faces)

                if mostrar_preview:
                    mostrar_previsualizacion(img, len(valid_faces))

                output_path = os.path.join(salida_rostros_difuminados, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, img)
                print(f"Guardada rostros difuminados: {input_path}")
            else:
                stats['imagenes_sin_rostros'] += 1
                output_path = os.path.join(salida_rostros_sindifuminar, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, img)
                print(f"Guardada rostros sin difuminar: {input_path}")

    mostrar_reporte(stats, output_dir)

def detectar_rostros(gray_img, face_cascade):
    return face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(50, 50),
        maxSize=(300, 300)
    )

def filtrar_rostros(faces):
    valid = []
    falsos = 0
    for (x, y, w, h) in faces:
        aspect_ratio = w / float(h)
        if ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]:
            valid.append((x, y, w, h))
        else:
            falsos += 1
    return valid, falsos

def desenfocar_rostros(img, faces):
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], BLUR_KERNEL, BLUR_SIGMA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def mostrar_previsualizacion(img, num_faces):
    preview = img.copy()
    cv2.putText(preview, f"Rostros: {num_faces}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Previsualización - Presione una tecla', preview)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

def mostrar_reporte(stats, output_dir):
    print("\n" + "="*50)
    print("📊 REPORTE FINAL")
    print("="*50)
    
    # Cálculo de métricas
    porcentaje_deteccion = (stats['imagenes_con_rostros'] / stats['total_imagenes']) * 100 if stats['total_imagenes'] > 0 else 0
    total_detecciones = stats['rostros_reales'] + stats['falsos_positivos']
    precision = (stats['rostros_reales'] / total_detecciones) * 100 if total_detecciones > 0 else 0
    aprendizaje = (porcentaje_deteccion + precision) / 2  # Puntuación promedio
    
    print(f"• Imágenes procesadas: {stats['total_imagenes']}")
    print(f"• Rostros detectados: {stats['rostros_reales']} (Falsos: {stats['falsos_positivos']})")
    print(f"• Cobertura: {porcentaje_deteccion:.2f}%")
    print(f"• Precisión: {precision:.2f}%")
    print("="*50)
    print(f"🌟 PORCENTAJE DE APRENDIZAJE: {aprendizaje:.2f}%")
    print("="*50 + "\n")


def main():
    while True:
        print("\n📷 MENÚ PRINCIPAL - DETECCIÓN DE ROSTROS")
        print("1. Iniciar procesamiento de imágenes")
        print("2. Salir")
        opcion = input("Seleccione una opción (1-2): ")

        if opcion == "1":
            dataset_dir = "Programas_e_Imagenes/DataSet"
            output_dir = "Programas_e_Imagenes/DataSet_desenfocado"
            print("\n" + "="*50)
            print(f"ENTRADA: {dataset_dir}")
            print(f"SALIDA: {output_dir}")
            print("="*50)
            procesar_imagenes(dataset_dir, output_dir, mostrar_preview=True)
        elif opcion == "2":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("⚠️ Opción no válida")

if __name__ == "__main__":
    main()
