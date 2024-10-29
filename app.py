from flask import Flask, render_template, request, send_file, jsonify, Response
import os
from werkzeug.utils import secure_filename
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import logging
import json

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max-limit
app.config['PROCESSED_FOLDER'] = 'processed'

# Asegurarse que los directorios necesarios existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Configuración del modelo
MODEL = "yolov8x.pt"
model = YOLO(MODEL)
model.fuse()

# Clases seleccionadas (sheep:18, cow:19)
selected_classes = [18, 19]

def draw_text_annotations(frame: np.ndarray, detections) -> np.ndarray:
    """Draw the current frame's pig count and individual pig labels"""
    # Draw total count
    position = (30, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    color = (0, 255, 0)  # Green color
    thickness = 3
    current_count = len(detections.tracker_id) if len(detections) > 0 else 0
    text = f"Total Unique Animals: {current_count}"
    cv2.putText(frame, text, position, font, scale, color, thickness)

    # Draw individual pig labels with sequential numbers
    if len(detections) > 0:
        indices = np.argsort([box[0] for box in detections.xyxy])
        for count, idx in enumerate(indices, start=1):
            x1, y1, x2, y2 = detections.xyxy[idx]
            label_position = (int(x1), int(y1) - 10)
            label_text = f"Pig #{count}"
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
    return frame

def process_video(source_path, target_path):
    logger.debug(f'Iniciando procesamiento de video: {source_path}')
    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f'Archivo no encontrado: {source_path}')
        
        # Obtener el total de frames del video
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise ValueError(f'No se pudo abrir el video: {source_path}')
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Variable para contar frames procesados
        processed_frames = 0

        byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        box_annotator = sv.BoxAnnotator(thickness=4)

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            nonlocal processed_frames
            processed_frames += 1
            
            # Calcular y guardar el progreso
            progress = (processed_frames / total_frames) * 100
            with open('progress.json', 'w') as f:
                json.dump({'progress': progress}, f)
            
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[np.isin(detections.class_id, selected_classes)]
            detections = byte_tracker.update_with_detections(detections)
            
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            annotated_frame = draw_text_annotations(
                annotated_frame,
                detections
            )
            return annotated_frame

        # Procesar el video
        sv.process_video(
            source_path=source_path,
            target_path=target_path,
            callback=callback
        )
        logger.debug('Procesamiento de video completado exitosamente')

    except Exception as e:
        logger.error(f'Error durante el procesamiento del video: {str(e)}')
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    try:
        if os.path.exists('progress.json'):
            with open('progress.json', 'r') as f:
                data = json.load(f)
                return jsonify(data)
        return jsonify({'progress': 0})
    except Exception as e:
        logger.error(f'Error al leer el progreso: {str(e)}')
        return jsonify({'progress': 0})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No se encontró el archivo'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
            return jsonify({'error': 'Formato de archivo no soportado. Use MP4, AVI, MOV o WMV'}), 400

        # Reiniciar el progreso
        with open('progress.json', 'w') as f:
            json.dump({'progress': 0}, f)

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f'processed_{filename}'
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Guardar y procesar el video
        file.save(input_path)
        try:
            process_video(input_path, output_path)
            return jsonify({
                'success': True,
                'processed_video': f'/processed/{output_filename}'
            })
        finally:
            # Limpiar archivo temporal
            if os.path.exists(input_path):
                os.remove(input_path)

    except Exception as e:
        logger.error(f'Error en el procesamiento: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/processed/<filename>')
def processed_file(filename):
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f'Error al enviar archivo: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)