import cv2 as cv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminho do modelo
MODEL_PATH = "model/hand_landmarker.task"

# Configuração base
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# Configuração do detector
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

# Cria detector
landmarker = vision.HandLandmarker.create_from_options(options)


def detectar_mao(frame):

    # Converte BGR -> RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Cria imagem MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detecta mãos
    resultado = landmarker.detect(mp_image)

    # Desenha landmarks
    if resultado.hand_landmarks:

        for hand_landmarks in resultado.hand_landmarks:

            altura, largura, _ = frame.shape

            for landmark in hand_landmarks:

                x = int(landmark.x * largura)
                y = int(landmark.y * altura)

                cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

    return frame