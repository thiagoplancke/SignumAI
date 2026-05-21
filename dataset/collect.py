import json
import os
import cv2 as cv
from capture.camera import abrir_camera
from tracking.hand_tracker import detectar_mao



from geometry.normalizar import (
    extrair_landmarks,
    normalizar_landmarks
)


DATASET_PATH = "labels.json"



def load_dataset():

    if os.path.exists(DATASET_PATH):

        with open(DATASET_PATH, "r") as f:
            return json.load(f)

    return []



def save_dataset(data):

    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)



def flatten(pontos):

    return [
        coord
        for ponto in pontos
        for coord in ponto
    ]



def collect_sample(hand_landmarks, label):

    dataset = load_dataset()

    pontos = extrair_landmarks(hand_landmarks)

    pontos_norm = normalizar_landmarks(pontos)

    features = flatten(pontos_norm)

    sample = {
        "label": label,
        "features": features
    }

    dataset.append(sample)

    save_dataset(dataset)

    print(f"[SALVO] {label} | total: {len(dataset)}")


def coletar_dados():

    status, frame = camera.read()

    

    frame, hand_landmarks = detectar_mao(frame)


    key = cv.waitKey(1) & 0xFF


    if 97 <= key <= 122:

        current_label = chr(key).upper()

        collecting = True
        samples_left = 50

        print(f"\n[COLETANDO] letra {current_label}")


    if collecting and len(hand_landmarks) > 0:

        frame_counter += 1

        mao = hand_landmarks[0]

        if frame_counter % save_interval == 0:

            collect_sample(mao, current_label)

            samples_left -= 1

            print(f"faltam: {samples_left}")

        if samples_left <= 0:

            collecting = False

            print(f"[FINALIZADO] letra {current_label}\n")


    cv.imshow("SignumAI", frame)
