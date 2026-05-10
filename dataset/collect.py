import json
import os

from geometry.normalizar import (
    extrair_landmarks,
    normalizar_landmarks
)

# ----------------------------
# CONFIGURAÇÃO
# ----------------------------
DATASET_PATH = "dataset/labels.json"


# ----------------------------
# CARREGAR DATASET EXISTENTE
# ----------------------------
def load_dataset():

    if os.path.exists(DATASET_PATH):

        with open(DATASET_PATH, "r") as f:
            return json.load(f)

    return []


# ----------------------------
# SALVAR DATASET
# ----------------------------
def save_dataset(data):

    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------
# CONVERTE LISTA 3D → VETOR
# ----------------------------
def flatten(pontos):

    return [
        coord
        for ponto in pontos
        for coord in ponto
    ]


# ----------------------------
# COLETAR SAMPLE
# ----------------------------
def collect_sample(hand_landmarks, label):

    dataset = load_dataset()

    # 1. extrair landmarks
    pontos = extrair_landmarks(hand_landmarks)

    # 2. normalizar
    pontos_norm = normalizar_landmarks(pontos)

    # 3. transformar em vetor
    features = flatten(pontos_norm)

    # 4. criar sample
    sample = {
        "label": label,
        "features": features
    }

    # 5. adicionar dataset
    dataset.append(sample)

    # 6. salvar
    save_dataset(dataset)

    print(f"[SALVO] {label} | total: {len(dataset)}")