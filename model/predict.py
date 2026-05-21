import json
import torch

from collections import Counter

from model.modelo import SignLanguageModel

from geometry.normalizar import (
    extrair_landmarks,
    normalizar_landmarks
)

# ----------------------------
# LOAD MODEL
# ----------------------------
with open("model/label_encoder.json", "r") as f:
    label_to_index = json.load(f)

index_to_label = {
    value: key
    for key, value in label_to_index.items()
}

model = SignLanguageModel()

model.load_state_dict(
    torch.load("model/signum_model.pth")
)

model.eval()

historico = []


# ----------------------------
# PREDICT
# ----------------------------
def testar_modelo(mao):

    pontos = extrair_landmarks(mao)

    pontos_norm = normalizar_landmarks(pontos)

    features = [
        coord
        for ponto in pontos_norm
        for coord in ponto
    ]

    X = torch.tensor(
        [features],
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(X)

        probs = torch.softmax(output, dim=1)

        confidence = torch.max(probs).item()

        predicted_index = torch.argmax(
            probs,
            dim=1
        ).item()

    letra = index_to_label[predicted_index]

    # ----------------------------
    # SMOOTHING
    # ----------------------------
    if confidence > 0.70:

        historico.append(letra)

        if len(historico) > 15:
            historico.pop(0)

        letra_estavel = Counter(
            historico
        ).most_common(1)[0][0]

    else:

        letra_estavel = "?"

    return letra_estavel, confidence