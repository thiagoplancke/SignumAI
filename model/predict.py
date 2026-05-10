import json
import torch
import cv2 as cv

from collections import Counter

from model.modelo import SignLanguageModel

from capture.camera import abrir_camera
from tracking.hand_tracker import detectar_mao

from geometry.normalizar import (
    extrair_landmarks,
    normalizar_landmarks
)

# ----------------------------
# LABELS
# ----------------------------
with open("model/label_encoder.json", "r") as f:
    label_to_index = json.load(f)

index_to_label = {
    value: key
    for key, value in label_to_index.items()
}

# ----------------------------
# MODELO
# ----------------------------
model = SignLanguageModel()

model.load_state_dict(
    torch.load("model/signum_model.pth")
)

model.eval()

# ----------------------------
# CÂMERA
# ----------------------------
camera = abrir_camera()

# ----------------------------
# HISTÓRICO TEMPORAL
# ----------------------------
historico = []

print("=== SIGNUM AI ===")
print("Pressione Q para sair")

# ----------------------------
# LOOP
# ----------------------------
while True:

    status, frame = camera.read()

    if not status:
        break

    frame, hand_landmarks = detectar_mao(frame)

    if len(hand_landmarks) > 0:

        mao = hand_landmarks[0]

        # ----------------------------
        # EXTRAÇÃO
        # ----------------------------
        pontos = extrair_landmarks(mao)

        # ----------------------------
        # NORMALIZAÇÃO
        # ----------------------------
        pontos_norm = normalizar_landmarks(pontos)

        # ----------------------------
        # FLATTEN
        # ----------------------------
        features = [
            coord
            for ponto in pontos_norm
            for coord in ponto
        ]

        # ----------------------------
        # TENSOR
        # ----------------------------
        X = torch.tensor(
            [features],
            dtype=torch.float32
        )

        # ----------------------------
        # INFERÊNCIA
        # ----------------------------
        with torch.no_grad():

            output = model(X)

            # probabilidades
            probs = torch.softmax(output, dim=1)

            # confiança máxima
            confidence = torch.max(probs).item()

            # índice previsto
            predicted_index = torch.argmax(
                probs,
                dim=1
            ).item()

        letra = index_to_label[predicted_index]

        # ----------------------------
        # FILTRO DE CONFIANÇA
        # ----------------------------
        if confidence > 0.70:

            historico.append(letra)

            # mantém últimos frames
            if len(historico) > 15:
                historico.pop(0)

            # letra mais comum
            letra_estavel = Counter(
                historico
            ).most_common(1)[0][0]

        else:

            letra_estavel = "?"

        # ----------------------------
        # TEXTO NA TELA
        # ----------------------------
        cv.putText(
            frame,
            f"Letra: {letra_estavel}",
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv.putText(
            frame,
            f"Conf: {confidence:.2f}",
            (50, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # ----------------------------
    # MOSTRAR FRAME
    # ----------------------------
    cv.imshow("SignumAI", frame)

    # ----------------------------
    # SAIR
    # ----------------------------
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# FINALIZAR
# ----------------------------
camera.release()
cv.destroyAllWindows()