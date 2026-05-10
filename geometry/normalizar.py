import math

def extrair_landmarks(landmarks):
    """
    Converte landmarks do MediaPipe para lista simples [(x,y,z), ...]
    """
    pontos = []
    for lm in landmarks:
        pontos.append((lm.x, lm.y, lm.z))
    return pontos


def normalizar_landmarks(pontos):
    """
    Normaliza a mão:
    1. centraliza no pulso (ponto 0)
    2. normaliza escala
    """

    if len(pontos) == 0:
        return []

    # ----------------------------
    # 1. Centralização (pulso como origem)
    # ----------------------------
    base_x, base_y, base_z = pontos[0]

    pontos = [
        (x - base_x, y - base_y, z - base_z)
        for (x, y, z) in pontos
    ]

    # ----------------------------
    # 2. Escala (distância do pulso até um ponto fixo)
    #    (aqui usamos ponto 9 como referência comum)
    # ----------------------------
    ref = pontos[9]  # pode ajustar depois

    distancia_base = math.sqrt(
        ref[0]**2 + ref[1]**2 + ref[2]**2
    )

    if distancia_base == 0:
        return pontos

    pontos = [
        (x / distancia_base, y / distancia_base, z / distancia_base)
        for (x, y, z) in pontos
    ]

    return pontos


