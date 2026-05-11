import random
import math


def augment(features):

    # ----------------------------
    # RECONSTRUIR PONTOS
    # ----------------------------
    pontos = []

    for i in range(0, len(features), 3):

        x = features[i]
        y = features[i + 1]
        z = features[i + 2]

        pontos.append([x, y, z])

    # ----------------------------
    # AUGMENTATION
    # ----------------------------

    # pequena rotação
    angle = random.uniform(-0.15, 0.15)

    cos = math.cos(angle)
    sin = math.sin(angle)

    # pequena escala
    scale = random.uniform(0.95, 1.05)

    pontos_aug = []

    for x, y, z in pontos:

        # ----------------------------
        # ROTAÇÃO 2D
        # ----------------------------
        new_x = x * cos - y * sin
        new_y = x * sin + y * cos

        # ----------------------------
        # ESCALA
        # ----------------------------
        new_x *= scale
        new_y *= scale
        z *= scale

        # ----------------------------
        # RUÍDO
        # ----------------------------
        new_x += random.uniform(-0.02, 0.02)
        new_y += random.uniform(-0.02, 0.02)
        z += random.uniform(-0.02, 0.02)

        pontos_aug.append([new_x, new_y, z])

    # ----------------------------
    # FLATTEN
    # ----------------------------
    features_aug = []

    for ponto in pontos_aug:

        features_aug.extend(ponto)

    return features_aug