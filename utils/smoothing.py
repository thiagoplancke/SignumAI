
historico = []

def media_angulos(angulo):
    historico.append(angulo)
    if len(historico) > 5:
        historico.pop(0)
    return sum(historico) / len(historico)