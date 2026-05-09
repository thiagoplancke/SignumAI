import math
def calcular_angulo(ponto1, ponto2, ponto3):
    # Calcula o vetor AB
    AB = (ponto2[0] - ponto1[0], ponto2[1] - ponto1[1])
    # Calcula o vetor BC
    BC = (ponto3[0] - ponto2[0], ponto3[1] - ponto2[1])
    # Calcula o produto escalar
    produto_escalar = AB[0] * BC[0] + AB[1] * BC[1]
    # Calcula as magnitudes
    magnitudes = ( (AB[0]**2 + AB[1]**2)**0.5 * (BC[0]**2 + BC[1]**2)**0.5 )
    # Calcula o ângulo em radianos
    angulo = math.acos(produto_escalar / magnitudes)
    # Converte para graus
    angulo_graus = math.degrees(angulo)
    return 180-angulo_graus