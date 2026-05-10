# SignumAI

Sistema de reconhecimento de gestos utilizando Visão Computacional, MediaPipe e PyTorch.

---

# Visão Geral

O SignumAI é um projeto de Inteligência Artificial focado em reconhecimento de gestos de mãos utilizando:

* OpenCV
* MediaPipe
* PyTorch
* Redes Neurais
* Processamento em tempo real

O objetivo inicial do projeto foi construir um detector de letras do alfabeto manual de Libras.

Durante o desenvolvimento, o projeto evoluiu para:

* Extração geométrica de landmarks
* Normalização espacial
* Criação automática de dataset
* Treinamento supervisionado
* Inferência em tempo real
* Estabilização temporal

---

# Tecnologias Utilizadas

## OpenCV

Responsável por:

* captura da câmera
* exibição de frames
* desenho na tela
* leitura de teclado

---

## MediaPipe

Responsável por:

* detectar mãos
* extrair landmarks
* fornecer coordenadas 3D

Cada mão possui:

* 21 landmarks
* coordenadas X, Y e Z

Total:

```text
21 × 3 = 63 features
```

---

## PyTorch

Responsável por:

* criação da rede neural
* treinamento
* inferência
* classificação das letras

---

# Estrutura do Projeto

```text
SignumAI/
│
├── capture/
│   └── camera.py
│
├── tracking/
│   └── hand_tracker.py
│
├── geometry/
│   ├── angulo.py
│   └── normalizar.py
│
├── dataset/
│   ├── collect.py
│   └── labels.json
│
├── model/
│   ├── modelo.py
│   ├── train.py
│   ├── predict.py
│   ├── signum_model.pth
│   └── label_encoder.json
│
├── utils/
│   └── smoothing.py
│
├── main.py
│
└── README.md
```

---

# Pipeline do Sistema

O fluxo completo do sistema funciona assim:

```text
Câmera
↓
MediaPipe detecta mão
↓
Extração dos landmarks
↓
Normalização espacial
↓
Transformação em vetor
↓
Rede neural
↓
Classificação da letra
↓
Filtro temporal
↓
Resultado final
```

---

# Landmarks da Mão

O MediaPipe retorna 21 pontos por mão.

Exemplo:

```text
0  -> pulso
1  -> base do polegar
4  -> ponta do polegar
8  -> ponta do indicador
12 -> ponta do médio
16 -> ponta do anelar
20 -> ponta do mindinho
```

---

# Normalização dos Dados

A normalização foi criada para permitir que o modelo reconheça gestos independentemente:

* da posição da mão
* da distância da câmera
* da escala da imagem

---

## Etapa 1 — Centralização

O pulso (landmark 0) é transformado na origem:

```text
(x - x0, y - y0, z - z0)
```

Isso remove dependência de posição.

---

## Etapa 2 — Escala

Todos os pontos são divididos por uma distância de referência.

Isso remove dependência de tamanho da mão e distância da câmera.

---

# Dataset

O dataset foi criado manualmente.

Cada amostra contém:

```json
{
  "label": "A",
  "features": [ ... 63 valores ... ]
}
```

---

# Estratégia de Coleta

Para melhorar generalização:

* foram utilizadas mão esquerda e direita
* diferentes posições
* diferentes rotações
* diferentes distâncias da câmera
* pequenas variações naturais

Isso ajuda o modelo a aprender padrões reais.

---

# Coleta Automática

O sistema possui coleta automática.

Ao pressionar uma tecla:

```text
A-Z
```

O sistema salva múltiplos frames automaticamente.

Exemplo:

```text
pressiona A
↓
50 amostras são salvas
```

Enquanto isso, o usuário move levemente a mão.

---

# Arquitetura da Rede Neural

O modelo foi implementado em PyTorch.

Estrutura:

```python
nn.Linear(63, 128)
ReLU()

nn.Linear(128, 192)
ReLU()

nn.Linear(192, 64)
ReLU()

nn.Linear(64, 26)
```

---

# Treinamento

O treinamento utiliza:

* CrossEntropyLoss
* Adam Optimizer
* treinamento supervisionado

---

# Resultado Inicial

Após ajustes de arquitetura e treinamento:

```text
Acurácia ~96%
```

---

# Problema de Oscilação

Durante inferência realtime, o modelo oscilava letras rapidamente.

Exemplo:

```text
A A B A C A A
```

Isso ocorre devido:

* ruído da câmera
* variação dos landmarks
* pequenas mudanças de pose

---

# Solução — Estabilização Temporal

Foi implementado:

```text
votação temporal
```

O sistema:

* guarda últimos frames
* conta letras mais frequentes
* escolhe a previsão dominante

Isso torna o resultado muito mais estável.

---

# Confiança da Rede

Também foi implementado:

```python
softmax()
```

para medir confiança da previsão.

Quando confiança é baixa:

```text
?
```

é mostrado ao invés de uma letra incorreta.

---

# Execução

## Instalar dependências

```bash
pip install opencv-python mediapipe torch numpy
```

---

# Rodar coleta de dataset

```bash
python main.py
```

Teclas:

```text
A-Z -> salvar gesto
1   -> sair
```

---

# Treinar modelo

```bash
python -m model.train
```

---

# Rodar inferência realtime

```bash
python -m model.predict
```

---

# Conceitos Aprendidos

Durante o desenvolvimento foram explorados:

* visão computacional
* landmarks 3D
* geometria computacional
* normalização espacial
* engenharia de features
* redes neurais
* inferência realtime
* suavização temporal
* pipelines de ML
* generalização
* criação de datasets

---

# Próximos Passos

## Melhorias de Classificação

* aumentar robustez
* adicionar mais dados
* melhorar normalização
* adicionar augmentations

---

## Reconhecimento de Palavras

Atualmente o sistema reconhece letras individuais.

Próximo passo:

```text
formação de palavras
```

---

## Libras Real

Libras real utiliza:

* movimento
* sequência temporal
* dinâmica da mão

Isso exigirá:

* datasets temporais
* múltiplos frames
* modelos sequenciais

Exemplos futuros:

* LSTM
* GRU
* Transformers

---

# Objetivo Educacional

O projeto foi desenvolvido com foco em:

* aprendizado profundo dos conceitos
* construção manual do pipeline
* entendimento da matemática e arquitetura
* evolução incremental do sistema

Ao invés de apenas utilizar modelos prontos.

---

# Autor

Projeto desenvolvido por Thiago.

---

# Licença

Projeto educacional e experimental.
