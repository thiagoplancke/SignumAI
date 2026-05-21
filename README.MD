# SignumAI

Sistema de reconhecimento de gestos utilizando Visão Computacional, MediaPipe e PyTorch.

O SignumAI é um projeto educacional e experimental criado para entender, na prática, como construir um pipeline de reconhecimento de gestos com câmera, extração geométrica de pontos da mão, normalização dos dados, treinamento supervisionado e inferência em tempo real.

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

Em vez de treinar uma rede neural diretamente com imagens da câmera, o projeto usa o MediaPipe para transformar a mão em pontos. Esses pontos descrevem a posição dos dedos, do pulso e das articulações. Depois disso, o SignumAI normaliza esses pontos e usa uma rede neural simples para classificar o gesto em uma das letras do alfabeto.

Essa abordagem deixa o projeto mais leve, mais interpretável e mais fácil de estudar, porque a rede aprende a geometria da mão em vez de aprender pixels crus da imagem.

---

# Tecnologias Utilizadas

## OpenCV

Responsável por:

* captura da câmera
* exibição de frames
* desenho na tela
* leitura de teclado

No projeto, o OpenCV abre a webcam, lê cada frame, mostra a janela de predição e escreve textos na tela, como a letra reconhecida e a confiança da rede.

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

Essas 63 features são a entrada da rede neural.

O MediaPipe não retorna uma imagem recortada da mão. Ele retorna pontos estruturados. Cada ponto representa uma região importante da mão, como pulso, articulações e pontas dos dedos.

---

## PyTorch

Responsável por:

* criação da rede neural
* treinamento
* inferência
* classificação das letras

O PyTorch recebe o vetor de 63 valores normalizados e produz 26 saídas, uma para cada letra do alfabeto.

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
│   ├── augmentation.py
│   ├── modelo.py
│   ├── train.py
│   ├── predict.py
│   ├── hand_landmarker.task
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

Observação: alguns arquivos surgiram em etapas anteriores de estudo, como cálculo de ângulos e suavização manual. Eles representam experimentos iniciais do projeto. O fluxo principal atual usa principalmente `main.py`, `capture/`, `tracking/`, `geometry/normalizar.py`, `dataset/`, e `model/`.

---

# Responsabilidade de Cada Parte

## `main.py`

Arquivo principal para rodar a predição em tempo real.

Fluxo atual:

```text
executa main.py
↓
usuário pressiona P no terminal
↓
câmera é aberta
↓
MediaPipe detecta a mão
↓
modelo classifica a letra
↓
resultado aparece na tela
```

Atualmente o `main.py` é focado apenas em predição. A coleta e o treinamento ficam separados em arquivos próprios.

---

## `capture/camera.py`

Responsável por abrir a webcam padrão:

```python
cv.VideoCapture(0)
```

O índice `0` normalmente representa a câmera principal do computador.

---

## `tracking/hand_tracker.py`

Responsável por carregar o modelo do MediaPipe e detectar landmarks da mão.

Ele usa o arquivo:

```text
model/hand_landmarker.task
```

Esse arquivo é o modelo de detecção de mãos utilizado pelo MediaPipe.

O módulo faz três coisas principais:

* converte o frame de BGR para RGB
* envia o frame para o detector do MediaPipe
* desenha os landmarks detectados na tela

---

## `geometry/normalizar.py`

Responsável por transformar os landmarks do MediaPipe em dados numéricos mais estáveis para o modelo.

Esse arquivo é uma das partes mais importantes do projeto, porque a qualidade da normalização influencia diretamente a qualidade da classificação.

---

## `dataset/collect.py`

Responsável por criar amostras para o dataset.

Cada amostra salva:

* a letra correspondente
* as 63 features normalizadas da mão

---

## `model/modelo.py`

Responsável pela arquitetura da rede neural.

Define o modelo que recebe as 63 features e devolve 26 valores, um para cada letra.

---

## `model/train.py`

Responsável por treinar a rede neural usando o dataset salvo em:

```text
dataset/labels.json
```

Ele carrega as amostras, aplica augmentation, transforma letras em índices numéricos, treina o modelo e salva os pesos em:

```text
model/signum_model.pth
```

---

## `model/predict.py`

Responsável por carregar o modelo treinado e fazer predição.

Ele também aplica:

* `softmax` para calcular confiança
* filtro temporal para estabilizar a letra exibida

---

## `model/augmentation.py`

Responsável por criar pequenas variações artificiais nos dados de treinamento.

Isso ajuda o modelo a generalizar melhor e não decorar exatamente as amostras coletadas.

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

Em termos práticos:

```text
imagem da câmera
↓
21 pontos da mão
↓
63 números
↓
63 números normalizados
↓
rede neural
↓
letra prevista
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

Cada ponto possui:

```text
x -> posição horizontal
y -> posição vertical
z -> profundidade relativa
```

Esses pontos permitem representar a mão como uma estrutura geométrica.

---

# Normalização dos Dados

A normalização foi criada para permitir que o modelo reconheça gestos independentemente:

* da posição da mão
* da distância da câmera
* da escala da imagem

Sem normalização, o mesmo gesto poderia gerar valores muito diferentes só porque a mão apareceu mais para a esquerda, mais para a direita, mais perto ou mais longe da câmera.

O objetivo da normalização é fazer o modelo aprender a forma relativa da mão, e não a posição absoluta dela na imagem.

---

## Etapa 1 — Centralização

O pulso (landmark 0) é transformado na origem:

```text
(x - x0, y - y0, z - z0)
```

Isso remove dependência de posição.

Exemplo conceitual:

```text
pulso antes:  (0.50, 0.80, -0.02)
pulso depois: (0.00, 0.00,  0.00)
```

Depois dessa etapa, todos os pontos passam a ser descritos em relação ao pulso.

Isso significa que a mesma mão no canto esquerdo ou no canto direito da tela tende a gerar uma representação parecida.

---

## Etapa 2 — Escala

Todos os pontos são divididos por uma distância de referência.

Isso remove dependência de tamanho da mão e distância da câmera.

No código atual, o ponto de referência usado é o landmark `9`, que fica na região da base do dedo médio.

A distância é calculada assim:

```text
distância = √(x² + y² + z²)
```

Depois cada coordenada é dividida por essa distância:

```text
x_normalizado = x / distância
y_normalizado = y / distância
z_normalizado = z / distância
```

Com isso, uma mão mais perto da câmera e uma mão mais longe passam a ter proporções mais comparáveis.

---

## Resultado da Normalização

Depois da normalização, o projeto transforma os pontos em uma lista única:

```text
[(x, y, z), (x, y, z), ...]
```

vira:

```text
[x, y, z, x, y, z, ...]
```

Essa lista final tem 63 valores e é usada como entrada da rede neural.

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

No estado atual do projeto, o dataset principal está em:

```text
dataset/labels.json
```

Ele contém amostras das letras de `A` até `Z`. Cada letra representa uma classe do problema de classificação.

---

# Estratégia de Coleta

Para melhorar generalização:

* foram utilizadas mão esquerda e direita
* diferentes posições
* diferentes rotações
* diferentes distâncias da câmera
* pequenas variações naturais

Isso ajuda o modelo a aprender padrões reais.

O objetivo da coleta não é fazer todas as amostras ficarem idênticas. Pelo contrário: é importante que exista variação natural, porque na inferência real a mão nunca aparece exatamente igual ao dataset.

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

Essa estratégia cria várias amostras parecidas, mas não idênticas, para a mesma letra.

---

# Augmentation dos Dados

Além da variação natural da coleta, o projeto usa augmentation durante o treinamento.

O augmentation está em:

```text
model/augmentation.py
```

Ele cria pequenas alterações artificiais nos dados para deixar o modelo menos rígido.

As alterações aplicadas são:

* pequena rotação
* pequena mudança de escala
* pequeno ruído nas coordenadas

---

## Rotação

O código escolhe um ângulo aleatório pequeno:

```python
angle = random.uniform(-0.15, 0.15)
```

Esse valor está em radianos. Aproximadamente, isso representa uma rotação entre `-8.6°` e `+8.6°`.

Essa rotação simula o caso em que o usuário faz a mesma letra com a mão um pouco inclinada.

---

## Escala

O código escolhe uma escala aleatória:

```python
scale = random.uniform(0.95, 1.05)
```

Isso deixa os pontos até 5% menores ou 5% maiores.

Essa etapa ajuda a simular pequenas diferenças de distância, tamanho da mão ou variação do detector.

---

## Ruído

O código adiciona pequenos valores aleatórios em `x`, `y` e `z`:

```python
random.uniform(-0.02, 0.02)
```

Esse ruído simula a instabilidade natural dos landmarks. Mesmo com a mão parada, o MediaPipe pode gerar pontos levemente diferentes entre um frame e outro.

---

## Por Que Augmentation Ajuda?

Sem augmentation, o modelo pode memorizar demais as amostras originais.

Com augmentation, ele aprende que pequenas diferenças ainda representam a mesma letra.

Exemplo:

```text
A levemente inclinado
A um pouco menor
A com pequeno ruído nos pontos
```

continuam sendo:

```text
A
```

---

# Arquitetura da Rede Neural

O modelo foi implementado em PyTorch.

Estrutura:

Estrutura atual do código:

```python
nn.Linear(63, 256)
ReLU()

nn.Linear(256, 128)
ReLU()

nn.Linear(128, 64)
ReLU()

nn.Linear(64, 26)
```

A entrada possui 63 valores:

```text
21 landmarks × 3 coordenadas = 63 features
```

A saída possui 26 valores:

```text
A-Z = 26 classes
```

Cada valor de saída representa a pontuação da rede para uma letra.

Registro anterior da arquitetura durante o desenvolvimento:

```python
nn.Linear(63, 128)
ReLU()

nn.Linear(128, 192)
ReLU()

nn.Linear(192, 64)
ReLU()

nn.Linear(64, 26)
```

Essa estrutura representa uma etapa anterior do projeto. A versão atual do arquivo `model/modelo.py` usa a arquitetura descrita acima como estrutura atual.

---

# Treinamento

O treinamento utiliza:

* CrossEntropyLoss
* Adam Optimizer
* treinamento supervisionado

Fluxo do treinamento:

```text
carrega dataset/labels.json
↓
aplica augmentation nas features
↓
transforma letras em números
↓
cria tensores PyTorch
↓
treina a rede neural
↓
calcula acurácia
↓
salva model/signum_model.pth
```

A função de perda usada é `CrossEntropyLoss`, comum em problemas de classificação multiclasse.

O otimizador usado é Adam, que ajusta os pesos da rede neural a cada época de treinamento.

---

# Resultado Inicial

Após ajustes de arquitetura e treinamento:

```text
Acurácia ~96%
```

Esse resultado indica que, no conjunto usado para treinamento, a rede conseguiu aprender bem os padrões das letras.

Como o projeto é experimental, essa acurácia deve ser interpretada com cuidado: para medir generalização real, o ideal seria separar dados de treino e teste.

---

# Inferência em Tempo Real

Durante a inferência, o projeto:

```text
abre a câmera
↓
detecta a mão
↓
normaliza os landmarks
↓
envia as 63 features ao modelo
↓
calcula a letra mais provável
↓
mostra letra e confiança na tela
```

O arquivo responsável por carregar o modelo e prever letras é:

```text
model/predict.py
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

Mesmo uma pequena mudança nos pontos detectados pode alterar a saída da rede por alguns frames.

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

No código atual, o histórico mantém as últimas previsões confiáveis e escolhe a letra mais frequente dentro desse histórico.

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

No código atual, a letra só entra no histórico de estabilização quando a confiança passa de um limite mínimo. Isso evita que previsões fracas contaminem o resultado final.

---

# Execução

## Instalar dependências

Com `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ou instalando as dependências principais manualmente:

```bash
pip install opencv-python mediapipe torch numpy
```

---

# Rodar predição em tempo real

```bash
python main.py
```

Ao iniciar, o programa pede:

```text
Pressione P para iniciar a predicao.
```

Depois de digitar `P`, a câmera é aberta e a predição começa.

Teclas durante a janela da câmera:

```text
1 -> sair
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

Observação: esta foi a ideia original documentada para a coleta. No estado atual do projeto, o `main.py` está focado em predição. A lógica de coleta fica separada no módulo `dataset/collect.py`.

---

# Treinar modelo

```bash
python -m model.train
```

Esse comando lê o dataset, treina a rede e atualiza o arquivo:

```text
model/signum_model.pth
```

---

# Rodar inferência realtime

```bash
python main.py
```

O arquivo `model/predict.py` contém a lógica de inferência, mas o ponto de entrada recomendado para usar a câmera é o `main.py`.

Também é possível chamar diretamente o módulo de predição:

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

Além disso, o projeto também ajuda a entender a diferença entre:

* trabalhar com imagem bruta
* trabalhar com features extraídas
* treinar modelos supervisionados
* aplicar pré-processamento antes da rede neural
* estabilizar uma saída em tempo real

---

# Limitações Atuais

O projeto reconhece letras individuais.

Ele ainda não reconhece Libras completa, porque Libras real envolve:

* movimento
* expressão
* contexto
* sequência temporal
* variações naturais entre pessoas

Além disso, algumas letras podem ser visualmente parecidas dependendo do ângulo da câmera e da posição da mão.

---

# Próximos Passos

## Melhorias de Classificação

* aumentar robustez
* adicionar mais dados
* melhorar normalização
* adicionar augmentations

Outras possíveis melhorias:

* separar dataset em treino e teste
* salvar métricas de treinamento
* gerar matriz de confusão
* avaliar quais letras mais confundem o modelo
* criar uma interface visual mais completa
* permitir troca entre modelos treinados

---

## Reconhecimento de Palavras

Atualmente o sistema reconhece letras individuais.

Próximo passo:

```text
formação de palavras
```

Isso exigiria acumular letras reconhecidas ao longo do tempo e criar regras para confirmar quando uma letra deve ser adicionada à palavra.

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

Para sinais dinâmicos, um único frame não é suficiente. O modelo precisaria observar uma sequência de frames para entender o movimento.

---

# Objetivo Educacional

O projeto foi desenvolvido com foco em:

* aprendizado profundo dos conceitos
* construção manual do pipeline
* entendimento da matemática e arquitetura
* evolução incremental do sistema

Ao invés de apenas utilizar modelos prontos.

O valor principal do projeto está em mostrar cada etapa do processo:

```text
câmera
↓
detecção
↓
normalização
↓
dataset
↓
treinamento
↓
predição
↓
estabilização
```

---

# Autor

Projeto desenvolvido por Thiago.

---

# Licença

Projeto educacional e experimental.
