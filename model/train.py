import json
import torch

from model import SignLanguageModel

with open("dataset/labels.json", "r") as f:
    dataset = json.load(f)


X = []
y = []
for sample in dataset:

    X.append(sample["features"])

    y.append(sample["label"])

labels = sorted(list(set(y)))
label_to_index = {
    label: index
    for index, label in enumerate(labels)
}

y = [label_to_index[label] for label in y]

X = torch.tensor(X, dtype=torch.float32)

y = torch.tensor(y, dtype=torch.long)

print(X.shape)
print(y.shape)

model = SignLanguageModel()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.00033
)

epochs = 6000

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(
    f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}"
)
    
with torch.no_grad():

    outputs = model(X)

    predictions = torch.argmax(outputs, dim=1)

    accuracy = (predictions == y).float().mean()

    print(f"Acurácia: {accuracy * 100:.2f}%")
torch.save(model.state_dict(), "model/signum_model.pth")