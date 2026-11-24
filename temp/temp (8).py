
# Simple Neural Network using PyTorch

#1. Aim
#To build and train a simple feed-forward neural network in PyTorch that classifies handwritten digits (0 to 9).

# Task: classify handwritten digits (0-9) from the scikit-learn digits dataset (64 pixel features)


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

torch.manual_seed(42)

digits = load_digits()
X, y = digits.data, digits.target
print('Dataset shape:', X.shape, ' Classes:', len(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Network: 64 -> 64 -> 32 -> 10
model = nn.Sequential(
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10))

print(model)
print('Trainable parameters:', sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs, losses, accs = 50, [], []
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    losses.append(loss.item())
    accs.append(acc)

    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d} Loss: {loss.item():.4f} Test accuracy: {acc:.4f}')

with torch.no_grad():
    pred = model(X_test).argmax(1).numpy()

print('\nFinal test accuracy: %.4f' % accuracy_score(y_test, pred))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, pred))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')

axes[1].plot(accs, color='green')
axes[1].set_title('Test Accuracy')
axes[1].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig('figs/exp8_nn.png', dpi=110)
plt.show()
