
# Convolutional Neural Network (CNN) for image classification using PyTorch

# 1. Aim
# To build and train a Convolutional Neural Network (CNN) in PyTorch for classifying handwritten digit images.

# Images: 8x8 grayscale handwritten digits (0-9) from scikit-learn

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

torch.manual_seed(42)

digits = load_digits()
images = digits.images / 16.0  # scale pixels to 0-1

X_train, X_test, y_train, y_test = train_test_split(
    images, digits.target,
    test_size=0.2,
    random_state=42,
    stratify=digits.target
)

# shape (N, 1 channel, 8, 8)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print('Train images:', tuple(X_train.shape),
      ' Test images:', tuple(X_test.shape))

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

history = []

for epoch in range(1, 16):
    model.train()
    total = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    history.append((total / len(X_train), acc))

    if epoch % 3 == 0:
        print(f'Epoch {epoch:2d} Loss: {total / len(X_train):.4f} Test accuracy: {acc:.4f}')

with torch.no_grad():
    pred = model(X_test).argmax(1)

print('\nFinal test accuracy: %.4f' % accuracy_score(y_test, pred))
print('\nClassification Report:')
print(classification_report(y_test, pred, digits=3))

fig, axes = plt.subplots(2, 8, figsize=(11, 3.4))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i, 0], cmap='gray_r')
    ok = pred[i] == y_test[i]
    ax.set_title(
        f'P:{pred[i].item()} A:{y_test[i].item()}',
        fontsize=9,
        color='green' if ok else 'red'
    )
    ax.axis('off')

plt.suptitle('CNN predictions on test images (P = predicted, A = actual)')
plt.tight_layout()
plt.savefig('figs/exp9_cnn.png', dpi=110)
plt.show()
