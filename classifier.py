import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist import load_images, load_labels


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        h = x.view(-1, 1, 28, 28)
        h = F.max_pool2d(F.relu(self.conv1(h)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.fc1(h.flatten(1)))
        return self.fc2(h)


def main():
    torch.manual_seed(0)
    os.makedirs("models", exist_ok=True)

    X_train = load_images("data/train-images-idx3-ubyte")
    y_train = load_labels("data/train-labels-idx1-ubyte")
    X_test = load_images("data/t10k-images-idx3-ubyte")
    y_test = load_labels("data/t10k-labels-idx1-ubyte")
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)

    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 4):
        total = 0.0
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        with torch.no_grad():
            accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
        print(f"Epoca {epoch}: cost {total / len(loader):.4f}, corect pe test {100 * accuracy:.2f}%")

    torch.save(model.state_dict(), "models/classifier.pt")
    print("Am salvat models/classifier.pt")


if __name__ == "__main__":
    main()