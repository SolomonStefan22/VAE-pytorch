import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from mnist import load_images
from vae import VAE, ConvVAE, vae_loss


def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    kind = sys.argv[2] if len(sys.argv) > 2 else "mlp"
    if kind not in ("mlp", "conv"):
        sys.exit("al doilea argument trebuie sa fie 'mlp' sau 'conv'")
    tag = f"{kind}_d{D}"
    print(f"Model: {kind}, dimensiunea codului: {D}")

    torch.manual_seed(0)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train = load_images("data/train-images-idx3-ubyte")
    X_test = load_images("data/t10k-images-idx3-ubyte")
    train_loader = DataLoader(TensorDataset(X_train), batch_size=100, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=100, shuffle=False)

    model = ConvVAE(D) if kind == "conv" else VAE(D)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        start = time.perf_counter()
        sum_rec = sum_kl = 0.0

        for (x,) in train_loader:
            logits, mu, v = model(x)
            rec, kl = vae_loss(logits, x, mu, v)
            optimizer.zero_grad()
            (rec + kl).backward()
            optimizer.step()
            sum_rec += rec.item()
            sum_kl += kl.item()

        seconds = time.perf_counter() - start
        n = len(train_loader)

        with torch.no_grad():
            test_sum = 0.0
            for (x,) in test_loader:
                logits, mu, v = model(x)
                rec, kl = vae_loss(logits, x, mu, v)
                test_sum += (rec + kl).item()
            test_total = test_sum / len(test_loader)

        print(f"Epoca {epoch}: refacere {sum_rec / n:.2f}, KL {sum_kl / n:.2f}, " f"total antrenare {(sum_rec + sum_kl) / n:.2f}, total test {test_total:.2f}, " f"timp {seconds:.1f} s")

    with torch.no_grad():
        x16 = X_test[:16]
        mu, _ = model.encode(x16)
        r16 = torch.sigmoid(model.decode(mu))
        save_image(torch.cat([x16, r16]).view(-1, 1, 28, 28), f"results/reconstructii_{tag}.png", nrow=16)

        z = torch.randn(64, D)
        samples = torch.sigmoid(model.decode(z))
        save_image(samples.view(-1, 1, 28, 28), f"results/generate_{tag}.png", nrow=8)

    torch.save(model.state_dict(), f"models/vae_{tag}.pt")
    print(f"Am salvat results/*_{tag}.png si models/vae_{tag}.pt")


if __name__ == "__main__":
    main()