import math
import sys

import torch
import torch.nn.functional as F

from classifier import Classifier
from mnist import load_images
from vae import VAE, ConvVAE

N = 1000


def report(title, probs):
    conf, pred = probs.max(dim=1)
    counts = torch.bincount(pred, minlength=10)
    q = counts.float() / len(pred)
    H = -sum(qk * math.log(qk) for qk in q.tolist() if qk > 0)
    print(title)
    print(f"  increderea medie:          {conf.mean().item():.3f}")
    print(f"  cifre clare (peste 0,9):   {100 * (conf > 0.9).float().mean().item():.1f}%")
    print("  cate din fiecare cifra:    " + " ".join(f"{k}:{c}" for k, c in enumerate(counts.tolist())))
    print(f"  numarul efectiv de cifre:  {math.exp(H):.2f} (din 10)\n")


def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    kinds = sys.argv[2:] if len(sys.argv) > 2 else ["mlp", "conv"]

    clf = Classifier()
    clf.load_state_dict(torch.load("models/classifier.pt"))
    clf.eval()

    with torch.no_grad():
        real = load_images("data/t10k-images-idx3-ubyte")[:N]
        report("Imagini reale (reperul):", F.softmax(clf(real), dim=1))

        for kind in kinds:
            model = ConvVAE(D) if kind == "conv" else VAE(D)
            model.load_state_dict(torch.load(f"models/vae_{kind}_d{D}.pt"))
            torch.manual_seed(1)
            samples = torch.sigmoid(model.decode(torch.randn(N, D)))
            report(f"Cifre generate, {kind}_d{D}:", F.softmax(clf(samples), dim=1))


if __name__ == "__main__":
    main()