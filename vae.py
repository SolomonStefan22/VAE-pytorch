import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.enc1 = nn.Linear(784, 400)
        self.enc_mu = nn.Linear(400, latent)
        self.enc_v = nn.Linear(400, latent)
        self.dec1 = nn.Linear(latent, 400)
        self.dec2 = nn.Linear(400, 784)

        for layer in (self.enc1, self.enc_mu, self.enc_v, self.dec1, self.dec2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        return self.enc_mu(h), self.enc_v(h)

    def decode(self, z):
        h = F.relu(self.dec1(z))
        return self.dec2(h)

    def forward(self, x):
        mu, v = self.encode(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * v) * eps
        return self.decode(z), mu, v


class ConvVAE(nn.Module):

    def __init__(self, latent):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_fc = nn.Linear(64 * 7 * 7, 256)
        self.enc_mu = nn.Linear(256, latent)
        self.enc_v = nn.Linear(256, latent)
        self.dec_fc1 = nn.Linear(latent, 256)
        self.dec_fc2 = nn.Linear(256, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        h = x.view(-1, 1, 28, 28)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.enc_fc(h.flatten(1)))
        return self.enc_mu(h), self.enc_v(h)

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = F.relu(self.deconv1(h.view(-1, 64, 7, 7)))
        return self.deconv2(h).flatten(1)

    def forward(self, x):
        mu, v = self.encode(x)
        z = mu + torch.exp(0.5 * v) * torch.randn_like(mu)
        return self.decode(z), mu, v


def vae_loss(logits, x, mu, v):
    B = x.shape[0]
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / B
    kl = 0.5 * torch.sum(torch.exp(v) + mu ** 2 - 1.0 - v) / B
    return rec, kl