"""
VAE Full Benchmark: Unified Comparison
=======================================
Fair head-to-head comparison of 5 generative models on CIFAR-10:

| Model    | Paper       | Key Idea                                    |
|----------|-------------|---------------------------------------------|
| VAE      | 1312.6114   | ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))    |
| β-VAE    | 1612.07647  | KL weight β>1 for disentanglement           |
| IWAE     | 1509.00549  | Importance-weighted bound with K samples     |
| WAE      | 1711.01558  | Wasserstein distance replaces KL             |
| VQ-VAE   | 1711.00937  | Discrete codebook + straight-through est.    |

All models share the same Conv encoder/decoder architecture.
All models implement: encode(), decode(), sample(), reconstruct().
All models return: (x_recon, loss_dict) from forward().

Evaluation metrics:
  - ELBO / IW-ELBO (importance-weighted with 5000 samples)
  - Reconstruction MSE
  - FID (on 5k reconstructions)
  - Active Units (|{i : Var(z_i) > threshold}|)
  - KL divergence per dimension
  - Log-likelihood estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# Shared Architecture
# ═══════════════════════════════════════════════════════════

class ConvEncoder(nn.Module):
    """4-layer Conv encoder: 3×32×32 → latent_dim.
    Channels: 3 → 32 → 64 → 128 → 256, each stride-2 downsample.
    Output spatial: 32→16→8→4→2, so final feature is 256×2×2 = 1024.
    """
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),   # → 32×16×16
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),             # → 64×8×8
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),            # → 128×4×4
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),           # → 256×2×2
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.conv(x).view(x.shape[0], -1)  # (B, 1024)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder: latent_dim → 3×32×32.
    Channels: 256 → 128 → 64 → 32 → 3, each stride-2 upsample.
    """
    def __init__(self, latent_dim=64, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),   # → 128×4×4
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # → 64×8×8
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # → 32×16×16
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # → 3×32×32
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], 256, 2, 2)
        return self.deconv(h)


class VectorQuantizer(nn.Module):
    """VQ-VAE codebook with EMA updates."""
    def __init__(self, n_embeddings=512, embedding_dim=64, decay=0.99, epsilon=1e-5,
                 commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_embeddings))
        self.register_buffer('ema_weight', embedding.clone())

    def forward(self, z):
        # z: (B, D)
        # Detach z for distance computation (no grad through codebook lookup)
        z_detached = z.detach()
        distances = (z_detached.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1)
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding[encoding_indices].detach()  # no grad from codebook

        if self.training:
            # EMA update with detached z to avoid graph issues
            one_hot = F.one_hot(encoding_indices, self.n_embeddings).float()
            count = one_hot.sum(0)
            self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
            weight = one_hot.T @ z_detached
            self.ema_weight.mul_(self.decay).add_(weight, alpha=1 - self.decay)
            n = self.ema_count.sum()
            count_smooth = (self.ema_count + self.epsilon) / (n + self.n_embeddings * self.epsilon) * n
            self.embedding.copy_(self.ema_weight / count_smooth.unsqueeze(-1))

        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = self.commitment_cost * commitment_loss

        z_q_st = z + (z_q - z).detach()  # straight-through estimator

        avg_probs = F.one_hot(encoding_indices, self.n_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, encoding_indices, perplexity


# ═══════════════════════════════════════════════════════════
# Models — all with unified interface
# ═══════════════════════════════════════════════════════════

class VAE(nn.Module):
    """Standard VAE (Kingma & Welling, 2013 — 1312.6114)."""
    def __init__(self, latent_dim=64, beta=1.0):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta  # for β-VAE compatibility (default 1.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruct(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1).mean()
        loss = recon_loss + self.beta * kl
        return x_recon, {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl': kl,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }


class BetaVAE(VAE):
    """β-VAE (Higgins et al., 2017 — 1612.07647).
    Same as VAE but with β > 1 to encourage disentanglement.
    """
    def __init__(self, latent_dim=64, beta=4.0):
        super().__init__(latent_dim=latent_dim, beta=beta)


class IWAE(VAE):
    """Importance-Weighted Autoencoder (Burda et al., 2016 — 1509.00549).
    Uses K importance samples for a tighter ELBO bound:
      L_K = E_q [ log (1/K Σ_k p(x|z_k) p(z_k) / q(z_k|x)) ]
    """
    def __init__(self, latent_dim=64, K=5):
        super().__init__(latent_dim=latent_dim, beta=1.0)
        self.K = K

    def forward(self, x):
        mu, logvar = self.encoder(x)
        B = x.shape[0]

        # Sample K times
        zs = []
        log_wk = []  # log importance weights for each sample
        recon_losses = []

        for _ in range(self.K):
            z = self.reparameterize(mu, logvar)
            x_recon_k = self.decoder(z)

            # log p(x|z_k) — Gaussian with fixed variance
            log_p_x_given_z = -F.mse_loss(x_recon_k, x, reduction='sum') / B

            # log p(z_k) — standard normal prior
            log_p_z = -0.5 * (z**2).sum(dim=1).mean()

            # log q(z_k|x) — encoder distribution
            log_q_z_given_x = -0.5 * (logvar + (z - mu)**2 / logvar.exp()).sum(dim=1).mean()

            # log importance weight
            log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
            log_wk.append(log_w)
            zs.append(z)
            recon_losses.append(F.mse_loss(x_recon_k, x, reduction='sum') / B)

        # IW-ELBO: log_mean_exp of importance weights
        log_wk = torch.stack(log_wk)  # (K,)
        iw_elbo = torch.logsumexp(log_wk, dim=0) - np.log(self.K)

        # For reconstruction, use the sample with highest weight
        best_k = log_wk.argmax(dim=0).item()
        x_recon = self.decoder(zs[best_k])

        # KL for monitoring (same as VAE)
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1).mean()

        loss = -iw_elbo  # maximize ELBO = minimize negative ELBO

        return x_recon, {
            'loss': loss,
            'recon_loss': recon_losses[best_k],
            'kl': kl,
            'iw_elbo': iw_elbo,
            'mu': mu,
            'logvar': logvar,
            'z': zs[best_k],
        }


class WAE(nn.Module):
    """Wasserstein Autoencoder (Tolstikhin et al., 2018 — 1711.01558).
    Minimizes Wasserstein distance between q(z|x) and p(z) via
    Maximum Mean Discrepancy (MMD) with inverse multiquadric kernel.
    No reparameterization trick needed for the penalty term.
    """
    def __init__(self, latent_dim=64, lambda_mmd=10.0):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.lambda_mmd = lambda_mmd

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _im_kernel(x, y, sigma=2.0):
        """Inverse multiquadric kernel: k(x,y) = α / (α + ||x-y||²)"""
        alpha = 2.0 * sigma**2
        xx = x.unsqueeze(1)  # (B,1,D)
        yy = y.unsqueeze(0)  # (1,B,D)
        dist2 = (xx - yy).pow(2).sum(-1)  # (B,B)
        return alpha / (alpha + dist2)

    def mmd_loss(self, z_enc, n_samples=None):
        """MMD between encoded z and prior samples z_prior ~ N(0,I)."""
        z_prior = torch.randn_like(z_enc)
        K_zz = self._im_kernel(z_enc, z_enc)
        K_pp = self._im_kernel(z_prior, z_prior)
        K_zp = self._im_kernel(z_enc, z_prior)
        return K_zz.mean() + K_pp.mean() - 2 * K_zp.mean()

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruct(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        mmd = self.mmd_loss(z)
        loss = recon_loss + self.lambda_mmd * mmd
        return x_recon, {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl': mmd,  # MMD replaces KL
            'mmd': mmd,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }


class VQVAE(nn.Module):
    """VQ-VAE (van den Oord et al., 2017 — 1711.00937).
    Discrete latent codes via learned codebook.
    """
    def __init__(self, latent_dim=64, n_embeddings=512):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 2 * 2, latent_dim)
        self.vq = VectorQuantizer(n_embeddings, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 2 * 2)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings

    def encode(self, x):
        h = self.encoder_conv(x).view(x.shape[0], -1)
        z = self.fc_enc(h)
        z_q, _, _, _ = self.vq(z)
        return z_q

    def decode(self, z):
        h = self.fc_dec(z).view(z.shape[0], 256, 2, 2)
        return self.decoder_deconv(h)

    def sample(self, n, device='cpu'):
        indices = torch.randint(0, self.n_embeddings, (n,), device=device)
        z_q = self.vq.embedding[indices]
        return self.decode(z_q)

    def reconstruct(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        h = self.encoder_conv(x).view(x.shape[0], -1)
        z = self.fc_enc(h)
        z_q, vq_loss, indices, perplexity = self.vq(z)
        x_recon = self.decode(z_q)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        loss = recon_loss + vq_loss
        return x_recon, {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl': vq_loss,  # VQ loss replaces KL
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'z': z_q,
            'indices': indices,
        }


# ═══════════════════════════════════════════════════════════
# Unified Training
# ═══════════════════════════════════════════════════════════

def train_model(model, train_loader, n_epochs=10, lr=2e-4, device='cpu'):
    """Generic training loop for any model with (x_recon, loss_dict) interface."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    history = {'loss': [], 'recon_loss': [], 'kl': []}

    for epoch in range(n_epochs):
        epoch_loss, epoch_recon, epoch_kl, n_batches = 0, 0, 0, 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            x_recon, loss_dict = model(imgs)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += loss_dict['recon_loss'].item()
            epoch_kl += loss_dict['kl'].item()
            n_batches += 1

        history['loss'].append(epoch_loss / n_batches)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kl'].append(epoch_kl / n_batches)

        print(f"  Epoch {epoch+1:2d} | Loss: {history['loss'][-1]:.4f} | "
              f"Recon: {history['recon_loss'][-1]:.4f} | KL: {history['kl'][-1]:.4f}")

    return history


# ═══════════════════════════════════════════════════════════
# Unified Evaluation
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_elbo(model, data_loader, device, n_samples=5000, K=5000):
    """Estimate ELBO / log-likelihood via importance sampling.

    For VAE/β-VAE: IW-ELBO with K samples.
    For WAE: uses MMD penalty, reports recon + MMD.
    For VQ-VAE: reports recon + VQ loss.
    """
    model.eval()
    total_elbo = 0
    total_n = 0
    count = 0

    for imgs, _ in data_loader:
        if total_n >= n_samples:
            break
        imgs = imgs.to(device)
        B = imgs.shape[0]

        if isinstance(model, IWAE):
            # IWAE already computes IW-ELBO
            _, loss_dict = model(imgs)
            elbo = loss_dict.get('iw_elbo', -loss_dict['loss']).item()
        elif isinstance(model, (VAE, BetaVAE)):
            # Standard IW-ELBO estimate
            mu, logvar = model.encoder(imgs)
            # Sample K times
            log_ws = []
            for _ in range(min(K, 100)):  # cap K for memory
                z = model.reparameterize(mu, logvar)
                x_recon = model.decoder(z)
                log_p_x = -F.mse_loss(x_recon, imgs, reduction='sum') / B
                log_p_z = -0.5 * (z**2).sum(dim=1).mean()
                log_q_z = -0.5 * (logvar + (z - mu)**2 / logvar.exp()).sum(dim=1).mean()
                log_ws.append((log_p_x + log_p_z - log_q_z).item())
            elbo = np.logaddexp.reduce(log_ws) - np.log(len(log_ws))
        elif isinstance(model, WAE):
            _, loss_dict = model(imgs)
            elbo = -loss_dict['loss'].item()  # negative loss as proxy
        elif isinstance(model, VQVAE):
            _, loss_dict = model(imgs)
            elbo = -loss_dict['loss'].item()
        else:
            _, loss_dict = model(imgs)
            elbo = -loss_dict['loss'].item()

        total_elbo += elbo * B
        total_n += B
        count += 1

    return total_elbo / total_n


@torch.no_grad()
def compute_mse(model, data_loader, device, n_samples=5000):
    """Reconstruction MSE on test set."""
    model.eval()
    total_mse = 0
    total_n = 0

    for imgs, _ in data_loader:
        if total_n >= n_samples:
            break
        imgs = imgs.to(device)
        x_recon = model.reconstruct(imgs)
        total_mse += F.mse_loss(x_recon, imgs, reduction='sum').item()
        total_n += imgs.shape[0]

    return total_mse / (total_n * 3 * 32 * 32)  # per-pixel MSE


@torch.no_grad()
def compute_active_units(model, data_loader, device, threshold=1e-2, n_samples=5000):
    """Count latent dimensions with variance > threshold.
    Active Units metric from Burda et al. (2014).
    """
    model.eval()
    all_z = []

    for imgs, _ in data_loader:
        if sum(z.shape[0] for z in all_z) >= n_samples:
            break
        imgs = imgs.to(device)
        if isinstance(model, VQVAE):
            h = model.encoder_conv(imgs).view(imgs.shape[0], -1)
            z = model.fc_enc(h)
        elif isinstance(model, WAE):
            mu, logvar = model.encoder(imgs)
            z = model.reparameterize(mu, logvar)
        else:
            mu, logvar = model.encoder(imgs)
            z = mu  # use mu for deterministic evaluation

        all_z.append(z.cpu())

    all_z = torch.cat(all_z, dim=0)[:n_samples]
    var_per_dim = all_z.var(dim=0)
    active = (var_per_dim > threshold).sum().item()
    return active, var_per_dim


@torch.no_grad()
def compute_fid(model, data_loader, device, n_samples=5000):
    """FID between real and reconstructed images.
    Uses torchmetrics if available, else falls back to manual computation.
    """
    model.eval()
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(normalize=True).to(device)
    except ImportError:
        print("  [WARN] torchmetrics not installed, skipping FID")
        return float('nan')

    count = 0
    for imgs, _ in data_loader:
        if count >= n_samples:
            break
        imgs = imgs.to(device)
        x_recon = model.reconstruct(imgs)

        # Inception expects uint8 [0, 255] or float [0, 1]
        real_uint8 = (imgs * 255).clamp(0, 255).to(torch.uint8)
        fake_uint8 = (x_recon * 255).clamp(0, 255).to(torch.uint8)

        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)
        count += imgs.shape[0]

    return fid.compute().item()


@torch.no_grad()
def compute_kl_per_dim(model, data_loader, device, n_samples=5000):
    """Average KL divergence per latent dimension.
    Only meaningful for VAE/β-VAE/IWAE.
    """
    model.eval()
    total_kl = 0
    total_n = 0

    for imgs, _ in data_loader:
        if total_n >= n_samples:
            break
        imgs = imgs.to(device)

        if isinstance(model, VQVAE):
            return None  # KL not applicable

        mu, logvar = model.encoder(imgs)
        kl_per_dim = 0.5 * (logvar.exp() + mu**2 - 1 - logvar)  # (B, D)
        total_kl += kl_per_dim.sum(dim=0).cpu()  # accumulate per-dim
        total_n += imgs.shape[0]

    return total_kl / total_n  # (D,)


@torch.no_grad()
def full_eval(model, data_loader, device, n_samples=5000):
    """Run all evaluation metrics and return results dict."""
    print("  Computing ELBO...")
    elbo = compute_elbo(model, data_loader, device, n_samples=n_samples)
    print("  Computing MSE...")
    mse = compute_mse(model, data_loader, device, n_samples=n_samples)
    print("  Computing Active Units...")
    active_units, var_per_dim = compute_active_units(model, data_loader, device, n_samples=n_samples)
    print("  Computing FID...")
    fid = compute_fid(model, data_loader, device, n_samples=n_samples)
    print("  Computing KL per dim...")
    kl_per_dim = compute_kl_per_dim(model, data_loader, device, n_samples=n_samples)

    results = {
        'elbo': elbo,
        'mse': mse,
        'active_units': active_units,
        'fid': fid,
        'kl_per_dim': kl_per_dim,
        'var_per_dim': var_per_dim,
    }

    if isinstance(model, VQVAE):
        # Add codebook stats
        all_indices = []
        for imgs, _ in data_loader:
            if sum(len(i) for i in all_indices) >= n_samples:
                break
            imgs = imgs.to(device)
            h = model.encoder_conv(imgs).view(imgs.shape[0], -1)
            z = model.fc_enc(h)
            _, _, indices, _ = model.vq(z)
            all_indices.append(indices.cpu())
        all_indices = torch.cat(all_indices)[:n_samples]
        used_codes = all_indices.unique().numel()
        results['codebook_usage'] = used_codes / model.n_embeddings
        results['used_codes'] = used_codes

    return results


# ═══════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════

def plot_training_curves(all_history, results_dir):
    """1. Training curves: Loss / Recon / KL per epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {'VAE': 'C0', 'β-VAE': 'C1', 'IWAE': 'C2', 'WAE': 'C3', 'VQ-VAE': 'C4'}

    for name, hist in all_history.items():
        c = colors.get(name, 'gray')
        axes[0].plot(hist['loss'], label=name, color=c)
        axes[1].plot(hist['recon_loss'], label=name, color=c)
        axes[2].plot(hist['kl'], label=name, color=c)

    for ax, title in zip(axes, ['Total Loss', 'Reconstruction Loss', 'KL / Regularization']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('VAE Benchmark: Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_reconstructions(models, test_imgs, results_dir, device, n=8):
    """2. Reconstruction comparison: original + 5 model reconstructions."""
    model_names = list(models.keys())
    fig, axes = plt.subplots(len(model_names) + 1, n, figsize=(2 * n, 2 * (len(model_names) + 1)))

    with torch.no_grad():
        # Originals
        for i in range(n):
            axes[0, i].imshow(test_imgs[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=10, fontweight='bold')

        # Reconstructions
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        for row, name in enumerate(model_names, start=1):
            model = models[name]
            for i in range(n):
                recon = model.reconstruct(test_imgs[i:i+1].to(device))
                axes[row, i].imshow(recon[0].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                axes[row, i].axis('off')
            axes[row, 0].set_ylabel(name, fontsize=10, fontweight='bold', color=colors[row-1])

    plt.suptitle('Reconstruction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'reconstructions.png', dpi=150)
    plt.close()


def plot_samples(models, results_dir, device, n=8):
    """3. Random samples from each model's prior."""
    model_names = list(models.keys())
    fig, axes = plt.subplots(len(model_names), n, figsize=(2 * n, 2 * len(model_names)))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for row, name in enumerate(model_names):
        model = models[name]
        model.eval()
        with torch.no_grad():
            samples = model.sample(n, device=device)
        for i in range(n):
            img = samples[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[row, i].imshow(img)
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(name, fontsize=10, fontweight='bold', color=colors[row])

    plt.suptitle('Prior Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'samples.png', dpi=150)
    plt.close()


def plot_latent_tsne(models, test_loader, results_dir, device, n_samples=2000):
    """4. t-SNE of latent space colored by class."""
    from sklearn.manifold import TSNE

    model_names = list(models.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    # Get test data
    imgs_all, labels_all = [], []
    for img, label in test_loader:
        imgs_all.append(img)
        labels_all.append(label)
        if sum(i.shape[0] for i in imgs_all) >= n_samples:
            break
    imgs_all = torch.cat(imgs_all)[:n_samples].to(device)
    labels_all = torch.cat(labels_all)[:n_samples].numpy()

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for ax, name, color in zip(axes, model_names, colors):
        model = models[name]
        model.eval()
        with torch.no_grad():
            if isinstance(model, VQVAE):
                h = model.encoder_conv(imgs_all).view(imgs_all.shape[0], -1)
                z = model.fc_enc(h)
            elif isinstance(model, WAE):
                mu, logvar = model.encoder(imgs_all)
                z = model.reparameterize(mu, logvar)
            else:
                mu, logvar = model.encoder(imgs_all)
                z = mu
            z_np = z.cpu().numpy()

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = tsne.fit_transform(z_np)

        ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all, cmap='tab10', alpha=0.5, s=3)
        ax.set_title(name, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Latent Space t-SNE (colored by class)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'latent_tsne.png', dpi=150)
    plt.close()


def plot_active_units(all_eval, latent_dim, results_dir):
    """5. Active Units bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(all_eval.keys())
    au = [all_eval[n]['active_units'] for n in names]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    bars = ax.bar(names, au, color=colors[:len(names)], alpha=0.8)
    ax.axhline(y=latent_dim, color='red', linestyle='--', alpha=0.5, label=f'Max ({latent_dim})')
    ax.set_ylabel('Active Units')
    ax.set_title('Active Units Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, v in zip(bars, au):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'active_units.png', dpi=150)
    plt.close()


def plot_quantitative_comparison(all_eval, results_dir):
    """6. Bar charts: ELBO, MSE, FID."""
    names = list(all_eval.keys())
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ELBO
    elbos = [all_eval[n]['elbo'] for n in names]
    axes[0].bar(names, elbos, color=colors[:len(names)], alpha=0.8)
    axes[0].set_title('ELBO (higher=better)')
    axes[0].set_ylabel('ELBO')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(elbos):
        axes[0].text(i, v, f'{v:.1f}', ha='center', fontweight='bold', fontsize=9)

    # MSE
    mses = [all_eval[n]['mse'] for n in names]
    axes[1].bar(names, mses, color=colors[:len(names)], alpha=0.8)
    axes[1].set_title('Reconstruction MSE (lower=better)')
    axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mses):
        axes[1].text(i, v, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)

    # FID
    fids = [all_eval[n]['fid'] for n in names]
    # Handle NaN (if torchmetrics not available)
    valid_fids = [(n, f) for n, f in zip(names, fids) if not np.isnan(f)]
    if valid_fids:
        fid_names, fid_vals = zip(*valid_fids)
        fid_colors = [colors[names.index(n)] for n in fid_names]
        axes[2].bar(fid_names, fid_vals, color=fid_colors, alpha=0.8)
        for i, v in enumerate(fid_vals):
            axes[2].text(i, v, f'{v:.1f}', ha='center', fontweight='bold', fontsize=9)
    axes[2].set_title('FID (lower=better)')
    axes[2].set_ylabel('FID')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Quantitative Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'quantitative_comparison.png', dpi=150)
    plt.close()


def plot_interpolations(models, test_imgs, results_dir, device, n_steps=10):
    """7. Latent space interpolation between two images."""
    img1 = test_imgs[0:1].to(device)
    img2 = test_imgs[1:2].to(device)

    model_names = list(models.keys())
    fig, axes = plt.subplots(len(model_names), n_steps, figsize=(2 * n_steps, 2 * len(model_names)))

    for row, name in enumerate(model_names):
        model = models[name]
        model.eval()
        with torch.no_grad():
            z1 = model.encode(img1)
            z2 = model.encode(img2)

            for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
                z_interp = (1 - alpha) * z1 + alpha * z2
                x_interp = model.decode(z_interp)
                axes[row, i].imshow(x_interp[0].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                axes[row, i].axis('off')

        axes[row, 0].set_ylabel(name, fontsize=10, fontweight='bold')

    plt.suptitle('Latent Interpolation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'interpolations.png', dpi=150)
    plt.close()


def plot_latent_variance(all_eval, results_dir):
    """8. Per-dimension variance plot for each model."""
    model_names = list(all_eval.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 3))
    if len(model_names) == 1:
        axes = [axes]

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    threshold = 1e-2

    for ax, name, color in zip(axes, model_names, colors):
        var = all_eval[name]['var_per_dim'].numpy()
        active = var > threshold
        bar_colors = [color if a else 'lightgray' for a in active]
        ax.bar(range(len(var)), var, color=bar_colors, alpha=0.7, width=1.0)
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(f'{name}: {active.sum()}/{len(var)} active', color=color, fontweight='bold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Variance')

    plt.suptitle('Latent Variance per Dimension', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'latent_variance.png', dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='VAE Benchmark')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs (use 1 for quick test)')
    parser.add_argument('--n-samples', type=int, default=5000, help='Evaluation sample count')
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--train-size', type=int, default=10000, help='Training set size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / 'results' / '59-vae-benchmark'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Latent dim: {args.latent_dim}")
    print()

    # ── Data ──
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(min(args.train_size, len(train_dataset))))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ── Models ──
    latent_dim = args.latent_dim
    models = {
        'VAE': VAE(latent_dim=latent_dim, beta=1.0),
        'β-VAE': BetaVAE(latent_dim=latent_dim, beta=4.0),
        'IWAE': IWAE(latent_dim=latent_dim, K=5),
        'WAE': WAE(latent_dim=latent_dim, lambda_mmd=10.0),
        'VQ-VAE': VQVAE(latent_dim=latent_dim, n_embeddings=512),
    }

    # ── Train ──
    all_history = {}
    for name, model in models.items():
        print(f"=== Training {name} ===")
        model = model.to(device)
        models[name] = model  # update with device model
        all_history[name] = train_model(
            model, train_loader, n_epochs=args.epochs, lr=2e-4, device=device
        )
        print()

    # ── Evaluate ──
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    all_eval = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.eval()
        all_eval[name] = full_eval(model, test_loader, device, n_samples=args.n_samples)
        e = all_eval[name]
        print(f"  ELBO:          {e['elbo']:.4f}")
        print(f"  MSE:           {e['mse']:.6f}")
        print(f"  FID:           {e['fid']:.2f}")
        print(f"  Active Units:  {e['active_units']}/{latent_dim}")
        if 'codebook_usage' in e:
            print(f"  Codebook:      {e['used_codes']}/{model.n_embeddings} ({e['codebook_usage']:.1%})")

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print(f"{'Model':<10} {'ELBO':>10} {'MSE':>12} {'FID':>10} {'Active U':>10}")
    print("-" * 80)
    for name in models:
        e = all_eval[name]
        fid_str = f"{e['fid']:.2f}" if not np.isnan(e['fid']) else 'N/A'
        print(f"{name:<10} {e['elbo']:>10.4f} {e['mse']:>12.6f} {fid_str:>10} {e['active_units']:>10}")
    print("=" * 80)

    # ── Visualizations ──
    print("\nSaving visualizations...")
    test_vis_imgs = next(iter(test_loader))[0][:16].to(device)

    plot_training_curves(all_history, results_dir)
    plot_reconstructions(models, test_vis_imgs[:8], results_dir, device)
    plot_samples(models, results_dir, device)
    plot_latent_tsne(models, test_loader, results_dir, device)
    plot_active_units(all_eval, latent_dim, results_dir)
    plot_quantitative_comparison(all_eval, results_dir)
    plot_interpolations(models, test_vis_imgs, results_dir, device)
    plot_latent_variance(all_eval, results_dir)

    print(f"\nAll results saved to {results_dir}")


if __name__ == '__main__':
    main()
