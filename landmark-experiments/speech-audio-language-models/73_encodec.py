"""
Minimal EnCodec Neural Audio Codec Reproduction
================================================
Reproduces the core ideas from "High Fidelity Neural Audio Compression"
(Defossez et al., 2022, 2210.13438):
1. Encoder: strided convolutions compress audio into latent representation
2. RVQ (Residual Vector Quantization): quantize latents at multiple levels
3. Decoder: transposed convolutions reconstruct audio from quantized codes
4. Multi-scale discriminators for adversarial training (simplified)
5. Reconstruction quality vs bitrate tradeoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Residual Vector Quantization ──

class VectorQuantizer(nn.Module):
    """Single codebook for vector quantization."""
    def __init__(self, n_codes=1024, code_dim=64):
        super().__init__()
        self.n_codes = n_codes
        self.codebook = nn.Parameter(torch.randn(n_codes, code_dim) * 0.1)

    def forward(self, z):
        """z: (B, D, T) -> quantized: (B, D, T), indices: (B, T), commit_loss: scalar"""
        B, D, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)

        # Find nearest codebook entry
        dist = (z_flat.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)
        # dist: (B*T, n_codes)
        indices = dist.argmin(dim=1)  # (B*T,)
        quantized = self.codebook[indices]  # (B*T, D)

        # Commitment loss: encourage encoder output to be close to codebook
        commit_loss = F.mse_loss(z_flat, quantized.detach())

        # Straight-through estimator: copy gradient from quantized to z
        quantized_st = z_flat + (quantized - z_flat).detach()

        quantized_st = quantized_st.reshape(B, T, D).permute(0, 2, 1)
        indices = indices.reshape(B, T)

        return quantized_st, indices, commit_loss


class ResidualVectorQuantizer(nn.Module):
    """Multi-level RVQ: quantize residual at each level."""
    def __init__(self, n_levels=4, n_codes=512, code_dim=64):
        super().__init__()
        self.n_levels = n_levels
        self.layers = nn.ModuleList([
            VectorQuantizer(n_codes, code_dim) for _ in range(n_levels)
        ])

    def forward(self, z):
        """
        z: (B, D, T)
        Returns: quantized (B, D, T), all_indices (n_levels, B, T),
                 total_commit_loss, residual norms per level
        """
        residual = z
        quantized_total = torch.zeros_like(z)
        all_indices = []
        total_commit_loss = 0.0
        residual_norms = []

        for vq in self.layers:
            q, indices, commit_loss = vq(residual)
            quantized_total = quantized_total + q
            residual = residual - q  # Update residual
            all_indices.append(indices)
            total_commit_loss = total_commit_loss + commit_loss
            residual_norms.append(residual.norm(dim=1).mean().item())

        return quantized_total, all_indices, total_commit_loss, residual_norms

    def encode(self, z):
        """Encode to codebook indices only."""
        residual = z
        all_indices = []
        for vq in self.layers:
            _, indices, _ = vq(residual)
            all_indices.append(indices)
            # Re-quantize to get accurate residual
            z_flat = residual.permute(0, 2, 1).reshape(-1, residual.shape[1])
            dist = (z_flat.unsqueeze(1) - vq.codebook.unsqueeze(0)).pow(2).sum(-1)
            idx = dist.argmin(dim=1)
            q = vq.codebook[idx].reshape(residual.shape[0], residual.shape[2], -1).permute(0, 2, 1)
            residual = residual - q
        return all_indices

    def decode(self, all_indices):
        """Decode from codebook indices."""
        z = None
        for vq, indices in zip(self.layers, all_indices):
            B, T = indices.shape
            q = vq.codebook[indices]  # (B, T, D)
            q = q.permute(0, 2, 1)  # (B, D, T)
            z = q if z is None else z + q
        return z


# ── Encoder ──

class Encoder(nn.Module):
    """Strided convolutional encoder: audio -> latent."""
    def __init__(self, in_channels=1, hidden_dim=64, code_dim=64):
        super().__init__()
        # Progressive downsampling: each layer halves the temporal resolution
        # Total downsampling: 2^4 = 16x
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=1, padding=3),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(hidden_dim * 4, code_dim, 5, stride=2, padding=2),
        )

    def forward(self, x):
        """x: (B, 1, T) -> (B, code_dim, T//16)"""
        return self.net(x)


# ── Decoder ──

class Decoder(nn.Module):
    """Transposed convolutional decoder: latent -> audio."""
    def __init__(self, out_channels=1, hidden_dim=64, code_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(code_dim, hidden_dim * 4, 5, stride=2, padding=2,
                               output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 4, 5, stride=2,
                               padding=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, 5, stride=2,
                               padding=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, 5, stride=2,
                               padding=2, output_padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, out_channels, 7, padding=3),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z):
        """z: (B, code_dim, T//16) -> (B, 1, T)"""
        return self.net(z)


# ── Multi-Scale Discriminator (simplified) ──

class MultiScaleDiscriminator(nn.Module):
    """Simplified multi-scale discriminator for adversarial training."""
    def __init__(self, in_channels=1):
        super().__init__()
        # Three discriminators at different temporal resolutions
        self.d1 = self._make_disc(in_channels)  # Original
        self.pool = nn.AvgPool1d(4, stride=4)
        self.d2 = self._make_disc(in_channels)  # 4x downsampled
        self.d3 = self._make_disc(in_channels)  # 16x downsampled

    def _make_disc(self, in_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, stride=4, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 11, stride=4, padding=5),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, stride=4, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, 3, padding=1),
        )

    def forward(self, x):
        """Returns list of discriminator outputs at different scales."""
        out1 = self.d1(x)
        x2 = self.pool(x)
        if x2.size(-1) < 16:
            out2 = None
        else:
            out2 = self.d2(x2)
        x3 = self.pool(x2) if x2.size(-1) >= 16 else None
        if x3 is not None and x3.size(-1) >= 16:
            out3 = self.d3(x3)
        else:
            out3 = None
        return [o for o in [out1, out2, out3] if o is not None]


# ── Full EnCodec Model ──

class EnCodecModel(nn.Module):
    def __init__(self, code_dim=64, n_levels=4, n_codes=512, hidden_dim=64):
        super().__init__()
        self.encoder = Encoder(1, hidden_dim, code_dim)
        self.rvq = ResidualVectorQuantizer(n_levels, n_codes, code_dim)
        self.decoder = Decoder(1, hidden_dim, code_dim)

    def forward(self, x):
        """
        x: (B, 1, T)
        Returns: reconstructed (B, 1, T), commit_loss, all_indices, residual_norms
        """
        z = self.encoder(x)
        z_q, all_indices, commit_loss, residual_norms = self.rvq(z)
        x_hat = self.decoder(z_q)
        return x_hat, commit_loss, all_indices, residual_norms


# ── Synthetic Audio Signals ──

def generate_synthetic_audio(batch_size, length, sample_rate=16000, device='cpu'):
    """Generate diverse synthetic audio signals."""
    audio = torch.zeros(batch_size, 1, length, device=device)

    for i in range(batch_size):
        signal_type = i % 4
        t = torch.linspace(0, length / sample_rate, length, device=device)

        if signal_type == 0:
            # Pure sine wave (random frequency 200-800 Hz)
            freq = np.random.uniform(200, 800)
            audio[i, 0] = 0.8 * torch.sin(2 * np.pi * freq * t)
        elif signal_type == 1:
            # Chirp: frequency sweeps from 200 to 1000 Hz
            f0, f1 = 200, 1000
            freq = f0 + (f1 - f0) * t / t[-1]
            phase = 2 * np.pi * torch.cumsum(freq / sample_rate, dim=0)
            audio[i, 0] = 0.7 * torch.sin(phase)
        elif signal_type == 2:
            # Multi-tone: sum of 3 harmonics
            f = np.random.uniform(150, 400)
            audio[i, 0] = (0.5 * torch.sin(2 * np.pi * f * t) +
                           0.3 * torch.sin(2 * np.pi * 2 * f * t) +
                           0.2 * torch.sin(2 * np.pi * 3 * f * t))
        else:
            # Amplitude-modulated signal
            carrier_freq = np.random.uniform(400, 800)
            mod_freq = np.random.uniform(3, 10)
            carrier = torch.sin(2 * np.pi * carrier_freq * t)
            modulator = 0.5 + 0.5 * torch.sin(2 * np.pi * mod_freq * t)
            audio[i, 0] = 0.8 * carrier * modulator

    return audio


# ── Training ──

def train_encodec(model, discriminator, n_steps=3000, lr=1e-3, device='cpu',
                  sample_rate=16000, audio_length=4096):
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    losses_recon = []
    losses_commit = []
    losses_total = []
    snr_values = []

    lambda_commit = 1.0
    lambda_adv = 0.1

    for step in range(n_steps):
        x_real = generate_synthetic_audio(16, audio_length, sample_rate, device)

        # ── Generator step ──
        x_hat, commit_loss, _, _ = model(x_real)

        # Reconstruction loss (L1 + L2)
        recon_loss = F.l1_loss(x_hat, x_real) + F.mse_loss(x_hat, x_real)

        # Adversarial loss (feature matching style)
        disc_real = discriminator(x_real)
        disc_fake = discriminator(x_hat.detach())
        adv_loss_g = 0.0
        for dr, df in zip(disc_real, disc_fake):
            # Hinge loss for generator
            adv_loss_g += -dr.mean()

        g_loss = recon_loss + lambda_commit * commit_loss + lambda_adv * adv_loss_g

        optimizer_g.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_g.step()

        # ── Discriminator step ──
        x_hat_det = x_hat.detach()
        disc_real = discriminator(x_real)
        disc_fake = discriminator(x_hat_det)

        d_loss = 0.0
        for dr, df in zip(disc_real, disc_fake):
            d_loss += (F.relu(1 - dr).mean() + F.relu(1 + df).mean())

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # ── Logging ──
        with torch.no_grad():
            signal_power = x_real.pow(2).mean()
            noise_power = (x_real - x_hat).pow(2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

        losses_recon.append(recon_loss.item())
        losses_commit.append(commit_loss.item())
        losses_total.append(g_loss.item())
        snr_values.append(snr.item())

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Recon: {recon_loss.item():.4f} | "
                  f"Commit: {commit_loss.item():.4f} | SNR: {snr.item():.1f} dB")

    return losses_recon, losses_commit, losses_total, snr_values


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "73-encodec"
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    audio_length = 4096

    # ── Setup ──
    print("=== EnCodec: Neural Audio Codec ===\n")

    # Train with 4 RVQ levels
    model = EnCodecModel(code_dim=64, n_levels=4, n_codes=512, hidden_dim=64).to(device)
    discriminator = MultiScaleDiscriminator().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"EnCodec parameters: {n_params:,}")
    print(f"RVQ levels: 4, Codebook size: 512, Code dim: 64")
    print(f"Downsampling: 16x (4 strided conv layers)")

    # ── Train ──
    print("\nTraining...")
    losses_recon, losses_commit, losses_total, snr_values = train_encodec(
        model, discriminator, n_steps=3000, device=device,
        sample_rate=sample_rate, audio_length=audio_length
    )

    # ── Bitrate vs Quality Tradeoff ──
    print("\nEvaluating bitrate vs quality tradeoff...")
    model.eval()
    x_test = generate_synthetic_audio(8, audio_length, sample_rate, device)

    bitrate_results = {}
    for n_levels in [1, 2, 3, 4]:
        # Create a temporary RVQ with fewer levels
        temp_rvq = ResidualVectorQuantizer(n_levels, 512, 64).to(device)

        with torch.no_grad():
            z = model.encoder(x_test)
            z_q, _, commit_loss, residual_norms = temp_rvq(z)
            x_hat = model.decoder(z_q)

            signal_power = x_test.pow(2).mean()
            noise_power = (x_test - x_hat).pow(2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()

        # Bitrate = n_levels * log2(n_codes) * frames_per_second
        frame_rate = sample_rate / 16  # 16x downsampling
        bits_per_frame = n_levels * np.log2(512)
        bitrate = bits_per_frame * frame_rate / 1000  # kbps

        bitrate_results[n_levels] = {
            'snr': snr,
            'bitrate': bitrate,
            'residual_norms': residual_norms
        }
        print(f"  RVQ levels={n_levels}: SNR={snr:.1f} dB, Bitrate={bitrate:.1f} kbps")

    # ── Visualization ──

    # 1. Training convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    window = 30

    ax = axes[0, 0]
    loss_smooth = np.convolve(losses_recon, np.ones(window)/window, mode='valid')
    ax.plot(loss_smooth, color='blue')
    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("L1 + L2 Loss (smoothed)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    commit_smooth = np.convolve(losses_commit, np.ones(window)/window, mode='valid')
    ax.plot(commit_smooth, color='orange')
    ax.set_title("Commitment Loss (Quantization)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    snr_smooth = np.convolve(snr_values, np.ones(window)/window, mode='valid')
    ax.plot(snr_smooth, color='green')
    ax.set_title("Signal-to-Noise Ratio")
    ax.set_xlabel("Step")
    ax.set_ylabel("SNR (dB, smoothed)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    total_smooth = np.convolve(losses_total, np.ones(window)/window, mode='valid')
    ax.plot(total_smooth, color='red')
    ax.set_title("Total Generator Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("EnCodec Training Convergence", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_convergence.png", dpi=150)
    plt.close()

    # 2. Bitrate vs Quality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    levels = sorted(bitrate_results.keys())
    bitrates = [bitrate_results[l]['bitrate'] for l in levels]
    snrs = [bitrate_results[l]['snr'] for l in levels]

    ax1.plot(bitrates, snrs, 'o-', color='blue', markersize=10)
    for i, l in enumerate(levels):
        ax1.annotate(f"L={l}", (bitrates[i], snrs[i]),
                    textcoords="offset points", xytext=(10, 5))
    ax1.set_xlabel("Bitrate (kbps)")
    ax1.set_ylabel("SNR (dB)")
    ax1.set_title("Reconstruction Quality vs Bitrate")
    ax1.grid(True, alpha=0.3)

    # Residual norms per level
    ax2.set_title("Residual Norm per RVQ Level")
    all_norms = [bitrate_results[l]['residual_norms'] for l in levels]
    max_levels = max(len(n) for n in all_norms)
    for i, l in enumerate(levels):
        norms = all_norms[i]
        ax2.bar(np.arange(len(norms)) + i * 0.2 - 0.3, norms,
                width=0.2, label=f"Config: {l} levels", alpha=0.8)
    ax2.set_xlabel("RVQ Level")
    ax2.set_ylabel("Residual Norm")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "bitrate_quality_tradeoff.png", dpi=150)
    plt.close()

    # 3. Waveform reconstruction visualization
    print("\nGenerating reconstruction visualization...")
    with torch.no_grad():
        x_test = generate_synthetic_audio(4, audio_length, sample_rate, device)
        x_hat, _, all_indices, _ = model(x_test)

    fig, axes = plt.subplots(4, 2, figsize=(14, 8))
    signal_names = ["Sine", "Chirp", "Multi-tone", "AM Signal"]

    for i in range(4):
        # Original
        axes[i, 0].plot(x_test[i, 0].cpu().numpy(), linewidth=0.5, color='blue')
        axes[i, 0].set_title(f"{signal_names[i]} - Original", fontsize=9)
        axes[i, 0].set_ylim(-1.1, 1.1)

        # Reconstructed
        axes[i, 1].plot(x_hat[i, 0].cpu().numpy(), linewidth=0.5, color='red')
        err = (x_test[i, 0] - x_hat[i, 0]).pow(2).mean().sqrt().item()
        axes[i, 1].set_title(f"{signal_names[i]} - Reconstructed (RMSE={err:.4f})",
                             fontsize=9)
        axes[i, 1].set_ylim(-1.1, 1.1)

    plt.suptitle("EnCodec: Original vs Reconstructed Audio", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "waveform_reconstruction.png", dpi=150)
    plt.close()

    # 4. Codebook usage visualization
    print("Analyzing codebook usage...")
    with torch.no_grad():
        x_test = generate_synthetic_audio(32, audio_length, sample_rate, device)
        z = model.encoder(x_test)
        _, all_indices, _, _ = model.rvq(z)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for level in range(4):
        idx = all_indices[level].cpu().numpy().flatten()
        unique, counts = np.unique(idx, return_counts=True)
        usage = np.zeros(512)
        for u, c in zip(unique, counts):
            usage[u] = c
        axes[level].bar(range(512), usage, width=1, color='steelblue', alpha=0.7)
        axes[level].set_title(f"Level {level} Codebook Usage")
        axes[level].set_xlabel("Code Index")
        axes[level].set_ylabel("Count")
        active = (usage > 0).sum()
        axes[level].set_xlim(0, 512)
        axes[level].text(0.95, 0.95, f"Active: {active}/512",
                        transform=axes[level].transAxes, ha='right', va='top',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.suptitle("RVQ Codebook Utilization per Level", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "codebook_usage.png", dpi=150)
    plt.close()

    # 5. RVQ decomposition: show how each level refines
    print("Visualizing RVQ level-by-level reconstruction...")
    with torch.no_grad():
        x_test = generate_synthetic_audio(1, audio_length, sample_rate, device)
        z = model.encoder(x_test)
        residual = z
        cumulative = torch.zeros_like(z)
        level_reconstructions = []

        for vq in model.rvq.layers:
            q, _, _ = vq(residual)
            cumulative = cumulative + q
            residual = residual - q
            x_level = model.decoder(cumulative)
            level_reconstructions.append(x_level[0, 0].cpu().numpy())

    original = x_test[0, 0].cpu().numpy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 8))
    axes[0].plot(original, linewidth=0.5, color='blue')
    axes[0].set_title("Original Signal")
    axes[0].set_ylim(-1.1, 1.1)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for level in range(4):
        axes[level + 1].plot(level_reconstructions[level], linewidth=0.5,
                            color=colors[level])
        rmse = np.sqrt(np.mean((original - level_reconstructions[level]) ** 2))
        axes[level + 1].set_title(f"After RVQ Level {level} (RMSE={rmse:.4f})")
        axes[level + 1].set_ylim(-1.1, 1.1)

    plt.suptitle("RVQ: Progressive Refinement with Each Level", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "rvq_progressive.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
