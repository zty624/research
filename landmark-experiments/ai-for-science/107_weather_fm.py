"""
Minimal Weather/Climate Foundation Model Reproduction
=====================================================
Reproduces core ideas from "Foundation Models for Weather and Climate"
(2309.10808, Nguyen et al.) and related works:
1. Spatiotemporal prediction: forecast future weather from past observations
2. 2D spatial grid as input (lat-lon) with channel features (temp, pressure, etc.)
3. Autoregressive rollout: predict next timestep from current state
4. Compare: persistent forecast vs climatology vs learned model
5. Show: rollout accuracy degradation, spatial error patterns
6. Demonstrate: spectral analysis of predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Weather Grid Model ──

class WeatherConvNet(nn.Module):
    """U-Net-style convolutional model for spatiotemporal weather prediction.
    Input: (B, C_in, H, W) grid of current weather state
    Output: (B, C_out, H, W) predicted next timestep
    """
    def __init__(self, in_channels=4, out_channels=4, hidden=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 4, hidden * 4, 3, padding=1), nn.GELU(),
        )
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 2, 3, padding=1), nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, 3, padding=1), nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Decode with skip connections
        d3 = self.dec3(e3)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1


class WeatherTransformer(nn.Module):
    """Vision Transformer-style model for weather grids.
    Treats spatial patches as tokens.
    """
    def __init__(self, in_channels=4, out_channels=4, patch_size=4,
                 d_model=128, n_heads=4, n_layers=4, grid_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        n_patches = (grid_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       dropout=0.1, activation='gelu', batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels * patch_size * patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        # Patchify
        patches = x.reshape(B, C, H // ps, ps, W // ps, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * ps * ps)
        # Transformer
        h = self.patch_embed(patches) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        # Depatchify
        out = self.head(h)
        out = out.reshape(B, H // ps, W // ps, -1, ps, ps)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, H, W)
        return out


# ── Synthetic Weather Data ──

class SyntheticWeather:
    """Generate synthetic spatiotemporal weather-like data.

    Simulates fields evolving with advection + diffusion on a 2D grid:
    - Temperature: warm equator, cold poles, advected by wind
    - Pressure: large-scale waves
    - Wind U, V: geostrophic-like flow
    """
    def __init__(self, grid_size=32, n_channels=4, device='cpu'):
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.device = device

        # Latitude-like coordinate
        lat = np.linspace(-np.pi / 2, np.pi / 2, grid_size)
        lon = np.linspace(0, 2 * np.pi, grid_size)
        self.LAT, self.LON = np.meshgrid(lat, lon, indexing='ij')

        # Wind field (slowly varying)
        self.wind_u = 0.5 * np.cos(self.LAT)  # westerlies
        self.wind_v = 0.1 * np.sin(2 * self.LAT) * np.cos(self.LON)

        # Perturbation phases
        self.phase = np.random.randn(10, 10) * 0.3

    def generate_frame(self, t):
        """Generate a single weather frame at time t."""
        gs = self.grid_size
        lat, lon = self.LAT, self.LON

        # Temperature: gradient + waves + perturbation
        temp = 300 - 40 * np.sin(lat)  # base: warm equator
        temp += 5 * np.sin(2 * lon - 0.3 * t + self.phase[0, 0])  # Rossby wave
        temp += 3 * np.cos(3 * lon + 0.5 * t + self.phase[1, 0])
        temp += 2 * np.sin(4 * lat + 0.2 * t)

        # Pressure: large-scale pattern
        pressure = 1013 + 15 * np.cos(2 * lat) * np.sin(lon - 0.2 * t)
        pressure += 8 * np.sin(3 * lon + 0.4 * t)

        # Wind U: geostrophic + perturbation
        u = self.wind_u * 10 + 3 * np.sin(lon - 0.3 * t)
        # Wind V: meridional
        v = self.wind_v * 10 + 2 * np.cos(2 * lat + 0.2 * t)

        # Stack channels
        frame = np.stack([temp, pressure, u, v], axis=0)  # (4, H, W)
        # Add small noise
        frame += np.random.randn(*frame.shape) * 0.5
        return frame.astype(np.float32)

    def generate_sequence(self, n_frames=12):
        """Generate a sequence of weather frames."""
        frames = [self.generate_frame(t) for t in range(n_frames)]
        return np.stack(frames)  # (T, C, H, W)

    def generate_batch(self, batch_size, seq_len=12):
        """Generate a batch of sequences."""
        batch = []
        for _ in range(batch_size):
            batch.append(self.generate_sequence(seq_len))
        return torch.tensor(np.stack(batch), dtype=torch.float32, device=self.device)


# ── Baselines ──

def persistence_forecast(x):
    """Baseline: predict current state as next state."""
    return x.clone()


def climatology_forecast(means, x):
    """Baseline: predict long-term average."""
    return means.expand_as(x)


# ── Training ──

def train_weather_model(model, weather, n_steps=3000, batch_size=16,
                        lr=1e-3, rollout_steps=1, device='cpu'):
    """Train weather model with single-step and multi-step rollout loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []
    accs = []

    for step in range(n_steps):
        data = weather.generate_batch(batch_size, seq_len=rollout_steps + 2)
        x = data[:, 0]  # (B, C, H, W) current state
        targets = data[:, 1:rollout_steps + 1]  # (B, T, C, H, W)

        total_loss = 0
        current = x
        for t in range(rollout_steps):
            pred = model(current)
            total_loss += F.mse_loss(pred, targets[:, t])
            # Teacher forcing: use ground truth for next input
            current = targets[:, t]

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())

        # Anomaly correlation (simplified)
        with torch.no_grad():
            pred = model(x)
            anomaly_pred = pred - pred.mean(dim=(2, 3), keepdim=True)
            anomaly_true = targets[:, 0] - targets[:, 0].mean(dim=(2, 3), keepdim=True)
            num = (anomaly_pred * anomaly_true).sum(dim=(2, 3))
            den = (anomaly_pred ** 2).sum(dim=(2, 3)).sqrt() * (anomaly_true ** 2).sum(dim=(2, 3)).sqrt() + 1e-8
            acc = (num / den).mean().item()
            accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {total_loss.item():.4f} | ACC: {acc:.3f}")

    return losses, accs


def evaluate_rollout(model, weather, n_steps=24, n_trials=5, device='cpu'):
    """Evaluate autoregressive rollout accuracy over multiple steps."""
    model.eval()
    errors = []

    for trial in range(n_trials):
        data = weather.generate_batch(1, seq_len=n_steps + 1)
        x0 = data[:, 0]
        current = x0

        trial_errors = []
        for t in range(1, n_steps + 1):
            with torch.no_grad():
                pred = model(current)
            target = data[:, t]
            rmse = ((pred - target) ** 2).mean().sqrt().item()
            trial_errors.append(rmse)
            # Autoregressive: use prediction as next input
            current = pred

        errors.append(trial_errors)

    return np.array(errors)  # (n_trials, n_steps)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "107-weather-fm"
    results_dir.mkdir(parents=True, exist_ok=True)

    grid_size = 32
    weather = SyntheticWeather(grid_size=grid_size, device=device)

    # Generate climatology
    print("=== Computing Climatology ===")
    n_clim = 500
    clim_frames = [weather.generate_frame(t) for t in range(n_clim)]
    clim_mean = torch.tensor(np.mean(clim_frames, axis=0), dtype=torch.float32, device=device)

    # ── Train Conv model ──
    print("\n=== Training WeatherConvNet (single-step) ===")
    conv_model = WeatherConvNet(in_channels=4, out_channels=4, hidden=64).to(device)
    conv_p = sum(p.numel() for p in conv_model.parameters())
    print(f"  Params: {conv_p:,}")
    conv_losses, conv_accs = train_weather_model(conv_model, weather, n_steps=2000, device=device)

    # ── Train ViT model ──
    print("\n=== Training WeatherTransformer ===")
    vit_model = WeatherTransformer(in_channels=4, out_channels=4, patch_size=4,
                                    d_model=128, n_heads=4, n_layers=4,
                                    grid_size=grid_size).to(device)
    vit_p = sum(p.numel() for p in vit_model.parameters())
    print(f"  Params: {vit_p:,}")
    vit_losses, vit_accs = train_weather_model(vit_model, weather, n_steps=2000, device=device)

    # ── Train with multi-step rollout ──
    print("\n=== Training WeatherConvNet (3-step rollout) ===")
    conv_rollout = WeatherConvNet(in_channels=4, out_channels=4, hidden=64).to(device)
    rollout_losses, rollout_accs = train_weather_model(
        conv_rollout, weather, n_steps=2000, rollout_steps=3, device=device)

    # ── Rollout evaluation ──
    print("\n=== Rollout Evaluation ===")
    conv_errors = evaluate_rollout(conv_model, weather, n_steps=24, device=device)
    vit_errors = evaluate_rollout(vit_model, weather, n_steps=24, device=device)
    rollout_errors = evaluate_rollout(conv_rollout, weather, n_steps=24, device=device)

    # Persistence baseline
    persist_errors = []
    for trial in range(5):
        data = weather.generate_batch(1, seq_len=25)
        trial_e = []
        for t in range(1, 25):
            rmse = ((data[:, 0] - data[:, t]) ** 2).mean().sqrt().item()
            trial_e.append(rmse)
        persist_errors.append(trial_e)
    persist_errors = np.array(persist_errors)

    # Climatology baseline
    clim_errors = []
    for trial in range(5):
        data = weather.generate_batch(1, seq_len=25)
        trial_e = []
        for t in range(1, 25):
            target = data[:, t]
            rmse = ((clim_mean - target) ** 2).mean().sqrt().item()
            trial_e.append(rmse)
        persist_errors_trial = trial_e
        clim_errors.append(trial_e)
    clim_errors = np.array(clim_errors)

    # ── Visualization ──

    # 1. Training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 30
    conv_s = np.convolve(conv_losses, np.ones(w)/w, mode='valid')
    vit_s = np.convolve(vit_losses, np.ones(w)/w, mode='valid')
    roll_s = np.convolve(rollout_losses, np.ones(w)/w, mode='valid')
    axes[0].plot(conv_s, label='ConvNet (1-step)', color='blue')
    axes[0].plot(vit_s, label='ViT', color='red')
    axes[0].plot(roll_s, label='ConvNet (3-step rollout)', color='green')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    conv_a = np.convolve(conv_accs, np.ones(w)/w, mode='valid')
    vit_a = np.convolve(vit_accs, np.ones(w)/w, mode='valid')
    roll_a = np.convolve(rollout_accs, np.ones(w)/w, mode='valid')
    axes[1].plot(conv_a, label='ConvNet (1-step)', color='blue')
    axes[1].plot(vit_a, label='ViT', color='red')
    axes[1].plot(roll_a, label='ConvNet (3-step rollout)', color='green')
    axes[1].set_title("Anomaly Correlation Coefficient")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("ACC (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Weather Foundation Model: Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training.png', dpi=150)
    plt.close()

    # 2. Rollout error degradation
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = range(1, 25)
    ax.plot(steps, persist_errors.mean(0), '--', label='Persistence', color='gray', alpha=0.7)
    ax.plot(steps, clim_errors.mean(0), '--', label='Climatology', color='orange', alpha=0.7)
    ax.plot(steps, conv_errors.mean(0), label='ConvNet (1-step trained)', color='blue')
    ax.plot(steps, vit_errors.mean(0), label='ViT', color='red')
    ax.plot(steps, rollout_errors.mean(0), label='ConvNet (3-step rollout)', color='green')
    ax.fill_between(steps, conv_errors.mean(0) - conv_errors.std(0),
                    conv_errors.mean(0) + conv_errors.std(0), alpha=0.15, color='blue')
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("RMSE")
    ax.set_title("Autoregressive Rollout Error Degradation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'rollout_degradation.png', dpi=150)
    plt.close()

    # 3. Spatial prediction visualization
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    data_vis = weather.generate_sequence(6)
    conv_model.eval()
    with torch.no_grad():
        x_t = torch.tensor(data_vis[:-1], dtype=torch.float32, device=device)
        pred = conv_model(x_t).cpu().numpy()

    for t in range(5):
        # Ground truth temperature
        ax = axes[0, t]
        ax.imshow(data_vis[t, 0], cmap='RdBu_r', vmin=260, vmax=340)
        ax.set_title(f"GT t={t}" if t == 0 else f"t={t}", fontsize=9)
        ax.axis('off')

        # Predicted temperature
        ax = axes[1, t]
        ax.imshow(pred[t, 0], cmap='RdBu_r', vmin=260, vmax=340)
        ax.set_title(f"Pred t={t}", fontsize=9)
        ax.axis('off')

        # Error
        ax = axes[2, t]
        err = np.abs(pred[t, 0] - data_vis[t + 1, 0])
        ax.imshow(err, cmap='hot', vmin=0, vmax=10)
        ax.set_title(f"Error t={t}", fontsize=9)
        ax.axis('off')

    # Labels
    axes[0, 5].text(0.5, 0.5, 'Ground\nTruth', ha='center', va='center',
                    transform=axes[0, 5].transAxes, fontsize=12)
    axes[1, 5].text(0.5, 0.5, 'Predicted', ha='center', va='center',
                    transform=axes[1, 5].transAxes, fontsize=12)
    axes[2, 5].text(0.5, 0.5, 'Absolute\nError', ha='center', va='center',
                    transform=axes[2, 5].transAxes, fontsize=12)
    for ax in axes[:, 5]:
        ax.axis('off')

    plt.suptitle('Weather Prediction: Temperature Field (Channel 0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'spatial_prediction.png', dpi=150)
    plt.close()

    # 4. Spectral analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute power spectrum of predictions vs ground truth
    gt_frame = data_vis[-1, 0]
    pred_frame = pred[-1, 0]

    gt_fft = np.abs(np.fft.fft2(gt_frame))
    pred_fft = np.abs(np.fft.fft2(pred_frame))

    # Radial power spectrum
    def radial_spectrum(fft2d):
        cy, cx = fft2d.shape[0] // 2, fft2d.shape[1] // 2
        Y, X = np.ogrid[:fft2d.shape[0], :fft2d.shape[1]]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        max_r = min(cx, cy)
        spectrum = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                spectrum[i] = fft2d[mask].mean()
        return spectrum

    gt_spec = radial_spectrum(np.fft.fftshift(gt_fft))
    pred_spec = radial_spectrum(np.fft.fftshift(pred_fft))
    freqs = np.arange(len(gt_spec))

    axes[0].semilogy(freqs, gt_spec, label='Ground Truth', color='blue')
    axes[0].semilogy(freqs, pred_spec, label='Predicted', color='red')
    axes[0].set_title("Power Spectrum: GT vs Prediction")
    axes[0].set_xlabel("Spatial Frequency")
    axes[0].set_ylabel("Power")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spectrum ratio
    ratio = pred_spec / (gt_spec + 1e-10)
    axes[1].plot(freqs, ratio, color='green')
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title("Spectrum Ratio: Pred / GT")
    axes[1].set_xlabel("Spatial Frequency")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 2)

    plt.suptitle('Spectral Analysis of Weather Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'spectral_analysis.png', dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    concept = (
        "Weather/Climate Foundation Model (2309.10808)\n"
        "=" * 55 + "\n\n"
        "Key Ideas:\n"
        "1. Spatiotemporal grid data → treat as image-like (lat, lon, channels)\n"
        "2. Autoregressive rollout: x_t → model → x_{t+1} → model → x_{t+2} ...\n"
        "3. U-Net or ViT backbone for spatial processing\n"
        "4. Error accumulates during rollout (compounding prediction drift)\n"
        "5. Multi-step rollout training reduces drift (teacher forcing vs scheduled sampling)\n\n"
        "Baselines:\n"
        "  • Persistence: x_{t+1} = x_t (RMSE grows linearly)\n"
        "  • Climatology: x_{t+1} = mean(x) (constant error)\n"
        "  • Learned model: captures dynamics, error grows sub-linearly\n\n"
        "Spectral insight: models often lose high-frequency detail,\n"
        "producing overly smooth predictions at long horizons."
    )
    ax.text(0.05, 0.95, concept, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
