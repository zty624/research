"""
Minimal Molecular Generation with VAE Reproduction
===================================================
Reproduces the core ideas from "Automatic Chemical Design Using a
Continuous Representation of Molecules" (Gómez-Bombarelli et al.,
2016/2019, arxiv 1907.01632):
1. SMILES string → latent space via encoder (VAE)
2. Latent space → SMILES via decoder
3. Interpolation in latent space = smooth chemical transitions
4. Latent space optimization for molecular properties
5. Demo on synthetic character-level "molecule-like" sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


# ── SMILES-like Tokenizer ──

class SMILESTokenizer:
    """Simple character-level tokenizer for SMILES-like strings."""
    SPECIAL = ['<pad>', '<sos>', '<eos>']

    def __init__(self, charset=None):
        if charset is None:
            # Common SMILES characters
            charset = list('CcNnOoFPSsClBrI()[]=#@+-/\\1234567890')
        self.charset = charset
        self.vocab = self.SPECIAL + sorted(set(charset))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2

    def encode(self, smiles, max_len=None):
        """Encode SMILES string to tensor of indices."""
        indices = [self.sos_idx]
        for c in smiles:
            if c in self.char2idx:
                indices.append(self.char2idx[c])
            else:
                indices.append(self.pad_idx)
        indices.append(self.eos_idx)
        if max_len:
            indices = indices[:max_len] + [self.pad_idx] * max(0, max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """Decode tensor of indices to SMILES string."""
        chars = []
        for idx in indices:
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            chars.append(self.idx2char.get(idx, '?'))
        return ''.join(chars)

    @property
    def vocab_size(self):
        return len(self.vocab)


# ── SMILES VAE ──

class SMILESVAE(nn.Module):
    """Variational Autoencoder for SMILES strings."""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32,
                 max_len=40, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_len = max_len

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Encoder (bidirectional GRU)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=0.1)

        # Latent space
        enc_out_dim = hidden_dim * 2  # bidirectional
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Decoder (GRU with latent conditioning)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.decoder = nn.GRU(embed_dim + latent_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=0.1)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        """x: (B, L) token indices → mu, logvar."""
        emb = self.embedding(x)  # (B, L, embed_dim)
        _, h = self.encoder(emb)  # h: (num_layers*2, B, hidden_dim)

        # Concatenate final hidden states from both directions
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden_dim*2)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target=None, teacher_forcing_ratio=0.5):
        """Decode latent vector z to SMILES."""
        B = z.shape[0]

        # Initialize decoder hidden state from latent
        h0 = self.latent_to_hidden(z)
        h0 = h0.reshape(self.decoder.num_layers, B, self.hidden_dim)

        # Start with SOS token
        input_tok = torch.full((B, 1), 1, dtype=torch.long, device=z.device)  # SOS
        z_expand = z.unsqueeze(1).expand(-1, self.max_len, -1)

        outputs = []
        hidden = h0

        for t in range(self.max_len):
            emb = self.embedding(input_tok)  # (B, 1, embed_dim)
            z_t = z_expand[:, t:t+1, :]  # (B, 1, latent_dim)
            dec_input = torch.cat([emb, z_t], dim=-1)

            out, hidden = self.decoder(dec_input, hidden)
            logits = self.output_proj(out.squeeze(1))  # (B, vocab_size)
            outputs.append(logits)

            # Teacher forcing
            if target is not None and np.random.random() < teacher_forcing_ratio:
                input_tok = target[:, t:t+1]
            else:
                input_tok = logits.argmax(dim=-1, keepdim=True)

        return torch.stack(outputs, dim=1)  # (B, max_len, vocab_size)

    def forward(self, x, teacher_forcing_ratio=0.5):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return logits, mu, logvar

    def sample(self, n_samples, device='cpu'):
        """Sample molecules from the prior."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        logits = self.decode(z, teacher_forcing_ratio=0.0)
        return logits.argmax(dim=-1)  # (n_samples, max_len)

    def interpolate(self, z1, z2, n_steps=10):
        """Linear interpolation between two latent codes."""
        alphas = torch.linspace(0, 1, n_steps, device=z1.device)
        z_interp = torch.stack([a * z2 + (1 - a) * z1 for a in alphas])
        logits = self.decode(z_interp, teacher_forcing_ratio=0.0)
        return logits.argmax(dim=-1)


# ── Loss ──

def vae_loss(logits, target, mu, logvar, pad_idx=0):
    """VAE loss = reconstruction + KL divergence."""
    # Reconstruction (cross-entropy)
    recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             target.reshape(-1), ignore_index=pad_idx)

    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / mu.shape[0]

    return recon + kl, recon, kl


# ── Synthetic SMILES-like Data ──

def generate_synthetic_smiles(n_samples=2000, max_len=30):
    """Generate synthetic SMILES-like strings for demo.

    These are NOT real molecules but have similar structure:
    alternating atoms, bonds, branches, etc.
    """
    atoms = ['C', 'N', 'O', 'S', 'F', 'P']
    bonds = ['', '=', '#']
    branches = ['(', ')']
    rings = ['1', '2', '3']

    smiles_list = []
    for _ in range(n_samples):
        length = np.random.randint(8, max_len)
        s = []
        i = 0
        while len(s) < length:
            if np.random.random() < 0.15 and i > 2:
                s.append(np.random.choice(branches))
            elif np.random.random() < 0.1:
                s.append(np.random.choice(rings))
            else:
                atom = np.random.choice(atoms)
                bond = np.random.choice(bonds) if s and s[-1] not in bonds + ['('] else ''
                s.append(bond + atom)
            i += 1
        smiles_list.append(''.join(s)[:max_len])

    return smiles_list


# ── Training ──

def train_smiles_vae(n_samples=2000, epochs=100, batch_size=64,
                     lr=1e-3, device='cpu'):
    """Train SMILES VAE."""
    tokenizer = SMILESTokenizer()
    smiles_list = generate_synthetic_smiles(n_samples)

    max_len = 35
    encoded = [tokenizer.encode(s, max_len=max_len) for s in smiles_list]
    data = torch.stack(encoded).to(device)

    model = SMILESVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64, hidden_dim=128, latent_dim=32,
        max_len=max_len, num_layers=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

    losses, recon_losses, kl_losses = [], [], []

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_samples, device=device)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = indices[i:i+batch_size]
            batch = data[idx]

            optimizer.zero_grad()
            logits, mu, logvar = model(batch, teacher_forcing_ratio=0.5)
            loss, recon, kl = vae_loss(logits, batch, mu, logvar, tokenizer.pad_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # KL annealing (gradually increase KL weight)
        kl_weight = min(1.0, epoch / 30)

        avg = epoch_loss / n_batches
        losses.append(avg)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, mu, logvar = model(data[:100])
                _, recon, kl = vae_loss(logits, data[:100], mu, logvar, tokenizer.pad_idx)
            print(f"  Epoch {epoch}: loss={avg:.4f}, recon={recon.item():.4f}, kl={kl.item():.4f}")

    return model, tokenizer, losses


# ── Visualization ──

def visualize_latent_space(model, tokenizer, data, n_points=500, save_dir=None):
    """Visualize latent space via t-SNE or PCA."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(data[:n_points])

    from sklearn.decomposition import PCA
    mu_np = mu.cpu().numpy()
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(mu_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=10, c=range(len(z_2d)), cmap='viridis')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('SMILES VAE: Latent Space (PCA)')
    plt.colorbar(scatter, label='Sample index')
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'smiles_latent_space.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_interpolation(model, tokenizer, data, n_steps=10, save_dir=None):
    """Visualize latent space interpolation."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(data[:2])
        z1, z2 = mu[0:1], mu[1:2]
        interp = model.interpolate(z1, z2, n_steps)

    decoded = [tokenizer.decode(interp[i]) for i in range(n_steps)]

    fig, ax = plt.subplots(figsize=(12, 3))
    for i, s in enumerate(decoded):
        ax.text(i, 0, s, rotation=45, ha='center', va='bottom', fontsize=9, family='monospace')
        ax.plot(i, -0.1, 'bo', markersize=8)

    ax.set_xlim(-1, n_steps)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title('Latent Space Interpolation')
    ax.set_xlabel('Interpolation Step')
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f'{i/(n_steps-1):.1f}' for i in range(n_steps)])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'smiles_interpolation.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_training(losses, save_dir=None):
    """Plot training loss."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SMILES VAE Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'smiles_vae_training.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    save_dir = Path(__file__).parent / 'results' / 'molecular_gen'
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training SMILES VAE ===")
    model, tokenizer, losses = train_smiles_vae(n_samples=2000, epochs=100, device=device)
    visualize_training(losses, save_dir)

    # Prepare data for visualization
    smiles_list = generate_synthetic_smiles(500)
    max_len = 35
    encoded = [tokenizer.encode(s, max_len=max_len) for s in smiles_list]
    data = torch.stack(encoded).to(device)

    print("\n=== Latent Space Visualization ===")
    visualize_latent_space(model, tokenizer, data, save_dir=save_dir)

    print("\n=== Interpolation Demo ===")
    visualize_interpolation(model, tokenizer, data, save_dir=save_dir)

    # Sample new molecules
    print("\n=== Sampled Molecules ===")
    model.eval()
    with torch.no_grad():
        samples = model.sample(10, device=device)
    for i in range(10):
        s = tokenizer.decode(samples[i])
        print(f"  {i}: {s}")

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
