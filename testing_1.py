import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# ---------------------------------------------------------
# 1. Hybrid Boa-M Architecture Components
# ---------------------------------------------------------

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        scale = x.abs().max() / 127.0 + 1e-8
        x_int = torch.clamp((x / scale).round(), -128, 127)
        return self.linear(x_int * scale)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm + self.eps))

class HybridAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.n_heads, self.window_size = n_heads, window_size
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = QuantLinear(d_model, 3 * d_model)
        self.proj = QuantLinear(d_model, d_model)

        # ALiBi Slopes
        slopes = torch.tensor([2**(-8/n_heads * (i+1)) for i in range(n_heads)])
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))

    def init_cache(self, batch_size, device):
        return {
            "k": torch.zeros(batch_size, self.n_heads, self.window_size, self.head_dim, device=device),
            "v": torch.zeros(batch_size, self.n_heads, self.window_size, self.head_dim, device=device),
            "pos": 0
        }

    def forward(self, x, cache):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        idx = cache["pos"] % self.window_size
        cache["k"][:, :, idx:idx+T] = k
        cache["v"][:, :, idx:idx+T] = v
        cache["pos"] += T

        attn = (q @ cache["k"].transpose(-2, -1)) * self.scale
        # ALiBi Bias Logic
        dist = torch.arange(self.window_size, device=x.device).view(1, 1, 1, self.window_size)
        attn += (dist - (cache["pos"]-1)) * self.slopes

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = (attn @ cache["v"]).transpose(1, 2).reshape(B, T, D)
        return self.proj(out), cache

class LinearRecurrence(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.decay = nn.Parameter(torch.ones(d_model) * -0.2)
        self.gate = nn.Parameter(torch.ones(d_model))
    def forward(self, x, h_prev):
        h_prev = torch.zeros_like(x) if h_prev is None else h_prev
        h_next = torch.exp(self.decay) * h_prev + self.gate * x
        return h_next, h_next

class HybridBlock(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = HybridAttention(d_model, n_heads, window_size)
        self.rec = LinearRecurrence(d_model)
        self.w1, self.w2, self.w3 = QuantLinear(d_model, 4*d_model), QuantLinear(d_model, 4*d_model), QuantLinear(4*d_model, d_model)

    def forward(self, x, cache):
        nx = self.ln1(x)
        h_a, cache['attn'] = self.attn(nx, cache['attn'])
        h_r, cache['rec'] = self.rec(nx, cache['rec'])
        x = x + h_a + h_r
        # SwiGLU
        nx = self.ln2(x)
        x = x + self.w3(F.silu(self.w1(nx)) * self.w2(nx))
        return x, cache

# ---------------------------------------------------------
# 2. Dataset Loader for CMS_DATA_float32.bin
# ---------------------------------------------------------

class CMSBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, seq_len=128):
        self.data = np.fromfile(file_path, dtype=np.float32)
        # Normalize data to byte-like range for embedding (0-255)
        self.data = ((self.data - self.data.min()) / (self.data.max() - self.data.min()) * 255).astype(np.uint8)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len]).long()
        y = torch.from_numpy(self.data[start + 1 : start + self.seq_len + 1]).long()
        return x, y

# ---------------------------------------------------------
# 3. Main Execution Script
# ---------------------------------------------------------

def execute_hybrid_boa_m():
    # Setup Paths
    DATA_PATH = "/boa-constrictor/experiments/cms_experiment/CMS_DATA_float32.bin"
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Creating simulated data for execution demonstration...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        np.random.randn(1000000).astype(np.float32).tofile(DATA_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Model
    model = nn.ModuleList([HybridBlock(256, 8, 128) for _ in range(4)]).to(device)
    embedding = nn.Embedding(256, 256).to(device)
    head = nn.Linear(256, 256).to(device)

    dataset = CMSBinaryDataset(DATA_PATH, seq_len=128)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embedding.parameters()) + list(head.parameters()), lr=1e-4)

    # Performance Monitoring
    latencies = []
    losses = []

    print(f"--- Starting Execution on {device} ---")
    model.train()

    for epoch in range(1):
        for i, (x, y) in enumerate(loader):
            if i > 50: break # Execution limit for demo

            x, y = x.to(device), y.to(device)
            start_time = time.time()

            # Init Caches for batch
            caches = [{'attn': b.attn.init_cache(x.size(0), device), 'rec': None} for b in model]

            # Forward Pass
            h = embedding(x)
            for layer_idx, layer in enumerate(model):
                h, caches[layer_idx] = layer(h, caches[layer_idx])

            logits = head(h)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            latencies.append((time.time() - start_time) * 1000)
            losses.append(loss.item())

            if i % 10 == 0:
                print(f"Step {i} | Loss: {loss.item():.4f} | Latency: {latencies[-1]:.2f}ms")

    # Final Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='blue', label='Cross Entropy Loss')
    plt.title("Training Convergence on CMS Data")
    plt.xlabel("Steps"); plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(latencies, color='orange', label='Inference Latency')
    plt.title("Hybrid Boa-M Timing Stability")
    plt.xlabel("Steps"); plt.ylabel("ms")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    execute_hybrid_boa_m()