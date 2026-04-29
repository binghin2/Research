"""
models_six.py — Six time-series classifiers for stress detection.

All classifiers share the same interface:
  Input:  [B, n_channels, seq_len]
  Output: [B]  (binary logit, no sigmoid applied)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. TimeMixerPP (ICLR 2025) ──────────────────────────────────────────────

class _FFTPeriods(nn.Module):
    def __init__(self, k=3): super().__init__(); self.k = k
    def forward(self, x):
        B, T, _ = x.shape; xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(-1).mean(0); amp[0] = 0
        _, idx = torch.topk(amp, min(self.k, amp.shape[0] - 1))
        return (T // idx.clamp(min=1)).cpu().numpy(), amp[idx]

class _DualAxisAttn(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.aw = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ah = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
    def forward(self, x):                                   # [B, H, W, D]
        B, H, W, D = x.shape
        h = x.reshape(B * H, W, D); h, _ = self.aw(h, h, h, need_weights=False)
        x = self.n1(x + h.reshape(B, H, W, D))
        w = x.permute(0, 2, 1, 3).reshape(B * W, H, D); w, _ = self.ah(w, w, w, need_weights=False)
        return self.n2(x + w.reshape(B, W, H, D).permute(0, 2, 1, 3))

class _TMMBlock(nn.Module):
    def __init__(self, d, k, h, drop):
        super().__init__()
        self.k = k; self.fft = _FFTPeriods(k)
        self.attn = _DualAxisAttn(d, h, drop); self.wt = nn.Linear(d, k)
    def forward(self, x):
        B, T, D = x.shape; periods, _ = self.fft(x); outs = []
        for p in periods:
            p = max(int(p), 2); pad = (p - T % p) % p
            xp = F.pad(x, (0, 0, 0, pad)) if pad else x
            T2 = xp.shape[1]; img = xp.reshape(B, T2 // p, p, D)
            outs.append(self.attn(img).reshape(B, T2, D)[:, :T, :])
        gates = F.softmax(self.wt(x), dim=-1).unsqueeze(2)  # [B, T, 1, k]
        return (torch.stack(outs, dim=-1) * gates).sum(-1) + x

class TimeMixerPPClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=128, n_blocks=2,
                 k_periods=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Linear(n_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([_TMMBlock(d_model, k_periods, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):                                   # [B, C, T]
        x = x.transpose(1, 2); h = self.emb(x) + self.pos[:, :x.shape[1]] 
        for blk in self.blocks: h = blk(h)
        return self.head(self.norm(h).mean(1)).squeeze(-1)


# ─── 2. Medformer (NeurIPS 2024) ─────────────────────────────────────────────
# Multi-granularity patching: one Transformer branch per patch scale, then fuse.

class _MedBranch(nn.Module):
    def __init__(self, seq_len, n_channels, patch_len, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.pl = patch_len; n_p = math.ceil(seq_len / patch_len)
        self.proj = nn.Linear(patch_len * n_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_p, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                         batch_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(enc, n_layers); self.norm = nn.LayerNorm(d_model)
    def forward(self, x):                                   # [B, C, T]
        B, C, T = x.shape; pad = (self.pl - T % self.pl) % self.pl
        if pad: x = F.pad(x, (0, pad))
        p = x.unfold(-1, self.pl, self.pl)                  # [B, C, N, pl]
        B, C, N, P = p.shape
        tokens = self.proj(p.permute(0, 2, 1, 3).reshape(B, N, C * P)) + self.pos[:, :N]
        return self.norm(self.tr(tokens)).mean(1)           # [B, D]

class MedformerClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=128,
                 patch_lens=(4, 8, 16), n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            _MedBranch(seq_len, n_channels, pl, d_model, n_heads, n_layers, dropout)
            for pl in patch_lens])
        k = len(patch_lens)
        self.fuse = nn.Linear(d_model * k, d_model)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):
        out = self.fuse(torch.cat([br(x) for br in self.branches], dim=-1))
        return self.head(out).squeeze(-1)


# ─── 3. TSLANet (ICML 2024) ──────────────────────────────────────────────────
# Adaptive Spectral Block (learnable FFT filter) + Interactive Convolution Block.

class _ASB(nn.Module):
    def __init__(self, n_patches, d_model):
        super().__init__()
        self.filt = nn.Parameter(torch.ones(n_patches // 2 + 1, d_model))
    def forward(self, x):                                   # [B, N, D]
        xf = torch.fft.rfft(x, dim=1) * torch.sigmoid(self.filt)
        return torch.fft.irfft(xf, n=x.shape[1], dim=1)

class _ICB(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.c1 = nn.Conv1d(d, d * 2, 1); self.c2 = nn.Conv1d(d, d, 1); self.dr = nn.Dropout(drop)
    def forward(self, x):                                   # [B, N, D]
        h = self.c1(x.transpose(1, 2))                     # [B, 2D, N]
        h1, h2 = h.chunk(2, dim=1)
        return x + self.dr(self.c2(h1 * torch.sigmoid(h2))).transpose(1, 2)

class _TSLABlock(nn.Module):
    def __init__(self, n_p, d, drop):
        super().__init__()
        self.asb = _ASB(n_p, d); self.icb = _ICB(d, drop)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
    def forward(self, x):
        return self.n2(self.icb(self.n1(x + self.asb(x))))

class TSLANetClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=128,
                 patch_len=8, patch_stride=4, n_blocks=3, dropout=0.1):
        super().__init__()
        self.pl = patch_len; self.ps = patch_stride
        n_p = (seq_len - patch_len) // patch_stride + 1
        self.proj = nn.Linear(patch_len * n_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_p, d_model) * 0.02)
        self.blocks = nn.ModuleList([_TSLABlock(n_p, d_model, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        B, C, T = x.shape
        p = x.unfold(-1, self.pl, self.ps)                  # [B, C, N, pl]
        B, C, N, P = p.shape
        h = self.proj(p.permute(0, 2, 1, 3).reshape(B, N, C * P)) + self.pos[:, :N]
        for blk in self.blocks: h = blk(h)
        return self.head(self.norm(h).mean(1)).squeeze(-1)


# ─── 4. ModernTCN (ICLR 2024) ────────────────────────────────────────────────
# Large-kernel depthwise conv (temporal mixing) + pointwise FFN (channel mixing).

class _MTCNBlock(nn.Module):
    def __init__(self, d, lk, drop):
        super().__init__()
        self.dw = nn.Conv1d(d, d, lk, padding=lk // 2, groups=d)
        self.n1 = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, d * 4, 1); self.pw2 = nn.Conv1d(d * 4, d, 1)
        self.n2 = nn.LayerNorm(d); self.dr = nn.Dropout(drop)
    def forward(self, x):                                   # [B, D, N]
        h = self.dw(self.n1(x.transpose(1, 2)).transpose(1, 2))
        x = x + self.dr(h)
        h = self.pw2(F.gelu(self.pw1(self.n2(x.transpose(1, 2)).transpose(1, 2))))
        return x + self.dr(h)

class ModernTCNClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=128,
                 patch_len=8, patch_stride=4, n_blocks=3, dropout=0.1):
        super().__init__()
        self.pl = patch_len; self.ps = patch_stride
        n_p = (seq_len - patch_len) // patch_stride + 1
        lk = min(51, n_p * 2 - 1) | 1                      # odd kernel, ≤ 51
        self.proj = nn.Linear(patch_len * n_channels, d_model)
        self.blocks = nn.ModuleList([_MTCNBlock(d_model, lk, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        B, C, T = x.shape
        p = x.unfold(-1, self.pl, self.ps)                  # [B, C, N, pl]
        B, C, N, P = p.shape
        h = self.proj(p.permute(0, 2, 1, 3).reshape(B, N, C * P)).transpose(1, 2)  # [B, D, N]
        for blk in self.blocks: h = blk(h)
        return self.head(self.norm(h.transpose(1, 2)).mean(1)).squeeze(-1)


# ─── 5. CrossGNN (NeurIPS 2023) ──────────────────────────────────────────────
# Per-channel temporal conv (intra) + learnable-adjacency cross-channel mixing (inter).

class _CrossGNNLayer(nn.Module):
    def __init__(self, d, n_ch, drop):
        super().__init__()
        self.tc = nn.Sequential(nn.Conv1d(d, d, 3, padding=1, groups=d), nn.GELU(), nn.Dropout(drop))
        self.adj = nn.Parameter(torch.eye(n_ch) + 0.01 * torch.randn(n_ch, n_ch))
        self.cross = nn.Linear(d, d)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
    def forward(self, x):                                   # [B, C, N, D]
        B, C, N, D = x.shape
        h = x.reshape(B * C, N, D).transpose(1, 2)
        h = self.tc(h).transpose(1, 2).reshape(B, C, N, D)
        x = self.n1(x + h)
        adj = F.softmax(self.adj, dim=-1)                   # [C, C]
        xc = self.cross(torch.einsum('ij,bjnd->bind', adj, x))
        return self.n2(x + xc)

class CrossGNNClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=64,
                 patch_len=8, patch_stride=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.pl = patch_len; self.ps = patch_stride; self.nc = n_channels
        self.proj = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList([_CrossGNNLayer(d_model, n_channels, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * n_channels, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        B, C, T = x.shape
        p = x.unfold(-1, self.pl, self.ps)                  # [B, C, N, pl]
        h = self.proj(p)                                    # [B, C, N, D]
        for lyr in self.layers: h = lyr(h)
        h = self.norm(h).mean(2)                            # [B, C, D]
        return self.head(h.reshape(B, C * h.shape[-1])).squeeze(-1)


# ─── 6. TimesNet (ICLR 2023) ─────────────────────────────────────────────────
# Reshape 1-D series to 2-D via FFT periods; apply Inception 2-D conv; fold back.

class _InceptionBlock(nn.Module):
    def __init__(self, d, ks=(1, 3, 5, 7)):
        super().__init__()
        each = d // len(ks); sizes = [each] * (len(ks) - 1) + [d - each * (len(ks) - 1)]
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, sz, k, padding=k // 2), nn.BatchNorm2d(sz), nn.GELU())
            for k, sz in zip(ks, sizes)])
    def forward(self, x): return torch.cat([c(x) for c in self.convs], dim=1)

class _TimesBlock(nn.Module):
    def __init__(self, d):
        super().__init__(); self.conv = _InceptionBlock(d); self.norm = nn.LayerNorm(d)
    def forward(self, x, periods):                          # x: [B, T, D]
        B, T, D = x.shape; outs = []
        for p in periods:
            p = max(int(p), 2); pad = (p - T % p) % p
            xp = F.pad(x.transpose(1, 2), (0, pad))        # [B, D, T+pad]
            T2 = xp.shape[2]
            img = self.conv(xp.reshape(B, D, T2 // p, p))  # [B, D, H, p]
            outs.append(img.reshape(B, D, T2)[:, :, :T].transpose(1, 2))
        return self.norm(x + torch.stack(outs).mean(0))

class TimesNetClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=64, n_blocks=2,
                 k_periods=3, dropout=0.1):
        super().__init__()
        self.k = k_periods; self.emb = nn.Linear(n_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([_TimesBlock(d_model) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def _periods(self, h):
        B, T, _ = h.shape; xf = torch.fft.rfft(h, dim=1)
        amp = xf.abs().mean(-1).mean(0); amp[0] = 0
        _, idx = torch.topk(amp, min(self.k, amp.shape[0] - 1))
        return (T // idx.clamp(min=1)).clamp(min=2).cpu().tolist()
    def forward(self, x):
        x = x.transpose(1, 2); h = self.emb(x) + self.pos[:, :x.shape[1]]
        periods = self._periods(h)
        for blk in self.blocks: h = blk(h, periods)
        return self.head(self.norm(h).mean(1)).squeeze(-1)


# ─── 7. Mamba-2 (pure PyTorch, ICLR 2024) ────────────────────────────────────
# CNN frontend for downsampling + selective-SSM blocks.

class _Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand; self.d_state = d_state
        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log   = nn.Parameter(torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)))
        self.D       = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj= nn.Linear(self.d_inner, d_model, bias=False)

    def _scan(self, u, dt, A, B, C, D):
        # Forces float32 inside the state-accumulation loop for numerical stability.
        u = u.float(); dt = dt.float(); A = A.float(); B = B.float(); C = C.float()
        B_sz, L, _ = u.shape
        h = torch.zeros(B_sz, self.d_inner, self.d_state, device=u.device)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A[None])    # [B, d, S]
            dB = dt[:, t, :, None] * B[:, t, None, :]      # [B, d, S]
            h  = dA * h + dB * u[:, t, :, None]
            ys.append((h * C[:, t, None, :]).sum(-1))       # [B, d]
        return torch.stack(ys, 1) + u * D[None, None, :]   # [B, L, d]

    def forward(self, x):                                   # [B, L, d_model]
        res = x; x = self.norm(x)
        xz  = self.in_proj(x); x_p, z = xz.chunk(2, dim=-1)
        x_t = self.conv1d(x_p.transpose(1, 2))[:, :, :x_p.shape[1]]
        x_p = F.silu(x_t.transpose(1, 2))
        tmp = self.x_proj(x_p)
        B_p = tmp[..., :self.d_state]; C_p = tmp[..., self.d_state:2 * self.d_state]
        dt  = F.softplus(self.dt_proj(tmp[..., -1:]))
        A   = -torch.exp(self.A_log)
        y   = self._scan(x_p, dt, A, B_p, C_p, self.D).to(x_p.dtype)
        return self.out_proj(y * F.silu(z)) + res

class Mamba2Classifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=64, d_state=16,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, d_model, 5, padding=2), nn.BatchNorm1d(d_model), nn.GELU(), nn.MaxPool1d(2),
        )
        self.blocks = nn.ModuleList([_Mamba2Block(d_model, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):                                   # [B, C, T]
        h = self.cnn(x).transpose(1, 2)                    # [B, T//4, d_model]
        for blk in self.blocks: h = blk(h)
        return self.head(self.norm(h).mean(1)).squeeze(-1)


# ─── 8. iTransformer (ICLR 2024) ─────────────────────────────────────────────
# Inverted Transformer: each channel is one token; its time series is the embedding.

class iTransformerClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=128, n_heads=8,
                 n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.var_embed = nn.Linear(seq_len, d_model)
        self.norm_in   = nn.LayerNorm(d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                         batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):                                   # [B, C, T]
        tokens = self.norm_in(self.var_embed(x))            # [B, C, d_model]
        out    = self.transformer(tokens).mean(1)           # [B, d_model]
        return self.head(out).squeeze(-1)


# ─── 9. PatchTST (ICLR 2023) ─────────────────────────────────────────────────
# Patch-based Transformer; multi-channel patches are flattened per patch token.

class PatchTSTClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, patch_len=16, patch_stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.pl = patch_len; self.ps = patch_stride
        n_p = (seq_len - patch_len) // patch_stride + 1
        self.patch_embed = nn.Linear(patch_len * n_channels, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_p, d_model) * 0.02)
        self.norm_in     = nn.LayerNorm(d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                         batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):                                   # [B, C, T]
        B, C, T = x.shape
        p = x.unfold(-1, self.pl, self.ps)                  # [B, C, N, pl]
        p = p.permute(0, 2, 1, 3).reshape(B, -1, C * self.pl)  # [B, N, C*pl]
        tokens = self.norm_in(self.patch_embed(p) + self.pos_embed[:, :p.shape[1]])
        out = self.transformer(tokens).mean(1)              # [B, d_model]
        return self.head(out).squeeze(-1)


# ─── 10. Chronos-2 (Amazon, 2024) — fine-tuning wrapper ──────────────────────
# Requires: pip install chronos-forecasting
# Wraps ChronosPipeline; top-2 T5 encoder layers + head are fine-tuned.

class Chronos2Classifier(nn.Module):
    def __init__(self, chronos_pipeline, subsample: int = 4, dropout: float = 0.3):
        super().__init__()
        self.subsample   = subsample
        self.tokenizer   = chronos_pipeline.tokenizer
        self.chron_model = chronos_pipeline.model
        d_model = chronos_pipeline.model.model.config.d_model

        n_total = len(chronos_pipeline.model.model.encoder.block)
        n_freeze = max(0, n_total - 2)
        for i, blk in enumerate(chronos_pipeline.model.model.encoder.block):
            if i < n_freeze:
                for p in blk.parameters(): p.requires_grad = False
        print(f"  Chronos: frozen {n_freeze}/{n_total} encoder blocks")

        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def _encode(self, ch: torch.Tensor) -> torch.Tensor:   # ch: [B, L]
        device = ch.device
        ids, mask, _ = self.tokenizer.context_input_transform(ch.detach().cpu())
        ids, mask = ids.to(device), mask.to(device)
        enc = self.chron_model.encode(input_ids=ids, attention_mask=mask)
        mf  = mask.unsqueeze(-1).float()
        return (enc * mf).sum(1) / mf.sum(1).clamp(min=1)  # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # [B, 1, T]
        x_sub = x[:, :, ::self.subsample].squeeze(1)       # [B, T//subsample]
        return self.head(self._encode(x_sub)).squeeze(-1)


# ─── 11. TransformerCNN ───────────────────────────────────────────────────────
# 1D CNN frontend (local feature extraction) + Transformer encoder (global context).

class TransformerCNNClassifier(nn.Module):
    def __init__(self, seq_len=240, n_channels=1, d_model=128,
                 n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                 n_conv_layers=2, cnn_kernel=5):
        super().__init__()
        cnn = []
        in_ch = n_channels
        for i in range(n_conv_layers):
            out_ch = d_model
            cnn += [nn.Conv1d(in_ch, out_ch, cnn_kernel, padding=cnn_kernel // 2),
                    nn.BatchNorm1d(out_ch), nn.GELU()]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                         batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):                                    # [B, C, T]
        h = self.cnn(x).transpose(1, 2)                     # [B, T, d_model]
        h = h + self.pos[:, :h.shape[1]]
        h = self.transformer(h)
        return self.head(self.norm(h).mean(1)).squeeze(-1)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(name: str, W: int = 240, n_channels: int = 1, **kwargs):
    common = dict(seq_len=W, n_channels=n_channels, dropout=kwargs.get('dropout', 0.1))
    if name == "TimeMixerPP":
        return TimeMixerPPClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            n_blocks=kwargs.get('n_blocks', 2),
            k_periods=kwargs.get('k_periods', 3),
            n_heads=kwargs.get('n_heads', 4))
    if name == "Medformer":
        return MedformerClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            patch_lens=[4, 8, 16],
            n_heads=kwargs.get('n_heads', 4),
            n_layers=kwargs.get('n_layers', 2))
    if name == "TSLANet":
        return TSLANetClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            patch_len=8, patch_stride=4,
            n_blocks=kwargs.get('n_blocks', 3))
    if name == "ModernTCN":
        return ModernTCNClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            patch_len=8, patch_stride=4,
            n_blocks=kwargs.get('n_blocks', 3))
    if name == "CrossGNN":
        return CrossGNNClassifier(**common,
            d_model=kwargs.get('d_model', 64),
            patch_len=8, patch_stride=4,
            n_layers=kwargs.get('n_layers', 2))
    if name == "TimesNet":
        return TimesNetClassifier(**common,
            d_model=kwargs.get('d_model', 64),
            n_blocks=kwargs.get('n_blocks', 2),
            k_periods=kwargs.get('k_periods', 3))
    if name == "Mamba2":
        return Mamba2Classifier(**common,
            d_model=kwargs.get('d_model', 64),
            d_state=kwargs.get('d_state', 16),
            n_layers=kwargs.get('n_layers', 3))
    if name == "iTransformer":
        return iTransformerClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            n_heads=kwargs.get('n_heads', 8),
            n_layers=kwargs.get('n_layers', 3),
            d_ff=kwargs.get('d_ff', 256))
    if name == "PatchTST":
        return PatchTSTClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            patch_len=16, patch_stride=8,
            n_heads=kwargs.get('n_heads', 8),
            n_layers=kwargs.get('n_layers', 3),
            d_ff=kwargs.get('d_ff', 256))
    if name == "Chronos2":
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Chronos2 requires pipeline= keyword argument")
        return Chronos2Classifier(pipeline, subsample=4,
                                   dropout=kwargs.get('dropout', 0.3))
    if name == "TransformerCNN":
        return TransformerCNNClassifier(**common,
            d_model=kwargs.get('d_model', 128),
            n_heads=kwargs.get('n_heads', 4),
            n_layers=kwargs.get('n_layers', 2),
            d_ff=kwargs.get('d_ff', 256),
            n_conv_layers=kwargs.get('n_conv_layers', 2),
            cnn_kernel=kwargs.get('cnn_kernel', 5))
    raise ValueError(f"Unknown model: {name}")
