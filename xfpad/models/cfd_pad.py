"""CFD-PAD — Channel-wise Feature Denoising PAD detector.

Reimplementation of

    F. Liu, Z. Kong, H. Liu, W. Zhang, L. Shen,
    "Fingerprint Presentation Attack Detection by Channel-wise Feature Denoising",
    IEEE TIFS 17:2963-2976, 2022.   Reference code: github.com/kongzhecn/cfd-pad

Why reimplemented rather than vendored: the reference code targets PyTorch 1.1 /
Python 3.6 (we run 2.7.1 / 3.9), declares no license, and expects a different
data contract. Its backbone is MobileNet-v2 at 224x224 — identical to ours — so
the method plugs into our pipeline and stays comparable with the other audited
backbones (same splits, seeds, protocol).

Structure (as in the paper, Fig. 2):
    G = feature_front   MobileNet-v2 features[0..16]  ->  f  (7x7x160)
    E = feature_back    features[17..18] + global pool ->  embedding (1280)
    C = classify        linear head

Training step:
  1. f = G(x); e = E(f); o = C(e)
  2. importance: for each channel i, zero channel i of f, re-run E and C, and
     accumulate |softmax_spoof(perturbed) - softmax_spoof(base)| into a running
     channel-gap array (accumulated across the whole epoch, as in the reference).
  3. keep the top-k channels of f (k=30 of 160), zero the rest -> f''
  4. L = CE(o, y) + w_t * triplet(E(f''), y_material) + w_2 * CE(C(E(f'')), y)

Two implementation-level deviations, both parameterised and off by default:
  * `importance_chunk`: the reference materialises N*160 samples in one forward,
    which needs batch_size=2 to fit in memory. We process the 160 perturbed
    copies in chunks, so the paper's batch size can be used without OOM. This
    changes nothing numerically.
  * `importance_every`: compute the importance pass every N batches instead of
    every batch. The channel-gap statistic is *accumulated over the epoch*, so
    sub-sampling it approximates the same quantity at ~N times lower cost.
    MUST be validated (channel overlap vs every=1) before being used in a run
    that is reported. Default 1 = faithful to the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

# Split point of MobileNet-v2: features[0..SPLIT-1] = G, features[SPLIT..] = E.
# features[16] outputs 160 channels at 7x7 for a 224x224 input — the map the
# paper denoises.
SPLIT = 17
N_CHANNELS = 160
DEFAULT_K = 30


def _mobilenet_backbone(in_channels: int) -> nn.Module:
    net = mobilenet_v2(weights=None)
    if in_channels != 3:
        old = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            in_channels, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False)
    return net


class CFDPad(nn.Module):
    """Channel-wise feature denoising detector.

    forward(x) returns a single logit (sigmoid -> P(bona fide)), matching
    PadDetector's convention so evaluation code is shared unchanged.
    """

    def __init__(self, in_channels: int = 1, k: int = DEFAULT_K) -> None:
        super().__init__()
        net = _mobilenet_backbone(in_channels)
        self.front = nn.Sequential(*list(net.features[:SPLIT]))
        self.back = nn.Sequential(*list(net.features[SPLIT:]))
        self.head = nn.Linear(1280, 1)
        self.k = int(k)
        self.register_buffer("channel_gap", torch.zeros(N_CHANNELS))

    # -- pieces -------------------------------------------------------------
    def embed(self, f: torch.Tensor) -> torch.Tensor:
        """E: feature map -> pooled embedding."""
        x = self.back(f)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def classify(self, e: torch.Tensor) -> torch.Tensor:
        """C: embedding -> single logit."""
        return self.head(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.embed(self.front(x)))

    # -- channel importance -------------------------------------------------
    @torch.no_grad()
    def update_channel_gap(self, f: torch.Tensor, base_logit: torch.Tensor,
                           chunk: int = 16) -> None:
        """Accumulate the per-channel importance of this batch into channel_gap.

        For every channel i we zero it in f, re-run E and C, and measure how far
        the spoof probability moves. Channels that move it most are the
        discriminative ones. Processed in chunks of channels to bound memory:
        the reference implementation builds all 160 perturbed copies at once,
        which is why it needs batch_size=2.
        """
        base_p = torch.sigmoid(base_logit).squeeze(1)           # (N,)
        gaps = torch.zeros(N_CHANNELS, device=f.device)
        n, c, h, w = f.shape
        for start in range(0, c, chunk):
            idx = torch.arange(start, min(start + chunk, c), device=f.device)
            m = idx.numel()
            rep = f.unsqueeze(1).expand(n, m, c, h, w).clone()   # (N,m,C,H,W)
            rep[:, torch.arange(m), idx, :, :] = 0.0             # zero channel i
            logits = self.classify(self.embed(rep.reshape(n * m, c, h, w)))
            p = torch.sigmoid(logits).view(n, m)
            gaps[idx] = (p - base_p.unsqueeze(1)).abs().sum(dim=0)
        self.channel_gap += gaps

    def denoise(self, f: torch.Tensor) -> torch.Tensor:
        """Zero every channel outside the current top-k of channel_gap."""
        keep = torch.topk(self.channel_gap, self.k).indices
        mask = torch.zeros(N_CHANNELS, device=f.device, dtype=f.dtype)
        mask[keep] = 1.0
        return f * mask.view(1, -1, 1, 1)

    def reset_channel_gap(self) -> None:
        self.channel_gap.zero_()


# ---------------------------------------------------------------------------
# PA-Adaptation loss
# ---------------------------------------------------------------------------

class PAAdaptationLoss(nn.Module):
    """Batch-hard triplet loss over material labels (paper's L_padp).

    Pulls bona fide samples together and pushes different PA materials apart, so
    that distinct attack types do not collapse into one cluster. Labels are the
    MATERIAL ids (0 = bona fide, 1..K = PAI), not the binary live/spoof target.
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        d = torch.cdist(emb, emb, p=2)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
        pos_mask = same & ~eye
        neg_mask = ~same
        if not pos_mask.any() or not neg_mask.any():
            return torch.zeros((), device=emb.device)
        hardest_pos = (d * pos_mask).max(dim=1).values
        d_neg = d.clone()
        d_neg[~neg_mask] = float("inf")
        hardest_neg = d_neg.min(dim=1).values
        valid = torch.isfinite(hardest_neg)
        if not valid.any():
            return torch.zeros((), device=emb.device)
        return F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin).mean()
