import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class AAFusion(nn.Module):
    """
    Adaptive Attention Fusion with head- and layer-weights + rollout.
    Works with HuggingFace ViT attentions: tuple(L) of [B, H, T, T].
    """
    def __init__(self, num_layers: int, num_heads: int, add_residual: bool = True):
        super().__init__()
        self.L = num_layers
        self.H = num_heads
        self.add_residual = add_residual
        # Learnable head weights per layer (L x H) and layer weights (L)
        self.head_logits = nn.Parameter(torch.zeros(self.L, self.H))
        self.layer_logits = nn.Parameter(torch.zeros(self.L))

    @staticmethod
    def _normalize_attn(A: torch.Tensor, eps: float = 1e-6):
        # row-normalize attention matrices to make rollout stable
        return A / (A.sum(dim=-1, keepdim=True) + eps)

    def forward(self, attn_tuple: tuple):
        """
        Returns:
          rollout: [B, T, T]
          cls_to_patch: [B, N]  (N = T-1), unnormalized in [0,1] after min-max
        """
        L = len(attn_tuple)
        assert L == self.L, f"Expected {self.L} layers of attention, got {L}."

        per_layer = []
        for l, A in enumerate(attn_tuple):
            # A: [B, H, T, T]
            B, H, T, _ = A.shape
            head_w = torch.softmax(self.head_logits[l], dim=-1)  # [H]
            A_fused = (A * head_w.view(1, H, 1, 1)).sum(dim=1)   # [B, T, T]

            if self.add_residual:
                I = torch.eye(T, device=A.device, dtype=A.dtype).unsqueeze(0)  # [1,T,T]
                A_fused = A_fused + I

            A_fused = self._normalize_attn(A_fused)  # [B, T, T]
            per_layer.append(A_fused)

        # Layer weighting before rollout (optional but helpful)
        layer_w = torch.softmax(self.layer_logits, dim=0)  # [L]
        # Weighted average as a skip connection for stability
        A_avg = 0
        for w, A in zip(layer_w, per_layer):
            A_avg = A_avg + w * A

        # Rollout
        rollout = per_layer[0]
        for A in per_layer[1:]:
            rollout = torch.bmm(rollout, A)  # [B, T, T]

        # Take CLS -> patch attention from rollout
        cls_to_patch = rollout[:, 0, 1:]  # [B, N]

        # Min–max normalise per-sample for a [0,1] heatmap
        vmin = cls_to_patch.amin(dim=1, keepdim=True)
        vmax = cls_to_patch.amax(dim=1, keepdim=True)
        cls_to_patch = (cls_to_patch - vmin) / (vmax - vmin + 1e-6)  # [B, N]

        return rollout, cls_to_patch

# Minimal helpers for attention rollout + upsampling

def get_rollout_attn(attn_tuple, add_residual=True, image_size=224, patch_size=16):
    """
    attn_tuple: tuple(L) of [B, H, T, T] from HF ViT (output_attentions=True)
    Returns attn_224: [B, 1, image_size, image_size] in [0,1]
    """
    L = len(attn_tuple)
    B, H, T, _ = attn_tuple[0].shape

    rollout = None
    I = torch.eye(T, device=attn_tuple[0].device, dtype=attn_tuple[0].dtype).unsqueeze(0)  # [1,T,T]

    for l in range(L):
        A = attn_tuple[l].mean(dim=1)  # mean over heads -> [B,T,T]
        if add_residual:
            A = A + I
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)  # row-normalise

        rollout = A if rollout is None else torch.bmm(rollout, A)  # [B,T,T]

    # CLS (0) -> patch tokens (1:)
    cls_to_patch = rollout[:, 0, 1:]                     # [B, N]
    grid = image_size // patch_size                      # 14 for 224/16
    attn_14 = cls_to_patch.view(B, 1, grid, grid)        # [B,1,14,14]

    # per-sample min-max normalise to [0,1] for stability
    vmin = attn_14.amin(dim=(2,3), keepdim=True)
    vmax = attn_14.amax(dim=(2,3), keepdim=True)
    attn_14 = (attn_14 - vmin) / (vmax - vmin + 1e-6)

    attn_224 = F.interpolate(attn_14, size=(image_size, image_size), mode='bilinear', align_corners=False)
    return attn_224


# Light BCE alignment loss between attention and weak mask
class AttentionAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    def forward(self, attn_224, weak_mask_224):
        return self.bce(attn_224.clamp(0,1), weak_mask_224)