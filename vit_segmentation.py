#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import ViTConfig, ViTModel
import os
import numpy as np
from torchvision import transforms
import cv2
import importlib.util
from AAF import AAFusion


# Custom Dataset Class
class CreateDataset(Dataset):
    def __init__(self, npy_dir, selected_scan_ids):
        self.npy_dir = npy_dir
        self.slice_files = [f for f in os.listdir(npy_dir) if f.endswith(
            '.npy') and f.split('_')[0] in selected_scan_ids]

        self.img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

        self.dummy_count = 0
        self.corrupted_log = []

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        file = self.slice_files[idx]
        sample_id = file.replace('.npy', '')
        path = os.path.join(self.npy_dir, file)

        try:
            data = np.load(path, allow_pickle=True)
            img = data[0]  # (3, H, W)
            opf_mask = data[5][0]  # (1, H, W)

            img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
            img = self.img_tf(img)

            mask = self.mask_tf(opf_mask)
            mask = (mask > 0).float()

            return img, mask, sample_id

        except Exception as e:
            print(
                f"[Corrupted] Deleted and replaced: {file} ({e})", flush=True)
            try:
                os.remove(path)
            except Exception as remove_err:
                print(f"Failed to delete {file}: {remove_err}", flush=True)

            self.dummy_count += 1
            self.corrupted_log.append((file, str(e)))

            dummy_img = torch.zeros((3, 224, 224))
            dummy_mask = torch.zeros((1, 224, 224))
            return dummy_img, dummy_mask, sample_id


# Vision Transformer Model
class ViTForSegmentation(nn.Module):
    def __init__(
        self,
        pretrained_model: str = "google/vit-base-patch16-224-in21k",
        dropout_p=0,
        num_classes: int = 1,
        # Compact ViT controls
        use_custom_vit: bool = True,
        custom_hidden_size: int = 256,
        custom_num_layers: int = 8,
        custom_num_heads: int = 4,
        custom_mlp_dim: int = 1024,
        custom_patch_size: int = 16,
    ):
        super().__init__()

        if use_custom_vit:
            cfg = ViTConfig(
                image_size=224,
                patch_size=custom_patch_size,
                hidden_size=custom_hidden_size,
                num_hidden_layers=custom_num_layers,
                num_attention_heads=custom_num_heads,
                intermediate_size=custom_mlp_dim,
                add_pooling_layer=False,
                output_attentions=True,
            )

            self.vit = ViTModel(cfg)
        else:
            self.vit = ViTModel.from_pretrained(
                pretrained_model,
                add_pooling_layer=False,
                output_attentions=True,
                attn_implementation="eager",
            )

        # Derive patch size from config
        self.patch_size = getattr(self.vit.config, 'patch_size', 16)
        self.image_size = 224
        self.num_tokens_no_cls = (self.image_size // self.patch_size) ** 2  # 14x14 = 196

        self.decoder = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),  # [B, 196, 256]
            nn.Dropout(p=dropout_p),
            nn.Unflatten(1, (14, 14)),  # [B, 14, 14, 256]
            Permute(0, 3, 1, 2),        # [B, 256, 14, 14]
            nn.Upsample(scale_factor=self.patch_size, mode='bilinear',align_corners=False),  # [B, 256, 224, 224]
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, sample_ids=None, save_dir=None, input_images=None):
        outputs = self.vit(pixel_values=x)
        attn_maps = outputs.attentions  # tuple(L): [B, H, T, T]
        hidden_states = outputs.last_hidden_state[:, 1:, :]  # [B, 196, C]
        logits = self.decoder(hidden_states)

        # Keep legacy simple mask saving for backward compatibility (non-GAR rollout is handled outside)
        if (not self.training) and sample_ids and save_dir:
            # Removed attention map saving mechanism (attn_mask.png)
            pass

        return logits, attn_maps


# Permute Layer
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


# Loss Function
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-8
        pred_bin = (pred > 0.5).float()
        intersection = (pred_bin * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)
        return bce_loss + dice_loss

# Utility to freeze first N layers of ViT
def freeze_vit_layers(model, freeze_n):
    for name, param in model.vit.encoder.named_parameters():
        layer_idx = int(name.split('.')[1]) if name.startswith("layer") else -1
        if 0 <= layer_idx < freeze_n:
            param.requires_grad = False