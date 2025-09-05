#!/usr/bin/env python
# coding: utf-8
# Author: Diksha Nigam
# Date: 2025-01-09

from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import ViTModel
from tqdm.notebook import tqdm
from torchvision import transforms
import random
import gc
import cv2
import sys
import importlib.util
from transformers import ViTConfig, ViTModel
from vit_segmentation import ViTForSegmentation, BCEDiceLoss, freeze_vit_layers

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# # Train and log
def train_vit_model(
    data_dir,
    selected_ids_txt,
    save_attn_dir=None,
    save_model_path=None,
    freeze_n_layers=0, 
    dropout_p=0,
    epochs=5,
    batch_size=4,
    lr=1e-4,
    resume_from=None,
    train=True,
    # Compact ViT controls
    use_custom_vit: bool = True,
    custom_hidden_size: int = 256,
    custom_num_layers: int = 6,
    custom_num_heads: int = 4,
    custom_mlp_dim: int = 1024,
    custom_patch_size: int = 16,
    loss_log_file: str = "loss_log.txt",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(selected_ids_txt, 'r') as f:
        all_ids = [line.strip().split(',')[0] for line in f if line.strip()]
    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42)
    print(
        f"Training scans: {len(train_ids)} | Validation scans: {len(val_ids)}", flush=True)

    train_dataset = CreateDataset(data_dir, set(train_ids))
    val_dataset = CreateDataset(data_dir, set(val_ids))

    print(f"Total training slices: {len(train_dataset)}", flush=True)
    print(f"Total validation slices: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ViTForSegmentation(
        dropout_p=dropout_p,
        use_custom_vit=use_custom_vit,
        custom_hidden_size=custom_hidden_size,
        custom_num_layers=custom_num_layers,
        custom_num_heads=custom_num_heads,
        custom_mlp_dim=custom_mlp_dim,
        custom_patch_size=custom_patch_size,
    ).to(device)
    freeze_vit_layers(model, freeze_n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss()

    if torch.cuda.is_available():
        print(
            f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
    else:
        print("Using CPU", flush=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}", flush=True)

    if save_attn_dir:
        os.makedirs(save_attn_dir, exist_ok=True)
    else:
        save_attn_dir = "attn_maps"
        os.makedirs(save_attn_dir, exist_ok=True)

    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
    else:
        save_model_path = "models"
        os.makedirs(save_model_path, exist_ok=True)
        checkpoint_dir = os.path.join(save_model_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    if use_custom_vit:
        print(
            f"Training custom ViT: hidden_size={custom_hidden_size}, layers={custom_num_layers}, heads={custom_num_heads}, mlp_dim={custom_mlp_dim}, patch_size={custom_patch_size}", flush=True)
    else:
        print("Training pretrained ViT: google/vit-base-patch16-224-in21k", flush=True)

    # Resume-from logic
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from checkpoint: {resume_from}", flush=True)
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
    else:
        start_epoch = 0

    best_val_dice = 0.0
    epochs_since_improvement = 0
    patience = 7

    if train:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0.0
            total_pixels = 0.0
            total_intersection = 0.0
            total_union = 0.0
            total_tp = 0.0
            total_fp = 0.0
            total_fn = 0.0
            total_gt_sum = 0.0
            total_pred_sum = 0.0

            for imgs, masks, ids in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", dynamic_ncols=True, file=sys.stdout, mininterval=0.1):
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds, _ = model(imgs, sample_ids=None,
                                 save_dir=None, input_images=imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                preds_bin = (preds > 0.3).float()
                total_correct += (preds_bin == masks).float().sum().item()
                total_pixels += masks.numel()
                total_intersection += (preds_bin * masks).sum().item()
                total_union += preds_bin.sum().item() + masks.sum().item()
                total_tp += (preds_bin * masks).sum().item()
                total_fp += (preds_bin * (1 - masks)).sum().item()
                total_fn += ((1 - preds_bin) * masks).sum().item()
                total_gt_sum += masks.sum().item()
                total_pred_sum += preds_bin.sum().item()

            avg_accuracy = total_correct / total_pixels
            avg_dice = (2 * total_intersection) / (total_union + 1e-8)
            precision = total_tp / (total_tp + total_fp + 1e-8)
            sensitivity = total_tp / (total_tp + total_fn + 1e-8)
            avg_loss = total_loss / len(train_loader)

            print(
                f"Epoch {epoch+1}/{epochs} [Train] - Loss: {avg_loss:.4f} - Acc: {avg_accuracy:.4f} - Dice: {avg_dice:.4f} - Precision: {precision:.4f} - Sensitivity: {sensitivity:.4f}", flush=True)
            with open("loss_vit_mini_dropout.txt", "a") as logf:
                logf.write(
                    f"Epoch {epoch+1}/{epochs} [Train] - Loss: {avg_loss:.4f} - Acc: {avg_accuracy:.4f} - Dice: {avg_dice:.4f} - Precision: {precision:.4f} - Sensitivity: {sensitivity:.4f}\n")

            # --- VALIDATION ---
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0.0
                val_pixels = 0.0
                val_intersection = 0.0
                val_union = 0.0
                val_tp = 0.0
                val_fp = 0.0
                val_fn = 0.0

                for imgs, masks, ids in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", dynamic_ncols=True, file=sys.stdout, mininterval=0.1):
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds, _ = model(imgs, sample_ids=None,
                                     save_dir=None, input_images=imgs)
                    loss = criterion(preds, masks)
                    val_loss += loss.item()

                    preds_bin = (preds > 0.3).float()
                    val_correct += (preds_bin == masks).float().sum().item()
                    val_pixels += masks.numel()
                    val_intersection += (preds_bin * masks).sum().item()
                    val_union += preds_bin.sum().item() + masks.sum().item()
                    val_tp += (preds_bin * masks).sum().item()
                    val_fp += (preds_bin * (1 - masks)).sum().item()
                    val_fn += ((1 - preds_bin) * masks).sum().item()

                val_acc = val_correct / val_pixels
                val_dice = (2 * val_intersection) / (val_union + 1e-8)
                val_prec = val_tp / (val_tp + val_fp + 1e-8)
                val_rec = val_tp / (val_tp + val_fn + 1e-8)
                val_loss /= len(val_loader)

                print(
                    f"Epoch {epoch+1} [Validation] - Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - Dice: {val_dice:.4f} - Precision: {val_prec:.4f} - Sensitivity: {val_rec:.4f}", flush=True)
                with open("loss_vit_mini_dropout.txt", "a") as logf:
                    logf.write(
                        f"Epoch {epoch+1} [Validation] - Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - Dice: {val_dice:.4f} - Precision: {val_prec:.4f} - Sensitivity: {val_rec:.4f}\n")

                if val_dice > best_val_dice and save_model_path:
                    best_val_dice = val_dice
                    epochs_since_improvement = 0
                    os.makedirs(os.path.dirname(
                        save_model_path), exist_ok=True)
                    best_model_path = os.path.join(
                        save_model_path, f"vit_best_val_{best_val_dice}.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_dice': best_val_dice,
                    }, best_model_path)
                    print(
                        f"Best model (val Dice {val_dice:.4f}) saved to {best_model_path}", flush=True)
                else:
                    epochs_since_improvement += 1
                    print(
                        f"No improvement for {epochs_since_improvement} epoch(s).", flush=True)
                    if epochs_since_improvement >= patience:
                        print(
                            f"Early stopping: No improvement for {patience} consecutive epochs.", flush=True)
                        break

            gc.collect()
            torch.cuda.empty_cache()

            if epoch + 1 > 10 and (epoch + 1) % 4 == 0 and save_model_path:
                checkpoint_dir = os.path.join(
                    os.path.dirname(save_model_path), "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"vit_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}", flush=True)

    if save_model_path and train:
        final_model_path = os.path.join(
            save_model_path, "vit_finetuned_final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}", flush=True)


# Training invocation with specified parameters
freeze_options = [0, 2, 4, 6]
dropout_options = [0.1, 0.3, 0.5]

for f in freeze_options:
    for d in dropout_options:
        print(f"\n[Running freeze={f} | dropout={d}]")
        train_vit_model(
            data_dir="filtered_dataset",
            selected_ids_txt="top_n_scans/full_train_dataset/exp07_highest_300_scans.txt",
            save_attn_dir=f"top_n_scans/full_train_dataset/vit_mini_without_dropouttop_n_scans/full_train_dataset/vit_mini_with_dropout_and_freeze/freeze_{f}_drop_{int(d*100)}/attn_maps",
            save_model_path=f"top_n_scans/full_train_dataset/vit_mini_without_dropouttop_n_scans/full_train_dataset/vit_mini_with_dropout_and_freeze/freeze_{f}_drop_{int(d*100)}/models",
            freeze_n_layers=f, 
            dropout_p=d,
            epochs=50,
            batch_size=8,
            lr=1e-4,
            resume_from=None,
            train=True,
            use_custom_vit=True,
            custom_hidden_size=256,
            custom_num_layers=8,
            custom_num_heads=4,
            custom_mlp_dim=1024,
            custom_patch_size=16,
            loss_log_file=f"top_n_scans/full_train_dataset/vit_mini_without_dropouttop_n_scans/full_train_dataset/vit_mini_with_dropout_and_freeze/freeze_{f}_drop_{int(d*100)}/loss.txt",
        )