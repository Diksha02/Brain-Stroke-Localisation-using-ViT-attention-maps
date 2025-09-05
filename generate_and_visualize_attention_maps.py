import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import ViTConfig, ViTModel
import pandas as pd
from vit_segmentation import ViTForSegmentation, Permute, BCEDiceLoss
from AAF import AAFusion


def binarize(x, thr=0.5):
    return (x > thr).astype(np.uint8)


def evaluate_attention_on_ids(model, device, npy_dir, sample_ids, output_size=(224, 224),
                              thr_cls="otsu", thr_roll="otsu"):
    """Returns dict with mean IoU/Dice for CLS/Rollout vs GT and vs OPF."""
    stats = {
        "CLS_vs_GT": [], "Roll_vs_GT": [],
        "CLS_vs_OPF": [], "Roll_vs_OPF": [],
        "OPF_vs_GT": []
    }
    model.eval()
    for sid in sample_ids:
        f = os.path.join(npy_dir, f"{sid}.npy")
        if not os.path.exists(f):
            print(f"[skip] missing {f}")
            continue
        data = np.load(f, allow_pickle=True)
        img_chw = data[0]     # (3,H,W)
        gt = data[1][0]       # (H,W)
        opf = data[5][0]      # (H,W)

        # prep tensors
        img = torch.tensor(img_chw).unsqueeze(
            0).float().to(device)  # [1,3,H,W]
        img = F.interpolate(img, size=output_size,
                            mode='bilinear', align_corners=False)
        img = TF.normalize(img, mean=[0.5]*3, std=[0.5]*3)
        gt_r = cv2.resize(gt,  output_size,
                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        opf_r = cv2.resize(
            opf, output_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        with torch.no_grad():
            preds, attns = model(img)

        # extract both maps
        cls_bin, cls_heat = extract_cls_attention_map(
            attns, sample_idx=0, output_size=output_size)   # (H,W) uint8 + float
        roll_bin, roll_heat = extract_rollout_attention_map(
            attns, sample_idx=0, output_size=output_size)

        # metrics
        stats["OPF_vs_GT"].append(calculate_metrics(opf_r, gt_r))
        stats["CLS_vs_GT"].append(calculate_metrics(cls_bin, gt_r))
        stats["Roll_vs_GT"].append(calculate_metrics(roll_bin, gt_r))
        stats["CLS_vs_OPF"].append(calculate_metrics(cls_bin, opf_r))
        stats["Roll_vs_OPF"].append(calculate_metrics(roll_bin, opf_r))

    # aggregate (mean)
    def agg(key):
        if not stats[key]:
            return (0.0, 0.0)
        ious = [x[0] for x in stats[key]]
        dices = [x[1] for x in stats[key]]
        return (float(np.mean(ious)), float(np.mean(dices)))

    summary = {k: {"IoU_mean": agg(k)[0], "Dice_mean": agg(k)[
        1]} for k in stats}
    return summary


def _row_normalise(A, eps=1e-6):
    return A / (A.sum(dim=-1, keepdim=True) + eps)


def _to_224(attn_14, output_size=(224, 224)):
    up = F.interpolate(attn_14.unsqueeze(0).unsqueeze(0), size=output_size,
                       mode='bilinear', align_corners=False)[0, 0]
    vmin, vmax = up.min(), up.max()
    up = (up - vmin) / (float(vmax - vmin) + 1e-8)
    return up

# def _threshold_map(attn_224_np, mode="otsu"):
#     if isinstance(mode, (float, int)):  # fixed numeric threshold
#         return (attn_224_np > float(mode)).astype(np.uint8)
#     if mode == "mean":
#         return (attn_224_np > float(attn_224_np.mean())).astype(np.uint8)
#     # Otsu
#     arr = (attn_224_np * 255.0).astype(np.uint8)
#     _, thr_mask = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return (thr_mask > 0).astype(np.uint8)


def _threshold_map_robust(attn_224_np, mode="auto", topk=0.01):
    x = attn_224_np.astype(np.float32)
    v = float(x.var())
    # if almost flat, keep top-k% pixels
    if v < 1e-6:
        thr = float(np.quantile(x, 1.0 - topk))
        return (x > thr).astype(np.uint8)

    if isinstance(mode, (float, int)):  # fixed numeric threshold
        return (x > float(mode)).astype(np.uint8)

    if mode in ("auto", "otsu"):
        arr = (x * 255.0).astype(np.uint8)
        _, mask = cv2.threshold(
            arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask01 = (mask > 0).astype(np.uint8)
        # if empty or almost full, fallback to top-k%
        s = int(mask01.sum())
        if s == 0 or s > 0.5 * mask01.size:
            thr = float(np.quantile(x, 1.0 - topk))
            return (x > thr).astype(np.uint8)
        return mask01

    thr = float(np.quantile(x, 1.0 - topk))
    return (x > thr).astype(np.uint8)


def quick_brain_mask(gray_224):
    arr = (gray_224 * 255).astype(np.uint8)
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return (th > 0).astype(np.uint8)


def downsample_mask_14x14(mask_224, patch=16):
    # majority vote per patch to a 14x14 mask
    m = mask_224.astype(np.float32)
    m = cv2.resize(m, (224//patch, 224//patch),
                   interpolation=cv2.INTER_AREA)  # ~mean
    return (m > 0.5).astype(np.float32)  # keep as float for multiplication


def extract_cls_attention_map(attns, sample_idx=0, output_size=(224, 224),
                              threshold_mode="auto", topk=0.01, brain_mask_14=None):
    last = attns[-1][sample_idx]     # [H,T,T]
    A = last.mean(dim=0)             # [T,T]
    cls_to_patch = A[0, 1:]          # [T-1]
    grid = int(np.sqrt(cls_to_patch.numel()))
    attn_14 = cls_to_patch.view(grid, grid)
    if brain_mask_14 is not None:    # suppress background tokens
        attn_14 = attn_14 * \
            torch.tensor(brain_mask_14, device=attn_14.device,
                         dtype=attn_14.dtype)
    attn_224 = _to_224(attn_14, output_size).cpu().numpy().astype(np.float32)
    attn_bin = _threshold_map_robust(attn_224, mode=threshold_mode, topk=topk)
    return attn_bin, attn_224


def extract_rollout_attention_map(attns, sample_idx=0, output_size=(224, 224),
                                  add_residual=True, start_layer=1,
                                  threshold_mode="auto", topk=0.01, brain_mask_14=None):
    L = len(attns)
    _, _, T, _ = attns[0].shape
    I = torch.eye(T, device=attns[0].device, dtype=attns[0].dtype)
    R = torch.eye(T, device=attns[0].device,
                  dtype=attns[0].dtype)  # start from I

    for l in range(start_layer, L):   # often skip the very first layer(s)
        A = attns[l][sample_idx].mean(dim=0)   # [T,T]
        if add_residual:
            A = A + I
        A = _row_normalise(A)
        R = A @ R  # left-multiply

    cls_to_patch = R[0, 1:]
    grid = int(np.sqrt(cls_to_patch.numel()))
    attn_14 = cls_to_patch.view(grid, grid)
    if brain_mask_14 is not None:
        attn_14 = attn_14 * \
            torch.tensor(brain_mask_14, device=attn_14.device,
                         dtype=attn_14.dtype)
    attn_224 = _to_224(attn_14, output_size).cpu().numpy().astype(np.float32)
    attn_bin = _threshold_map_robust(attn_224, mode=threshold_mode, topk=topk)
    return attn_bin, attn_224


# Helper function to calculate IoU and Dice scores
def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)

    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()

    iou = intersection / union if union != 0 else 0
    union_sum = pred_bin.sum() + target_bin.sum()
    dice = (2 * intersection) / union_sum if union_sum != 0 else 0

    return iou, dice

# Helper function to create an overlay image


def create_overlay(image, mask, color, alpha=0.5, apply_threshold=True):
    if image.ndim == 2:  # Convert grayscale to RGB for consistent overlay
        image_rgb = cv2.cvtColor(
            (image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = (image * 255).astype(np.uint8)

    overlay = image_rgb.copy().astype(np.float32)

    if apply_threshold:
        mask_bin = (mask > 0).astype(np.uint8)
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        colored_mask[mask_bin > 0] = color
    else:
        # For heatmaps, apply color gradient based on intensity
        # First normalize mask to 0-255 if it's not already
        mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_norm = (mask_norm * 255).astype(np.uint8)

        # Apply a colormap (e.g., JET for heatmaps)
        heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
        # Blend heatmap with background image
        colored_mask = heatmap

    # Resize colored_mask to match image_rgb if necessary (for consistency in overlays)
    if colored_mask.shape[:2] != image_rgb.shape[:2]:
        colored_mask = cv2.resize(
            colored_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Ensure colored_mask is RGB if image_rgb is RGB
    if image_rgb.shape[-1] == 3 and colored_mask.ndim == 2:  # Grayscale mask
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2RGB)
    elif image_rgb.shape[-1] == 3 and colored_mask.shape[-1] == 4:  # RGBA mask
        colored_mask = colored_mask[..., :3]  # Remove alpha channel

    # Ensure both are float32 before addWeighted
    image_rgb_float = image_rgb.astype(np.float32)
    colored_mask_float = colored_mask.astype(np.float32)

    combined = cv2.addWeighted(
        image_rgb_float, 1 - alpha, colored_mask_float, alpha, 0)
    return np.clip(combined / 255.0, 0, 1)  # Normalize back to 0-1 for display

# Attention Rollout Function


def attention_rollout(attn_matrices, start_layer=0):
    """
    Computes joint attention using the attention rollout method.
    attn_matrices: list of [num_heads, num_tokens, num_tokens] tensors
    Returns: [num_tokens, num_tokens] joint attention matrix
    """
    result = torch.eye(attn_matrices[0].size(-1)).to(attn_matrices[0].device)
    for attn in attn_matrices[start_layer:]:
        attn_heads_avg = attn.mean(dim=0)  # [num_tokens, num_tokens]
        attn_heads_avg = attn_heads_avg + \
            torch.eye(attn_heads_avg.size(0)).to(attn_heads_avg.device)
        attn_heads_avg = attn_heads_avg / \
            attn_heads_avg.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_heads_avg, result)
    return result


# Main visualization function
def visualize_and_save_sample(
    model: nn.Module,
    device: torch.device,
    npy_dir: str,
    sample_id: str,
    output_dir: str,
    output_size=(224, 224),
    # Model loading parameters (passed to ViTForSegmentation)
    pretrained_model: str = "google/vit-base-patch16-224-in21k",
    num_classes: int = 1,
    use_custom_vit: bool = True,
    custom_hidden_size: int = 256,
    custom_num_layers: int = 8,
    custom_num_heads: int = 4,
    custom_mlp_dim: int = 1024,
    custom_patch_size: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)

    npy_file_path = os.path.join(npy_dir, f"{sample_id}.npy")

    if not os.path.exists(npy_file_path):
        print(f"Error: NPY file not found for {sample_id} at {npy_file_path}")
        return

    data = np.load(npy_file_path, allow_pickle=True)
    original_img_chw = data[0]  # (C, H, W)
    gt_mask = data[1][0]     # (H, W)
    opf_mask = data[5][0]    # (H, W)

    # --- Prepare image for model input and display ---
    # For model: (1, C, H, W) normalized to [-1, 1]
    # For display: (H, W, C) normalized to [0, 1]

    # Resize masks to output_size for consistent overlay and metric calculation
    gt_mask_resized = cv2.resize(
        gt_mask, output_size, interpolation=cv2.INTER_NEAREST)
    opf_mask_resized = cv2.resize(
        opf_mask, output_size, interpolation=cv2.INTER_NEAREST)

    # Preprocess image for display: (H, W, C) and normalize
    original_img_display = np.transpose(
        original_img_chw, (1, 2, 0))  # (H, W, 3)
    original_img_display = (original_img_display - original_img_display.min()) / (
        original_img_display.max() - original_img_display.min() + 1e-8)
    original_img_display = cv2.resize(original_img_display, output_size)

    # Quick brain mask to limit attention maps to brain region
    brain_mask_224 = quick_brain_mask(
        original_img_display[..., 0])  # any channel is fine
    brain_mask_14 = downsample_mask_14x14(brain_mask_224, patch=16)

    # Convert original_img to tensor for model input
    img_tensor = torch.tensor(original_img_chw).unsqueeze(
        0).float().to(device)  # [1, 3, H, W]
    # Normalize to [-1, 1] as per ViTForSegmentation's img_tf
    img_tensor = F.interpolate(
        img_tensor, size=output_size, mode='bilinear', align_corners=False)
    img_tensor = TF.normalize(
        img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # --- Generate Attention Maps Dynamically ---
    model.eval()
    with torch.no_grad():  # Initial forward pass for attns, not for GAR gradients yet
        preds, attns = model(img_tensor)

    # Decide what to compute based on toggles
    cls_bin = cls_heat = roll_bin = roll_heat = None

    if ATTN_METHOD in ("cls", "both"):
        cls_bin, cls_heat = extract_cls_attention_map(
            attns, sample_idx=0, output_size=output_size,
            threshold_mode="auto", topk=0.01, brain_mask_14=brain_mask_14
        )
    if ATTN_METHOD in ("rollout", "both"):
        roll_bin, roll_heat = extract_rollout_attention_map(
            attns, sample_idx=0, output_size=output_size,
            add_residual=True, start_layer=1,            # try start_layer=1 or 2
            threshold_mode="auto", topk=0.01, brain_mask_14=brain_mask_14
        )

    # # Attention Rollout Map
    # rollout_attn_map = extract_rollout_attention_map(attns, output_size=output_size)
    print("CLS stats:", cls_heat.min(), cls_heat.max(), cls_heat.var())
    print("ROL stats:", roll_heat.min(), roll_heat.max(), roll_heat.var())

    # --- Create and Save Visualizations ---
    # 1. Ground Truth Mask Overlay
    iou_gt, dice_gt = calculate_metrics(
        gt_mask_resized, gt_mask_resized)  # GT vs GT for sanity
    overlay_gt = create_overlay(
        original_img_display, gt_mask_resized, color=(0, 255, 0))
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_gt)
    plt.title(
        f"Sample: {sample_id} - GT Overlay (Green)\nIoU: {iou_gt:.2f}, Dice: {dice_gt:.2f}", fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(
        output_dir, f"{sample_id}_gt_overlay.png"), bbox_inches='tight')
    plt.close()

    # 2. OPF Mask Overlay
    iou_opf, dice_opf = calculate_metrics(opf_mask_resized, gt_mask_resized)
    overlay_opf = create_overlay(
        original_img_display, opf_mask_resized, color=(255, 0, 0))
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_opf)
    plt.title(
        f"Sample: {sample_id} - OPF Overlay (Red)\nIoU: {iou_opf:.2f}, Dice: {dice_opf:.2f}", fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(
        output_dir, f"{sample_id}_opf_overlay.png"), bbox_inches='tight')
    plt.close()

    # 3. CLS map
    if cls_bin is not None:
        iou_cls_attn, dice_cls_attn = calculate_metrics(
            cls_bin, gt_mask_resized)
        overlay_cls_attn = create_overlay(
            original_img_display,
            cls_heat if SAVE_HEATMAPS else cls_bin,
            color=(0, 255, 255),  # cyan
            apply_threshold=not SAVE_HEATMAPS
        )
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_cls_attn)
        plt.title(
            f"Sample: {sample_id} - Cls Attn Overlay (Cyan)\nIoU: {iou_cls_attn:.2f}, Dice: {dice_cls_attn:.2f}", fontsize=18)
        plt.axis('off')
        plt.savefig(os.path.join(
            output_dir, f"{sample_id}_cls_attn_map.png"), bbox_inches='tight')
        plt.close()

    # 4. Rollout map
    if roll_bin is not None:
        iou_rollout_attn, dice_rollout_attn = calculate_metrics(
            roll_bin, gt_mask_resized)
        overlay_rollout_attn = create_overlay(
            original_img_display,
            roll_heat if SAVE_HEATMAPS else roll_bin,
            color=(255, 165, 0),  # orange/yellow
            apply_threshold=not SAVE_HEATMAPS
        )
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_rollout_attn)
        plt.title(
            f"Sample: {sample_id} - Rollout Attn Overlay (Yellow)\nIoU: {iou_rollout_attn:.2f}, Dice: {dice_rollout_attn:.2f}", fontsize=18)
        plt.axis('off')
        plt.savefig(os.path.join(
            output_dir, f"{sample_id}_rollout_attn_map.png"), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # --- Configuration --- #
    DEVICE = torch.device('cpu')
    # UPDATE THIS PATH to the best saved model
    MODEL_PATH = "/mnt/iusers01/fse-ugpgt01/compsci01/f32077dn/scratch/top_n_scans/full_train_dataset/exp07_highest_300_scans/freeze_0_drop_50/vit_best_val.pth"

    NPY_DATA_DIR = "filtered_dataset"  # Directory containing .npy slice files
    # Directory to save the output images
    OUTPUT_SAVE_DIR = "visualized_attention_maps/vit_mini_with_aaf_freeze0_drop50"

    ATTN_METHOD = "both"   # "cls" | "rollout" | "both"
    THRESHOLD_MODE = "otsu"   # "mean" | "otsu" | 0.5 (float)
    # if True, use gradient colormap for attention instead of binary mask
    SAVE_HEATMAPS = False

    # --- Model Configuration (must match how the model was trained!) ---
    # If trained with custom ViT:
    USE_CUSTOM_VIT = True  # Set to False for baseline model
    CUSTOM_HIDDEN_SIZE = 256
    CUSTOM_NUM_LAYERS = 8
    CUSTOM_NUM_HEADS = 4
    CUSTOM_MLP_DIM = 1024
    CUSTOM_PATCH_SIZE = 16
    Dropout_P = 0.5  # Match the dropout used during training
    # Model intialisation
    print("Loading model for visualization...")
    model = ViTForSegmentation(
        dropout_p=Dropout_P,
        use_custom_vit=USE_CUSTOM_VIT,
        custom_hidden_size=CUSTOM_HIDDEN_SIZE,
        custom_num_layers=CUSTOM_NUM_LAYERS,
        custom_num_heads=CUSTOM_NUM_HEADS,
        custom_mlp_dim=CUSTOM_MLP_DIM,
        custom_patch_size=CUSTOM_PATCH_SIZE,
        num_classes=1,  # Assuming binary segmentation
    ).to(DEVICE)

    # Load model weights
    checkpoint_data = torch.load(MODEL_PATH, map_location=DEVICE)
    # Check if the checkpoint is a full dictionary or just the state_dict
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)
    model.eval()
    print("Model loaded successfully.")

    # List of sample IDs to visualize
    sample_ids_to_visualize = [
        "r031s035_71",
        "r048s018_79",
        "r052s014_72",
        "r035s006_86",
        "r044s002_116",
        "r009s083_52",
    ]

    for sample_id in sample_ids_to_visualize:
        print(f"Visualizing and saving for sample: {sample_id}")
        visualize_and_save_sample(
            model=model,
            device=DEVICE,
            npy_dir=NPY_DATA_DIR,
            sample_id=sample_id,
            output_dir=OUTPUT_SAVE_DIR,
            # Pass model init parameters for consistency if needed for dynamic class loading (ViTOL PADL)
            use_custom_vit=USE_CUSTOM_VIT,
            custom_hidden_size=CUSTOM_HIDDEN_SIZE,
            custom_num_layers=CUSTOM_NUM_LAYERS,
            custom_num_heads=CUSTOM_NUM_HEADS,
            custom_mlp_dim=CUSTOM_MLP_DIM,
            custom_patch_size=CUSTOM_PATCH_SIZE,
            num_classes=1,
        )
    print(f"Visualizations saved to: {OUTPUT_SAVE_DIR}")

    summary = evaluate_attention_on_ids(
        model=model,
        device=DEVICE,
        npy_dir=NPY_DATA_DIR,
        sample_ids=sample_ids_to_visualize,
        output_size=(224, 224)
    )
    print("\n=== Attention Evaluation Summary (means over selected slices) ===")
    for k, v in summary.items():
        print(f"{k:>12} | IoU={v['IoU_mean']:.3f}  Dice={v['Dice_mean']:.3f}")

    # (optional) Save CSV for your Results table
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(
        OUTPUT_SAVE_DIR, "attention_eval_summary_drop0.5_freeze0.csv")
    pd.DataFrame.from_dict(summary, orient='index').to_csv(csv_path)
    print(f"Saved summary to {csv_path}")
