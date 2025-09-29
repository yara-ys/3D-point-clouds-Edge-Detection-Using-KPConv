
# Evaluate a saved checkpoint and export final summary plots.

import os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import Config
from utils.metrics import metrics_from_counts, MeterBinary
from utils.viz import plot_confusion_2x2, plot_metrics_bar, plot_pr_curve
from utils.ply_io import save_errormap_ply
from datasets.ABC import ABCDataset, abc_collate
from models.architecture import build_from_cfg


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_to_device(batch, device):
    T_KEYS = ("points", "features", "neighbors", "pools", "upsamples", "labels")
    out = {}
    for k, v in batch.items():
        if k in T_KEYS:
            if isinstance(v, list):
                out[k] = [t.to(device, non_blocking=True) for t in v]
            elif torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


@torch.no_grad()
def _forward(model, batch, device):
    xb = _move_to_device(batch, device)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1)[:, 1]
    coords = xb["points"][0].cpu().numpy()
    labels = xb["labels"].cpu().numpy()
    return probs.cpu().numpy(), labels, coords


def test_run(run_dir: str,
             ckpt_name: str = "model_best.pt",
             split: str = "Validation",
             threshold: float = 0.25,
             export_diff_ply: bool = True,
             export_pred_ply: bool = False,
             save_plots: bool = True) -> dict:
    """
    Load a checkpoint, evaluate it on a split, and export final plots 
    """
    t0 = time.time()
    device = _device()

    # config + dataset
    cfg = Config()
    ds = ABCDataset(cfg, split=split,
                    use_ssm=getattr(cfg, "use_ssm", False),
                    use_normals=getattr(cfg, "use_normals", False),
                    pool_cap=getattr(cfg, "pool_cap", 32))
    loader = DataLoader(ds,
                        batch_size=getattr(cfg, "batch_size", 2),
                        shuffle=False,
                        num_workers=getattr(cfg, "num_workers", 0),
                        pin_memory=(device.type == "cuda"),
                        collate_fn=getattr(ds, "collate_fn", abc_collate))

    # model + checkpoint
    in_dim = ds[0]["features"][0].shape[1]
    model = build_from_cfg(cfg, in_dim).to(device)
    ckpt_path = os.path.join(run_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # accumulators
    all_probs, all_labels = [], []
    TP=TN=FP=FN=0

    export_dir = os.path.join(run_dir, "exports", split)
    os.makedirs(export_dir, exist_ok=True)


    for bi, batch in enumerate(loader):
        probs, y, coords = _forward(model, batch, device)
        pred = (probs >= float(threshold)).astype(np.int32)
        all_probs.append(probs)
        all_labels.append(y)

        TP += int(((y == 1) & (pred == 1)).sum())
        TN += int(((y == 0) & (pred == 0)).sum())
        FP += int(((y == 0) & (pred == 1)).sum())
        FN += int(((y == 1) & (pred == 0)).sum())

        if export_diff_ply:
            save_errormap_ply(os.path.join(export_dir, f"sample{bi:04d}_diff.ply"), coords, pred, y)
        if export_pred_ply:
            save_errormap_ply(os.path.join(export_dir, f"sample{bi:04d}_pred.ply"), coords, pred, pred)

    # metrics
    stats = metrics_from_counts(TP, TN, FP, FN)
    stats.update({
        "threshold": threshold,
        "split": split,
        "elapsed_sec": float(time.time() - t0),
        "tp": TP, "tn": TN, "fp": FP, "fn": FN,
    })

    y_all = np.concatenate(all_labels) if all_labels else np.zeros((0,))
    p_all = np.concatenate(all_probs) if all_probs else np.zeros((0,))

    # final plots
    if save_plots:
        plot_confusion_2x2(TP, TN, FP, FN, normalize=False,
                           save_path=os.path.join(export_dir, "confusion_final.png"))
        plot_confusion_2x2(TP, TN, FP, FN, normalize=True,
                           save_path=os.path.join(export_dir, "confusion_final_norm.png"))
        plot_metrics_bar(stats, save_path=os.path.join(export_dir, "metrics_bar_final.png"))
        if p_all.size and y_all.size:
            plot_pr_curve(y_all, p_all, save_path=os.path.join(export_dir, "pr_curve_final.png"))

    print(f"[tester] {split}: F1={stats['f1']:.3f} "
          f"P={stats['precision']:.3f} R={stats['recall']:.3f} "
          f"IoU={stats['iou']:.3f} MCC={stats['mcc']:.3f}")

    return stats
