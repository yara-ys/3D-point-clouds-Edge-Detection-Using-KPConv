
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def plot_confusion_2x2(tp: int, tn: int, fp: int, fn: int,
                       normalize: bool = False,
                       title: str | None = None,
                       save_path: str | None = None):
    """
    Show a 2x2 confusion matrix 
    """
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.float64)
    if normalize:
        rs = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / rs

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["GT 0", "GT 1"])
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth")
    ax.set_title(title or ("Confusion (norm)" if normalize else "Confusion"))
    for i in range(2):
        for j in range(2):
            txt = f"{cm[i,j]:.3f}" if normalize else f"{int(round(cm[i,j]))}"
            ax.text(j, i, txt, ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.close(fig)

def plot_metrics_bar(metrics: dict,
                     keys=("f1","iou","precision","recall","mcc"),
                     title: str = "Metrics",
                     save_path: str | None = None):
    vals = [float(metrics[k]) for k in keys if k in metrics]
    ks   = [k for k in keys if k in metrics]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(ks, vals)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.close(fig)

def plot_pr_curve(y_true, y_score, num_thresholds: int = 256,
                  title: str = "Precision–Recall",
                  save_path: str | None = None):
    """
    Minimal PR curve (binary)
    """
    y = _to_np(y_true).astype(np.int32).ravel()
    s = _to_np(y_score).astype(np.float64).ravel()
    thr = np.linspace(s.max(), s.min(), num_thresholds)  
    P, R = [], []
    for t in thr:
        pred = (s >= t)
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        P.append(prec); R.append(rec)
    P, R = np.array(P), np.array(R)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(R, P)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return thr, P, R