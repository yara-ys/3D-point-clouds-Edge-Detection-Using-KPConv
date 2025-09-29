
import os
import time
import json
import random
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext

from utils.metrics import MeterBinary
from utils.viz import plot_confusion_2x2, plot_metrics_bar, plot_pr_curve
from utils.ply_io import save_errormap_ply


#-----------Seed for Reproducibility--------------
def set_seed(seed: int = 42):
    import os as _os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] Global seed set to {seed}")


# ------- Dice Loss --------------- 
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        p = torch.softmax(logits, dim=1)[:, 1]
        t = (targets == 1).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        return 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)


# -------- Trainer -----------------
class Trainer:

    def __init__(self,
                 cfg,
                 model: nn.Module,
                 train_set,
                 val_set,
                 build_optimizer_fn=None,
                 run_root: Optional[str] = None,
                 device: Optional[str] = None):
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # dataloaders
        collate_train = getattr(train_set, "collate_fn", None)
        collate_val = getattr(val_set, "collate_fn", None)
        if collate_train is None or collate_val is None:
            try:
                from datasets.ABC import abc_collate
            except Exception:
                abc_collate = None
            collate_train = collate_train or abc_collate
            collate_val = collate_val or abc_collate

        self.batch_size = int(getattr(cfg, "batch_size", 2))
        self.num_workers = int(getattr(cfg, "num_workers", 0))

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_train,
            worker_init_fn=(lambda wid: set_seed(getattr(cfg, "seed", 42) + wid))
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_val,
            worker_init_fn=(lambda wid: set_seed(getattr(cfg, "seed", 42) + wid))
        )

        # optimizer & scheduler
        if build_optimizer_fn is not None:
            self.optimizer, self.scheduler = build_optimizer_fn(cfg, model)
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(getattr(cfg, "learning_rate", 1e-3)),
                weight_decay=float(getattr(cfg, "weight_decay", 1e-4)),
            )
            sched_type = str(getattr(cfg, "sched", "cosine"))
            if sched_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=int(getattr(cfg, "max_epochs", 60))
                )
            elif sched_type == "step":
                step_size = int(getattr(cfg, "step_size", 20))
                gamma = float(getattr(cfg, "gamma", 0.5))
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
            else:
                self.scheduler = None

        # losses
        self.ce = nn.CrossEntropyLoss(weight=self._estimate_class_weights(self.train_loader))
        self.use_dice = bool(getattr(cfg, "use_dice", True))
        self.dice = DiceLoss()
        self.grad_clip = float(getattr(cfg, "grad_clip", 5.0) or 0.0)

        # AMP
        if self.device.type == "cuda":
            from torch import amp as _amp
            self.amp_context = lambda: _amp.autocast("cuda")
            self.scaler = _amp.GradScaler("cuda", enabled=True)
        else:
            self.amp_context = lambda: nullcontext()

            class _Dummy:
                def scale(self, x): return x
                def unscale_(self, opt): pass
                def step(self, opt): opt.step()
                def update(self): pass

            self.scaler = _Dummy()

        # run dir
        root = run_root or getattr(cfg, "run_root", None)
        if root is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            root = os.path.join(project_root, "runs")
        self.run_dir = os.path.join(root, str(getattr(cfg, "run_name", "exp_default")))
        os.makedirs(self.run_dir, exist_ok=True)

        # save config snapshot
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

        # metrics state
        self.best_f1 = -1.0
        self.in_dim = None 

    # ----------- helpers ----------- 
    def _estimate_class_weights(self, loader, max_batches=20, cap=10.0) -> torch.Tensor:
        import collections
        cnt = collections.Counter()
        for i, batch in enumerate(loader):
            y = batch["labels"].view(-1).numpy()
            cnt.update(y.tolist())
            if i + 1 >= max_batches:
                break
        neg = max(cnt.get(0, 1), 1)
        pos = max(cnt.get(1, 1), 1)
        w1 = min(float(neg) / float(pos), cap)
        return torch.tensor([1.0, w1], device=self.device, dtype=torch.float32)

    def _loss_fn(self, logits, y):
        if self.use_dice:
            return 0.5 * self.ce(logits, y) + 0.5 * self.dice(logits, y)
        return self.ce(logits, y)

    def _move_to_device(self, batch: dict) -> dict:
        T_KEYS = ("points", "features", "neighbors", "pools", "upsamples", "labels")
        xb = {}
        for k, v in batch.items():
            if k in T_KEYS:
                if isinstance(v, list):
                    xb[k] = [t.to(self.device, non_blocking=True) for t in v]
                elif torch.is_tensor(v):
                    xb[k] = v.to(self.device, non_blocking=True)
                else:
                    xb[k] = v
            else:
                xb[k] = v
        return xb

#####

    def _train_one_epoch(self) -> float:
        self.model.train(True)
        loss_sum, n = 0.0, 0

        for batch in self.train_loader:
            xb = self._move_to_device(batch)
            y = xb["labels"].long()

            with self.amp_context():
                logits = self.model(xb)
                loss = self._loss_fn(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_sum += float(loss.item())
            n += 1

            del batch, xb, y, logits, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return loss_sum / max(1, n)

    @torch.no_grad()
    def _validate(self) -> Tuple[dict, Tuple[np.ndarray, np.ndarray]]:
        self.model.train(False)
        meter = MeterBinary(threshold=None)  
        total_loss, n_items = 0.0, 0

        all_probs, all_labels = [], []
        snapshot_saved = False

        for batch in self.val_loader:
            xb = self._move_to_device(batch)
            y = xb["labels"].long()

            logits = self.model(xb)
            loss = F.cross_entropy(logits, y)

            meter.update(logits, y)

            # collect for PR curve
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            lab = y.detach().cpu().numpy()
            all_probs.append(prob)
            all_labels.append(lab)

            # save one colored diff snapshot per epoch
            if not snapshot_saved and "points" in xb and isinstance(xb["points"], list):
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
                pts0 = xb["points"][0].detach().cpu().numpy()
                save_errormap_ply(
                    os.path.join(self.run_dir, f"val_snapshot_ep{getattr(self, '_epoch_idx', 0):03d}_diff.ply"),
                    pts0, pred, lab
                )
                snapshot_saved = True

            total_loss += float(loss.item()) * y.shape[0]
            n_items += y.shape[0]

            del batch, xb, y, logits, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        stats = meter.compute()  # includes tp/tn/fp/fn, precision/recall/f1/iou/mcc,...
        stats["loss"] = total_loss / max(1, n_items)
        y_all = np.concatenate(all_labels) if all_labels else np.zeros((0,), np.int64)
        p_all = np.concatenate(all_probs) if all_probs else np.zeros((0,), np.float32)
        return stats, (y_all, p_all)

    # ------------- training loop -----------------
    def fit(self):
        max_epochs = int(getattr(self.cfg, "max_epochs", 60))
        print(f"Training for {max_epochs} epochs on device={self.device}")

        for epoch in range(1, max_epochs + 1):
            self._epoch_idx = epoch 

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            train_loss = self._train_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()

            val_stats, (y_all, p_all) = self._validate()
            dt = time.time() - t0

            msg = (f"[ep {epoch:02d}] train {train_loss:.4f} | val {val_stats['loss']:.4f} | "
                   f"P {val_stats['precision']:.3f} R {val_stats['recall']:.3f} "
                   f"F1 {val_stats['f1']:.3f} IoU {val_stats['iou']:.3f} MCC {val_stats['mcc']:.3f}")
            if self.device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg += f" | peak {peak_gb:.2f} GB"
            print(msg + f" | {dt:.1f}s")

            # ------save checkpoints ---
            torch.save(
                {"epoch": epoch,
                 "model_state": self.model.state_dict(),
                 "in_dim": self.in_dim,
                 "layer_radii": getattr(self.cfg, "layer_radii", None)},
                os.path.join(self.run_dir, "model_last.pt")
            )
            if val_stats["f1"] > self.best_f1:
                self.best_f1 = val_stats["f1"]
                torch.save(
                    {"epoch": epoch,
                     "model_state": self.model.state_dict(),
                     "in_dim": self.in_dim,
                     "layer_radii": getattr(self.cfg, "layer_radii", None),
                     "best_f1": float(self.best_f1)},
                    os.path.join(self.run_dir, "model_best.pt")
                )
                print(f" ↑ Saved BEST (F1={self.best_f1:.3f})")

            # --------- plots per epoch ---
            tp, tn, fp, fn = int(val_stats["tp"]), int(val_stats["tn"]), int(val_stats["fp"]), int(val_stats["fn"])
            plot_confusion_2x2(tp, tn, fp, fn, normalize=False,
                               save_path=os.path.join(self.run_dir, f"confusion_ep{epoch:03d}.png"))
            plot_confusion_2x2(tp, tn, fp, fn, normalize=True,
                               save_path=os.path.join(self.run_dir, f"confusion_norm_ep{epoch:03d}.png"))
            plot_metrics_bar(val_stats, save_path=os.path.join(self.run_dir, f"metrics_bar_ep{epoch:03d}.png"))
            if p_all.size and y_all.size:
                plot_pr_curve(y_all, p_all, save_path=os.path.join(self.run_dir, f"pr_curve_ep{epoch:03d}.png"))
