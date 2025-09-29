
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def preds_binary(logits_or_scores: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    """
    Convert logits or scores to hard binary predictions:
    - If input is [N,2] logits: argmax by default, or threshold on softmax[:,1] if threshold is set
    - If input is [N] scores/probs for class=1: threshold at given value (default 0.5)
    """
    x = logits_or_scores
    if x.ndim == 2:  
        if threshold is None:
            return x.argmax(1).to(torch.int64)
        p1 = F.softmax(x, dim=1)[:, 1]
    else:           
        p1 = x.view(-1).float()
    thr = 0.5 if threshold is None else float(threshold)
    return (p1 >= thr).to(torch.int64)

@torch.no_grad()
def conf_binary(pred: torch.Tensor, target: torch.Tensor, ignore_label: int | None = None):
    """
    TP/TN/FP/FN from binary preds & targets
    """
    pred = pred.view(-1).to(torch.int64)
    tgt  = target.view(-1).to(torch.int64)
    if ignore_label is not None:
        mask = tgt != int(ignore_label)
        pred = pred[mask]; tgt = tgt[mask]
    tp = int(((pred == 1) & (tgt == 1)).sum())
    tn = int(((pred == 0) & (tgt == 0)).sum())
    fp = int(((pred == 1) & (tgt == 0)).sum())
    fn = int(((pred == 0) & (tgt == 1)).sum())
    return tp, tn, fp, fn

def metrics_from_counts(tp: int, tn: int, fp: int, fn: int, eps: float = 1e-9) -> dict:
    """
    Precision, Recall, F1, IoU, MCC, Accuracy, Specificity from TP/TN/FP/FN.
    """
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    iou  = tp / (tp + fp + fn + eps)
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    spec = tn / (tn + fp + eps)
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = mcc_num / (mcc_den + eps)
    return {
        "precision": prec, "recall": rec, "f1": f1, "iou": iou, "mcc": mcc,
        "accuracy": acc, "specificity": spec,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }

class MeterBinary:
    def __init__(self, ignore_label: int | None = None, threshold: float | None = None):
        self.ignore_label = ignore_label
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

    @torch.no_grad()
    def update(self, logits_or_preds: torch.Tensor, target: torch.Tensor):
        if logits_or_preds.ndim >= 2:  # logits [N,2]
            pred = preds_binary(logits_or_preds, threshold=self.threshold)
        else:                          # already hard preds [N]
            pred = logits_or_preds.to(torch.int64)
        tp, tn, fp, fn = conf_binary(pred, target, ignore_label=self.ignore_label)
        self.tp += tp; self.tn += tn; self.fp += fp; self.fn += fn

    def compute(self) -> dict:
        return metrics_from_counts(self.tp, self.tn, self.fp, self.fn)