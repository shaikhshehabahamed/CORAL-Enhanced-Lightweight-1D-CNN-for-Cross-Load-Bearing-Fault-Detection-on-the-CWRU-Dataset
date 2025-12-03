# main_coral.py  (updated with plotting, baseline training & fixes)
import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pandas as pd
except Exception:
    pd = None

from helper import CWRUDataset
from nn_model import CNN_1D_3L, CNN_1D_2L
from train_helper import fit_coral, validate  # validate alias

# New imports (ensure these files exist in same folder)
from train_supervised import fit_supervised
from train_dann import fit_dann

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(n_in: int, which: str):
    if which == "2L":
        return CNN_1D_2L(n_in=n_in)
    return CNN_1D_3L(n_in=n_in)


@torch.no_grad()
def eval_collect(model, dl, loss_func, device):
    """Evaluation that tolerates models returning logits or (logits, ...)"""
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = model(xb)
        # If model returns tuple (e.g., DANN: (logits, dom_logits, feats)), take first element
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = loss_func(logits, yb)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds = logits.argmax(dim=1)
        all_true.append(yb.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([])
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, acc, y_true, y_pred


@torch.no_grad()
def extract_features(model, dl, device, use_gap=True):
    """
    Returns (feats_numpy, labels_numpy)
    Works for:
      - backbone models (CNN_1D_3L/CNN_1D_2L) when called with return_feats=True
      - CNN_DANN (it will use model.feature_extractor)
    """
    model.eval()
    feats_all = []
    labs_all = []

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)

        # Try straightforward call first
        try:
            out = model(xb, return_feats=True, use_gap=use_gap)
            # many backbones return (logits, feats)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                feats = out[1]
            else:
                # fallback if model returns unexpected shape
                feats = out
        except TypeError:
            # maybe DANN-like model where forward signature is (x, lambd=..)
            if hasattr(model, "feature_extractor"):
                out = model.feature_extractor(xb, return_feats=True, use_gap=use_gap)
                feats = out[1] if isinstance(out, (tuple, list)) else out
            else:
                # final fallback: call model(x) and hope second element is feats
                out = model(xb)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    # DANN returns (logits, dom_logits, feats)
                    feats = out[-1]  # often last element
                else:
                    raise RuntimeError("Unable to extract features for t-SNE from model outputs.")

        feats_all.append(feats.cpu().numpy())
        labs_all.append(yb.cpu().numpy())

    feats_all = np.concatenate(feats_all, axis=0)
    labs_all = np.concatenate(labs_all, axis=0)
    return feats_all, labs_all


def plot_tsne_and_save(feats, labels, out_path, title="t-SNE", perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb = tsne.fit_transform(feats)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette="deep", s=18, alpha=0.8, linewidth=0)
    plt.title(title)
    plt.legend(title="label", loc="best", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_and_save(cm, labels_short, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_short, yticklabels=labels_short)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CNN1D + CORAL / DANN / SUP on CWRU")

    # Data
    parser.add_argument("--data", type=str,
        default=r"C:\Users\sheha\OneDrive\Documents\GitHub\Enhancing-Bearing-Fault-Diagnosis-Across-Loads-Using-a-Lightweight-CNN-and-CORAL\Data")
    parser.add_argument("--source_load", type=int, default=0)
    parser.add_argument("--target_load", type=int, default=1)

    # Windowing
    parser.add_argument("--segment_length", type=int, default=2048)
    parser.add_argument("--per_window_norm", action="store_true")
    parser.add_argument("--src_size", type=str, default=None)
    parser.add_argument("--tgt_size", type=str, default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_coral", type=float, default=0.5)
    parser.add_argument("--model", choices=["2L", "3L"], default="3L")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--out_dir", type=str, default=".")

    # NEW OPTIONS
    parser.add_argument("--no_coral", action="store_true", help="Supervised baseline")
    parser.add_argument("--dann", action="store_true", help="Use DANN baseline")
    parser.add_argument("--no_gap", action="store_true", help="Disable GAP for CORAL features")

    return parser.parse_args()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    args = parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Source {args.source_load} → Target {args.target_load}")

    overlap = args.segment_length // 2

    # Datasets
    src_ds = CWRUDataset(args.data, args.segment_length, True, args.source_load,
                         overlap, args.src_size, args.per_window_norm)
    tgt_ds = CWRUDataset(args.data, args.segment_length, True, args.target_load,
                         overlap, args.tgt_size, args.per_window_norm)

    # Grouped splits
    gss_s = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    s_train_idx, s_val_idx = next(gss_s.split(src_ds.X, src_ds.y, groups=src_ds.filenames))
    src_train_ds = Subset(src_ds, s_train_idx)
    src_val_ds = Subset(src_ds, s_val_idx)

    gss_t = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    t_adapt_idx, t_test_idx = next(gss_t.split(tgt_ds.X, tgt_ds.y, groups=tgt_ds.filenames))
    tgt_adapt_ds = Subset(tgt_ds, t_adapt_idx)
    tgt_test_ds = Subset(tgt_ds, t_test_idx)

    # Dataloaders
    src_dl = DataLoader(src_train_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_workers, pin_memory=pin_mem)
    src_val_dl = DataLoader(src_val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=args.num_workers)
    tgt_dl = DataLoader(tgt_adapt_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_workers)
    tgt_eval_dl = DataLoader(tgt_test_ds, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.num_workers)

    # ----------------------------
    # Baseline supervised model (TRAIN & SAVE) - used for "before adaptation" t-SNE
    # Only train baseline if we will perform adaptation OR if user requested supervised run.
    # ----------------------------
    baseline_model = make_model(args.segment_length, args.model).to(device)
    baseline_opt = torch.optim.AdamW(baseline_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.no_coral:
        # supervised-only run: train on source and save baseline (this is the main model)
        print("[INFO] Training supervised baseline (no adaptation)...")
        baseline_model = fit_supervised(args.epochs, baseline_model, baseline_opt, src_dl, src_val_dl,
                                        CrossEntropyLoss(weight=None), args.early_stop_patience)
        # save supervised baseline model
        torch.save(baseline_model.state_dict(), os.path.join(args.out_dir, "baseline_model.pt"))

        # Evaluate baseline on target test and save confusion matrix image & t-SNE
        print("\n[INFO] Evaluating supervised baseline on TARGET TEST:")
        ce = CrossEntropyLoss()
        test_loss, test_acc, y_true, y_pred = eval_collect(baseline_model, tgt_eval_dl, ce, device)
        print(f"[RESULT - Baseline] Target loss: {test_loss:.4f} | acc: {test_acc:.4f}")
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
        labels_short = ["N","B","IR","OR"]
        plot_confusion_and_save(cm, labels_short, os.path.join(args.out_dir, "confusion_matrix_baseline.png"),
                               title="Confusion Matrix (Supervised Baseline on Target)")
        # t-SNE on baseline features
        feats_before, labels_before = extract_features(baseline_model, tgt_eval_dl, device, use_gap=not args.no_gap)
        plot_tsne_and_save(feats_before, labels_before, os.path.join(args.out_dir, "tsne_baseline.png"),
                           title="t-SNE (Supervised Baseline Features on Target)")
        return

    # If here: either doing CORAL or DANN. For CORAL we will train baseline first to visualize "before".
    if not args.dann:
        print("[INFO] Training supervised baseline (for t-SNE BEFORE adaptation)...")
        baseline_model = fit_supervised(args.epochs, baseline_model, baseline_opt, src_dl, src_val_dl,
                                        CrossEntropyLoss(weight=None), args.early_stop_patience)
        torch.save(baseline_model.state_dict(), os.path.join(args.out_dir, "baseline_model.pt"))
        feats_before, labels_before = extract_features(baseline_model, tgt_eval_dl, device, use_gap=not args.no_gap)
        plot_tsne_and_save(feats_before, labels_before, os.path.join(args.out_dir, "tsne_before_coral.png"),
                           title="t-SNE Before CORAL (Supervised Baseline)")

    # ----------------------------
    # Main model (DANN or CORAL or supervised as selected)
    # ----------------------------
    if args.dann:
        from nn_model import CNN_DANN
        model = CNN_DANN(args.segment_length, use_gap=not args.no_gap).to(device)
    else:
        model = make_model(args.segment_length, args.model).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Weighted CE (for class imbalance)
    cls_counts = np.bincount(np.array(src_ds.y)[s_train_idx], minlength=4)
    cls_weights = (cls_counts.sum() / (np.maximum(1, cls_counts) * 4.0)).astype("float32")
    ce = CrossEntropyLoss(weight=torch.tensor(cls_weights, device=device))
    dom_loss = CrossEntropyLoss()  # for DANN

    # Train mode selection
    if args.dann:
        print("[INFO] Training with DANN baseline...")
        model = fit_dann(args.epochs, model, opt, src_dl, tgt_dl,
                         ce, dom_loss, src_val_dl, args.early_stop_patience)
    else:
        print("[INFO] Training with CORAL...")
        fit_coral(args.epochs, model, opt, src_dl, tgt_dl, args.lambda_coral,
                  ce, src_val_dl, args.early_stop_patience, use_gap=not args.no_gap)

    # -------------------------------------------------------------
    # Evaluate final model on target test set and save plots
    # -------------------------------------------------------------
    print("\n[INFO] Evaluating FINAL model on TARGET TEST:")
    test_loss, test_acc, y_true, y_pred = eval_collect(model, tgt_eval_dl, ce, device)
    print(f"[RESULT - Final] Target loss: {test_loss:.4f} | acc: {test_acc:.4f}")

    # Save model weights (final)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))

    # Confusion matrix image
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    labels_short = ["N","B","IR","OR"]
    plot_confusion_and_save(cm, labels_short, os.path.join(args.out_dir, "confusion_matrix_final.png"),
                           title="Confusion Matrix (Final Model on Target)")

    # t-SNE AFTER adaptation (final model)
    feats_after, labels_after = extract_features(model, tgt_eval_dl, device, use_gap=not args.no_gap)
    plot_tsne_and_save(feats_after, labels_after, os.path.join(args.out_dir, "tsne_after_coral.png"),
                       title="t-SNE After Adaptation (Final Model)")

    # print classification report to console (no txt file)
    report = classification_report(y_true, y_pred, target_names=labels_short, digits=4)
    print(report)


if __name__ == "__main__":
    main()