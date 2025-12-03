# train_dann.py
import torch
from torch.cuda.amp import autocast, GradScaler
from train_helper import validate


def fit_dann(
    epochs,
    model,
    opt,
    src_dl,
    tgt_dl,
    cls_loss,
    dom_loss,
    src_val_dl,
    early_stop_patience=10
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    patience = 0
    best_state = None

    print("EPOCH\tClsLoss\tDomLoss\tTotal\tValLoss\tValAcc")

    for epoch in range(epochs):

        model.train()
        total_cls = 0.0
        total_dom = 0.0

        # match source & target batches
        n_batches = min(len(src_dl), len(tgt_dl))

        # λ schedule for DANN
        p = epoch / epochs
        p = torch.tensor(p, dtype=torch.float32, device=device)  # ensure Tensor
        lambd = 2 / (1 + torch.exp(-10 * p)) - 1
        lambd = lambd.item()  # convert back to float

        for (xs, ys), (xt, _) in zip(src_dl, tgt_dl):
            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            opt.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):

                # Forward pass (source)
                logits_s, dom_s, feats_s = model(xs, lambd=lambd)
                loss_cls = cls_loss(logits_s, ys)

                # domain labels
                domain_s = torch.zeros(len(xs), dtype=torch.long, device=device)
                domain_t = torch.ones(len(xt), dtype=torch.long, device=device)

                # Forward pass (target)
                _, dom_t, feats_t = model(xt, lambd=lambd)

                loss_dom_s = dom_loss(dom_s, domain_s)
                loss_dom_t = dom_loss(dom_t, domain_t)
                loss_dom = loss_dom_s + loss_dom_t

                loss = loss_cls + loss_dom

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_cls += loss_cls.item()
            total_dom += loss_dom.item()

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = validate(model, src_val_dl, cls_loss)

        print(
            f"{epoch}\t"
            f"{total_cls/n_batches:.4f}\t"
            f"{total_dom/n_batches:.4f}\t"
            f"{(total_cls+total_dom)/n_batches:.4f}\t"
            f"{val_loss:.4f}\t"
            f"{val_acc:.4f}"
        )

        # -------------------------
        # EARLY STOPPING
        # -------------------------
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience += 1

        if patience >= early_stop_patience:
            print("[EARLY STOP] Restoring best DANN model.")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    # ======================================================
    # ALWAYS RESTORE + RETURN MODEL (FIXES YOUR NoneType BUG)
    # ======================================================
    if best_state is not None:
        model.load_state_dict(best_state)

    return model