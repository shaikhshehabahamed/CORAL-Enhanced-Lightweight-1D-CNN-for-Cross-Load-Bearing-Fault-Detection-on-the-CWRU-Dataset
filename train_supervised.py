# train_supervised.py
import torch
from train_helper import validate
from torch.cuda.amp import autocast, GradScaler

def fit_supervised(epochs, model, opt, src_dl, src_val_dl, loss_func, early_stop_patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    best_state = None
    patience = 0

    print("EPOCH\tTrainLoss\tValLoss\tValAcc")

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0
        n_samples = 0

        for xb, yb in src_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                _out = model(xb)
                logits = _out[0] if isinstance(_out, (tuple, list)) else _out
                loss = loss_func(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = yb.size(0)
            n_samples += bs
            total_loss += loss.item() * bs

        train_loss = total_loss / max(1, n_samples)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = validate(model, src_val_dl, loss_func)

        print(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}")

        # early stop
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= early_stop_patience:
            print("[EARLY STOP] Restoring best supervised model.")
            model.load_state_dict(best_state)
            break

    return model