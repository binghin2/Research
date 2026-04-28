"""
optuna_utils.py — Reusable Optuna hyperparameter search utilities for models_six.py.

Provides:
  suggest_hp(trial, model_name, W, n_channels)
      → dict of suggested hyper-parameters (architecture + training)

  make_objective(model_name, x_tr, y_tr, x_va, y_va, W, n_channels,
                 device, alpha, n_epochs, patience, use_bf16)
      → Optuna objective callable (maximises val F1)

  run_study(objective, n_trials, study_name, direction, pruner)
      → optuna.Study

  report_study(study)
      → prints / returns a summary DataFrame
"""

import math
from copy import deepcopy

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models_six import build_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── FocalLoss (same as Temp_TenModels.ipynb) ────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt  = torch.sigmoid(logits) * y + (1 - torch.sigmoid(logits)) * (1 - y)
        fl  = (1 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            fl = (self.alpha * y + (1 - self.alpha) * (1 - y)) * fl
        return fl.mean()


# ─── Hyperparameter search spaces ────────────────────────────────────────────

def suggest_hp(trial: optuna.Trial, model_name: str,
               W: int = 240, n_channels: int = 1) -> dict:
    """
    Suggest a full set of hyper-parameters for *model_name*.

    Returns a flat dict with keys:
      - training HPs: lr, wd, batch_size, dropout
      - architecture HPs: model-specific keys (d_model, n_heads, …)
    All keys are also stored in trial.user_attrs so they appear in
    the Optuna study DataFrame.
    """
    hp = {}

    # ── shared training HP ───────────────────────────────────────────────────
    hp["lr"]         = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    hp["wd"]         = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    hp["dropout"]    = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    # ── architecture HP (model-specific) ─────────────────────────────────────
    if model_name == "TimeMixerPP":
        hp["d_model"]   = trial.suggest_categorical("d_model",   [64, 128, 256])
        hp["n_blocks"]  = trial.suggest_int("n_blocks",  1, 4)
        hp["k_periods"] = trial.suggest_int("k_periods", 2, 5)
        hp["n_heads"]   = trial.suggest_categorical("n_heads", [2, 4, 8])

    elif model_name == "Medformer":
        hp["d_model"]  = trial.suggest_categorical("d_model",  [64, 128, 256])
        hp["n_heads"]  = trial.suggest_categorical("n_heads",  [2, 4, 8])
        hp["n_layers"] = trial.suggest_int("n_layers", 1, 4)
        # patch_lens: pick from predefined combinations
        pl_choice = trial.suggest_categorical("patch_lens_key",
                                              ["4-8", "4-8-16", "8-16", "8-16-32"])
        hp["patch_lens"] = [int(v) for v in pl_choice.split("-")]

    elif model_name == "TSLANet":
        hp["d_model"]      = trial.suggest_categorical("d_model",      [64, 128, 256])
        hp["patch_len"]    = trial.suggest_categorical("patch_len",    [4, 8, 16])
        hp["patch_stride"] = trial.suggest_categorical("patch_stride", [2, 4, 8])
        hp["n_blocks"]     = trial.suggest_int("n_blocks", 1, 4)

    elif model_name == "ModernTCN":
        hp["d_model"]      = trial.suggest_categorical("d_model",      [64, 128, 256])
        hp["patch_len"]    = trial.suggest_categorical("patch_len",    [4, 8, 16])
        hp["patch_stride"] = trial.suggest_categorical("patch_stride", [2, 4, 8])
        hp["n_blocks"]     = trial.suggest_int("n_blocks", 1, 4)

    elif model_name == "CrossGNN":
        hp["d_model"]      = trial.suggest_categorical("d_model",      [32, 64, 128])
        hp["patch_len"]    = trial.suggest_categorical("patch_len",    [4, 8, 16])
        hp["patch_stride"] = trial.suggest_categorical("patch_stride", [2, 4, 8])
        hp["n_layers"]     = trial.suggest_int("n_layers", 1, 3)

    elif model_name == "TimesNet":
        hp["d_model"]   = trial.suggest_categorical("d_model",   [32, 64, 128])
        hp["n_blocks"]  = trial.suggest_int("n_blocks",  1, 4)
        hp["k_periods"] = trial.suggest_int("k_periods", 2, 5)

    elif model_name == "Mamba2":
        hp["d_model"]  = trial.suggest_categorical("d_model",  [32, 64, 128])
        hp["d_state"]  = trial.suggest_categorical("d_state",  [8, 16, 32])
        hp["n_layers"] = trial.suggest_int("n_layers", 1, 5)

    elif model_name == "iTransformer":
        hp["d_model"]  = trial.suggest_categorical("d_model",  [64, 128, 256])
        hp["n_heads"]  = trial.suggest_categorical("n_heads",  [2, 4, 8])
        hp["n_layers"] = trial.suggest_int("n_layers", 1, 4)
        hp["d_ff"]     = trial.suggest_categorical("d_ff",     [128, 256, 512])

    elif model_name == "PatchTST":
        hp["d_model"]      = trial.suggest_categorical("d_model",      [64, 128, 256])
        hp["patch_len"]    = trial.suggest_categorical("patch_len",    [8, 16, 32])
        hp["patch_stride"] = trial.suggest_categorical("patch_stride", [4, 8, 16])
        hp["n_heads"]      = trial.suggest_categorical("n_heads",      [2, 4, 8])
        hp["n_layers"]     = trial.suggest_int("n_layers", 1, 4)
        hp["d_ff"]         = trial.suggest_categorical("d_ff",         [128, 256, 512])

    else:
        raise ValueError(f"suggest_hp: unknown model '{model_name}'")

    # Persist all keys into trial for CSV export
    for k, v in hp.items():
        trial.set_user_attr(k, str(v))

    return hp


def _build_model_from_hp(model_name: str, W: int, n_channels: int, hp: dict):
    """Build a model instance from a suggest_hp dict (no Chronos2 support)."""
    kw = dict(seq_len=W, n_channels=n_channels, dropout=hp["dropout"])

    if model_name == "TimeMixerPP":
        kw.update(d_model=hp["d_model"], n_blocks=hp["n_blocks"],
                  k_periods=hp["k_periods"], n_heads=hp["n_heads"])
    elif model_name == "Medformer":
        kw.update(d_model=hp["d_model"], patch_lens=hp["patch_lens"],
                  n_heads=hp["n_heads"], n_layers=hp["n_layers"])
    elif model_name == "TSLANet":
        kw.update(d_model=hp["d_model"], patch_len=hp["patch_len"],
                  patch_stride=hp["patch_stride"], n_blocks=hp["n_blocks"])
    elif model_name == "ModernTCN":
        kw.update(d_model=hp["d_model"], patch_len=hp["patch_len"],
                  patch_stride=hp["patch_stride"], n_blocks=hp["n_blocks"])
    elif model_name == "CrossGNN":
        kw.update(d_model=hp["d_model"], patch_len=hp["patch_len"],
                  patch_stride=hp["patch_stride"], n_layers=hp["n_layers"])
    elif model_name == "TimesNet":
        kw.update(d_model=hp["d_model"], n_blocks=hp["n_blocks"],
                  k_periods=hp["k_periods"])
    elif model_name == "Mamba2":
        kw.update(d_model=hp["d_model"], d_state=hp["d_state"],
                  n_layers=hp["n_layers"])
    elif model_name == "iTransformer":
        kw.update(d_model=hp["d_model"], n_heads=hp["n_heads"],
                  n_layers=hp["n_layers"], d_ff=hp["d_ff"])
    elif model_name == "PatchTST":
        kw.update(d_model=hp["d_model"], patch_len=hp["patch_len"],
                  patch_stride=hp["patch_stride"], n_heads=hp["n_heads"],
                  n_layers=hp["n_layers"], d_ff=hp["d_ff"])
    else:
        raise ValueError(f"_build_model_from_hp: unknown model '{model_name}'")

    # Use the classifier constructors directly (build_model uses fixed defaults)
    from models_six import (
        TimeMixerPPClassifier, MedformerClassifier, TSLANetClassifier,
        ModernTCNClassifier, CrossGNNClassifier, TimesNetClassifier,
        Mamba2Classifier, iTransformerClassifier, PatchTSTClassifier,
    )
    _MAP = {
        "TimeMixerPP":  TimeMixerPPClassifier,
        "Medformer":    MedformerClassifier,
        "TSLANet":      TSLANetClassifier,
        "ModernTCN":    ModernTCNClassifier,
        "CrossGNN":     CrossGNNClassifier,
        "TimesNet":     TimesNetClassifier,
        "Mamba2":       Mamba2Classifier,
        "iTransformer": iTransformerClassifier,
        "PatchTST":     PatchTSTClassifier,
    }
    return _MAP[model_name](**kw)


# ─── Low-level training helpers ──────────────────────────────────────────────

def _make_loader(x: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def _train_epoch(model: nn.Module, loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 use_bf16: bool = True) -> float:
    model.train()
    tot = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item() * len(xb)
        n   += len(xb)
    return tot / max(n, 1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader,
              criterion: nn.Module,
              device: torch.device,
              use_bf16: bool = True) -> dict:
    model.eval()
    tot = n = tp = tn = fp = fn = 0
    all_prob = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            lg = model(xb).float()
        tot += criterion(lg, yb).item() * len(xb)
        n   += len(xb)
        prob = torch.sigmoid(lg).cpu().numpy()
        all_prob.extend(prob.tolist())
        pred = (prob >= 0.5).astype(int)
        yi   = yb.long().cpu().numpy()
        tp += int(((pred == 1) & (yi == 1)).sum())
        tn += int(((pred == 0) & (yi == 0)).sum())
        fp += int(((pred == 1) & (yi == 0)).sum())
        fn += int(((pred == 0) & (yi == 1)).sum())
    acc = (tp + tn) / max(n, 1)
    pre = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "loss":      tot / max(n, 1),
        "accuracy":  acc,
        "f1":        2 * pre * rec / max(pre + rec, 1e-8),
        "precision": pre,
        "recall":    rec,
    }


# ─── Objective factory ───────────────────────────────────────────────────────

def make_objective(
    model_name: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    W: int = 240,
    n_channels: int = 1,
    device: torch.device = None,
    alpha: float = 0.5,
    n_epochs: int = 30,
    patience: int = 7,
    use_bf16: bool = True,
):
    """
    Returns an Optuna objective function that:
      1. Samples HPs via suggest_hp()
      2. Trains the model with early stopping
      3. Reports intermediate val F1 for pruning
      4. Returns best val F1 as the optimisation target

    Parameters
    ----------
    model_name  : one of the 9 scratch models in models_six.py
    x_tr, y_tr  : training arrays  [N, 1, W], [N]
    x_va, y_va  : validation arrays
    W           : window size (= seq_len)
    n_channels  : number of input channels
    device      : torch.device (auto-detected if None)
    alpha       : FocalLoss alpha (set to neg-class fraction)
    n_epochs    : maximum training epochs per trial
    patience    : early-stopping patience
    use_bf16    : use bfloat16 mixed precision (auto-disabled on CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = use_bf16 and device.type == "cuda"

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hp(trial, model_name, W=W, n_channels=n_channels)

        loader_tr = _make_loader(x_tr, y_tr, hp["batch_size"], shuffle=True)
        loader_va = _make_loader(x_va, y_va, hp["batch_size"], shuffle=False)
        criterion = FocalLoss(gamma=2.0, alpha=alpha).to(device)

        try:
            model = _build_model_from_hp(model_name, W, n_channels, hp).to(device)
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"model build failed: {e}")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=max(2, patience // 2))

        best_f1    = 0.0
        best_state = None
        wait       = 0

        for epoch in range(1, n_epochs + 1):
            _train_epoch(model, loader_tr, optimizer, criterion,
                         device, use_bf16)
            val_m = _evaluate(model, loader_va, criterion, device, use_bf16)
            scheduler.step(val_m["loss"])

            # Optuna intermediate report + pruning
            trial.report(val_m["f1"], step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_m["f1"] > best_f1:
                best_f1    = val_m["f1"]
                best_state = deepcopy(model.state_dict())
                wait       = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Persist best validation metrics for easy inspection
        trial.set_user_attr("best_val_f1",     best_f1)
        trial.set_user_attr("best_epoch",       epoch - wait)
        return best_f1

    return objective


# ─── Study runner ────────────────────────────────────────────────────────────

def run_study(
    objective,
    n_trials: int = 50,
    study_name: str = "optuna_study",
    direction: str = "maximize",
    pruner: optuna.pruners.BasePruner = None,
    sampler: optuna.samplers.BaseSampler = None,
    storage: str = None,
) -> optuna.Study:
    """
    Create (or load) an Optuna study and optimise *objective* for *n_trials*.

    Parameters
    ----------
    objective   : callable returned by make_objective()
    n_trials    : number of Optuna trials
    study_name  : name for the study (used for DB storage)
    direction   : "maximize" (val F1) or "minimize"
    pruner      : e.g. optuna.pruners.MedianPruner(n_startup_trials=5)
    sampler     : e.g. optuna.samplers.TPESampler(seed=42)
    storage     : optional SQLite path, e.g. "sqlite:///optuna.db"

    Returns
    -------
    optuna.Study
    """
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True, gc_after_trial=True)
    return study


# ─── Reporting ───────────────────────────────────────────────────────────────

def report_study(study: optuna.Study, top_k: int = 5) -> "pd.DataFrame":
    """
    Print a summary of the Optuna study and return a DataFrame of all trials.

    Parameters
    ----------
    study : completed optuna.Study
    top_k : number of top trials to highlight

    Returns
    -------
    pd.DataFrame  (one row per completed trial, sorted by value desc)
    """
    import pandas as pd

    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials found.")
        return pd.DataFrame()

    rows = []
    for t in trials:
        row = {"trial": t.number, "value": t.value}
        row.update(t.params)
        row.update({f"ua_{k}": v for k, v in t.user_attrs.items()})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Study: {study.study_name}")
    print(f"  Completed trials : {len(trials)}")
    print(f"  Best trial       : #{best.number}  value={best.value:.4f}")
    print(f"  Best params      :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    print(f"Top-{top_k} trials:")
    print(df.head(top_k).to_string(index=False))
    return df
