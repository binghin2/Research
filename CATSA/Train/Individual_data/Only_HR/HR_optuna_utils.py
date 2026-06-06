"""
optuna_utils.py — Hyperparameter search spaces and objective factory for Stage1 Optuna search.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

COMMON_SEARCH_SPACE: dict[str, Any] = {
    "lr":      ("float", 1e-4, 1e-2, True),   # (type, low, high, log)
    "wd":      ("float", 1e-5, 1e-2, True),
    "dropout": ("float", 0.0,  0.4,  False),
    "bs":      ("categorical", [32, 64, 128]),
    "patience": ("int", 50, 100),
}

# Per-model search spaces on top of COMMON_SEARCH_SPACE.
# Each entry: param_name -> (type, *args)  same convention as COMMON_SEARCH_SPACE
MODEL_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "TimeMixerPP": {
        "d_model":   ("categorical", [64, 128, 256]),
        "n_blocks":  ("int", 1, 4),
        "k_periods": ("int", 2, 5),
        "n_heads":   ("categorical", [2, 4, 8]),
    },
    "Medformer": {
        "d_model":  ("categorical", [64, 128, 256]),
        "n_heads":  ("categorical", [2, 4, 8]),
        "n_layers": ("int", 1, 4),
    },
    "TSLANet": {
        "d_model":  ("categorical", [64, 128, 256]),
        "n_blocks": ("int", 1, 5),
    },
    "ModernTCN": {
        "d_model":  ("categorical", [64, 128, 256]),
        "n_blocks": ("int", 1, 5),
    },
    "CrossGNN": {
        "d_model":  ("categorical", [32, 64, 128]),
        "n_layers": ("int", 1, 4),
    },
    "TimesNet": {
        "d_model":   ("categorical", [32, 64, 128]),
        "n_blocks":  ("int", 1, 4),
        "k_periods": ("int", 2, 5),
    },
    "Mamba2": {
        "d_model":  ("categorical", [32, 64, 128]),
        "d_state":  ("categorical", [8, 16, 32]),
        "n_layers": ("int", 1, 5),
    },
    "iTransformer": {
        "d_model":  ("categorical", [64, 128, 256]),
        "n_heads":  ("categorical", [2, 4, 8]),
        "n_layers": ("int", 1, 4),
        "d_ff":     ("categorical", [128, 256, 512]),
    },
    "PatchTST": {
        "d_model":  ("categorical", [64, 128, 256]),
        "n_heads":  ("categorical", [2, 4, 8]),
        "n_layers": ("int", 1, 4),
        "d_ff":     ("categorical", [128, 256, 512]),
    },
    "Chronos2": {
        "dropout": ("float", 0.1, 0.5, False),
    },
    "TransformerCNN": {
        "d_model":       ("categorical", [64, 128, 256]),
        "n_heads":       ("categorical", [2, 4, 8]),
        "n_layers":      ("int", 1, 4),
        "d_ff":          ("categorical", [128, 256, 512]),
        "n_conv_layers": ("int", 1, 3),
        "cnn_kernel":    ("categorical", [3, 5, 7]),
    },
}


def sample_hp(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Sample all hyperparameters for model_name from trial."""
    hp: dict[str, Any] = {}

    # Common params
    for param, spec in COMMON_SEARCH_SPACE.items():
        hp[param] = _suggest(trial, param, spec)

    # Model-specific params (override common if key collides, e.g. Chronos2 dropout)
    for param, spec in MODEL_SEARCH_SPACES.get(model_name, {}).items():
        hp[param] = _suggest(trial, param, spec)

    return hp


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "float":
        _, low, high, log = spec
        return trial.suggest_float(name, low, high, log=log)
    if kind == "int":
        _, low, high = spec
        return trial.suggest_int(name, low, high)
    if kind == "categorical":
        _, choices = spec
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unknown spec kind: {kind}")


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none')
        pt = torch.exp(-bce)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Validation F1
# ---------------------------------------------------------------------------

@torch.no_grad()
def _val_f1(model: nn.Module, loader: DataLoader,
            use_bf16: bool, device: torch.device) -> float:
    model.eval()
    all_pred, all_true = [], []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = (logits.sigmoid() >= 0.5).long().cpu()
            all_pred.append(preds)
            all_true.append(yb)
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def evaluate_on_e4(model, e4_data_per_subject, device, use_bf16: bool = True, chunk_size: int = 256):
    """
    Evaluate model on EmpaticaE4 cross-dataset test set.
    Returns per-subject and overall metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    all_true, all_pred, all_prob = [], [], []
    per_subject = []

    for sub, (x, y_true) in e4_data_per_subject.items():
        if len(x) == 0:
            per_subject.append({"subject": sub, "n_windows": 0})
            continue

        probs = []
        for i in range(0, len(x), chunk_size):
            xb = torch.from_numpy(x[i:i + chunk_size]).to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                    lg = model(xb).float()
            probs.append(torch.sigmoid(lg).cpu().numpy())

        prob = np.concatenate(probs)
        y_pred = (prob >= 0.5).astype(int)

        per_subject.append(
            {
                "subject": sub,
                "n_windows": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }
        )

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        all_prob.extend(prob.tolist())

    if not all_true:
        return {"overall": None, "per_subject": per_subject}

    overall = {
        "accuracy": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred, zero_division=0),
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall": recall_score(all_true, all_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
    }
    try:
        overall["auroc"] = roc_auc_score(all_true, all_prob)
    except Exception:
        overall["auroc"] = float("nan")

    return {"overall": overall, "per_subject": per_subject}


def finalize_best_checkpoint(study: optuna.Study, model_name: str, save_dir: str | Path):
    """Study 종료 후 best trial의 checkpoint를 영구 보관, 나머지는 삭제."""
    save_dir = Path(save_dir)
    tmp_dir = save_dir / "tmp_checkpoints" / model_name
    final_dir = save_dir / "best_models"
    final_dir.mkdir(parents=True, exist_ok=True)

    try:
        best_trial = study.best_trial
    except ValueError:
        return None

    src = tmp_dir / f"trial_{best_trial.number}.pt"
    dst = final_dir / f"{model_name}_best.pt"

    if src.exists():
        import shutil

        shutil.copy(src, dst)
        for f in tmp_dir.glob("trial_*.pt"):
            if f.exists():
                f.unlink()
        return dst

    return None


def generate_stage1_summary(study_results, e4_results, save_dir):
    """
    Generate comprehensive summary CSV combining Optuna search results and E4 evaluation.

    Args:
        study_results: dict {model_name: optuna.Study}
        e4_results: dict {model_name: dict with 'overall' and 'per_subject'}
        save_dir: Path to save directory

    Returns:
        pd.DataFrame sorted by E4 F1 descending
    """
    import json

    rows = []
    common_keys = ['lr', 'batch_size', 'dropout', 'weight_decay', 'focal_gamma']

    for model_name, study in study_results.items():
        if study is None:
            rows.append({
                'model': model_name,
                'status': 'FAILED',
                'best_val_f1': float('nan'),
                'e4_f1': float('nan'),
            })
            continue

        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        try:
            best_params = study.best_params
            best_val_f1 = study.best_value
            best_trial_num = study.best_trial.number
        except (ValueError, AttributeError):
            best_params = {}
            best_val_f1 = float('nan')
            best_trial_num = -1

        best_common = {
            'lr': best_params.get('lr', float('nan')),
            'batch_size': best_params.get('bs', float('nan')),
            'dropout': best_params.get('dropout', float('nan')),
            'weight_decay': best_params.get('wd', float('nan')),
            'focal_gamma': best_params.get('focal_gamma', 2.0),
        }
        best_specific = {k: v for k, v in best_params.items() if k not in common_keys and k not in {'bs', 'wd'}}

        e4 = e4_results.get(model_name)
        e4_overall = e4['overall'] if (e4 and e4.get('overall')) else {}

        total_time_min = study.user_attrs.get('total_time_min', float('nan'))

        rows.append({
            'model': model_name,
            'status': 'OK',
            'best_val_f1': best_val_f1,
            'best_trial_num': best_trial_num,
            'n_trials_completed': n_complete,
            'n_trials_pruned': n_pruned,
            'n_trials_failed': n_failed,
            **{f'best_{k}': v for k, v in best_common.items()},
            'best_model_specific': json.dumps(best_specific),
            'e4_accuracy': e4_overall.get('accuracy', float('nan')),
            'e4_f1': e4_overall.get('f1', float('nan')),
            'e4_precision': e4_overall.get('precision', float('nan')),
            'e4_recall': e4_overall.get('recall', float('nan')),
            'e4_auroc': e4_overall.get('auroc', float('nan')),
            'total_time_min': total_time_min,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('e4_f1', ascending=False, na_position='last').reset_index(drop=True)

    save_path = Path(save_dir) / 'stage1_summary.csv'
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    return df


def save_per_subject_results(e4_results, save_dir):
    """Save per-subject E4 performance for each model in long format."""
    rows = []
    for model_name, result in e4_results.items():
        if result is None or 'per_subject' not in result:
            continue
        for sub_result in result['per_subject']:
            rows.append({'model': model_name, **sub_result})

    df = pd.DataFrame(rows)
    save_path = Path(save_dir) / 'stage1_per_subject.csv'
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return df


def plot_stage1_comparison(summary_df, study_results, save_dir, signal_name='TEMP'):
    """
    Generate 3-panel comparison plot:
    - Left: Bar chart of E4 metrics (Accuracy, F1, AUROC) per model
    - Center: Optimization history overlay (all models)
    - Right: F1 ranking horizontal bar
    """
    import matplotlib.pyplot as plt

    valid_df = summary_df.dropna(subset=['e4_f1']).copy()
    if len(valid_df) == 0:
        print('No valid results to plot.')
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    models = valid_df['model'].tolist()
    x = np.arange(len(models))
    width = 0.25

    axes[0].bar(x - width, valid_df['e4_accuracy'], width, label='Accuracy', color='#4C78A8', alpha=0.85)
    axes[0].bar(x, valid_df['e4_f1'], width, label='F1', color='#F58518', alpha=0.85)
    axes[0].bar(x + width, valid_df['e4_auroc'], width, label='AUROC', color='#54A24B', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=40, ha='right', fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Score')
    axes[0].set_title('E4 Test Performance (sorted by F1)', fontsize=11)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    import optuna
    colors = plt.cm.tab10(np.linspace(0, 1, len(study_results)))
    for (name, study), color in zip(study_results.items(), colors):
        if study is None:
            continue
        best_so_far = []
        current_best = -float('inf')
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
                current_best = max(current_best, trial.value)
            best_so_far.append(current_best if current_best != -float('inf') else None)

        valid_x = [i + 1 for i, v in enumerate(best_so_far) if v is not None]
        valid_y = [v for v in best_so_far if v is not None]

        if valid_x:
            axes[1].plot(valid_x, valid_y, label=name, color=color,
                         linewidth=1.8, marker='o', markersize=4, alpha=0.85)

    axes[1].set_xlabel('Trial Number')
    axes[1].set_ylabel('Best Val F1 So Far')
    axes[1].set_title('Optimization History (CATSA Validation)', fontsize=11)
    axes[1].legend(fontsize=8, loc='lower right', ncol=2)
    axes[1].grid(linestyle='--', alpha=0.4)

    f1_vals = valid_df['e4_f1'].values[::-1]
    names_sorted = models[::-1]
    bar_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(models)))

    axes[2].barh(range(len(models)), f1_vals, color=bar_colors)
    axes[2].set_yticks(range(len(models)))
    axes[2].set_yticklabels(names_sorted, fontsize=10)
    axes[2].set_xlim(0, 1)
    axes[2].axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    axes[2].set_xlabel('E4 Test F1')
    axes[2].set_title('F1 Ranking', fontsize=11)
    axes[2].grid(axis='x', linestyle='--', alpha=0.4)

    for i, v in enumerate(f1_vals):
        if not np.isnan(v):
            axes[2].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    fig.suptitle(f'Stage 1 Baseline Comparison — 11 Models on CATSA→EmpaticaE4 ({signal_name})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = Path(save_dir) / 'stage1_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_per_subject_heatmap(per_subject_df, save_dir, metric='f1'):
    """Heatmap of model x subject performance."""
    import matplotlib.pyplot as plt

    if len(per_subject_df) == 0:
        print('No per-subject data to plot.')
        return

    pivot = per_subject_df.pivot(index='model', columns='subject', values=metric)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(pivot) + 2))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='black' if 0.3 < v < 0.7 else 'white', fontsize=9)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f'Per-subject {metric.upper()} (rows sorted by mean)')

    means = pivot.mean(axis=1).values
    for i, m in enumerate(means):
        ax.text(pivot.shape[1] - 0.3, i, f'  μ={m:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()

    save_path = Path(save_dir) / f'stage1_per_subject_heatmap_{metric}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def save_hp_importance_plots(study_results, save_dir):
    """Save per-model Optuna visualization plots."""
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as ovm

    out_dir = Path(save_dir) / 'optuna_results'
    out_dir.mkdir(exist_ok=True, parents=True)

    for model_name, study in study_results.items():
        if study is None:
            continue

        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete < 2:
            print(f'  [{model_name}] Skipping plots (only {n_complete} completed trials)')
            continue

        try:
            ovm.plot_optimization_history(study)
            plt.title(f'{model_name} - Optimization History')
            plt.tight_layout()
            plt.savefig(out_dir / f'{model_name}_history.png', dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'  [{model_name}] history plot failed: {e}')

        if n_complete >= 3:
            try:
                ovm.plot_param_importances(study)
                plt.title(f'{model_name} - HP Importance')
                plt.tight_layout()
                plt.savefig(out_dir / f'{model_name}_importance.png', dpi=120, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f'  [{model_name}] importance plot failed: {e}')

        try:
            ovm.plot_parallel_coordinate(study)
            plt.title(f'{model_name} - Parallel Coordinates')
            plt.tight_layout()
            plt.savefig(out_dir / f'{model_name}_parallel.png', dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'  [{model_name}] parallel plot failed: {e}')

    print(f'Plots saved to: {out_dir}')


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    W: int,
    n_channels: int,
    alpha: float,
    device: torch.device,
    save_dir: str | Path,
    epochs: int = 30,
    patience: int = 7,
    chronos_pipeline=None,
):
    """
    Returns (objective_fn, best_f1_ref, best_hp_ref).
    best_f1_ref[0] and best_hp_ref[0] are updated in-place after each trial.
    """
    from HR_models_ten import build_model  # local import to avoid circular deps

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    is_chronos = (model_name == "Chronos2")
    use_bf16 = (not is_chronos) and torch.cuda.is_available()

    best_f1_ref: list[float] = [-1.0]
    best_hp_ref: list[dict] = [{}]

    tmp_dir = save_dir / "tmp_checkpoints" / model_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).long()
    xv = torch.from_numpy(x_val).float()
    yv = torch.from_numpy(y_val).long()

    val_ds = TensorDataset(xv, yv)

    def objective(trial: optuna.Trial) -> float:
        set_seed(42)
        hp = sample_hp(trial, model_name)
        bs = int(hp.pop("bs"))
        lr = float(hp.pop("lr"))
        wd = float(hp.pop("wd"))

        g = torch.Generator()
        g.manual_seed(42)

        train_ds = TensorDataset(xt, yt)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True,
                                  worker_init_fn=seed_worker, generator=g)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                  num_workers=2, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g)

        # Build model
        if is_chronos:
            if chronos_pipeline is None:
                raise optuna.TrialPruned()
            model = build_model(model_name, W=W, n_channels=n_channels,
                                pipeline=chronos_pipeline, **hp)
            # Split optimizer: encoder lr×0.1, head lr×1
            enc_params, head_params = [], []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    (enc_params if 'head' not in n else head_params).append(p)
            optimizer = torch.optim.AdamW([
                {'params': enc_params, 'lr': lr * 0.1},
                {'params': head_params, 'lr': lr},
            ], weight_decay=wd)
        else:
            model = build_model(model_name, W=W, n_channels=n_channels, **hp)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        model = model.to(device)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_f1  = -1.0
        no_improve = 0
        ckpt_path = tmp_dir / f"trial_{trial.number}.pt"

        for epoch in range(epochs):
            # --- train ---
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = criterion(model(xb), yb)
                else:
                    loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # --- validate ---
            val_f1 = _val_f1(model, val_loader, use_bf16 and not is_chronos, device)

            # Pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                if ckpt_path.exists():
                    ckpt_path.unlink()
                raise optuna.TrialPruned()

            if val_f1 > best_f1:
                best_f1 = val_f1
                no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "hp": {**hp, "lr": lr, "wd": wd, "bs": bs},
                        "epoch": epoch,
                        "val_f1": val_f1,
                        "trial_number": trial.number,
                        "model_name": model_name,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Update global best
        if best_f1 > best_f1_ref[0]:
            best_f1_ref[0] = best_f1
            best_hp_ref[0] = {**hp, 'lr': lr, 'wd': wd, 'bs': bs}

        del model
        torch.cuda.empty_cache()
        return best_f1

    return objective, best_f1_ref, best_hp_ref
