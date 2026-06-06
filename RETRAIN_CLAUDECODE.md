# iTransformer 재훈련 작업 지시서 (Claude Code 핸드오프)

## 개요

`HR_HRV_Norm.ipynb` 훈련 코드에서 두 가지 결함을 수정하고 iTransformer 단일 모델로 재훈련한다.

- 수정 파일: `HR_HRV_Norm.ipynb`
- 추론 파일: `Stress_classification.ipynb`
- 사용 모델: **iTransformer 단독**
- 데이터: CATSA (train/val) → EmpaticaE4 (test), cross-dataset

---

## 수정 위치 및 내용

### [1] Cell 2 — MODELS 리스트 변경

```python
# ❌ 기존
MODELS = [
    'TimeMixerPP', 'Medformer', 'TSLANet', 'ModernTCN',
    'CrossGNN', 'TimesNet', 'Mamba2', 'iTransformer', 'PatchTST', 'Chronos2', 'TransformerCNN',
]

# ✅ 수정 후
MODELS = ['iTransformer']
```

---

### [2] Cell 3 — `build_all_arrays` 함수 교체

문제:
- 정규화 시 baseline + 모든 stress task를 합쳐 mu/sigma 계산 → leakage
- val split이 전체 윈도우의 뒤 10%를 그대로 자름 → 피험자 섞임

```python
# ❌ 기존 build_all_arrays — 아래 함수 전체를 교체
def build_all_arrays(root: Path, w: int, stride: int, val_ratio: float = 0.10):
    xs, ys = [], []
    skipped = []
    for sub in discover_subjects(root):
        try:
            task_sigs = {task: load_modalities(sub / task) for task in TASKS}
            # Per-subject normalization across all tasks
            all_sig = np.concatenate(list(task_sigs.values()), axis=1)
            mu    = np.mean(all_sig, axis=1, keepdims=True)
            sigma = np.std(all_sig,  axis=1, keepdims=True) + 1e-8
            for task, sig in task_sigs.items():
                norm = (sig - mu) / sigma
                x = make_windows_3d(norm, w, stride)
                if len(x):
                    xs.append(x)
                    ys.append(np.full(len(x), int(task in STRESS_TASKS), np.int64))
        except Exception as e:
            skipped.append(f'{sub.name}: {e}')
    if skipped:
        print('Skipped subjects:')
        for item in skipped[:10]:
            print(f'  {item}')
        if len(skipped) > 10:
            print(f'  ... and {len(skipped) - 10} more')
    if not xs:
        raise RuntimeError(f'No CATSA data could be loaded from {root}')
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    split = max(1, int(len(X) * (1 - val_ratio)))
    return X[:split], Y[:split], X[split:], Y[split:]
```

```python
# ✅ 수정 후 build_all_arrays — 위 함수를 아래로 교체
def build_all_arrays(root: Path, w: int, stride: int, val_ratio: float = 0.10):
    per_sub_X, per_sub_Y = [], []
    skipped = []
    for sub in discover_subjects(root):
        try:
            task_sigs = {task: load_modalities(sub / task) for task in TASKS}

            # ✅ 수정 1: Baseline task만으로 mu/sigma 계산 (leakage 제거)
            baseline_sig = task_sigs['Baseline']
            mu    = np.mean(baseline_sig, axis=1, keepdims=True)
            sigma = np.std(baseline_sig,  axis=1, keepdims=True) + 1e-8

            sub_xs, sub_ys = [], []
            for task, sig in task_sigs.items():
                norm = (sig - mu) / sigma
                x = make_windows_3d(norm, w, stride)
                if len(x):
                    sub_xs.append(x)
                    sub_ys.append(np.full(len(x), int(task in STRESS_TASKS), np.int64))

            if sub_xs:
                per_sub_X.append(np.concatenate(sub_xs))
                per_sub_Y.append(np.concatenate(sub_ys))
        except Exception as e:
            skipped.append(f'{sub.name}: {e}')

    if skipped:
        print('Skipped subjects:')
        for item in skipped[:10]:
            print(f'  {item}')
        if len(skipped) > 10:
            print(f'  ... and {len(skipped) - 10} more')
    if not per_sub_X:
        raise RuntimeError(f'No CATSA data could be loaded from {root}')

    # ✅ 수정 2: 피험자 단위 split (윈도우 누수 없음)
    n_subs = len(per_sub_X)
    n_val  = max(1, int(n_subs * val_ratio))
    train_subs = per_sub_X[:-n_val]
    train_ys   = per_sub_Y[:-n_val]
    val_subs   = per_sub_X[-n_val:]
    val_ys     = per_sub_Y[-n_val:]

    x_train = np.concatenate(train_subs)
    y_train = np.concatenate(train_ys)
    x_val   = np.concatenate(val_subs)
    y_val   = np.concatenate(val_ys)

    print(f'  피험자: train={n_subs - n_val}명, val={n_val}명')
    return x_train, y_train, x_val, y_val
```

---

### [3] Cell 4 — `e4_windows` 함수 교체

문제: test(E4)도 stress+rest 전체로 정규화 → CATSA 훈련과 규칙 불일치

```python
# ❌ 기존 e4_windows
def e4_windows(sig: np.ndarray, labels: np.ndarray, W: int, S: int):
    mask  = labels >= 0
    mu    = np.mean(sig[:, mask], axis=1, keepdims=True) if mask.any() else np.mean(sig, axis=1, keepdims=True)
    sigma = np.std(sig[:, mask],  axis=1, keepdims=True) + 1e-8 if mask.any() else np.std(sig, axis=1, keepdims=True) + 1e-8
    norm = ((sig - mu) / sigma).astype(np.float32)
    xs, ys = [], []
    for i in range(0, norm.shape[-1] - W + 1, S):
        u = np.unique(labels[i:i + W])
        if len(u) == 1 and u[0] in (0, 1):
            xs.append(norm[:, i:i + W])
            ys.append(int(u[0]))
    if not xs:
        return np.empty((0, N_CHANNELS, W), np.float32), np.empty(0, np.int64)
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)
```

```python
# ✅ 수정 후 e4_windows — rest(label==0) 구간만으로 정규화
def e4_windows(sig: np.ndarray, labels: np.ndarray, W: int, S: int):
    # ✅ rest 구간(label==0)만으로 mu/sigma 계산 → CATSA Baseline 정규화와 동일 규칙
    rest_mask = labels == 0
    if not rest_mask.any():
        rest_mask = labels >= 0   # fallback: rest 구간 없으면 전체 사용
    mu    = np.mean(sig[:, rest_mask], axis=1, keepdims=True)
    sigma = np.std(sig[:, rest_mask],  axis=1, keepdims=True) + 1e-8
    norm  = ((sig - mu) / sigma).astype(np.float32)
    xs, ys = [], []
    for i in range(0, norm.shape[-1] - W + 1, S):
        u = np.unique(labels[i:i + W])
        if len(u) == 1 and u[0] in (0, 1):
            xs.append(norm[:, i:i + W])
            ys.append(int(u[0]))
    if not xs:
        return np.empty((0, N_CHANNELS, W), np.float32), np.empty(0, np.int64)
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)
```

---

### [4] Stress_classification.ipynb — Cell 5 정규화 교체

추론 노트북도 훈련과 동일 규칙(baseline 구간만 정규화)으로 맞춰야 함.
프로토콜 기준: baseline = 측정 시작 후 60~360초 (초반 1분 적응 구간 제외).

```python
# ❌ 기존 Cell 5 정규화
sig      = np.vstack([hr_1hz[None, :], hrv_1hz[None, :]])
mu       = sig.mean(axis=1, keepdims=True)
sigma    = sig.std(axis=1,  keepdims=True) + 1e-8
sig_norm = ((sig - mu) / sigma).astype(np.float32)

# ✅ 수정 후 — baseline 구간(60~360s)만으로 mu/sigma 추정
sig      = np.vstack([hr_1hz[None, :], hrv_1hz[None, :]])
base_mask = (t_1hz >= 60) & (t_1hz < 360)   # baseline 후반 5분
if base_mask.sum() < 10:
    base_mask = np.ones(len(t_1hz), dtype=bool)  # fallback
    print("경고: baseline 구간 샘플 부족 — 전체 신호로 정규화")
mu       = sig[:, base_mask].mean(axis=1, keepdims=True)
sigma    = sig[:, base_mask].std(axis=1,  keepdims=True) + 1e-8
sig_norm = ((sig - mu) / sigma).astype(np.float32)
print(f"정규화 기준: baseline {base_mask.sum()}s  "
      f"HR mean={mu[0,0]:.1f}, std={sigma[0,0]:.1f}  "
      f"HRV mean={mu[1,0]:.2f}, std={sigma[1,0]:.2f}")
```

---

## 수정 후 완료 기준

- [ ] `build_all_arrays` — baseline-only 정규화 적용 확인
- [ ] `build_all_arrays` — val이 train과 피험자 겹치지 않음 확인 (출력 메시지로 검증)
- [ ] `e4_windows` — rest(label==0) 기준 정규화 적용 확인
- [ ] `Stress_classification.ipynb` Cell 5 — baseline 구간 정규화 적용 확인
- [ ] MODELS = ['iTransformer'] 단일 모델 확인
- [ ] 재훈련 실행 후 E4 test F1 / AUROC / Accuracy 보고

## 건드리지 말 것

- `detect_hrv_from_hr` 함수 (HRV 계산 방식은 별도 트랙, 지금 재훈련과 무관)
- `load_modalities`, `make_windows_3d` 함수
- Optuna 훈련 루프, 체크포인트 저장 로직
- `HR_HRV_models_ten.py` 모델 정의 파일
