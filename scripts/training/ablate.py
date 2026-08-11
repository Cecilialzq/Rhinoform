"""Passthrough runner around train.py that adds one capability the
stock script lacks: --source-pca-dim 0 (ablate the source-PCA conditioning so the
condition vector is control-landmark deltas only). For every other config it is a
transparent passthrough to train.main().

Usage is identical to train.py, e.g.:
  python scripts/training/ablate.py --repo data --out large/no_source --source-pca-dim 0
  python scripts/training/ablate.py --repo data --out large/no_edge --edge-weight 0
  python scripts/training/ablate.py --repo data --out large/strain --strain-weight 0.05
"""
import sys
from pathlib import Path

PRIZE = str(Path(__file__).resolve().parent)
if PRIZE not in sys.path:
    sys.path.insert(0, PRIZE)

import numpy as np
from rhinoform import train as nfe

# 1) allow source-pca-dim 0 -> return None instead of raising (delta-pca-dim is
#    untouched because callers never pass 0 for it).
_orig_fit = nfe.fit_truncated_pca
def _fit(mat, k):
    if int(k) <= 0:
        return None
    return _orig_fit(mat, k)
nfe.fit_truncated_pca = _fit

# 2) when source_pca is None, build the condition from control deltas only,
#    mirroring pair_conditions' return signature exactly.
_orig_pair = nfe.pair_conditions
def _pair_conditions(by_id, pairs, source_pca, cond_mean=None, cond_std=None):
    if source_pca is not None:
        return _orig_pair(by_id, pairs, source_pca, cond_mean, cond_std)
    landmarks = next(iter(by_id.values()))["landmarks"]
    ctrl, source_vertices, target_delta = [], [], []
    for s, t in pairs:
        src = by_id[s]["vertices"]; tgt = by_id[t]["vertices"]
        source_vertices.append(src)
        d = tgt - src
        target_delta.append(d)
        ctrl.append(d[landmarks].reshape(-1))
    ctrl_arr = np.stack(ctrl, axis=0)
    cond = ctrl_arr.copy()
    if cond_mean is None:
        cond_mean = cond.mean(axis=0, keepdims=True)
        cond_std = np.maximum(cond.std(axis=0, keepdims=True), 1e-6)
    cond = (cond - cond_mean) / cond_std
    source_code = np.zeros((len(pairs), 0), dtype=np.float32)
    return (
        cond.astype(np.float32),
        np.stack(source_vertices, axis=0).astype(np.float32),
        np.stack(target_delta, axis=0).astype(np.float32),
        ctrl_arr.astype(np.float32),
        source_code,
        cond_mean,
        cond_std,
    )
nfe.pair_conditions = _pair_conditions

if __name__ == "__main__":
    raise SystemExit(nfe.main())
