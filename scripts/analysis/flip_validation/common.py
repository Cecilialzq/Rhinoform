"""Shared helpers for the Gate2/Gate3/A1 pathfinding protocol (2026-06-15).

All heavy artefacts are read from a LOCAL repo copy (default /tmp/gate23a1) to
avoid the Google-Drive file-stream latency. Outputs are written to local disk
and packaged once at the end.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SEED = 20260615
LOCAL = Path(os.environ.get("GATE23A1_LOCAL", "/tmp/gate23a1"))
REPO = LOCAL / "repo"                      # manifest.json + meshes/
SPLIT_MANIFEST = LOCAL / "split_manifest.json"
CVAE_PKG = LOCAL / "pkg" / "cvae_base.pt"
RBSR_PKG = LOCAL / "pkg" / "rbsr_gate_model.pt"
OUT = LOCAL / "out"

SUBUNITS = ("root", "dorsum", "tip", "alar_left", "alar_right")

# Make the production code importable.
_WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WS / "code"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_provenance() -> dict:
    import numpy as _np
    import scipy as _sp

    prov = {
        "seed": SEED,
        "python": sys.version.split()[0],
        "numpy": _np.__version__,
        "scipy": _sp.__version__,
        "inputs_sha256": {},
    }
    try:
        import torch as _torch

        prov["torch"] = _torch.__version__
    except Exception:
        prov["torch"] = None
    for name, path in (
        ("data_manifest", REPO / "manifest.json"),
        ("split_manifest", SPLIT_MANIFEST),
        ("cvae_package", CVAE_PKG),
        ("rbsr_package", RBSR_PKG),
    ):
        if Path(path).exists():
            prov["inputs_sha256"][name] = sha256_file(path)
    return prov


def load_split_ids() -> dict[str, list[str]]:
    d = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return {
        "train": [str(x) for x in d["train_pool_ids"]],
        "val": [str(x) for x in d["val_ids"]],
        "test": [str(x) for x in d["test_ids"]],
    }


def load_meshes(ids: list[str]) -> dict[str, dict]:
    """Load only the requested identities from the local repo (no full scan)."""
    by_id: dict[str, dict] = {}
    for sid in ids:
        sid = str(sid)
        path = REPO / "meshes" / f"{sid}_neutral.npz"
        data = np.load(path, allow_pickle=False)
        by_id[sid] = {
            "subject_id": sid,
            "vertices": np.asarray(data["vertices"], dtype=np.float64),
            "faces": np.asarray(data["faces"], dtype=np.int64),
            "landmarks": np.asarray(data["local_landmarks"], dtype=np.int64),
            "subunits": {k: np.asarray(data[f"subunit_{k}"], dtype=np.int64) for k in SUBUNITS},
        }
    return by_id


def template_topology(by_id: dict[str, dict]) -> dict:
    first = next(iter(by_id.values()))
    return {
        "faces": first["faces"],
        "landmarks": first["landmarks"],
        "subunits": first["subunits"],
        "n_vertices": first["vertices"].shape[0],
    }


def roi_index(by_id: dict[str, dict]) -> np.ndarray:
    """Union of all subunit vertex indices (the analysed ROI)."""
    top = template_topology(by_id)
    idx = np.unique(np.concatenate([top["subunits"][k] for k in SUBUNITS]))
    return idx.astype(np.int64)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = values[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def identity_bootstrap_ci(per_pair: np.ndarray, pair_ids: list[tuple[str, str]],
                          n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float, float]:
    """Identity-level bootstrap: resample identities, average pairs touching them."""
    per_pair = np.asarray(per_pair, dtype=np.float64)
    ids = sorted({i for pair in pair_ids for i in pair}, key=lambda s: int(s))
    id_to_pairs: dict[str, list[int]] = {i: [] for i in ids}
    for k, (a, b) in enumerate(pair_ids):
        id_to_pairs[a].append(k)
        id_to_pairs[b].append(k)
    rng = np.random.default_rng(seed)
    n = len(ids)
    boots = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n, n)
        sel = []
        for c in chosen:
            sel.extend(id_to_pairs[ids[c]])
        if sel:
            boots.append(per_pair[sel].mean())
    boots = np.asarray(boots)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(per_pair.mean()), lo, hi


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT
