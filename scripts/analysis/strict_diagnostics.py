#!/usr/bin/env python3
"""Strict sparse-to-dense diagnostic experiment suite.

This file is intentionally self-contained so it can run in Colab without
depending on older project internals that may be Google Drive placeholders.

Outputs are written under a user-selected run directory. Every completed unit
writes a DONE marker, CSV/JSON summaries, figures, and model-free diagnostics
where possible. Existing completed units are skipped unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _require(pkg: str, import_name: Optional[str] = None) -> Any:
    try:
        return importlib.import_module(import_name or pkg)
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            f"Missing dependency: {pkg}. In Colab run `pip install {pkg}` first."
        ) from exc


pd = _require("pandas")
plt_mod = _require("matplotlib", "matplotlib.pyplot")
plt = plt_mod
sk_decomp = _require("scikit-learn", "sklearn.decomposition")
sk_ensemble = _require("scikit-learn", "sklearn.ensemble")
sk_kernel = _require("scikit-learn", "sklearn.kernel_ridge")
sk_linear = _require("scikit-learn", "sklearn.linear_model")
sk_nn = _require("scikit-learn", "sklearn.neural_network")
sk_pre = _require("scikit-learn", "sklearn.preprocessing")
sp_sparse = _require("scipy", "scipy.sparse")
sp_linalg = _require("scipy", "scipy.sparse.linalg")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch optional for non-neural experiments
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


LANDMARK_LOCAL_DEFAULT = list(range(9))
RBSR_REGIONS = ["root", "dorsum", "tip", "alar_left", "alar_right"]
LC_SCALES = [30, 60, 120, 240, 480, 676]
LC_TRAIN_PAIRS = [30, 60, 120, 240, 480, 652]
BASIS_GRID = [2, 4, 8, 16, 32]
LATENT_GRID = [4, 8, 16, 32, 64]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=json_default))


def atomic_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        atomic_text(path, "")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_fig_atomic(path: Path, dpi: int = 180) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".png", dir=str(path.parent))
    os.close(fd)
    plt.savefig(tmp, dpi=dpi, bbox_inches="tight")
    plt.close()
    os.replace(tmp, path)


def done_marker(path: Path) -> Path:
    return path / "DONE.json"


@contextlib.contextmanager
def experiment_guard(path: Path, overwrite: bool = False):
    ensure_dir(path)
    marker = done_marker(path)
    if marker.exists() and not overwrite:
        log(f"SKIP complete: {path}")
        yield False
        return
    lock = path / "RUNNING.json"
    atomic_json(lock, {"started_at": now(), "pid": os.getpid()})
    try:
        yield True
        atomic_json(marker, {"completed_at": now(), "path": str(path)})
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()
    except Exception as exc:
        atomic_json(path / "FAILED.json", {"failed_at": now(), "error": repr(exc)})
        raise


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    s = str(value)
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def normalize_index_list(obj: Any) -> List[int]:
    if obj is None:
        return []
    if isinstance(obj, str):
        text = obj.strip()
        # Plain metadata strings should not become bogus vertex IDs. Only accept
        # strings that are themselves a single integer token.
        return [int(text)] if re.fullmatch(r"-?\d+", text) else []
    if isinstance(obj, (int, np.integer)):
        return [int(obj)]
    if isinstance(obj, (float, np.floating)):
        return [int(obj)] if float(obj).is_integer() else []
    if isinstance(obj, dict):
        for key in ["vertices", "roi_vertices", "indices", "idx", "values"]:
            if key in obj:
                return normalize_index_list(obj[key])
        if all(str(k).isdigit() for k in obj.keys()):
            return [int(k) for k, v in obj.items() if bool(v)]
        out: List[int] = []
        for v in obj.values():
            if isinstance(v, (list, tuple, dict, np.ndarray, int, float, np.integer, np.floating)):
                out.extend(normalize_index_list(v))
        return sorted(set(out))
    if isinstance(obj, np.ndarray):
        return [int(x) for x in obj.reshape(-1).tolist()]
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(normalize_index_list(x))
        return out
    return []


def load_roi_metadata(root: Path) -> Dict[str, Any]:
    data = read_json(root / "roi" / "vertices.json")
    if not isinstance(data, dict):
        raise ValueError("roi/vertices.json must be a dict with roi_indices")
    return data


def load_roi_indices(root: Path) -> np.ndarray:
    data = load_roi_metadata(root)
    if "roi_indices" in data:
        idx = normalize_index_list(data["roi_indices"])
    elif "vertices" in data:
        idx = normalize_index_list(data["vertices"])
    else:
        raise ValueError("roi/vertices.json has no roi_indices/vertices field")
    if not idx:
        raise ValueError("roi/vertices.json parsed to an empty list")
    return np.asarray(idx, dtype=np.int64)


def auto_landmarks_from_roi(root: Path, roi_global: np.ndarray) -> np.ndarray:
    data = load_roi_metadata(root)
    raw = data.get("landmarks_27_35")
    if not isinstance(raw, dict):
        log("WARNING: roi/vertices.json has no landmarks_27_35; falling back to local landmark indices 0..8")
        return np.asarray(LANDMARK_LOCAL_DEFAULT, dtype=np.int64)
    global_to_local = {int(g): i for i, g in enumerate(roi_global.tolist())}
    locals_: List[int] = []
    missing: Dict[str, Any] = {}
    for key in sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        g = int(raw[key])
        if g in global_to_local:
            locals_.append(global_to_local[g])
        else:
            missing[str(key)] = g
    if missing:
        raise ValueError(f"Some landmarks_27_35 are not in roi_indices: {missing}")
    if len(locals_) != 9:
        raise ValueError(f"Expected 9 landmarks from landmarks_27_35, got {len(locals_)}")
    return np.asarray(locals_, dtype=np.int64)


def load_subunits(root: Path, roi_global: np.ndarray, n_roi: int) -> Dict[str, np.ndarray]:
    path = root / "roi" / "subunits.json"
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("roi/subunits.json must be a dict")
    groups = raw.get("groups", raw)
    if not isinstance(groups, dict):
        raise ValueError("roi/subunits.json groups field must be a dict")
    global_to_local = {int(g): i for i, g in enumerate(roi_global.tolist())}
    out: Dict[str, np.ndarray] = {}
    for name, value in groups.items():
        if isinstance(value, dict):
            if "vertex_indices" in value:
                idx = normalize_index_list(value["vertex_indices"])
            elif "vertices" in value:
                idx = normalize_index_list(value["vertices"])
            elif "indices" in value:
                idx = normalize_index_list(value["indices"])
            else:
                idx = []
        else:
            idx = normalize_index_list(value)
        if not idx:
            continue
        arr = np.asarray(idx, dtype=np.int64)
        if int(arr.max()) >= n_roi:
            arr = np.asarray([global_to_local[int(x)] for x in arr if int(x) in global_to_local], dtype=np.int64)
        arr = arr[(arr >= 0) & (arr < n_roi)]
        if arr.size:
            out[str(name)] = np.unique(arr)
    if not out:
        log("WARNING: roi/subunits.json yielded no usable subunit vertex sets; region diagnostics will be skipped.")
    return out


def load_mesh_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    vertex_keys = ["vertices", "verts", "v", "V", "mesh_vertices", "points", "xyz"]
    face_keys = ["faces", "triangles", "tris", "f", "F"]
    vertices = None
    faces = None
    for key in vertex_keys:
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 2 and arr.shape[1] == 3:
                vertices = arr.astype(np.float32)
                break
    if vertices is None:
        for key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 1000:
                vertices = arr.astype(np.float32)
                break
    for key in face_keys:
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 2 and arr.shape[1] == 3:
                faces = arr.astype(np.int64)
                break
    if vertices is None:
        raise ValueError(f"Could not find vertices in {path}; keys={z.files}")
    return vertices, faces


def mesh_id_from_path(path: Path) -> str:
    return parse_id(path.stem)


def load_identity_table(root: Path, roi_global: np.ndarray) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    mesh_dir = root / "data" / "meshes"
    paths = sorted(mesh_dir.glob("*_neutral.npz"))
    if not paths:
        paths = sorted(mesh_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No mesh npz files under {mesh_dir}")
    by_id: Dict[str, np.ndarray] = {}
    faces = None
    log(f"Loading {len(paths)} neutral meshes")
    for i, path in enumerate(paths):
        vertices, f = load_mesh_npz(path)
        if vertices.shape[0] == len(roi_global):
            # Current project cache stores neutral meshes already cropped to ROI.
            roi_vertices = vertices
        elif vertices.shape[0] > int(np.max(roi_global)):
            # Raw full-topology mesh: crop with frozen ROI global indices.
            roi_vertices = vertices[roi_global]
        else:
            raise IndexError(
                f"{path} has {vertices.shape[0]} vertices, which is neither ROI-sized "
                f"({len(roi_global)}) nor large enough for max ROI index {int(np.max(roi_global))}."
            )
        by_id[mesh_id_from_path(path)] = roi_vertices.astype(np.float32)
        if faces is None and f is not None:
            faces = f
        if (i + 1) % 100 == 0:
            log(f"  loaded {i + 1}/{len(paths)} meshes")
    return by_id, faces


def extract_pairs_from_obj(obj: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        pair_keys = [
            ("source", "target"),
            ("source_id", "target_id"),
            ("src", "dst"),
            ("src_id", "tgt_id"),
            ("a", "b"),
        ]
        for a, b in pair_keys:
            if a in obj and b in obj:
                pairs.append((parse_id(obj[a]), parse_id(obj[b])))
                return pairs
        for key in ["pairs", "train_pairs", "val_pairs", "validation_pairs", "test_pairs", "items"]:
            if key in obj:
                pairs.extend(extract_pairs_from_obj(obj[key]))
        if not pairs:
            for v in obj.values():
                if isinstance(v, (dict, list, tuple)):
                    pairs.extend(extract_pairs_from_obj(v))
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 2 and not isinstance(obj[0], (dict, list, tuple)):
            pairs.append((parse_id(obj[0]), parse_id(obj[1])))
        else:
            for item in obj:
                pairs.extend(extract_pairs_from_obj(item))
    return pairs


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = extract_pairs_from_obj(read_json(path))
    uniq = []
    seen = set()
    for a, b in pairs:
        if a == b:
            continue
        key = (a, b)
        if key not in seen:
            uniq.append(key)
            seen.add(key)
    if not uniq:
        raise ValueError(f"No source-target pairs found in {path}")
    return uniq


def extract_named_pairs(obj: Any, keys: Sequence[str]) -> List[Tuple[str, str]]:
    if not isinstance(obj, dict):
        return []
    for key in keys:
        if key in obj:
            return dedupe_pairs(extract_pairs_from_obj(obj[key]))
    return []


def dedupe_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    uniq: List[Tuple[str, str]] = []
    seen = set()
    for a, b in pairs:
        if a == b:
            continue
        key = (parse_id(a), parse_id(b))
        if key not in seen:
            uniq.append(key)
            seen.add(key)
    return uniq


def identity_sort_key(sid: str) -> Tuple[int, str]:
    text = parse_id(sid)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def ordered_identity_pairs(ids: Sequence[str], max_pairs: Optional[int], seed: int) -> List[Tuple[str, str]]:
    ids_sorted = sorted([parse_id(x) for x in ids], key=identity_sort_key)
    pairs = [(a, b) for a in ids_sorted for b in ids_sorted if a != b]
    if max_pairs is not None and int(max_pairs) > 0 and len(pairs) > int(max_pairs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(i)] for i in idx]
    return pairs


def load_manifest_identity_groups(root: Path) -> Dict[str, List[str]]:
    manifest_path = root / "data" / "manifest.json"
    split_manifest_path = root / "splits" / "facescape_847" / "split_manifest.json"
    groups = {"train_pool": [], "validation": [], "test": []}
    if split_manifest_path.exists():
        obj = read_json(split_manifest_path)
        groups["train_pool"] = [parse_id(x) for x in obj.get("train_pool_ids", [])]
        groups["validation"] = [parse_id(x) for x in obj.get("val_ids", [])]
        groups["test"] = [parse_id(x) for x in obj.get("test_ids", [])]
    elif manifest_path.exists():
        obj = read_json(manifest_path)
        for row in obj.get("rows", []):
            sid = parse_id(row.get("subject_id"))
            split = str(row.get("split", ""))
            if split == "train pool" or split == "clean-prior train":
                groups["train_pool"].append(sid)
            elif split == "clean-prior validation":
                groups["validation"].append(sid)
            elif split == "main test":
                groups["test"].append(sid)
    else:
        raise FileNotFoundError("No data/manifest.json or splits/facescape_847/split_manifest.json found")
    for key, ids in groups.items():
        groups[key] = sorted(set(ids), key=identity_sort_key)
    if not groups["train_pool"] or not groups["test"]:
        raise ValueError(f"Manifest identity split is incomplete: { {k: len(v) for k, v in groups.items()} }")
    return groups


def candidate_pair_paths(root: Path, chain: int, scale: int, split: str = "primary") -> List[Path]:
    pair_dir = root / "splits" / "facescape_847" / "pairs"
    names = [
        f"{split}_chain_{chain}_scale{scale}.json",
        f"{split}_chain{chain}_scale{scale}.json",
    ]
    if scale == 30:
        names.append(f"{split}_anchor_scale30.json")
    return [pair_dir / n for n in names]


def find_pair_path(root: Path, chain: int, scale: int, split: str = "primary") -> Path:
    for path in candidate_pair_paths(root, chain, scale, split):
        if path.exists():
            return path
    raise FileNotFoundError(f"No pair JSON for split={split} chain={chain} scale={scale}")


def split_pairs(pairs: Sequence[Tuple[str, str]], val_fraction: float, seed: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_fraction)))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


def pair_identities(pairs: Sequence[Tuple[str, str]]) -> set:
    ids = set()
    for a, b in pairs:
        ids.add(parse_id(a))
        ids.add(parse_id(b))
    return ids


def identity_disjoint_split_pairs(
    pairs: Sequence[Tuple[str, str]],
    val_fraction: float,
    seed: int,
    allowed_ids: Optional[set] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Any]]:
    ids = sorted(pair_identities(pairs) if allowed_ids is None else (pair_identities(pairs) & set(allowed_ids)))
    if len(ids) < 4:
        raise ValueError(f"Need at least 4 identities for identity_disjoint split; got {len(ids)}")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(ids) * val_fraction)))
    test_ids = {ids[i] for i in idx[:n_test]}
    train_ids = {ids[i] for i in idx[n_test:]}
    train = [(a, b) for a, b in pairs if a in train_ids and b in train_ids]
    test = [(a, b) for a, b in pairs if a in test_ids and b in test_ids]
    dropped = len(pairs) - len(train) - len(test)
    if not train or not test:
        raise ValueError(
            "identity_disjoint split produced an empty train/test pair set. "
            f"train_pairs={len(train)} test_pairs={len(test)} dropped_cross_pairs={dropped}"
        )
    audit = {
        "split_mode": "identity_disjoint",
        "train_identity_count": len(train_ids),
        "test_identity_count": len(test_ids),
        "identity_overlap_count": len(train_ids & test_ids),
        "train_identities": sorted(train_ids),
        "test_identities": sorted(test_ids),
        "n_train_pairs": len(train),
        "n_test_pairs": len(test),
        "n_dropped_cross_split_pairs": dropped,
    }
    return train, test, audit


def load_pair_splits(
    path: Path,
    mode: str,
    val_fraction: float,
    seed: int,
    available_ids: Optional[set] = None,
    root: Optional[Path] = None,
    max_manifest_test_pairs: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Any]]:
    data = read_json(path)
    if mode == "manifest_identity":
        if root is None:
            raise ValueError("root is required for --eval-split-mode manifest_identity")
        groups = load_manifest_identity_groups(root)
        train_allowed = set(groups["train_pool"])
        test_allowed = set(groups["test"])
        train = [(a, b) for a, b in dedupe_pairs(extract_pairs_from_obj(data)) if a in train_allowed and b in train_allowed]
        if not train:
            train = ordered_identity_pairs(groups["train_pool"], None, seed)
        test = ordered_identity_pairs(groups["test"], max_manifest_test_pairs, seed + 1)
        val_pairs = ordered_identity_pairs(groups["validation"], max_manifest_test_pairs, seed + 2) if groups["validation"] else []
        audit = {
            "split_mode": "manifest_identity",
            "source": "data/manifest.json or splits/facescape_847/split_manifest.json",
            "train_identity_count": len(groups["train_pool"]),
            "validation_identity_count": len(groups["validation"]),
            "test_identity_count": len(groups["test"]),
            "identity_overlap_count": 0,
            "train_identities": groups["train_pool"],
            "validation_identities": groups["validation"],
            "test_identities": groups["test"],
            "n_validation_pairs_not_used_for_final_test": len(val_pairs),
            "max_manifest_test_pairs": max_manifest_test_pairs,
            "note": "Matches the legacy CVAE/train.py identity-disjoint split: training identities come from train pool; test pairs are constructed only within main test identities.",
        }
    elif mode == "use_json_splits":
        train = extract_named_pairs(data, ["train_pairs", "train", "fit_pairs"])
        test = extract_named_pairs(data, ["test_pairs", "test"])
        val = extract_named_pairs(data, ["val_pairs", "validation_pairs", "validation"])
        if not train or not test:
            raise ValueError(f"{path} does not contain explicit train_pairs and test_pairs for --eval-split-mode use_json_splits")
        if val:
            train = dedupe_pairs(list(train) + list(val))
        audit = {
            "split_mode": "use_json_splits",
            "json_train_pairs": len(train),
            "json_test_pairs": len(test),
            "json_val_pairs_merged_into_train": len(val),
        }
    else:
        pairs = load_pairs(path)
        if mode == "identity_disjoint":
            train, test, audit = identity_disjoint_split_pairs(pairs, val_fraction, seed, available_ids)
        elif mode == "pair_level":
            train, test = split_pairs(pairs, val_fraction, seed)
            train_ids = pair_identities(train)
            test_ids = pair_identities(test)
            audit = {
                "split_mode": "pair_level",
                "identity_overlap_count": len(train_ids & test_ids),
                "train_identity_count": len(train_ids),
                "test_identity_count": len(test_ids),
                "n_train_pairs": len(train),
                "n_test_pairs": len(test),
                "warning": "Pair-level split can leak identities across train/test; do not claim identity-disjoint generalisation.",
            }
        else:
            raise ValueError(f"Unknown eval split mode: {mode}")
    if available_ids is not None:
        train = [(a, b) for a, b in train if a in available_ids and b in available_ids]
        test = [(a, b) for a, b in test if a in available_ids and b in available_ids]
    train_ids = pair_identities(train)
    test_ids = pair_identities(test)
    audit.update({
        "pair_json": str(path),
        "eval_split_mode": mode,
        "n_train_pairs": len(train),
        "n_test_pairs": len(test),
        "train_identity_count": len(train_ids),
        "test_identity_count": len(test_ids),
        "identity_overlap_count": len(train_ids & test_ids),
        "identity_overlap": sorted(train_ids & test_ids),
    })
    if not train or not test:
        raise ValueError(f"Resolved empty train/test split for {path}: train={len(train)} test={len(test)}")
    return train, test, audit


def resolve_eval_pairs(args: argparse.Namespace, bundle: DatasetBundle) -> Tuple[Path, List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Any]]:
    path = find_pair_path(args.root, args.chain, args.scale, args.split)
    train_pairs, test_pairs, audit = load_pair_splits(
        path,
        args.eval_split_mode,
        args.val_fraction,
        args.seed,
        available_ids=set(bundle.meshes.keys()),
        root=args.root,
        max_manifest_test_pairs=args.max_manifest_test_pairs if args.max_manifest_test_pairs > 0 else None,
    )
    return path, train_pairs, test_pairs, audit


@dataclass
class DatasetBundle:
    root: Path
    roi_global: np.ndarray
    landmarks: np.ndarray
    free_idx: np.ndarray
    subunits: Dict[str, np.ndarray]
    meshes: Dict[str, np.ndarray]
    faces_global: Optional[np.ndarray]
    roi_faces: Optional[np.ndarray]
    roi_xyz_mean: np.ndarray
    roi_xyz_std: np.ndarray


def build_roi_faces(faces_global: Optional[np.ndarray], roi_global: np.ndarray) -> Optional[np.ndarray]:
    if faces_global is None:
        return None
    n_roi = len(roi_global)
    if faces_global.size and int(np.max(faces_global)) < n_roi:
        # Faces are already local to the ROI-cropped mesh.
        return faces_global.astype(np.int64)
    mapping = {int(g): i for i, g in enumerate(roi_global.tolist())}
    keep = []
    for tri in faces_global:
        a, b, c = [int(x) for x in tri]
        if a in mapping and b in mapping and c in mapping:
            keep.append([mapping[a], mapping[b], mapping[c]])
    if not keep:
        return None
    return np.asarray(keep, dtype=np.int64)


def load_bundle(root: Path, landmarks: Any = "auto") -> DatasetBundle:
    roi_global = load_roi_indices(root)
    meshes, faces_global = load_identity_table(root, roi_global)
    first = next(iter(meshes.values()))
    n_roi = first.shape[0]
    if landmarks == "auto" or landmarks is None:
        landmark_arr = auto_landmarks_from_roi(root, roi_global)
    else:
        landmark_arr = np.asarray(landmarks, dtype=np.int64)
    if int(landmark_arr.max()) >= n_roi:
        global_to_local = {int(g): i for i, g in enumerate(roi_global.tolist())}
        landmark_arr = np.asarray([global_to_local[int(x)] for x in landmark_arr], dtype=np.int64)
    free_idx = np.asarray([i for i in range(n_roi) if i not in set(landmark_arr.tolist())], dtype=np.int64)
    subunits = load_subunits(root, roi_global, n_roi)
    roi_faces = build_roi_faces(faces_global, roi_global)
    stack = np.stack(list(meshes.values()), axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0) + 1e-6
    log(f"ROI vertices={n_roi}; landmarks={len(landmark_arr)}; free={len(free_idx)}; subunits={list(subunits)}")
    return DatasetBundle(root, roi_global, landmark_arr, free_idx, subunits, meshes, faces_global, roi_faces, mean, std)


def make_pair_arrays(
    bundle: DatasetBundle,
    pairs: Sequence[Tuple[str, str]],
    source_pca: Optional[Any] = None,
    source_pca_dim: int = 16,
    fit_pca: bool = False,
) -> Dict[str, Any]:
    valid = [(a, b) for a, b in pairs if a in bundle.meshes and b in bundle.meshes]
    if not valid:
        available = sorted(list(bundle.meshes.keys()))[:10]
        sample = list(pairs[:10])
        raise ValueError(f"No pair ids match loaded meshes; sample_pairs={sample}; sample_mesh_ids={available}")
    if len(valid) < len(pairs):
        log(f"WARNING: dropped {len(pairs) - len(valid)}/{len(pairs)} pairs because ids were missing from meshes")
    src = np.stack([bundle.meshes[a] for a, _ in valid], axis=0)
    tgt = np.stack([bundle.meshes[b] for _, b in valid], axis=0)
    delta = tgt - src
    controls = delta[:, bundle.landmarks, :].reshape(len(valid), -1)
    src_flat = src.reshape(len(valid), -1)
    if source_pca_dim > 0:
        if fit_pca:
            source_pca = sk_decomp.PCA(n_components=min(source_pca_dim, len(valid), src_flat.shape[1]), random_state=0)
            source_scores = source_pca.fit_transform(src_flat)
        else:
            if source_pca is None:
                raise ValueError("source_pca is required when fit_pca=False")
            source_scores = source_pca.transform(src_flat)
    else:
        source_pca = None
        source_scores = np.zeros((len(valid), 0), dtype=np.float32)
    cond = np.concatenate([controls, source_scores], axis=1).astype(np.float32)
    return {
        "pairs": valid,
        "source": src.astype(np.float32),
        "target": tgt.astype(np.float32),
        "delta": delta.astype(np.float32),
        "controls": controls.astype(np.float32),
        "source_flat": src_flat.astype(np.float32),
        "source_pca": source_pca,
        "source_scores": source_scores.astype(np.float32),
        "cond": cond,
    }


def _subunit_edge_stats(vertices: np.ndarray, edges: Optional[np.ndarray], idx: np.ndarray) -> np.ndarray:
    if edges is None or len(idx) == 0:
        return np.zeros((vertices.shape[0], 4), dtype=np.float32)
    mask = np.zeros(vertices.shape[1], dtype=bool)
    mask[idx] = True
    e = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    if len(e) == 0:
        return np.zeros((vertices.shape[0], 4), dtype=np.float32)
    lengths = np.linalg.norm(vertices[:, e[:, 0], :] - vertices[:, e[:, 1], :], axis=-1)
    return np.stack([
        lengths.mean(axis=1),
        lengths.std(axis=1),
        np.percentile(lengths, 10, axis=1),
        np.percentile(lengths, 90, axis=1),
    ], axis=1).astype(np.float32)


def _vertex_laplacian_norms(vertices: np.ndarray, edges: Optional[np.ndarray]) -> np.ndarray:
    if edges is None:
        return np.zeros(vertices.shape[:2], dtype=np.float32)
    n = vertices.shape[1]
    acc = np.zeros_like(vertices, dtype=np.float32)
    deg = np.zeros(n, dtype=np.float32)
    for a, b in edges:
        acc[:, a, :] += vertices[:, b, :] - vertices[:, a, :]
        acc[:, b, :] += vertices[:, a, :] - vertices[:, b, :]
        deg[a] += 1
        deg[b] += 1
    acc /= np.maximum(deg[None, :, None], 1.0)
    return np.linalg.norm(acc, axis=-1).astype(np.float32)


def _face_vertex_normals(vertices: np.ndarray, faces: Optional[np.ndarray]) -> np.ndarray:
    if faces is None or len(faces) == 0:
        return np.zeros_like(vertices, dtype=np.float32)
    normals = np.zeros_like(vertices, dtype=np.float32)
    tri = vertices[:, faces, :]
    fn = np.cross(tri[:, :, 1, :] - tri[:, :, 0, :], tri[:, :, 2, :] - tri[:, :, 0, :])
    fn /= np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-12
    for corner in range(3):
        np.add.at(normals, (slice(None), faces[:, corner], slice(None)), fn)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    return normals.astype(np.float32)


def _subunit_area_stats(vertices: np.ndarray, faces: Optional[np.ndarray], idx: np.ndarray) -> np.ndarray:
    if faces is None or len(idx) == 0:
        return np.zeros((vertices.shape[0], 4), dtype=np.float32)
    mask = np.zeros(vertices.shape[1], dtype=bool)
    mask[idx] = True
    f = faces[mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]]
    if len(f) == 0:
        return np.zeros((vertices.shape[0], 4), dtype=np.float32)
    tri = vertices[:, f, :]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, :, 1, :] - tri[:, :, 0, :], tri[:, :, 2, :] - tri[:, :, 0, :]), axis=-1)
    return np.stack([
        area.sum(axis=1),
        area.mean(axis=1),
        area.std(axis=1),
        np.percentile(area, 90, axis=1),
    ], axis=1).astype(np.float32)


def _subunit_shape_pca_stats(points: np.ndarray) -> np.ndarray:
    rows = []
    for pts in points:
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(pts) - 1)
        eig = np.linalg.eigvalsh(cov).astype(np.float32)
        eig = np.sort(np.maximum(eig, 0.0))[::-1]
        scale = float(np.sum(eig) + 1e-12)
        ratios = eig / scale
        bbox = pts.max(axis=0) - pts.min(axis=0)
        sorted_bbox = np.sort(np.maximum(bbox, 1e-12))[::-1]
        rows.append(np.concatenate([
            eig,
            ratios,
            np.asarray([
                sorted_bbox[0] / sorted_bbox[1],
                sorted_bbox[1] / sorted_bbox[2],
                sorted_bbox[0] / sorted_bbox[2],
            ], dtype=np.float32),
        ]))
    return np.asarray(rows, dtype=np.float32)


def append_legal_rbsr_features(bundle: DatasetBundle, data: Dict[str, Any], mode: str) -> Dict[str, Any]:
    alias = {
        "base": "f0",
        "deformation": "f1",
        "local_shape": "f2",
        "local_shape_deformation": "f3",
    }
    mode = alias.get(mode, mode)
    if mode == "f0":
        return data
    if mode not in {"f1", "f2", "f3"}:
        raise ValueError(f"Unknown RBSR feature mode: {mode}")
    source = data["source"]
    controls = data["controls"].reshape(len(source), -1, 3)
    src_norm = (source - bundle.roi_xyz_mean[None]) / bundle.roi_xyz_std[None]
    edges = build_edges(bundle.roi_faces)
    lap_norm = _vertex_laplacian_norms(src_norm, edges)
    normals = _face_vertex_normals(src_norm, bundle.roi_faces)
    feats: List[np.ndarray] = []

    ctrl_norm = np.linalg.norm(controls, axis=-1)
    # F1: legal deformation-scale features from the nine observed handles.
    left_motion = ctrl_norm[:, [3, 4, 5]].mean(axis=1, keepdims=True) if ctrl_norm.shape[1] >= 6 else ctrl_norm.mean(axis=1, keepdims=True)
    right_motion = ctrl_norm[:, [-3, -2, -1]].mean(axis=1, keepdims=True) if ctrl_norm.shape[1] >= 6 else ctrl_norm.mean(axis=1, keepdims=True)
    feats.extend([
        ctrl_norm,
        ctrl_norm.mean(axis=1, keepdims=True),
        ctrl_norm.max(axis=1, keepdims=True),
        ctrl_norm.std(axis=1, keepdims=True),
        np.linalg.norm(controls[:, :1, :] - controls[:, -1:, :], axis=-1),
        left_motion,
        right_motion,
        np.abs(left_motion - right_motion),
        np.linalg.norm(controls.mean(axis=1), axis=1, keepdims=True),
    ])
    if ctrl_norm.shape[1] >= 9:
        feats.extend([
            ctrl_norm[:, 0:3].mean(axis=1, keepdims=True),
            ctrl_norm[:, 3:6].mean(axis=1, keepdims=True),
            ctrl_norm[:, 6:9].mean(axis=1, keepdims=True),
        ])

    if mode in {"f2", "f3"}:
        # F2: source local geometry features per anatomical subunit.
        for name in RBSR_REGIONS:
            idx = bundle.subunits.get(name)
            if idx is None or len(idx) == 0:
                continue
            pts = src_norm[:, idx, :]
            centroid = pts.mean(axis=1)
            spread = pts.std(axis=1)
            bbox = pts.max(axis=1) - pts.min(axis=1)
            curvature = lap_norm[:, idx]
            nrm = normals[:, idx, :]
            feats.extend([
                centroid,
                spread,
                bbox,
                _subunit_shape_pca_stats(pts),
                curvature.mean(axis=1, keepdims=True),
                curvature.std(axis=1, keepdims=True),
                np.percentile(curvature, 90, axis=1, keepdims=True),
                nrm.mean(axis=1),
                nrm.std(axis=1),
                _subunit_edge_stats(src_norm, edges, idx),
                _subunit_area_stats(src_norm, bundle.roi_faces, idx),
            ])

    if mode == "f3":
        # F3: landmark-relative source geometry and control relation features.
        lm_src = src_norm[:, bundle.landmarks, :]
        lm_delta = controls
        pairwise_src = []
        pairwise_delta = []
        for i in range(len(bundle.landmarks)):
            for j in range(i + 1, len(bundle.landmarks)):
                pairwise_src.append(np.linalg.norm(lm_src[:, i, :] - lm_src[:, j, :], axis=1, keepdims=True))
                pairwise_delta.append(np.linalg.norm(lm_delta[:, i, :] - lm_delta[:, j, :], axis=1, keepdims=True))
        if pairwise_src:
            feats.append(np.concatenate(pairwise_src, axis=1))
            feats.append(np.concatenate(pairwise_delta, axis=1))
        for name in RBSR_REGIONS:
            idx = bundle.subunits.get(name)
            if idx is None or len(idx) == 0:
                continue
            pts = src_norm[:, idx, :]
            dists = np.linalg.norm(pts[:, :, None, :] - lm_src[:, None, :, :], axis=-1)
            feats.extend([
                dists.mean(axis=1),
                dists.min(axis=1),
                dists.std(axis=1),
            ])

    left = bundle.subunits.get("alar_left")
    right = bundle.subunits.get("alar_right")
    if mode in {"f2", "f3"} and left is not None and right is not None and len(left) and len(right):
        left_cent = src_norm[:, left, :].mean(axis=1)
        right_cent = src_norm[:, right, :].mean(axis=1)
        feats.append(np.linalg.norm(left_cent - right_cent, axis=1, keepdims=True))
        feats.append(left_cent - right_cent)

    extra = np.concatenate([f.astype(np.float32).reshape(len(source), -1) for f in feats], axis=1)
    out = dict(data)
    out["cond"] = np.concatenate([data["cond"], extra], axis=1).astype(np.float32)
    out["rbsr_feature_mode"] = mode
    out["rbsr_extra_feature_dim"] = int(extra.shape[1])
    return out


def strict_overwrite(pred_delta: np.ndarray, true_delta: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    out = pred_delta.copy()
    out[:, landmarks, :] = true_delta[:, landmarks, :]
    return out


def rmse(pred: np.ndarray, true: np.ndarray, idx: Optional[np.ndarray] = None) -> float:
    if idx is not None:
        pred = pred[:, idx, :]
        true = true[:, idx, :]
    return float(np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=-1))))


def per_vertex_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=-1), axis=0))


def subunit_metrics(pred: np.ndarray, true: np.ndarray, subunits: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {f"rmse_{k}": rmse(pred, true, v) for k, v in subunits.items() if len(v)}


def build_edges(faces: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if faces is None:
        return None
    edges = set()
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        for u, v in [(a, b), (b, c), (c, a)]:
            if u != v:
                edges.add(tuple(sorted((u, v))))
    if not edges:
        return None
    return np.asarray(sorted(edges), dtype=np.int64)


def uniform_laplacian_matrix(n_vertices: int, faces: Optional[np.ndarray]) -> Any:
    edges = build_edges(faces)
    if edges is None:
        raise ValueError("ROI faces are required for Laplacian/ARAP baselines")
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float64)
    adj = sp_sparse.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    return sp_sparse.diags(deg) - adj


def selector_matrix(indices: np.ndarray, n_vertices: int) -> Any:
    indices = np.asarray(indices, dtype=np.int64)
    rows = np.arange(len(indices), dtype=np.int64)
    data = np.ones(len(indices), dtype=np.float64)
    return sp_sparse.coo_matrix((data, (rows, indices)), shape=(len(indices), n_vertices)).tocsr()


def solve_linear_handle_baseline(
    controls: np.ndarray,
    landmarks: np.ndarray,
    n_vertices: int,
    smooth_operator: Any,
    handle_weight: float,
    ridge: float,
) -> np.ndarray:
    p = selector_matrix(landmarks, n_vertices)
    a = smooth_operator.T @ smooth_operator + float(handle_weight) * (p.T @ p) + float(ridge) * sp_sparse.eye(n_vertices, format="csr")
    solve = sp_linalg.factorized(a.tocsc())
    pred = np.zeros((len(controls), n_vertices, 3), dtype=np.float64)
    rhs_handle = float(handle_weight) * p.T
    for i, ctrl in enumerate(controls):
        rhs = rhs_handle @ ctrl
        for d in range(3):
            pred[i, :, d] = solve(rhs[:, d])
    return pred.astype(np.float32)


def arap_predict_vectorised_strict(
    sources: np.ndarray,
    controls: np.ndarray,
    landmarks: np.ndarray,
    faces: Optional[np.ndarray],
    laplacian: Any,
    init_delta: np.ndarray,
    handle_weight: float,
    ridge: float,
    n_iter: int,
) -> np.ndarray:
    if faces is None:
        raise ValueError("ROI faces are required for ARAP baseline")
    edges = build_edges(faces)
    if edges is None:
        raise ValueError("Could not build ROI edges for ARAP baseline")
    n_pairs, n_vertices, _ = sources.shape
    e0, e1 = edges[:, 0], edges[:, 1]
    pmat = selector_matrix(landmarks, n_vertices)
    a = laplacian + float(handle_weight) * (pmat.T @ pmat) + float(ridge) * sp_sparse.eye(n_vertices, format="csr")
    solve = sp_linalg.factorized(a.tocsc())
    out = np.zeros((n_pairs, n_vertices, 3), dtype=np.float32)
    for k in range(n_pairs):
        src = sources[k].astype(np.float64)
        target_handles = src[landmarks] + controls[k]
        q = src + init_delta[k].astype(np.float64)
        d_src = src[e0] - src[e1]
        for _ in range(int(n_iter)):
            d_q = q[e0] - q[e1]
            outer = d_q[:, :, None] * d_src[:, None, :]
            cov = np.zeros((n_vertices, 3, 3), dtype=np.float64)
            np.add.at(cov, e0, outer)
            np.add.at(cov, e1, outer)
            u, _, vt = np.linalg.svd(cov)
            r = np.matmul(u, vt)
            flip = np.linalg.det(r) < 0.0
            if np.any(flip):
                u[flip, :, -1] *= -1.0
                r[flip] = np.matmul(u[flip], vt[flip])
            term = 0.5 * np.matmul((r[e0] + r[e1]), d_src[:, :, None])[:, :, 0]
            b = np.zeros((n_vertices, 3), dtype=np.float64)
            np.add.at(b, e0, term)
            np.add.at(b, e1, -term)
            b += float(handle_weight) * (pmat.T @ target_handles)
            for d in range(3):
                q[:, d] = solve(b[:, d])
        out[k] = (q - src).astype(np.float32)
        if (k + 1) % 50 == 0:
            log(f"  ARAP completed {k + 1}/{n_pairs}")
    return out


def edge_strain(pred_shape: np.ndarray, true_shape: np.ndarray, edges: Optional[np.ndarray]) -> float:
    if edges is None:
        return float("nan")
    p = np.linalg.norm(pred_shape[:, edges[:, 0], :] - pred_shape[:, edges[:, 1], :], axis=-1)
    t = np.linalg.norm(true_shape[:, edges[:, 0], :] - true_shape[:, edges[:, 1], :], axis=-1)
    return float(np.mean(np.abs(p - t) / (t + 1e-6)))


def normal_consistency(pred_shape: np.ndarray, true_shape: np.ndarray, faces: Optional[np.ndarray]) -> float:
    if faces is None:
        return float("nan")
    tri = faces
    p0, p1, p2 = pred_shape[:, tri[:, 0], :], pred_shape[:, tri[:, 1], :], pred_shape[:, tri[:, 2], :]
    t0, t1, t2 = true_shape[:, tri[:, 0], :], true_shape[:, tri[:, 1], :], true_shape[:, tri[:, 2], :]
    pn = np.cross(p1 - p0, p2 - p0)
    tn = np.cross(t1 - t0, t2 - t0)
    pn = pn / (np.linalg.norm(pn, axis=-1, keepdims=True) + 1e-9)
    tn = tn / (np.linalg.norm(tn, axis=-1, keepdims=True) + 1e-9)
    cos = np.sum(pn * tn, axis=-1)
    return float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))


def evaluate_prediction(
    bundle: DatasetBundle,
    data: Dict[str, Any],
    pred_delta: np.ndarray,
    name: str,
) -> Dict[str, float]:
    true_delta = data["delta"]
    pred_delta = strict_overwrite(pred_delta, true_delta, bundle.landmarks)
    pred_shape = data["source"] + pred_delta
    true_shape = data["target"]
    edges = build_edges(bundle.roi_faces)
    row = {
        "method": name,
        "n_pairs": len(true_delta),
        "strict_free_rmse": rmse(pred_delta, true_delta, bundle.free_idx),
        "full_roi_rmse_with_hard_landmarks": rmse(pred_delta, true_delta, None),
        "landmark_rmse_after_overwrite": rmse(pred_delta, true_delta, bundle.landmarks),
        "edge_strain": edge_strain(pred_shape, true_shape, edges),
        "normal_consistency": normal_consistency(pred_shape, true_shape, bundle.roi_faces),
    }
    row.update(subunit_metrics(pred_delta, true_delta, bundle.subunits))
    return row


def per_pair_metric_rows(
    bundle: DatasetBundle,
    data: Dict[str, Any],
    pred_delta: np.ndarray,
    method: str,
) -> List[Dict[str, Any]]:
    true_delta = data["delta"]
    pred_delta = strict_overwrite(pred_delta, true_delta, bundle.landmarks)
    pred_shape = data["source"] + pred_delta
    true_shape = data["target"]
    edges = build_edges(bundle.roi_faces)
    rows: List[Dict[str, Any]] = []
    for i, (src_id, tgt_id) in enumerate(data["pairs"]):
        row: Dict[str, Any] = {
            "pair_index": i,
            "source_id": src_id,
            "target_id": tgt_id,
            "method": method,
            "strict_free_rmse": rmse(pred_delta[i:i + 1], true_delta[i:i + 1], bundle.free_idx),
            "full_roi_rmse_with_hard_landmarks": rmse(pred_delta[i:i + 1], true_delta[i:i + 1], None),
            "landmark_rmse_after_overwrite": rmse(pred_delta[i:i + 1], true_delta[i:i + 1], bundle.landmarks),
            "edge_strain": edge_strain(pred_shape[i:i + 1], true_shape[i:i + 1], edges),
            "normal_consistency": normal_consistency(pred_shape[i:i + 1], true_shape[i:i + 1], bundle.roi_faces),
        }
        for name, idx in bundle.subunits.items():
            row[f"rmse_{name}"] = rmse(pred_delta[i:i + 1], true_delta[i:i + 1], idx)
        rows.append(row)
    return rows


def paired_bootstrap_and_permutation(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    metric: str,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> Dict[str, Any]:
    a = np.asarray([float(r[metric]) for r in rows_a], dtype=np.float64)
    b = np.asarray([float(r[metric]) for r in rows_b], dtype=np.float64)
    delta = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=np.float64)
    n = len(delta)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(delta[idx]))
    obs = float(np.mean(delta))
    # Sign-flip paired permutation for mean delta under symmetric null.
    count = 0
    for _ in range(int(n_perm)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=n)
        stat = abs(float(np.mean(delta * signs)))
        if stat >= abs(obs):
            count += 1
    return {
        "metric": metric,
        "n_pairs": int(n),
        "mean_a_minus_b": obs,
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "permutation_p_two_sided": float((count + 1) / (int(n_perm) + 1)),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
    }


@dataclass
class RidgeModel:
    x_scaler: Any
    y_scaler: Any
    model: Any
    source_pca: Any
    source_pca_dim: int


def train_ridge(bundle: DatasetBundle, train_pairs: Sequence[Tuple[str, str]], alpha: float, source_pca_dim: int) -> Tuple[RidgeModel, Dict[str, Any]]:
    train = make_pair_arrays(bundle, train_pairs, source_pca_dim=source_pca_dim, fit_pca=True)
    x = train["cond"]
    y = train["delta"].reshape(len(train["pairs"]), -1)
    xs = sk_pre.StandardScaler().fit(x)
    ys = sk_pre.StandardScaler().fit(y)
    model = sk_linear.Ridge(alpha=alpha)
    model.fit(xs.transform(x), ys.transform(y))
    return RidgeModel(xs, ys, model, train["source_pca"], source_pca_dim), train


def predict_ridge(bundle: DatasetBundle, ridge: RidgeModel, pairs: Sequence[Tuple[str, str]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = make_pair_arrays(bundle, pairs, source_pca=ridge.source_pca, source_pca_dim=ridge.source_pca_dim, fit_pca=False)
    pred = ridge.y_scaler.inverse_transform(ridge.model.predict(ridge.x_scaler.transform(data["cond"])))
    pred = pred.reshape(len(data["pairs"]), -1, 3).astype(np.float32)
    pred = strict_overwrite(pred, data["delta"], bundle.landmarks)
    return pred, data


def train_predict_ridge_augmented(
    bundle: DatasetBundle,
    train_pairs: Sequence[Tuple[str, str]],
    test_pairs: Sequence[Tuple[str, str]],
    alpha: float,
    source_pca_dim: int,
    feature_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    train = make_pair_arrays(bundle, train_pairs, source_pca_dim=source_pca_dim, fit_pca=True)
    test = make_pair_arrays(bundle, test_pairs, source_pca=train["source_pca"], source_pca_dim=source_pca_dim, fit_pca=False)
    train_aug = append_legal_rbsr_features(bundle, train, feature_mode)
    test_aug = append_legal_rbsr_features(bundle, test, feature_mode)
    xs = sk_pre.StandardScaler().fit(train_aug["cond"])
    y = train["delta"].reshape(len(train["pairs"]), -1)
    ys = sk_pre.StandardScaler().fit(y)
    model = sk_linear.Ridge(alpha=alpha).fit(xs.transform(train_aug["cond"]), ys.transform(y))
    pred = ys.inverse_transform(model.predict(xs.transform(test_aug["cond"]))).reshape(len(test["pairs"]), -1, 3).astype(np.float32)
    pred = strict_overwrite(pred, test["delta"], bundle.landmarks)
    meta = {
        "feature_mode": feature_mode,
        "base_condition_dim": int(train["cond"].shape[1]),
        "augmented_condition_dim": int(train_aug["cond"].shape[1]),
        "extra_feature_dim": int(train_aug.get("rbsr_extra_feature_dim", 0)),
    }
    return pred, test, meta


def residual_heatmap(bundle: DatasetBundle, pred_delta: np.ndarray, data: Dict[str, Any], out: Path) -> None:
    residual = pred_delta - data["delta"]
    pv = per_vertex_rmse(pred_delta, data["delta"])
    rows = []
    for i, value in enumerate(pv.tolist()):
        label = "free" if i in set(bundle.free_idx.tolist()) else "landmark"
        unit = "unassigned"
        for name, idx in bundle.subunits.items():
            if i in set(idx.tolist()):
                unit = name
                break
        rows.append({"vertex": i, "global_vertex": int(bundle.roi_global[i]), "rmse": value, "kind": label, "subunit": unit})
    atomic_csv(out / "per_vertex_residual_rmse.csv", rows)
    top = sorted(rows, key=lambda r: r["rmse"], reverse=True)[:100]
    atomic_csv(out / "top100_residual_vertices.csv", top)

    xyz = data["source"].mean(axis=0)
    for view_name, a, b in [("xy_front", 0, 1), ("xz_profile", 0, 2), ("yz_profile", 1, 2)]:
        plt.figure(figsize=(7.2, 6.2))
        sc = plt.scatter(xyz[:, a], xyz[:, b], c=pv, s=10, cmap="magma")
        plt.scatter(xyz[bundle.landmarks, a], xyz[bundle.landmarks, b], c="cyan", s=35, edgecolors="black", label="hard landmarks")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(sc, label="Ridge residual RMSE")
        plt.title(f"Ridge residual heatmap ({view_name})")
        plt.legend(loc="best")
        save_fig_atomic(out / f"ridge_residual_heatmap_{view_name}.png")

    sub_rows = []
    for name, idx in bundle.subunits.items():
        if len(idx):
            sub_rows.append({"subunit": name, "mean_vertex_rmse": float(np.mean(pv[idx])), "p95_vertex_rmse": float(np.percentile(pv[idx], 95)), "n_vertices": len(idx)})
    atomic_csv(out / "subunit_residual_summary.csv", sub_rows)
    if sub_rows:
        plt.figure(figsize=(8, 4.5))
        names = [r["subunit"] for r in sub_rows]
        vals = [r["mean_vertex_rmse"] for r in sub_rows]
        plt.bar(names, vals, color="#4c78a8")
        plt.ylabel("Mean per-vertex residual RMSE")
        plt.title("Ridge residual by anatomical subunit")
        plt.xticks(rotation=25, ha="right")
        save_fig_atomic(out / "ridge_residual_by_subunit.png")

    energy = np.sum(residual ** 2, axis=(0, 2))
    order = np.argsort(-energy)
    cum = np.cumsum(energy[order]) / (np.sum(energy) + 1e-12)
    plt.figure(figsize=(7, 4.2))
    plt.plot(np.arange(1, len(cum) + 1), cum, linewidth=2)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axhline(0.9, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Top residual vertices")
    plt.ylabel("Cumulative residual energy")
    plt.title("How concentrated is Ridge residual energy?")
    save_fig_atomic(out / "ridge_residual_energy_concentration.png")

    atomic_json(out / "residual_heatmap_summary.json", {
        "n_pairs": len(data["pairs"]),
        "n_roi_vertices": int(len(bundle.roi_global)),
        "n_free_vertices": int(len(bundle.free_idx)),
        "top_1pct_energy_fraction": float(cum[max(0, int(math.ceil(0.01 * len(cum))) - 1)]),
        "top_5pct_energy_fraction": float(cum[max(0, int(math.ceil(0.05 * len(cum))) - 1)]),
        "top_10pct_energy_fraction": float(cum[max(0, int(math.ceil(0.10 * len(cum))) - 1)]),
    })


@dataclass
class RegionExpert:
    name: str
    vertices: np.ndarray
    pca: Any
    regressor: Any
    x_scaler: Any
    c_scaler: Any
    predictor: str = "ridge"


@dataclass
class AdmissionModel:
    name: str
    vertices: np.ndarray
    x_scaler: Any
    regressor: Any
    improvement_scale: float


@dataclass
class VertexGateModel:
    x_scaler: Any
    regressor: Any
    improvement_scale: float


class TorchCoefficientRegressor:
    def __init__(self, model: Any, device: str):
        self.model = model
        self.device = device

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tx = torch.as_tensor(x.astype(np.float32), device=self.device)
            out = self.model(tx).detach().cpu().numpy()
        return out


def fit_torch_coefficient_regressor(
    x: np.ndarray,
    coeff_scaled: np.ndarray,
    coeff_mean: np.ndarray,
    coeff_scale: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    residual_flat: np.ndarray,
    local_edges: Optional[np.ndarray],
    local_faces: Optional[np.ndarray],
    seed: int,
    epochs: int,
    lr: float,
    residual_weight: float,
    edge_weight: float,
    normal_weight: float,
) -> TorchCoefficientRegressor:
    if torch is None:
        raise RuntimeError("mlp_supervised requires torch")
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_t = torch.as_tensor(x.astype(np.float32), device=device)
    c_t = torch.as_tensor(coeff_scaled.astype(np.float32), device=device)
    c_mean = torch.as_tensor(coeff_mean.astype(np.float32), device=device)
    c_scale = torch.as_tensor(coeff_scale.astype(np.float32), device=device)
    comp = torch.as_tensor(pca_components.astype(np.float32), device=device)
    mean = torch.as_tensor(pca_mean.astype(np.float32), device=device)
    y = torch.as_tensor(residual_flat.astype(np.float32), device=device)
    if local_edges is not None and len(local_edges):
        edge_t = torch.as_tensor(local_edges.astype(np.int64), device=device)
        y_vertex = y.view(y.shape[0], -1, 3)
    else:
        edge_t = None
        y_vertex = None
    if local_faces is not None and len(local_faces):
        face_t = torch.as_tensor(local_faces.astype(np.int64), device=device)
        if y_vertex is None:
            y_vertex = y.view(y.shape[0], -1, 3)
    else:
        face_t = None
    width = 256
    model = nn.Sequential(
        nn.Linear(x.shape[1], width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, coeff_scaled.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred_c = model(x_t)
        coeff_loss = F.mse_loss(pred_c, c_t)
        pred_unscaled = pred_c * c_scale + c_mean
        recon = pred_unscaled @ comp + mean
        recon_loss = F.mse_loss(recon, y)
        loss = coeff_loss + float(residual_weight) * recon_loss
        if edge_t is not None and float(edge_weight) > 0:
            recon_vertex = recon.view(recon.shape[0], -1, 3)
            pred_edge = recon_vertex[:, edge_t[:, 0], :] - recon_vertex[:, edge_t[:, 1], :]
            true_edge = y_vertex[:, edge_t[:, 0], :] - y_vertex[:, edge_t[:, 1], :]
            loss = loss + float(edge_weight) * F.mse_loss(pred_edge, true_edge)
        if face_t is not None and float(normal_weight) > 0:
            recon_vertex = recon.view(recon.shape[0], -1, 3)
            pred_tri = recon_vertex[:, face_t, :]
            true_tri = y_vertex[:, face_t, :]
            pred_n = torch.cross(pred_tri[:, :, 1, :] - pred_tri[:, :, 0, :], pred_tri[:, :, 2, :] - pred_tri[:, :, 0, :], dim=-1)
            true_n = torch.cross(true_tri[:, :, 1, :] - true_tri[:, :, 0, :], true_tri[:, :, 2, :] - true_tri[:, :, 0, :], dim=-1)
            pred_n = F.normalize(pred_n, dim=-1, eps=1e-8)
            true_n = F.normalize(true_n, dim=-1, eps=1e-8)
            normal_loss = torch.mean(1.0 - torch.sum(pred_n * true_n, dim=-1).clamp(-1.0, 1.0))
            loss = loss + float(normal_weight) * normal_loss
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return TorchCoefficientRegressor(model, device)


def fit_region_expert(
    cond: np.ndarray,
    residual: np.ndarray,
    vertices: np.ndarray,
    n_basis: int,
    ridge_alpha: float,
    name: str,
    predictor: str = "ridge",
    local_edges: Optional[np.ndarray] = None,
    local_faces: Optional[np.ndarray] = None,
    mlp_edge_weight: float = 0.0,
    mlp_normal_weight: float = 0.0,
) -> RegionExpert:
    y = residual[:, vertices, :].reshape(len(cond), -1)
    n_comp = min(n_basis, y.shape[0], y.shape[1])
    pca = sk_decomp.PCA(n_components=n_comp, random_state=0).fit(y)
    coeff = pca.transform(y)
    xs = sk_pre.StandardScaler().fit(cond)
    cs = sk_pre.StandardScaler().fit(coeff)
    x = xs.transform(cond)
    c = cs.transform(coeff)
    if predictor == "ridge":
        reg = sk_linear.Ridge(alpha=ridge_alpha)
    elif predictor == "kernel_rbf":
        gamma = 1.0 / max(1, x.shape[1])
        reg = sk_kernel.KernelRidge(alpha=max(1e-6, ridge_alpha * 1e-3), kernel="rbf", gamma=gamma)
    elif predictor == "extra_trees":
        reg = sk_ensemble.ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=0,
            n_jobs=-1,
        )
    elif predictor == "mlp":
        reg = sk_nn.MLPRegressor(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=True,
            random_state=0,
        )
    elif predictor == "mlp_supervised":
        reg = fit_torch_coefficient_regressor(
            x,
            c,
            cs.mean_,
            cs.scale_,
            pca.components_,
            pca.mean_,
            y,
            local_edges,
            local_faces,
            seed=0,
            epochs=600,
            lr=1e-3,
            residual_weight=1.0,
            edge_weight=mlp_edge_weight,
            normal_weight=mlp_normal_weight,
        )
        return RegionExpert(name, vertices.copy(), pca, reg, xs, cs, predictor)
    else:
        raise ValueError(f"Unknown RBSR coefficient predictor: {predictor}")
    reg.fit(x, c)
    return RegionExpert(name, vertices.copy(), pca, reg, xs, cs, predictor)


def local_edges_for_vertices(bundle: DatasetBundle, vertices: np.ndarray) -> Optional[np.ndarray]:
    edges = build_edges(bundle.roi_faces)
    if edges is None or len(vertices) == 0:
        return None
    vertices = np.asarray(vertices, dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(vertices.tolist())}
    rows = []
    for a, b in edges:
        ia = pos.get(int(a))
        ib = pos.get(int(b))
        if ia is not None and ib is not None:
            rows.append((ia, ib))
    if not rows:
        return None
    return np.asarray(rows, dtype=np.int64)


def local_faces_for_vertices(bundle: DatasetBundle, vertices: np.ndarray) -> Optional[np.ndarray]:
    if bundle.roi_faces is None or len(vertices) == 0:
        return None
    vertices = np.asarray(vertices, dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(vertices.tolist())}
    rows = []
    for tri in bundle.roi_faces:
        mapped = [pos.get(int(v)) for v in tri]
        if all(v is not None for v in mapped):
            rows.append(tuple(int(v) for v in mapped))
    if not rows:
        return None
    return np.asarray(rows, dtype=np.int64)


def fit_region_experts(
    bundle: DatasetBundle,
    cond: np.ndarray,
    residual: np.ndarray,
    regions: Dict[str, np.ndarray],
    n_basis: int,
    ridge_alpha: float,
    predictor: str,
    mlp_edge_weight: float = 0.0,
    mlp_normal_weight: float = 0.0,
) -> List[RegionExpert]:
    return [
        fit_region_expert(
            cond,
            residual,
            idx,
            int(n_basis),
            ridge_alpha,
            name,
            predictor,
            local_edges=local_edges_for_vertices(bundle, idx),
            local_faces=local_faces_for_vertices(bundle, idx),
            mlp_edge_weight=mlp_edge_weight,
            mlp_normal_weight=mlp_normal_weight,
        )
        for name, idx in regions.items()
    ]


def predict_region_expert(expert: RegionExpert, cond: np.ndarray) -> np.ndarray:
    coeff = expert.c_scaler.inverse_transform(expert.regressor.predict(expert.x_scaler.transform(cond)))
    return expert.pca.inverse_transform(coeff).reshape(len(cond), len(expert.vertices), 3)


def region_vertex_sets(bundle: DatasetBundle) -> Dict[str, np.ndarray]:
    sets = {}
    for name in RBSR_REGIONS:
        if name in bundle.subunits:
            sets[name] = np.intersect1d(bundle.subunits[name], bundle.free_idx)
    if "alar" not in sets:
        parts = [bundle.subunits[k] for k in ["alar_left", "alar_right"] if k in bundle.subunits]
        if parts:
            sets["alar"] = np.intersect1d(np.unique(np.concatenate(parts)), bundle.free_idx)
    return {k: v for k, v in sets.items() if len(v)}


def require_regions(bundle: DatasetBundle, out: Path) -> Dict[str, np.ndarray]:
    regions = region_vertex_sets(bundle)
    if not regions:
        atomic_json(out / "SKIPPED.json", {
            "reason": "No usable anatomical region sets found for RBSR/oracle routing.",
            "available_subunits": {k: int(len(v)) for k, v in bundle.subunits.items()},
            "expected_any_of": RBSR_REGIONS + ["alar_left", "alar_right"],
        })
        log(f"SKIP {out.name}: no usable region vertex sets")
    return regions


def root_coverage_audit(bundle: DatasetBundle) -> Dict[str, Any]:
    regions = region_vertex_sets(bundle)
    root = bundle.subunits.get("root", np.asarray([], dtype=np.int64))
    root_free = np.intersect1d(root, bundle.free_idx)
    root_landmarks = np.intersect1d(root, bundle.landmarks)
    covered = np.asarray([], dtype=np.int64)
    if regions:
        covered = np.unique(np.concatenate(list(regions.values())))
    root_covered = np.intersect1d(root_free, covered)
    return {
        "root_subunit_vertices": int(len(root)),
        "root_free_vertices": int(len(root_free)),
        "root_landmark_vertices": int(len(root_landmarks)),
        "root_in_rbsr_region_list": "root" in RBSR_REGIONS,
        "root_region_available": "root" in regions,
        "root_free_vertices_covered_by_rbsr": int(len(root_covered)),
        "root_free_coverage_fraction": float(len(root_covered) / max(1, len(root_free))),
        "root_is_hard_fixed": bool(len(root_free) == 0 and len(root) > 0),
        "hard_fixed_vertices_are_only_landmarks": True,
        "rbsr_region_list": list(RBSR_REGIONS),
        "rbsr_regions_loaded": {k: int(len(v)) for k, v in regions.items()},
        "diagnosis": (
            "root is included in RBSR basis/routing"
            if "root" in regions and len(root_covered) == len(root_free)
            else "root is not fully covered by RBSR basis/routing"
        ),
    }


def apply_oracle_region_experts(
    bundle: DatasetBundle,
    ridge_pred: np.ndarray,
    data: Dict[str, Any],
    experts: Sequence[RegionExpert],
) -> np.ndarray:
    out = ridge_pred.copy()
    used = np.zeros(out.shape[1], dtype=bool)
    for expert in experts:
        corr = predict_region_expert(expert, data["cond"])
        out[:, expert.vertices, :] = ridge_pred[:, expert.vertices, :] + corr
        used[expert.vertices] = True
    out = strict_overwrite(out, data["delta"], bundle.landmarks)
    return out


def apply_region_experts_with_gates(
    bundle: DatasetBundle,
    ridge_pred: np.ndarray,
    data: Dict[str, Any],
    experts: Sequence[RegionExpert],
    gates: np.ndarray,
) -> np.ndarray:
    out = ridge_pred.copy()
    for j, expert in enumerate(experts):
        corr = predict_region_expert(expert, data["cond"])
        gate = gates[:, j].astype(np.float32).reshape(-1, 1, 1)
        out[:, expert.vertices, :] = ridge_pred[:, expert.vertices, :] + gate * corr
    return strict_overwrite(out, data["delta"], bundle.landmarks)


def projection_upper_bound(
    ridge_pred: np.ndarray,
    true_delta: np.ndarray,
    experts: Sequence[RegionExpert],
) -> np.ndarray:
    out = ridge_pred.copy()
    residual = true_delta - ridge_pred
    for expert in experts:
        y = residual[:, expert.vertices, :].reshape(len(residual), -1)
        yhat = expert.pca.inverse_transform(expert.pca.transform(y)).reshape(len(residual), len(expert.vertices), 3)
        out[:, expert.vertices, :] = ridge_pred[:, expert.vertices, :] + yhat
    return out


def per_pair_region_rmse(pred_delta: np.ndarray, true_delta: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    diff = pred_delta[:, vertices, :] - true_delta[:, vertices, :]
    return np.sqrt(np.mean(diff * diff, axis=(1, 2)))


def coefficient_prediction_rows(
    experts: Sequence[RegionExpert],
    cond: np.ndarray,
    true_delta: np.ndarray,
    ridge_pred: np.ndarray,
) -> List[Dict[str, Any]]:
    residual = true_delta - ridge_pred
    rows: List[Dict[str, Any]] = []
    for expert in experts:
        y = residual[:, expert.vertices, :].reshape(len(residual), -1)
        true_coeff = expert.pca.transform(y)
        pred_coeff = expert.c_scaler.inverse_transform(expert.regressor.predict(expert.x_scaler.transform(cond)))
        denom = np.sum((true_coeff - true_coeff.mean(axis=0, keepdims=True)) ** 2, axis=0)
        numer = np.sum((true_coeff - pred_coeff) ** 2, axis=0)
        r2 = 1.0 - numer / np.maximum(denom, 1e-12)
        rows.append({
            "region": expert.name,
            "predictor": expert.predictor,
            "n_vertices": int(len(expert.vertices)),
            "n_coefficients": int(true_coeff.shape[1]),
            "coeff_mae": float(np.mean(np.abs(true_coeff - pred_coeff))),
            "coeff_rmse": float(np.sqrt(np.mean((true_coeff - pred_coeff) ** 2))),
            "coeff_r2_mean": float(np.mean(r2)),
            "coeff_r2_median": float(np.median(r2)),
            "coeff_r2_min": float(np.min(r2)),
            "coeff_r2_max": float(np.max(r2)),
        })
    return rows


def fit_admission_models(
    cond: np.ndarray,
    ridge_pred: np.ndarray,
    rbsr_pred: np.ndarray,
    true_delta: np.ndarray,
    experts: Sequence[RegionExpert],
    alpha: float,
) -> List[AdmissionModel]:
    models: List[AdmissionModel] = []
    for expert in experts:
        ridge_err = per_pair_region_rmse(ridge_pred, true_delta, expert.vertices)
        rbsr_err = per_pair_region_rmse(rbsr_pred, true_delta, expert.vertices)
        improvement = ridge_err - rbsr_err
        xs = sk_pre.StandardScaler().fit(cond)
        reg = sk_linear.Ridge(alpha=alpha).fit(xs.transform(cond), improvement)
        scale = float(np.std(improvement))
        models.append(AdmissionModel(expert.name, expert.vertices.copy(), xs, reg, max(scale, 1e-6)))
    return models


def predict_admission_gates(models: Sequence[AdmissionModel], cond: np.ndarray, mode: str) -> np.ndarray:
    cols = []
    for model in models:
        pred_improvement = model.regressor.predict(model.x_scaler.transform(cond))
        if mode == "hard":
            gate = (pred_improvement > 0.0).astype(np.float32)
        elif mode == "soft":
            gate = 1.0 / (1.0 + np.exp(-pred_improvement / model.improvement_scale))
            gate = gate.astype(np.float32)
        else:
            raise ValueError(f"Unknown admission mode: {mode}")
        cols.append(gate)
    return np.stack(cols, axis=1)


def fit_global_admission_model(
    cond: np.ndarray,
    ridge_pred: np.ndarray,
    rbsr_pred: np.ndarray,
    true_delta: np.ndarray,
    free_idx: np.ndarray,
    alpha: float,
) -> AdmissionModel:
    ridge_err = np.asarray([rmse(ridge_pred[i:i + 1], true_delta[i:i + 1], free_idx) for i in range(len(true_delta))])
    rbsr_err = np.asarray([rmse(rbsr_pred[i:i + 1], true_delta[i:i + 1], free_idx) for i in range(len(true_delta))])
    improvement = ridge_err - rbsr_err
    xs = sk_pre.StandardScaler().fit(cond)
    reg = sk_linear.Ridge(alpha=alpha).fit(xs.transform(cond), improvement)
    return AdmissionModel("global", np.asarray([], dtype=np.int64), xs, reg, max(float(np.std(improvement)), 1e-6))


def predict_global_admission_gates(model: AdmissionModel, cond: np.ndarray, n_regions: int, mode: str) -> np.ndarray:
    gate = predict_admission_gates([model], cond, mode)[:, :1]
    return np.repeat(gate, int(n_regions), axis=1)


def fit_vertex_gate_model(
    cond: np.ndarray,
    ridge_pred: np.ndarray,
    rbsr_pred: np.ndarray,
    true_delta: np.ndarray,
    alpha: float,
) -> VertexGateModel:
    ridge_err = np.linalg.norm(ridge_pred - true_delta, axis=-1)
    rbsr_err = np.linalg.norm(rbsr_pred - true_delta, axis=-1)
    improvement = ridge_err - rbsr_err
    xs = sk_pre.StandardScaler().fit(cond)
    reg = sk_linear.Ridge(alpha=alpha).fit(xs.transform(cond), improvement)
    return VertexGateModel(xs, reg, max(float(np.std(improvement)), 1e-6))


def apply_vertex_gate(
    bundle: DatasetBundle,
    ridge_pred: np.ndarray,
    rbsr_pred: np.ndarray,
    true_delta: np.ndarray,
    model: VertexGateModel,
    cond: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred_improvement = model.regressor.predict(model.x_scaler.transform(cond))
    if mode == "hard":
        gate = (pred_improvement > 0.0).astype(np.float32)
    elif mode == "soft":
        gate = (1.0 / (1.0 + np.exp(-pred_improvement / model.improvement_scale))).astype(np.float32)
    else:
        raise ValueError(f"Unknown vertex gate mode: {mode}")
    out = ridge_pred + gate[:, :, None] * (rbsr_pred - ridge_pred)
    out = strict_overwrite(out, true_delta, bundle.landmarks)
    summary = {
        "mode": mode,
        "mean_gate": float(np.mean(gate[:, bundle.free_idx])),
        "active_fraction_gate_gt_0_5": float(np.mean(gate[:, bundle.free_idx] > 0.5)),
        "min_gate": float(np.min(gate[:, bundle.free_idx])),
        "max_gate": float(np.max(gate[:, bundle.free_idx])),
    }
    return out, summary


def oracle_admission_gates(
    ridge_pred: np.ndarray,
    rbsr_pred: np.ndarray,
    true_delta: np.ndarray,
    experts: Sequence[RegionExpert],
) -> np.ndarray:
    cols = []
    for expert in experts:
        ridge_err = per_pair_region_rmse(ridge_pred, true_delta, expert.vertices)
        rbsr_err = per_pair_region_rmse(rbsr_pred, true_delta, expert.vertices)
        cols.append((rbsr_err < ridge_err).astype(np.float32))
    return np.stack(cols, axis=1)


def admission_gate_rows(
    gates: np.ndarray,
    experts: Sequence[RegionExpert],
    method: str,
) -> List[Dict[str, Any]]:
    rows = []
    for j, expert in enumerate(experts):
        col = gates[:, j]
        rows.append({
            "method": method,
            "region": expert.name,
            "mean_gate": float(np.mean(col)),
            "min_gate": float(np.min(col)),
            "max_gate": float(np.max(col)),
            "active_fraction_gate_gt_0_5": float(np.mean(col > 0.5)),
        })
    return rows


def residual_basis_capacity_audit(
    bundle: DatasetBundle,
    train: Dict[str, Any],
    ridge_pred_train: np.ndarray,
    basis_grid: Sequence[int],
) -> Dict[str, Any]:
    residual = train["delta"] - ridge_pred_train
    regions = region_vertex_sets(bundle)
    rows = []
    for name, idx in regions.items():
        y = residual[:, idx, :].reshape(len(residual), -1)
        rank_cap = int(min(y.shape[0], y.shape[1]))
        pca = sk_decomp.PCA(n_components=rank_cap, random_state=0).fit(y)
        ev = np.cumsum(pca.explained_variance_ratio_)
        row: Dict[str, Any] = {
            "region": name,
            "n_vertices": int(len(idx)),
            "residual_dim": int(y.shape[1]),
            "n_train_pairs": int(y.shape[0]),
            "max_train_residual_rank_cap": rank_cap,
        }
        for k in basis_grid:
            kk = min(int(k), rank_cap)
            row[f"train_residual_energy_explained_k{int(k)}"] = float(ev[kk - 1]) if kk > 0 else 0.0
        row["k_for_90pct_train_energy"] = int(np.searchsorted(ev, 0.90) + 1)
        row["k_for_95pct_train_energy"] = int(np.searchsorted(ev, 0.95) + 1)
        row["k_for_99pct_train_energy"] = int(np.searchsorted(ev, 0.99) + 1)
        rows.append(row)
    return {
        "note": "This is a training residual PCA capacity audit. Test projection upper bound is measured separately in basis_gap_root_included.csv.",
        "regions": rows,
    }


def risk_summary_rows(
    per_pair_by_method: Dict[str, List[Dict[str, Any]]],
    baseline_method: str = "Ridge_strict",
    metric: str = "strict_free_rmse",
) -> List[Dict[str, Any]]:
    baseline = np.asarray([float(r[metric]) for r in per_pair_by_method[baseline_method]], dtype=np.float64)
    rows = []
    for method, items in per_pair_by_method.items():
        vals = np.asarray([float(r[metric]) for r in items], dtype=np.float64)
        rows.append({
            "method": method,
            "metric": metric,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
            "failure_rate_worse_than_ridge": float(np.mean(vals > baseline)) if method != baseline_method else 0.0,
            "large_failure_rate_worse_than_ridge_by_0_05": float(np.mean(vals > baseline + 0.05)) if method != baseline_method else 0.0,
        })
    return rows


def plot_feature_ablation(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame([r for r in rows if "strict_free_rmse" in r])
    if df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    plt.bar(df["feature_mode"], df["strict_free_rmse"], color="#4c78a8")
    plt.ylabel("Strict free-vertex RMSE")
    plt.xlabel("Feature set")
    plt.title("RBSR legal input feature ablation")
    save_fig_atomic(out / "feature_ablation_strict_free_rmse.png")


def run_residual_heatmap(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path) -> None:
    out = out_root / "01_ridge_residual_heatmap"
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        path, train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
        ridge, _ = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
        pred, data = predict_ridge(bundle, ridge, test_pairs)
        metrics = evaluate_prediction(bundle, data, pred, "ridge_strict")
        atomic_json(out / "config.json", vars(args) | {"pair_json": str(path), "hash": stable_hash(vars(args))})
        atomic_json(out / "split_audit.json", split_audit)
        atomic_json(out / "metrics.json", metrics)
        residual_heatmap(bundle, pred, data, out)


def run_region_oracle(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path, n_basis: int = 16) -> None:
    out = out_root / "02_oracle_region_routing"
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        regions = require_regions(bundle, out)
        if not regions:
            return
        path, train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
        ridge, train = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
        train_ridge_pred = ridge.y_scaler.inverse_transform(ridge.model.predict(ridge.x_scaler.transform(train["cond"]))).reshape(len(train["pairs"]), -1, 3)
        train_ridge_pred = strict_overwrite(train_ridge_pred, train["delta"], bundle.landmarks)
        train_resid = train["delta"] - train_ridge_pred
        ridge_pred, test = predict_ridge(bundle, ridge, test_pairs)
        rbsr_train = append_legal_rbsr_features(bundle, train, args.rbsr_feature_mode)
        rbsr_test = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
        experts = fit_region_experts(
            bundle, rbsr_train["cond"], train_resid, regions, n_basis,
            args.ridge_alpha, args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
        test_for_rbsr = dict(test)
        test_for_rbsr["cond"] = rbsr_test["cond"]
        oracle_pred = apply_oracle_region_experts(bundle, ridge_pred, test_for_rbsr, experts)
        upper_pred = projection_upper_bound(ridge_pred, test["delta"], experts)
        upper_pred = strict_overwrite(upper_pred, test["delta"], bundle.landmarks)
        rows = [
            evaluate_prediction(bundle, test, ridge_pred, "ridge_strict"),
            evaluate_prediction(bundle, test, oracle_pred, f"region_oracle_routing_basis{n_basis}"),
            evaluate_prediction(bundle, test, upper_pred, f"region_projection_upper_bound_basis{n_basis}"),
        ]
        atomic_csv(out / "oracle_region_routing_metrics.csv", rows)
        atomic_json(out / "split_audit.json", split_audit)
        atomic_json(out / "summary.json", {"regions": {k: len(v) for k, v in regions.items()}, "basis": n_basis, "pair_json": str(path)})
        plot_metric_bars(rows, "strict_free_rmse", out / "oracle_region_routing_free_rmse.png", "Strict free-vertex RMSE")
        plot_metric_bars(rows, "normal_consistency", out / "oracle_region_routing_normal_consistency.png", "Normal consistency loss")


def run_basis_sweep(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path) -> None:
    out = out_root / "03_rbsr_basis_sweep"
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        regions = require_regions(bundle, out)
        if not regions:
            return
        path, train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
        ridge, train = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
        train_pred = ridge.y_scaler.inverse_transform(ridge.model.predict(ridge.x_scaler.transform(train["cond"]))).reshape(len(train["pairs"]), -1, 3)
        train_pred = strict_overwrite(train_pred, train["delta"], bundle.landmarks)
        train_resid = train["delta"] - train_pred
        test_pred, test = predict_ridge(bundle, ridge, test_pairs)
        rbsr_train = append_legal_rbsr_features(bundle, train, args.rbsr_feature_mode)
        rbsr_test = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
        test_for_rbsr = dict(test)
        test_for_rbsr["cond"] = rbsr_test["cond"]
        atomic_json(out / "split_audit.json", split_audit)
        rows = [evaluate_prediction(bundle, test, test_pred, "ridge_strict")]
        for k in args.basis_grid:
            log(f"Basis sweep k={k}")
            experts = fit_region_experts(
                bundle, rbsr_train["cond"], train_resid, regions, k,
                args.ridge_alpha, args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
            pred = apply_oracle_region_experts(bundle, test_pred, test_for_rbsr, experts)
            upper = projection_upper_bound(test_pred, test["delta"], experts)
            upper = strict_overwrite(upper, test["delta"], bundle.landmarks)
            row = evaluate_prediction(bundle, test, pred, f"rbsr_predicted_basis{k}")
            row["basis"] = k
            row["kind"] = "predicted_coefficients"
            rows.append(row)
            row2 = evaluate_prediction(bundle, test, upper, f"rbsr_projection_upper_basis{k}")
            row2["basis"] = k
            row2["kind"] = "projection_upper_bound"
            rows.append(row2)
        atomic_csv(out / "basis_sweep_metrics.csv", rows)
        plot_basis_sweep(rows, out)


def plot_metric_bars(rows: Sequence[Dict[str, Any]], metric: str, path: Path, ylabel: str) -> None:
    rows = [r for r in rows if metric in r and np.isfinite(float(r[metric]))]
    plt.figure(figsize=(9, 4.5))
    plt.bar([r["method"] for r in rows], [float(r[metric]) for r in rows], color="#4c78a8")
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    save_fig_atomic(path)


def plot_basis_sweep(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[df.get("basis").notna()] if "basis" in df.columns else df.iloc[0:0]
    if df.empty:
        return
    for metric in ["strict_free_rmse", "edge_strain", "normal_consistency"]:
        if metric not in df:
            continue
        plt.figure(figsize=(7, 4.5))
        for kind, group in df.groupby("kind"):
            group = group.sort_values("basis")
            plt.plot(group["basis"], group[metric], marker="o", label=kind)
        plt.xlabel("Residual basis count")
        plt.ylabel(metric)
        plt.title(f"RBSR basis sweep: {metric}")
        plt.legend()
        save_fig_atomic(out / f"basis_sweep_{metric}.png")


class VectorCVAE(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int, latent_dim: int, width: int, layers: int):
        super().__init__()
        enc_layers: List[nn.Module] = []
        in_dim = cond_dim + out_dim
        for _ in range(layers):
            enc_layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        dec_layers: List[nn.Module] = []
        in_dim = cond_dim + latent_dim
        for _ in range(layers):
            dec_layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        dec_layers.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, cond: Any, y: Any) -> Tuple[Any, Any, Any]:
        h = self.encoder(torch.cat([cond, y], dim=1))
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -8.0, 8.0)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        pred = self.decoder(torch.cat([cond, z], dim=1))
        return pred, mu, logvar

    def predict(self, cond: Any) -> Any:
        z = torch.zeros((cond.shape[0], self.mu.out_features), dtype=cond.dtype, device=cond.device)
        return self.decoder(torch.cat([cond, z], dim=1))


def torch_edges(edges: Optional[np.ndarray], device: str) -> Optional[Any]:
    if edges is None or torch is None:
        return None
    return torch.as_tensor(edges, dtype=torch.long, device=device)


def torch_faces(faces: Optional[np.ndarray], device: str) -> Optional[Any]:
    if faces is None or torch is None:
        return None
    return torch.as_tensor(faces, dtype=torch.long, device=device)


def geometry_loss_terms(
    pred_delta_flat: Any,
    true_delta_flat: Any,
    source_flat: Any,
    n_vertices: int,
    edges: Optional[Any],
    faces: Optional[Any],
) -> Dict[str, Any]:
    pred_delta = pred_delta_flat.view(-1, n_vertices, 3)
    true_delta = true_delta_flat.view(-1, n_vertices, 3)
    pred_shape = source_flat.view(-1, n_vertices, 3) + pred_delta
    true_shape = source_flat.view(-1, n_vertices, 3) + true_delta
    out = {}
    if edges is not None:
        pe = torch.linalg.norm(pred_shape[:, edges[:, 0], :] - pred_shape[:, edges[:, 1], :], dim=-1)
        te = torch.linalg.norm(true_shape[:, edges[:, 0], :] - true_shape[:, edges[:, 1], :], dim=-1)
        out["edge"] = torch.mean(torch.abs(pe - te) / (te + 1e-6))
        lap = torch.zeros_like(pred_shape)
        deg = torch.zeros((n_vertices,), dtype=pred_shape.dtype, device=pred_shape.device)
        for side in [0, 1]:
            src = edges[:, side]
            dst = edges[:, 1 - side]
            lap.index_add_(1, src, pred_shape[:, dst, :])
            deg.index_add_(0, src, torch.ones_like(src, dtype=pred_shape.dtype))
        lap = lap / deg.clamp_min(1).view(1, -1, 1) - pred_shape
        out["lap"] = torch.mean(lap ** 2)
    if faces is not None:
        p0, p1, p2 = pred_shape[:, faces[:, 0], :], pred_shape[:, faces[:, 1], :], pred_shape[:, faces[:, 2], :]
        t0, t1, t2 = true_shape[:, faces[:, 0], :], true_shape[:, faces[:, 1], :], true_shape[:, faces[:, 2], :]
        pn = torch.cross(p1 - p0, p2 - p0, dim=-1)
        tn = torch.cross(t1 - t0, t2 - t0, dim=-1)
        pn = F.normalize(pn, dim=-1, eps=1e-8)
        tn = F.normalize(tn, dim=-1, eps=1e-8)
        out["normal"] = torch.mean(1.0 - torch.sum(pn * tn, dim=-1).clamp(-1.0, 1.0))
    return out


LOSS_MATRIX = {
    "roi_only": {"roi": 1.0, "landmark": 0.0, "edge": 0.0, "normal": 0.0, "lap": 0.0},
    "roi_landmark": {"roi": 1.0, "landmark": 5.0, "edge": 0.0, "normal": 0.0, "lap": 0.0},
    "roi_edge": {"roi": 1.0, "landmark": 5.0, "edge": 0.1, "normal": 0.0, "lap": 0.0},
    "roi_normal": {"roi": 1.0, "landmark": 5.0, "edge": 0.0, "normal": 0.1, "lap": 0.0},
    "roi_laplacian": {"roi": 1.0, "landmark": 5.0, "edge": 0.0, "normal": 0.0, "lap": 0.01},
}


def train_cvae_once(
    bundle: DatasetBundle,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    latent_dim: int,
    width: int,
    layers: int,
    loss_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    out: Path,
    device: str,
    return_state: bool = False,
) -> Any:
    if torch is None:
        raise SystemExit("PyTorch is required for CVAE experiments")
    set_seed(seed)
    weights = LOSS_MATRIX[loss_name]
    n_vertices = len(bundle.roi_global)
    cond_scaler = sk_pre.StandardScaler().fit(train_data["cond"])
    y_scaler = sk_pre.StandardScaler().fit(train_data["delta"].reshape(len(train_data["pairs"]), -1))
    x_train = cond_scaler.transform(train_data["cond"]).astype(np.float32)
    y_train = y_scaler.transform(train_data["delta"].reshape(len(train_data["pairs"]), -1)).astype(np.float32)
    src_train = train_data["source"].reshape(len(train_data["pairs"]), -1).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(src_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = VectorCVAE(x_train.shape[1], y_train.shape[1], latent_dim, width, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    edges_t = torch_edges(build_edges(bundle.roi_faces), device)
    faces_t = torch_faces(bundle.roi_faces, device)
    y_mean_t = torch.as_tensor(y_scaler.mean_.astype(np.float32), device=device)
    y_scale_t = torch.as_tensor(y_scaler.scale_.astype(np.float32), device=device)
    lm_flat = np.concatenate([np.arange(i * 3, i * 3 + 3) for i in bundle.landmarks])
    free_flat = np.concatenate([np.arange(i * 3, i * 3 + 3) for i in bundle.free_idx])
    lm_t = torch.as_tensor(lm_flat, dtype=torch.long, device=device)
    free_t = torch.as_tensor(free_flat, dtype=torch.long, device=device)
    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            pred, mu, logvar = model(xb, yb)
            roi_loss = F.mse_loss(pred[:, free_t], yb[:, free_t])
            lm_loss = F.mse_loss(pred[:, lm_t], yb[:, lm_t])
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = weights["roi"] * roi_loss + weights["landmark"] * lm_loss + 1e-4 * kl
            if weights["edge"] or weights["normal"] or weights["lap"]:
                pred_un = pred * y_scale_t + y_mean_t
                true_un = yb * y_scale_t + y_mean_t
                geom = geometry_loss_terms(pred_un, true_un, sb, n_vertices, edges_t, faces_t)
                for key in ["edge", "normal", "lap"]:
                    if weights[key] and key in geom:
                        loss = loss + weights[key] * geom[key]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            pred_val, _ = predict_cvae_arrays(model, cond_scaler, y_scaler, val_data["cond"], device)
            pred_val = pred_val.reshape(len(val_data["pairs"]), n_vertices, 3)
            pred_val = strict_overwrite(pred_val, val_data["delta"], bundle.landmarks)
            val_rmse = rmse(pred_val, val_data["delta"], bundle.free_idx)
            rec = {"epoch": epoch, "train_loss": total / len(ds), "val_strict_free_rmse": val_rmse}
            history.append(rec)
            log(f"CVAE {loss_name} z={latent_dim} w={width} L={layers} epoch={epoch}/{epochs} val_free_rmse={val_rmse:.5f}")
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, out / "best_model.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, _ = predict_cvae_arrays(model, cond_scaler, y_scaler, val_data["cond"], device)
    atomic_csv(out / "training_curve.csv", history)
    meta = {
        "best_val_strict_free_rmse": best_val,
        "latent_dim": latent_dim,
        "width": width,
        "layers": layers,
        "loss_name": loss_name,
    }
    pred = pred.reshape(len(val_data["pairs"]), n_vertices, 3)
    if return_state:
        return pred, meta, model, cond_scaler, y_scaler
    return pred, meta


def predict_cvae_arrays(model: Any, cond_scaler: Any, y_scaler: Any, cond: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.as_tensor(cond_scaler.transform(cond).astype(np.float32), device=device)
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            pred = model.predict(x[start:start + 256])
            outs.append(pred.detach().cpu().numpy())
    arr = np.concatenate(outs, axis=0)
    return y_scaler.inverse_transform(arr).astype(np.float32), arr


def cvae_condition_data(
    bundle: DatasetBundle,
    train_pairs: Sequence[Tuple[str, str]],
    val_pairs: Sequence[Tuple[str, str]],
    source_pca_dim: int,
    condition_variant: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dim = cvae_source_pca_dim(source_pca_dim, condition_variant)
    train = make_pair_arrays(bundle, train_pairs, source_pca_dim=dim, fit_pca=True)
    val = make_pair_arrays(bundle, val_pairs, source_pca=train["source_pca"], source_pca_dim=dim, fit_pca=False)
    train = apply_cvae_condition_variant(bundle, train, train, condition_variant)
    val = apply_cvae_condition_variant(bundle, train, val, condition_variant)
    return train, val


def cvae_source_pca_dim(source_pca_dim: int, condition_variant: str) -> int:
    if condition_variant == "base":
        return source_pca_dim
    elif condition_variant.startswith("source_pca"):
        return int(condition_variant.replace("source_pca", ""))
    elif condition_variant == "source_flat_pca128":
        return 128
    return source_pca_dim


def apply_cvae_condition_variant(
    bundle: DatasetBundle,
    train_ref: Dict[str, Any],
    data: Dict[str, Any],
    condition_variant: str,
) -> Dict[str, Any]:
    if condition_variant == "subunit_error_oracle":
        return append_subunit_oracle_features(bundle, train_ref, data)
    elif condition_variant == "local_coordinate_summary":
        return append_local_coordinate_summary(bundle, data)
    return data


def cvae_eval_data_from_train(
    bundle: DatasetBundle,
    train_ref: Dict[str, Any],
    pairs: Sequence[Tuple[str, str]],
    source_pca_dim: int,
    condition_variant: str,
) -> Dict[str, Any]:
    dim = cvae_source_pca_dim(source_pca_dim, condition_variant)
    data = make_pair_arrays(bundle, pairs, source_pca=train_ref["source_pca"], source_pca_dim=dim, fit_pca=False)
    return apply_cvae_condition_variant(bundle, train_ref, data, condition_variant)


def append_local_coordinate_summary(bundle: DatasetBundle, data: Dict[str, Any]) -> Dict[str, Any]:
    src_norm = (data["source"] - bundle.roi_xyz_mean[None]) / bundle.roi_xyz_std[None]
    stats = []
    for name in RBSR_REGIONS:
        idx = bundle.subunits.get(name)
        if idx is not None and len(idx):
            stats.append(src_norm[:, idx, :].mean(axis=1))
            stats.append(src_norm[:, idx, :].std(axis=1))
    if stats:
        extra = np.concatenate(stats, axis=1).astype(np.float32)
        data = dict(data)
        data["cond"] = np.concatenate([data["cond"], extra], axis=1).astype(np.float32)
    return data


def append_subunit_oracle_features(bundle: DatasetBundle, train_ref: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    # Strong diagnostic only: exposes true per-subunit control residual summaries.
    feats = []
    for name in RBSR_REGIONS:
        idx = bundle.subunits.get(name)
        if idx is not None and len(idx):
            feats.append(data["delta"][:, idx, :].mean(axis=1))
            feats.append(data["delta"][:, idx, :].std(axis=1))
    if feats:
        data = dict(data)
        data["cond"] = np.concatenate([data["cond"], np.concatenate(feats, axis=1).astype(np.float32)], axis=1).astype(np.float32)
    return data


def run_cvae_grid(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path, mode: str) -> None:
    out = out_root / ("05_cvae_loss_matrix" if mode == "loss" else "04_cvae_capacity_scan" if mode == "capacity" else "07_cvae_stronger_input_oracle")
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        if torch is None:
            raise SystemExit("PyTorch is required for CVAE experiments")
        device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        path, train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
        fit_pairs, val_pairs = split_pairs(train_pairs, args.cvae_val_fraction, args.seed + 101)
        atomic_json(out / "split_audit.json", split_audit)
        atomic_json(out / "cvae_split_audit.json", {
            "early_stopping_used_test_pairs": False,
            "fit_pairs": len(fit_pairs),
            "validation_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "cvae_val_fraction": float(args.cvae_val_fraction),
            "note": "CVAE checkpoints are selected on train-internal validation pairs; test pairs are evaluated once after checkpoint selection.",
        })
        rows = []
        if mode == "capacity":
            jobs = [(z, w, l, "roi_landmark", "base") for z in args.latent_grid for w in args.width_grid for l in args.layer_grid]
        elif mode == "loss":
            jobs = [(args.default_latent, args.default_width, args.default_layers, loss, "base") for loss in LOSS_MATRIX]
        else:
            variants = ["base", "source_pca64", "source_flat_pca128", "local_coordinate_summary", "subunit_error_oracle"]
            jobs = [(args.default_latent, args.default_width, args.default_layers, "roi_landmark", v) for v in variants]
        for z, w, layers, loss_name, variant in jobs:
            job_out = out / f"z{z}_w{w}_L{layers}_{loss_name}_{variant}"
            with experiment_guard(job_out, args.overwrite) as job_active:
                if not job_active:
                    summary_path = job_out / "summary.json"
                    if summary_path.exists():
                        rows.append(read_json(summary_path))
                    continue
                log(f"CVAE job mode={mode} z={z} width={w} layers={layers} loss={loss_name} variant={variant}")
                train, val = cvae_condition_data(bundle, fit_pairs, val_pairs, args.source_pca_dim, variant)
                pred_val, meta, model, cond_scaler, y_scaler = train_cvae_once(
                    bundle, train, val, z, w, layers, loss_name, args.epochs, args.batch_size, args.lr, args.seed, job_out, device,
                    return_state=True,
                )
                test_data = cvae_eval_data_from_train(bundle, train, test_pairs, args.source_pca_dim, variant)
                pred_test_flat, _ = predict_cvae_arrays(model, cond_scaler, y_scaler, test_data["cond"], device)
                pred = pred_test_flat.reshape(len(test_data["pairs"]), len(bundle.roi_global), 3)
                pred = strict_overwrite(pred, test_data["delta"], bundle.landmarks)
                metrics = evaluate_prediction(bundle, test_data, pred, "cvae")
                summary = dict(meta)
                summary.update(metrics)
                summary.update({
                    "condition_variant": variant,
                    "device": device,
                    "pair_json": str(path),
                    "fit_pairs": len(fit_pairs),
                    "validation_pairs": len(val_pairs),
                    "test_pairs": len(test_pairs),
                    "early_stopping_used_test_pairs": False,
                })
                atomic_json(job_out / "summary.json", summary)
                rows.append(summary)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        atomic_csv(out / "summary.csv", rows)
        plot_cvae_summary(rows, out, mode)


def plot_cvae_summary(rows: Sequence[Dict[str, Any]], out: Path, mode: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "strict_free_rmse" in df:
        plt.figure(figsize=(9, 5))
        if mode == "capacity":
            for (w, l), g in df.groupby(["width", "layers"]):
                g = g.sort_values("latent_dim")
                plt.plot(g["latent_dim"], g["strict_free_rmse"], marker="o", label=f"w={w}, L={l}")
            plt.xlabel("Latent dimension")
        elif mode == "loss":
            plt.bar(df["loss_name"], df["strict_free_rmse"], color="#4c78a8")
            plt.xticks(rotation=25, ha="right")
        else:
            plt.bar(df["condition_variant"], df["strict_free_rmse"], color="#4c78a8")
            plt.xticks(rotation=25, ha="right")
        plt.ylabel("Strict free-vertex RMSE")
        plt.title(f"CVAE {mode} diagnostic")
        if mode == "capacity":
            plt.legend()
        save_fig_atomic(out / f"cvae_{mode}_strict_free_rmse.png")
    for metric in ["edge_strain", "normal_consistency"]:
        if metric in df and df[metric].notna().any():
            plt.figure(figsize=(9, 5))
            label_col = "loss_name" if mode == "loss" else "condition_variant" if mode == "oracle" else "latent_dim"
            plt.bar(df[label_col].astype(str), df[metric], color="#f58518")
            plt.ylabel(metric)
            plt.xticks(rotation=25, ha="right")
            save_fig_atomic(out / f"cvae_{mode}_{metric}.png")


def run_rbsr_learning_curve(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path) -> None:
    out = out_root / "06_rbsr_learning_curve"
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        regions = require_regions(bundle, out)
        if not regions:
            return
        rows = []
        audit_rows = []
        if args.lc_mode == "identity_scale":
            work_items = []
            for scale in args.lc_scales:
                path = find_pair_path(args.root, args.chain, scale, args.split)
                train_pairs, test_pairs, split_audit = load_pair_splits(
                    path,
                    args.eval_split_mode,
                    args.val_fraction,
                    args.seed,
                    available_ids=set(bundle.meshes.keys()),
                    root=args.root,
                    max_manifest_test_pairs=args.max_manifest_test_pairs if args.max_manifest_test_pairs > 0 else None,
                )
                meta = read_json(path)
                audit_rows.append({
                    "mode": "identity_scale",
                    "scale": int(scale),
                    "n_ids": len(meta.get("ids", [])) if isinstance(meta, dict) else None,
                    "json_pairs": len(load_pairs(path)),
                    "train_pairs_after_split": len(train_pairs),
                    "test_pairs_after_split": len(test_pairs),
                    "identity_overlap_count": split_audit.get("identity_overlap_count"),
                    "eval_split_mode": split_audit.get("eval_split_mode"),
                    "pair_budget_policy": meta.get("pair_budget_policy") if isinstance(meta, dict) else None,
                })
                work_items.append((int(scale), len(train_pairs), train_pairs, test_pairs, str(path)))
        else:
            path, full_train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
            meta = read_json(path)
            audit_rows.append({
                "mode": "pair_count",
                "scale": int(args.scale),
                "n_ids": len(meta.get("ids", [])) if isinstance(meta, dict) else None,
                "json_pairs": len(load_pairs(path)),
                "full_train_pairs_after_split": len(full_train_pairs),
                "fixed_test_pairs": len(test_pairs),
                "identity_overlap_count": split_audit.get("identity_overlap_count"),
                "eval_split_mode": split_audit.get("eval_split_mode"),
                "pair_budget_policy": meta.get("pair_budget_policy") if isinstance(meta, dict) else None,
            })
            work_items = []
            for n_train in args.lc_train_pairs:
                n = min(int(n_train), len(full_train_pairs))
                if n <= 1:
                    continue
                work_items.append((n, n, full_train_pairs[:n], test_pairs, str(path)))

        for x_value, n_train_used, train_pairs, test_pairs, pair_path in work_items:
            log(f"RBSR learning curve mode={args.lc_mode} train_pairs={n_train_used} x={x_value}")
            ridge, train = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
            train_pred = ridge.y_scaler.inverse_transform(ridge.model.predict(ridge.x_scaler.transform(train["cond"]))).reshape(len(train["pairs"]), -1, 3)
            train_pred = strict_overwrite(train_pred, train["delta"], bundle.landmarks)
            train_resid = train["delta"] - train_pred
            rbsr_train = append_legal_rbsr_features(bundle, train, args.rbsr_feature_mode)
            experts = fit_region_experts(
                bundle, rbsr_train["cond"], train_resid, regions, args.default_basis,
                args.ridge_alpha, args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
            ridge_pred, test = predict_ridge(bundle, ridge, test_pairs)
            rbsr_test = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
            test_for_rbsr = dict(test)
            test_for_rbsr["cond"] = rbsr_test["cond"]
            rbsr_pred = apply_oracle_region_experts(bundle, ridge_pred, test_for_rbsr, experts)
            for row in [
                evaluate_prediction(bundle, test, ridge_pred, "ridge_strict"),
                evaluate_prediction(bundle, test, rbsr_pred, "rbsr_region_oracle"),
            ]:
                row["curve_mode"] = args.lc_mode
                row["x_value"] = x_value
                row["scale"] = x_value if args.lc_mode == "identity_scale" else int(args.scale)
                row["train_pairs"] = len(train_pairs)
                row["test_pairs"] = len(test_pairs)
                row["pair_json"] = pair_path
                rows.append(row)
            atomic_csv(out / "learning_curve_partial.csv", rows)
        atomic_csv(out / "learning_curve_metrics.csv", rows)
        atomic_csv(out / "learning_curve_pair_audit.csv", audit_rows)
        plot_learning_curve(rows, out)


def plot_learning_curve(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    x_col = "x_value" if "x_value" in df.columns else "scale"
    x_label = "Training pairs" if "curve_mode" in df.columns and str(df["curve_mode"].iloc[0]) == "pair_count" else "Training identity scale"
    for metric in ["strict_free_rmse", "edge_strain", "normal_consistency"]:
        if metric not in df:
            continue
        plt.figure(figsize=(7, 4.5))
        for method, g in df.groupby("method"):
            g = g.sort_values(x_col)
            plt.plot(g[x_col], g[metric], marker="o", label=method)
        plt.xlabel(x_label)
        plt.ylabel(metric)
        plt.title(f"RBSR learning curve: {metric}")
        plt.legend()
        save_fig_atomic(out / f"rbsr_learning_curve_{metric}.png")


def run_preflight(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path) -> None:
    out = out_root / "00_preflight"
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        report: Dict[str, Any] = {
            "root": str(args.root),
            "n_meshes": int(len(bundle.meshes)),
            "roi_vertices": int(len(bundle.roi_global)),
            "landmarks": [int(x) for x in bundle.landmarks.tolist()],
            "free_vertices": int(len(bundle.free_idx)),
            "subunits": {k: int(len(v)) for k, v in bundle.subunits.items()},
            "region_sets": {k: int(len(v)) for k, v in region_vertex_sets(bundle).items()},
            "has_roi_faces": bool(bundle.roi_faces is not None),
            "roi_faces": int(len(bundle.roi_faces)) if bundle.roi_faces is not None else 0,
        }
        pair_path, train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
        report.update({
            "pair_json": str(pair_path),
            "n_pairs_total_in_pair_json": int(len(load_pairs(pair_path))),
            "n_train_pairs_split": int(len(train_pairs)),
            "n_test_pairs_split": int(len(test_pairs)),
            "sample_train_pairs": train_pairs[:5],
            "sample_test_pairs": test_pairs[:5],
            "split_audit": split_audit,
        })
        dry_n_train = min(len(train_pairs), 256)
        dry_n_test = min(len(test_pairs), 128)
        ridge, _ = train_ridge(bundle, train_pairs[:dry_n_train], args.ridge_alpha, min(args.source_pca_dim, 8))
        pred, data = predict_ridge(bundle, ridge, test_pairs[:dry_n_test])
        report["dry_run_metrics"] = evaluate_prediction(bundle, data, pred, "ridge_preflight")
        atomic_json(out / "preflight_report.json", report)
        log(f"Preflight OK: meshes={report['n_meshes']} roi={report['roi_vertices']} free={report['free_vertices']} pairs={report['n_pairs_total']}")


def evaluate_rbsr_for_basis_grid(
    bundle: DatasetBundle,
    train: Dict[str, Any],
    test: Dict[str, Any],
    ridge_pred_train: np.ndarray,
    ridge_pred_test: np.ndarray,
    basis_grid: Sequence[int],
    ridge_alpha: float,
    predictor: str,
    feature_mode: str,
    mlp_edge_weight: float = 0.0,
    mlp_normal_weight: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    train_resid = train["delta"] - ridge_pred_train
    rbsr_train = append_legal_rbsr_features(bundle, train, feature_mode)
    rbsr_test = append_legal_rbsr_features(bundle, test, feature_mode)
    test_for_rbsr = dict(test)
    test_for_rbsr["cond"] = rbsr_test["cond"]
    regions = region_vertex_sets(bundle)
    if not regions:
        raise ValueError("No regions available for RBSR")
    rows = [evaluate_prediction(bundle, test, ridge_pred_test, "ridge_strict")]
    preds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in basis_grid:
        log(f"Final RBSR root-included basis k={k}")
        experts = fit_region_experts(
            bundle, rbsr_train["cond"], train_resid, regions, int(k),
            ridge_alpha, predictor, mlp_edge_weight, mlp_normal_weight,
        )
        pred = apply_oracle_region_experts(bundle, ridge_pred_test, test_for_rbsr, experts)
        upper = projection_upper_bound(ridge_pred_test, test["delta"], experts)
        pred = strict_overwrite(pred, test["delta"], bundle.landmarks)
        upper = strict_overwrite(upper, test["delta"], bundle.landmarks)
        row = evaluate_prediction(bundle, test, pred, f"rbsr_predicted_k{k}")
        row["basis"] = int(k)
        row["kind"] = "predicted_coefficients"
        rows.append(row)
        row2 = evaluate_prediction(bundle, test, upper, f"rbsr_projection_upper_k{k}")
        row2["basis"] = int(k)
        row2["kind"] = "projection_upper_bound"
        rows.append(row2)
        preds[int(k)] = (pred, upper)
    return rows, preds


def plot_basis_gap_three_lines(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    ridge_val = float(df[df["method"] == "ridge_strict"]["strict_free_rmse"].iloc[0])
    k_vals = sorted(int(x) for x in df["basis"].dropna().unique())
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(k_vals, [ridge_val] * len(k_vals), marker="o", label="Ridge")
    for kind, label in [
        ("predicted_coefficients", "Predicted RBSR"),
        ("coefficient_supervised", "Coefficient-supervised RBSR"),
        ("staged_prediction", "Staged-selected RBSR"),
        ("validation_selected", "Validation-selected RBSR"),
        ("projection_upper_bound", "Projection upper bound"),
    ]:
        g = df[df["kind"] == kind].sort_values("basis")
        if not g.empty:
            plt.plot(g["basis"], g["strict_free_rmse"], marker="o", label=label)
    plt.xlabel("Residual basis count K")
    plt.ylabel("Strict free-vertex RMSE")
    plt.title("Residual basis potential vs predictable gain")
    plt.legend()
    save_fig_atomic(out / "basis_gap_root_included_strict_free_rmse.png")


def run_feature_ablation_matrix(
    bundle: DatasetBundle,
    train: Dict[str, Any],
    test: Dict[str, Any],
    ridge_pred_train: np.ndarray,
    ridge_pred_test: np.ndarray,
    regions: Dict[str, np.ndarray],
    args: argparse.Namespace,
    out: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    train_resid = train["delta"] - ridge_pred_train
    for mode in args.feature_ablation_modes:
        log(f"Feature ablation mode={mode}")
        rtrain = append_legal_rbsr_features(bundle, train, mode)
        rtest = append_legal_rbsr_features(bundle, test, mode)
        test_for_rbsr = dict(test)
        test_for_rbsr["cond"] = rtest["cond"]
        experts = fit_region_experts(
            bundle, rtrain["cond"], train_resid, regions, max(args.basis_grid),
            args.ridge_alpha, args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
        pred = apply_oracle_region_experts(bundle, ridge_pred_test, test_for_rbsr, experts)
        row = evaluate_prediction(bundle, test, pred, f"RBSR_feature_{mode}_K{max(args.basis_grid)}")
        row["feature_mode"] = mode
        row["extra_feature_dim"] = int(rtest.get("rbsr_extra_feature_dim", 0))
        rows.append(row)
        for coeff_row in coefficient_prediction_rows(experts, rtest["cond"], test["delta"], ridge_pred_test):
            coeff_row["feature_mode"] = mode
            coeff_row["method"] = row["method"]
            rows.append(coeff_row)
    atomic_csv(out / "feature_ablation.csv", rows)
    plot_feature_ablation(rows, out)
    return rows


def run_coefficient_supervised_basis_curve(
    bundle: DatasetBundle,
    train: Dict[str, Any],
    test: Dict[str, Any],
    ridge_pred_train: np.ndarray,
    ridge_pred_test: np.ndarray,
    regions: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    train_resid = train["delta"] - ridge_pred_train
    rtrain = append_legal_rbsr_features(bundle, train, args.rbsr_feature_mode)
    rtest = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
    test_for_rbsr = dict(test)
    test_for_rbsr["cond"] = rtest["cond"]
    for k in args.basis_grid:
        log(f"Coefficient-supervised basis curve k={k}")
        experts = fit_region_experts(
            bundle, rtrain["cond"], train_resid, regions, int(k), args.ridge_alpha,
            "mlp_supervised", args.mlp_edge_weight, args.mlp_normal_weight,
        )
        pred = apply_oracle_region_experts(bundle, ridge_pred_test, test_for_rbsr, experts)
        row = evaluate_prediction(bundle, test, pred, f"rbsr_coeff_supervised_k{k}")
        row["basis"] = int(k)
        row["kind"] = "coefficient_supervised"
        rows.append(row)
    return rows


def run_staged_basis_selection(
    bundle: DatasetBundle,
    train_pairs: Sequence[Tuple[str, str]],
    test: Dict[str, Any],
    ridge_pred_test: np.ndarray,
    regions: Dict[str, np.ndarray],
    args: argparse.Namespace,
    out: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    fit_pairs, val_pairs = split_pairs(train_pairs, args.staged_val_fraction, args.seed + 31)
    ridge_fit, fit_data = train_ridge(bundle, fit_pairs, args.ridge_alpha, args.source_pca_dim)
    fit_pred = ridge_fit.y_scaler.inverse_transform(
        ridge_fit.model.predict(ridge_fit.x_scaler.transform(fit_data["cond"]))
    ).reshape(len(fit_data["pairs"]), -1, 3)
    fit_pred = strict_overwrite(fit_pred, fit_data["delta"], bundle.landmarks)
    fit_resid = fit_data["delta"] - fit_pred
    val_ridge_pred, val_data = predict_ridge(bundle, ridge_fit, val_pairs)
    fit_rbsr = append_legal_rbsr_features(bundle, fit_data, args.rbsr_feature_mode)
    val_rbsr = append_legal_rbsr_features(bundle, val_data, args.rbsr_feature_mode)
    val_for_rbsr = dict(val_data)
    val_for_rbsr["cond"] = val_rbsr["cond"]
    rows: List[Dict[str, Any]] = []
    best_k = int(args.basis_grid[0])
    best_score = float("inf")
    for k in args.basis_grid:
        experts = fit_region_experts(
            bundle, fit_rbsr["cond"], fit_resid, regions, int(k), args.ridge_alpha,
            args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
        pred = apply_oracle_region_experts(bundle, val_ridge_pred, val_for_rbsr, experts)
        row = evaluate_prediction(bundle, val_data, pred, f"staged_val_k{k}")
        row["basis"] = int(k)
        row["kind"] = "staged_validation"
        rows.append(row)
        score = float(row["strict_free_rmse"])
        if score < best_score:
            best_score = score
            best_k = int(k)

    # Refit selected stage on the full training split and evaluate once on test.
    full_ridge, full_train = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
    full_pred = full_ridge.y_scaler.inverse_transform(
        full_ridge.model.predict(full_ridge.x_scaler.transform(full_train["cond"]))
    ).reshape(len(full_train["pairs"]), -1, 3)
    full_pred = strict_overwrite(full_pred, full_train["delta"], bundle.landmarks)
    full_resid = full_train["delta"] - full_pred
    full_rbsr = append_legal_rbsr_features(bundle, full_train, args.rbsr_feature_mode)
    test_rbsr = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
    test_for_rbsr = dict(test)
    test_for_rbsr["cond"] = test_rbsr["cond"]
    experts = fit_region_experts(
        bundle, full_rbsr["cond"], full_resid, regions, best_k, args.ridge_alpha,
        args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
    pred = apply_oracle_region_experts(bundle, ridge_pred_test, test_for_rbsr, experts)
    row = evaluate_prediction(bundle, test, pred, f"RBSR_staged_selected_K{best_k}")
    row["basis"] = best_k
    row["kind"] = "staged_prediction"
    rows.append(row)
    atomic_csv(out / "staged_basis_selection.csv", rows)
    return pred, rows, best_k


def apply_gate_mode(
    bundle: DatasetBundle,
    ridge_pred: np.ndarray,
    rbsr_pred: np.ndarray,
    data_for_rbsr: Dict[str, Any],
    true_data: Dict[str, Any],
    experts: Sequence[RegionExpert],
    gate_mode: str,
    train_cond: Optional[np.ndarray] = None,
    train_ridge_pred: Optional[np.ndarray] = None,
    train_rbsr_pred: Optional[np.ndarray] = None,
    train_true: Optional[np.ndarray] = None,
    alpha: float = 100.0,
) -> np.ndarray:
    if gate_mode == "none":
        return rbsr_pred
    if train_cond is None or train_ridge_pred is None or train_rbsr_pred is None or train_true is None:
        raise ValueError(f"Gate mode {gate_mode} requires gate training data")
    if gate_mode.startswith("global_"):
        mode = gate_mode.replace("global_", "")
        model = fit_global_admission_model(train_cond, train_ridge_pred, train_rbsr_pred, train_true, bundle.free_idx, alpha)
        gates = predict_global_admission_gates(model, data_for_rbsr["cond"], len(experts), mode)
    elif gate_mode.startswith("subunit_"):
        mode = gate_mode.replace("subunit_", "")
        models = fit_admission_models(train_cond, train_ridge_pred, train_rbsr_pred, train_true, experts, alpha)
        gates = predict_admission_gates(models, data_for_rbsr["cond"], mode)
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    return apply_region_experts_with_gates(bundle, ridge_pred, data_for_rbsr, experts, gates)


def run_validation_selected_rbsr(
    bundle: DatasetBundle,
    train_pairs: Sequence[Tuple[str, str]],
    test: Dict[str, Any],
    ridge_pred_test: np.ndarray,
    regions: Dict[str, np.ndarray],
    args: argparse.Namespace,
    out: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    fit_pairs, val_pairs = split_pairs(train_pairs, args.model_select_val_fraction, args.seed + 43)
    ridge_fit, fit_data = train_ridge(bundle, fit_pairs, args.ridge_alpha, args.source_pca_dim)
    fit_ridge_pred = ridge_fit.y_scaler.inverse_transform(
        ridge_fit.model.predict(ridge_fit.x_scaler.transform(fit_data["cond"]))
    ).reshape(len(fit_data["pairs"]), -1, 3)
    fit_ridge_pred = strict_overwrite(fit_ridge_pred, fit_data["delta"], bundle.landmarks)
    fit_resid = fit_data["delta"] - fit_ridge_pred
    val_ridge_pred, val_data = predict_ridge(bundle, ridge_fit, val_pairs)
    fit_rbsr = append_legal_rbsr_features(bundle, fit_data, args.rbsr_feature_mode)
    val_rbsr = append_legal_rbsr_features(bundle, val_data, args.rbsr_feature_mode)
    val_for_rbsr = dict(val_data)
    val_for_rbsr["cond"] = val_rbsr["cond"]
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"score": float("inf"), "basis": None, "predictor": None, "gate_mode": None}
    gate_modes = list(args.model_select_gate_modes)
    for predictor in args.model_select_predictors:
        for k in args.basis_grid:
            log(f"Validation model selection k={k} predictor={predictor}")
            experts = fit_region_experts(
                bundle, fit_rbsr["cond"], fit_resid, regions, int(k), args.ridge_alpha,
                predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
            fit_rbsr_pred = apply_oracle_region_experts(bundle, fit_ridge_pred, dict(fit_data, cond=fit_rbsr["cond"]), experts)
            val_rbsr_pred = apply_oracle_region_experts(bundle, val_ridge_pred, val_for_rbsr, experts)
            for gate_mode in gate_modes:
                pred = apply_gate_mode(
                    bundle, val_ridge_pred, val_rbsr_pred, val_for_rbsr, val_data, experts, gate_mode,
                    train_cond=fit_rbsr["cond"],
                    train_ridge_pred=fit_ridge_pred,
                    train_rbsr_pred=fit_rbsr_pred,
                    train_true=fit_data["delta"],
                    alpha=args.ridge_alpha,
                )
                row = evaluate_prediction(bundle, val_data, pred, f"model_select_k{k}_{predictor}_{gate_mode}")
                row.update({"basis": int(k), "predictor": predictor, "gate_mode": gate_mode, "split": "validation"})
                rows.append(row)
                score = float(row["strict_free_rmse"])
                if score < best["score"]:
                    best = {"score": score, "basis": int(k), "predictor": predictor, "gate_mode": gate_mode}

    # Refit selected configuration on the full train split; evaluate once on held-out test.
    ridge_full, full_train = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
    full_ridge_pred = ridge_full.y_scaler.inverse_transform(
        ridge_full.model.predict(ridge_full.x_scaler.transform(full_train["cond"]))
    ).reshape(len(full_train["pairs"]), -1, 3)
    full_ridge_pred = strict_overwrite(full_ridge_pred, full_train["delta"], bundle.landmarks)
    full_resid = full_train["delta"] - full_ridge_pred
    full_rbsr = append_legal_rbsr_features(bundle, full_train, args.rbsr_feature_mode)
    test_rbsr = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
    test_for_rbsr = dict(test)
    test_for_rbsr["cond"] = test_rbsr["cond"]
    experts = fit_region_experts(
        bundle, full_rbsr["cond"], full_resid, regions, int(best["basis"]), args.ridge_alpha,
        str(best["predictor"]), args.mlp_edge_weight, args.mlp_normal_weight,
        )
    test_rbsr_pred = apply_oracle_region_experts(bundle, ridge_pred_test, test_for_rbsr, experts)
    full_rbsr_pred = apply_oracle_region_experts(bundle, full_ridge_pred, dict(full_train, cond=full_rbsr["cond"]), experts)
    selected_pred = apply_gate_mode(
        bundle, ridge_pred_test, test_rbsr_pred, test_for_rbsr, test, experts, str(best["gate_mode"]),
        train_cond=full_rbsr["cond"],
        train_ridge_pred=full_ridge_pred,
        train_rbsr_pred=full_rbsr_pred,
        train_true=full_train["delta"],
        alpha=args.ridge_alpha,
    )
    test_row = evaluate_prediction(bundle, test, selected_pred, "RBSR_validation_selected")
    test_row.update({"basis": int(best["basis"]), "predictor": best["predictor"], "gate_mode": best["gate_mode"], "split": "test"})
    rows.append(test_row)
    atomic_csv(out / "validation_model_selection.csv", rows)
    atomic_json(out / "validation_selected_config.json", best)
    return selected_pred, rows, best


def run_final_strict_analysis(args: argparse.Namespace, bundle: DatasetBundle, out_root: Path) -> None:
    out = out_root / "08_final_strict_analysis"
    with experiment_guard(out, args.overwrite) as active:
        if not active:
            return
        path, train_pairs, test_pairs, split_audit = resolve_eval_pairs(args, bundle)
        split_meta = read_json(path)
        json_pairs_count = len(load_pairs(path))
        atomic_json(out / "split_audit.json", split_audit)
        ridge, train = train_ridge(bundle, train_pairs, args.ridge_alpha, args.source_pca_dim)
        ridge_pred_train = ridge.y_scaler.inverse_transform(ridge.model.predict(ridge.x_scaler.transform(train["cond"]))).reshape(len(train["pairs"]), -1, 3)
        ridge_pred_train = strict_overwrite(ridge_pred_train, train["delta"], bundle.landmarks)
        ridge_pred_test, test = predict_ridge(bundle, ridge, test_pairs)

        n_vertices = len(bundle.roi_global)
        controls = test["delta"][:, bundle.landmarks, :]
        rigid_pred = np.zeros_like(test["delta"], dtype=np.float32)
        rigid_pred = strict_overwrite(rigid_pred, test["delta"], bundle.landmarks)
        lap = uniform_laplacian_matrix(n_vertices, bundle.roi_faces)
        lap_pred = solve_linear_handle_baseline(controls, bundle.landmarks, n_vertices, lap, args.handle_weight, args.system_ridge)
        lap_pred = strict_overwrite(lap_pred, test["delta"], bundle.landmarks)
        log("Running ARAP baseline")
        arap_pred = arap_predict_vectorised_strict(
            test["source"], controls, bundle.landmarks, bundle.roi_faces, lap, lap_pred,
            args.handle_weight, args.system_ridge, args.arap_iter,
        )
        arap_pred = strict_overwrite(arap_pred, test["delta"], bundle.landmarks)

        atomic_json(out / "root_coverage_audit.json", root_coverage_audit(bundle))
        atomic_json(out / "residual_basis_capacity_audit.json", residual_basis_capacity_audit(bundle, train, ridge_pred_train, args.basis_grid))
        basis_rows, rbsr_preds = evaluate_rbsr_for_basis_grid(
            bundle, train, test, ridge_pred_train, ridge_pred_test, args.basis_grid,
            args.ridge_alpha, args.rbsr_predictor, args.rbsr_feature_mode, args.mlp_edge_weight, args.mlp_normal_weight
        )
        if args.run_coeff_supervised_curve:
            coeff_curve_rows = run_coefficient_supervised_basis_curve(
                bundle, train, test, ridge_pred_train, ridge_pred_test, region_vertex_sets(bundle), args
            )
            basis_rows.extend(coeff_curve_rows)
            atomic_csv(out / "coefficient_supervised_basis_curve.csv", coeff_curve_rows)
            atomic_csv(out / "basis_gap_root_included.csv", basis_rows)
            plot_basis_gap_three_lines(basis_rows, out)
        atomic_csv(out / "basis_gap_root_included.csv", basis_rows)
        plot_basis_gap_three_lines(basis_rows, out)
        rbsr_pred_k = rbsr_preds[int(max(args.basis_grid))][0]
        rbsr_upper_k = rbsr_preds[int(max(args.basis_grid))][1]

        train_resid = train["delta"] - ridge_pred_train
        regions = region_vertex_sets(bundle)
        feature_ablation_rows = run_feature_ablation_matrix(
            bundle, train, test, ridge_pred_train, ridge_pred_test, regions, args, out
        )
        staged_pred, staged_rows, staged_k = run_staged_basis_selection(
            bundle, train_pairs, test, ridge_pred_test, regions, args, out
        )
        selected_pred, selected_rows, selected_config = run_validation_selected_rbsr(
            bundle, train_pairs, test, ridge_pred_test, regions, args, out
        )
        basis_rows.extend([r for r in staged_rows if r.get("kind") == "staged_prediction"])
        for row in selected_rows:
            if row.get("split") == "test":
                row["kind"] = "validation_selected"
                basis_rows.append(row)
        atomic_csv(out / "basis_gap_root_included.csv", basis_rows)
        plot_basis_gap_three_lines(basis_rows, out)
        rbsr_train = append_legal_rbsr_features(bundle, train, args.rbsr_feature_mode)
        rbsr_test = append_legal_rbsr_features(bundle, test, args.rbsr_feature_mode)
        test_for_rbsr = dict(test)
        test_for_rbsr["cond"] = rbsr_test["cond"]
        base_experts = fit_region_experts(
            bundle, rbsr_train["cond"], train_resid, regions, int(max(args.basis_grid)),
            args.ridge_alpha, args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
        atomic_csv(
            out / "coefficient_prediction_diagnostics.csv",
            coefficient_prediction_rows(base_experts, rbsr_test["cond"], test["delta"], ridge_pred_test),
        )
        train_fit_pairs, gate_pairs = split_pairs(train_pairs, args.admission_val_fraction, args.seed + 17)
        gate_ridge, gate_fit = train_ridge(bundle, train_fit_pairs, args.ridge_alpha, args.source_pca_dim)
        gate_fit_pred = gate_ridge.y_scaler.inverse_transform(
            gate_ridge.model.predict(gate_ridge.x_scaler.transform(gate_fit["cond"]))
        ).reshape(len(gate_fit["pairs"]), -1, 3)
        gate_fit_pred = strict_overwrite(gate_fit_pred, gate_fit["delta"], bundle.landmarks)
        gate_fit_resid = gate_fit["delta"] - gate_fit_pred
        gate_fit_rbsr = append_legal_rbsr_features(bundle, gate_fit, args.rbsr_feature_mode)
        gate_experts = fit_region_experts(
            bundle, gate_fit_rbsr["cond"], gate_fit_resid, regions, int(max(args.basis_grid)),
            args.ridge_alpha, args.rbsr_predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
        gate_ridge_pred, gate_data = predict_ridge(bundle, gate_ridge, gate_pairs)
        gate_data_rbsr = append_legal_rbsr_features(bundle, gate_data, args.rbsr_feature_mode)
        gate_data_for_rbsr = dict(gate_data)
        gate_data_for_rbsr["cond"] = gate_data_rbsr["cond"]
        gate_rbsr_pred = apply_oracle_region_experts(bundle, gate_ridge_pred, gate_data_for_rbsr, gate_experts)
        admission_models = fit_admission_models(
            gate_data_rbsr["cond"], gate_ridge_pred, gate_rbsr_pred, gate_data["delta"], gate_experts, args.ridge_alpha
        )
        global_admission_model = fit_global_admission_model(
            gate_data_rbsr["cond"], gate_ridge_pred, gate_rbsr_pred, gate_data["delta"], bundle.free_idx, args.ridge_alpha
        )
        global_hard_gates = predict_global_admission_gates(global_admission_model, rbsr_test["cond"], len(base_experts), "hard")
        global_soft_gates = predict_global_admission_gates(global_admission_model, rbsr_test["cond"], len(base_experts), "soft")
        hard_gates = predict_admission_gates(admission_models, rbsr_test["cond"], "hard")
        soft_gates = predict_admission_gates(admission_models, rbsr_test["cond"], "soft")
        oracle_gates = oracle_admission_gates(ridge_pred_test, rbsr_pred_k, test["delta"], base_experts)
        rbsr_global_hard_admit = apply_region_experts_with_gates(bundle, ridge_pred_test, test_for_rbsr, base_experts, global_hard_gates)
        rbsr_global_soft_admit = apply_region_experts_with_gates(bundle, ridge_pred_test, test_for_rbsr, base_experts, global_soft_gates)
        rbsr_hard_admit = apply_region_experts_with_gates(bundle, ridge_pred_test, test_for_rbsr, base_experts, hard_gates)
        rbsr_soft_admit = apply_region_experts_with_gates(bundle, ridge_pred_test, test_for_rbsr, base_experts, soft_gates)
        rbsr_oracle_admit = apply_region_experts_with_gates(bundle, ridge_pred_test, test_for_rbsr, base_experts, oracle_gates)
        gate_rows: List[Dict[str, Any]] = []
        gate_rows.extend(admission_gate_rows(global_hard_gates, base_experts, "legal_global_hard_admission"))
        gate_rows.extend(admission_gate_rows(global_soft_gates, base_experts, "legal_global_soft_admission"))
        gate_rows.extend(admission_gate_rows(hard_gates, base_experts, "legal_hard_admission"))
        gate_rows.extend(admission_gate_rows(soft_gates, base_experts, "legal_soft_admission"))
        gate_rows.extend(admission_gate_rows(oracle_gates, base_experts, "oracle_admission"))
        vertex_gate_preds: Dict[str, np.ndarray] = {}
        if args.run_vertex_gate:
            vertex_model = fit_vertex_gate_model(
                gate_data_rbsr["cond"], gate_ridge_pred, gate_rbsr_pred, gate_data["delta"], args.ridge_alpha
            )
            for mode in ["hard", "soft"]:
                pred, summary = apply_vertex_gate(
                    bundle, ridge_pred_test, rbsr_pred_k, test["delta"], vertex_model, rbsr_test["cond"], mode
                )
                name = f"RBSR_vertex_{mode}_admission_K{max(args.basis_grid)}_with_root"
                vertex_gate_preds[name] = pred
                summary["method"] = name
                gate_rows.append(summary)
        atomic_csv(out / "admission_gate_summary.csv", gate_rows)

        predictor_rows: List[Dict[str, Any]] = []
        predictor_preds: Dict[str, np.ndarray] = {}
        for predictor in args.coeff_predictors:
            log(f"Coefficient predictor diagnostic: {predictor}")
            experts = fit_region_experts(
                bundle, rbsr_train["cond"], train_resid, regions, int(max(args.basis_grid)),
                args.ridge_alpha, predictor, args.mlp_edge_weight, args.mlp_normal_weight,
        )
            pred = apply_oracle_region_experts(bundle, ridge_pred_test, test_for_rbsr, experts)
            predictor_preds[predictor] = pred
            row = evaluate_prediction(bundle, test, pred, f"RBSR_coeff_{predictor}_K{max(args.basis_grid)}")
            row["coefficient_predictor"] = predictor
            predictor_rows.append(row)
            for coeff_row in coefficient_prediction_rows(experts, rbsr_test["cond"], test["delta"], ridge_pred_test):
                coeff_row["method"] = f"RBSR_coeff_{predictor}_K{max(args.basis_grid)}"
                predictor_rows.append(coeff_row)
        atomic_csv(out / "coefficient_predictor_scan.csv", predictor_rows)

        ridge_aug_pred = None
        ridge_aug_meta: Dict[str, Any] = {}
        if args.rbsr_feature_mode not in {"base", "f0"} or args.run_augmented_baselines:
            ridge_aug_pred, _, ridge_aug_meta = train_predict_ridge_augmented(
                bundle, train_pairs, test_pairs, args.ridge_alpha, args.source_pca_dim, args.rbsr_feature_mode
            )
            atomic_json(out / "ridge_augmented_config.json", ridge_aug_meta)

        device = args.device if args.device != "auto" else ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        cvae_out = out / "best_cvae_z4_w512_L3"
        ensure_dir(cvae_out)
        cvae_fit_pairs, cvae_val_pairs = split_pairs(train_pairs, args.cvae_val_fraction, args.seed + 61)
        atomic_json(out / "cvae_split_audit.json", {
            "early_stopping_used_test_pairs": False,
            "fit_pairs": len(cvae_fit_pairs),
            "validation_pairs": len(cvae_val_pairs),
            "test_pairs": len(test_pairs),
            "cvae_val_fraction": float(args.cvae_val_fraction),
            "test_pair_source": "resolved eval test split only; not used for checkpoint selection",
        })
        train_cvae, val_cvae = cvae_condition_data(bundle, cvae_fit_pairs, cvae_val_pairs, args.source_pca_dim, "base")
        _cvae_val_pred, _cvae_meta, cvae_model, cvae_cond_scaler, cvae_y_scaler = train_cvae_once(
            bundle, train_cvae, val_cvae,
            latent_dim=4, width=512, layers=3, loss_name="roi_landmark",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            out=cvae_out, device=device, return_state=True,
        )
        test_cvae = cvae_eval_data_from_train(bundle, train_cvae, test_pairs, args.source_pca_dim, "base")
        cvae_pred_flat, _ = predict_cvae_arrays(cvae_model, cvae_cond_scaler, cvae_y_scaler, test_cvae["cond"], device)
        cvae_pred = cvae_pred_flat.reshape(len(test_cvae["pairs"]), len(bundle.roi_global), 3)
        cvae_pred = strict_overwrite(cvae_pred, test["delta"], bundle.landmarks)
        cvae_aug_pred = None
        if args.run_cvae_augmented:
            cvae_aug_out = out / f"best_cvae_augmented_{args.rbsr_feature_mode}_z4_w512_L3"
            ensure_dir(cvae_aug_out)
            train_cvae_aug = append_legal_rbsr_features(bundle, train_cvae, args.rbsr_feature_mode)
            val_cvae_aug = append_legal_rbsr_features(bundle, val_cvae, args.rbsr_feature_mode)
            _cvae_aug_val_pred, _cvae_aug_meta, cvae_aug_model, cvae_aug_cond_scaler, cvae_aug_y_scaler = train_cvae_once(
                bundle, train_cvae_aug, val_cvae_aug,
                latent_dim=4, width=512, layers=3, loss_name="roi_landmark",
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                out=cvae_aug_out, device=device, return_state=True,
            )
            test_cvae_aug = append_legal_rbsr_features(bundle, test_cvae, args.rbsr_feature_mode)
            cvae_aug_pred_flat, _ = predict_cvae_arrays(cvae_aug_model, cvae_aug_cond_scaler, cvae_aug_y_scaler, test_cvae_aug["cond"], device)
            cvae_aug_pred = cvae_aug_pred_flat.reshape(len(test_cvae_aug["pairs"]), len(bundle.roi_global), 3)
            cvae_aug_pred = strict_overwrite(cvae_aug_pred, test["delta"], bundle.landmarks)
        hybrid_pred = strict_overwrite((1.0 - args.hybrid_alpha) * ridge_pred_test + args.hybrid_alpha * cvae_pred, test["delta"], bundle.landmarks)

        methods = {
            "Rigid/source_copy_strict": rigid_pred,
            "Laplacian_strict": lap_pred,
            f"ARAP_iter{args.arap_iter}_strict": arap_pred,
            "Ridge_strict": ridge_pred_test,
            "Best_CVAE_z4_w512_L3": cvae_pred,
            f"Hybrid_strict_alpha{args.hybrid_alpha:g}": hybrid_pred,
            f"RBSR_predicted_K{max(args.basis_grid)}_with_root": rbsr_pred_k,
            f"RBSR_global_hard_admission_K{max(args.basis_grid)}_with_root": rbsr_global_hard_admit,
            f"RBSR_global_soft_admission_K{max(args.basis_grid)}_with_root": rbsr_global_soft_admit,
            f"RBSR_legal_hard_admission_K{max(args.basis_grid)}_with_root": rbsr_hard_admit,
            f"RBSR_legal_soft_admission_K{max(args.basis_grid)}_with_root": rbsr_soft_admit,
            f"RBSR_oracle_admission_K{max(args.basis_grid)}_with_root": rbsr_oracle_admit,
            f"RBSR_staged_selected_K{staged_k}_with_root": staged_pred,
            f"RBSR_validation_selected_K{selected_config['basis']}_{selected_config['predictor']}_{selected_config['gate_mode']}_with_root": selected_pred,
            f"RBSR_projection_upper_K{max(args.basis_grid)}_with_root": rbsr_upper_k,
        }
        if ridge_aug_pred is not None:
            methods[f"Ridge_augmented_{args.rbsr_feature_mode}"] = ridge_aug_pred
        if cvae_aug_pred is not None:
            methods[f"Best_CVAE_augmented_{args.rbsr_feature_mode}_z4_w512_L3"] = cvae_aug_pred
        methods.update(vertex_gate_preds)
        table_rows: List[Dict[str, Any]] = []
        per_pair_by_method: Dict[str, List[Dict[str, Any]]] = {}
        for name, pred in methods.items():
            metrics = evaluate_prediction(bundle, test, pred, name)
            table_rows.append(metrics)
            rows = per_pair_metric_rows(bundle, test, pred, name)
            per_pair_by_method[name] = rows
            atomic_csv(out / f"per_pair_{re.sub('[^A-Za-z0-9_]+', '_', name)}.csv", rows)
        atomic_csv(out / "final_strict_table.csv", table_rows)
        atomic_csv(out / "risk_summary.csv", risk_summary_rows(per_pair_by_method))

        ridge_rows = per_pair_by_method["Ridge_strict"]
        rbsr_rows = per_pair_by_method[f"RBSR_predicted_K{max(args.basis_grid)}_with_root"]
        tests = {
            "ridge_minus_rbsr_predicted": paired_bootstrap_and_permutation(
                ridge_rows, rbsr_rows, "strict_free_rmse", args.seed, args.n_boot, args.n_perm
            ),
            "ridge_minus_rbsr_legal_hard_admission": paired_bootstrap_and_permutation(
                ridge_rows,
                per_pair_by_method[f"RBSR_legal_hard_admission_K{max(args.basis_grid)}_with_root"],
                "strict_free_rmse",
                args.seed + 1,
                args.n_boot,
                args.n_perm,
            ),
            "ridge_minus_rbsr_legal_soft_admission": paired_bootstrap_and_permutation(
                ridge_rows,
                per_pair_by_method[f"RBSR_legal_soft_admission_K{max(args.basis_grid)}_with_root"],
                "strict_free_rmse",
                args.seed + 2,
                args.n_boot,
                args.n_perm,
            ),
            "ridge_minus_rbsr_global_hard_admission": paired_bootstrap_and_permutation(
                ridge_rows,
                per_pair_by_method[f"RBSR_global_hard_admission_K{max(args.basis_grid)}_with_root"],
                "strict_free_rmse",
                args.seed + 3,
                args.n_boot,
                args.n_perm,
            ),
            "ridge_minus_rbsr_global_soft_admission": paired_bootstrap_and_permutation(
                ridge_rows,
                per_pair_by_method[f"RBSR_global_soft_admission_K{max(args.basis_grid)}_with_root"],
                "strict_free_rmse",
                args.seed + 4,
                args.n_boot,
                args.n_perm,
            ),
            "ridge_minus_rbsr_staged_selected": paired_bootstrap_and_permutation(
                ridge_rows,
                per_pair_by_method[f"RBSR_staged_selected_K{staged_k}_with_root"],
                "strict_free_rmse",
                args.seed + 5,
                args.n_boot,
                args.n_perm,
            ),
            "ridge_minus_rbsr_validation_selected": paired_bootstrap_and_permutation(
                ridge_rows,
                per_pair_by_method[f"RBSR_validation_selected_K{selected_config['basis']}_{selected_config['predictor']}_{selected_config['gate_mode']}_with_root"],
                "strict_free_rmse",
                args.seed + 8,
                args.n_boot,
                args.n_perm,
            ),
        }
        if args.run_vertex_gate:
            for offset, mode in enumerate(["hard", "soft"], start=6):
                method_name = f"RBSR_vertex_{mode}_admission_K{max(args.basis_grid)}_with_root"
                tests[f"ridge_minus_rbsr_vertex_{mode}_admission"] = paired_bootstrap_and_permutation(
                    ridge_rows,
                    per_pair_by_method[method_name],
                    "strict_free_rmse",
                    args.seed + offset,
                    args.n_boot,
                    args.n_perm,
                )
        atomic_json(out / "paired_tests.json", tests)
        atomic_json(out / "learning_curve_scale_audit.json", {
            "pair_json": str(path),
            "scale": split_meta.get("scale"),
            "n_ids": len(split_meta.get("ids", [])) if isinstance(split_meta, dict) else None,
            "n_pairs_in_json": json_pairs_count,
            "eval_split_mode": args.eval_split_mode,
            "identity_overlap_count": split_audit.get("identity_overlap_count"),
            "n_train_pairs_resolved": len(train_pairs),
            "n_test_pairs_resolved": len(test_pairs),
            "all_ordered_pairs": split_meta.get("all_ordered_pairs") if isinstance(split_meta, dict) else None,
            "budget": split_meta.get("budget") if isinstance(split_meta, dict) else None,
            "pair_budget_policy": split_meta.get("pair_budget_policy") if isinstance(split_meta, dict) else None,
            "note": "With manifest_identity, training pairs come from the train-pool pair manifest and test pairs come from main-test identities; pair_count learning curve changes train pair count explicitly.",
        })
        atomic_json(out / "summary.json", {
            "pair_json": str(path),
            "n_train_pairs": len(train["pairs"]),
            "n_test_pairs": len(test["pairs"]),
            "eval_split_mode": args.eval_split_mode,
            "identity_overlap_count": split_audit.get("identity_overlap_count"),
            "regions": {k: int(len(v)) for k, v in region_vertex_sets(bundle).items()},
            "root_coverage_audit": root_coverage_audit(bundle),
            "hybrid_alpha": args.hybrid_alpha,
            "basis_grid": [int(x) for x in args.basis_grid],
            "rbsr_predictor": args.rbsr_predictor,
            "coeff_predictors": list(args.coeff_predictors),
            "rbsr_feature_mode": args.rbsr_feature_mode,
            "feature_ablation_modes": list(args.feature_ablation_modes),
            "staged_selected_k": int(staged_k),
            "validation_selected_config": selected_config,
            "mlp_edge_weight": float(args.mlp_edge_weight),
            "mlp_normal_weight": float(args.mlp_normal_weight),
            "run_vertex_gate": bool(args.run_vertex_gate),
            "run_augmented_baselines": bool(args.run_augmented_baselines),
            "run_cvae_augmented": bool(args.run_cvae_augmented),
            "paired_tests": tests,
        })


def run_all(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.out)
    atomic_json(args.out / "run_config.json", vars(args) | {"started_at": now()})
    bundle = load_bundle(args.root, args.landmarks)
    atomic_json(args.out / "strict_task_definition.json", {
        "roi_vertices": int(len(bundle.roi_global)),
        "landmarks": [int(x) for x in bundle.landmarks.tolist()],
        "free_vertices": int(len(bundle.free_idx)),
        "definition": "9 landmark vertices are hard-overwritten from ground truth controls; primary RMSE is evaluated only on ROI minus landmarks.",
        "subunits": {k: len(v) for k, v in bundle.subunits.items()},
        "has_roi_faces": bundle.roi_faces is not None,
    })
    selected = set(args.experiments)
    if "all" in selected or "preflight" in selected:
        run_preflight(args, bundle, args.out)
    if "all" in selected or "residual_heatmap" in selected:
        run_residual_heatmap(args, bundle, args.out)
    if "all" in selected or "oracle_region_routing" in selected:
        run_region_oracle(args, bundle, args.out, args.default_basis)
    if "all" in selected or "basis_sweep" in selected:
        run_basis_sweep(args, bundle, args.out)
    if "all" in selected or "cvae_capacity" in selected:
        run_cvae_grid(args, bundle, args.out, "capacity")
    if "all" in selected or "cvae_loss_matrix" in selected:
        run_cvae_grid(args, bundle, args.out, "loss")
    if "all" in selected or "rbsr_learning_curve" in selected:
        run_rbsr_learning_curve(args, bundle, args.out)
    if "all" in selected or "cvae_oracle_inputs" in selected:
        run_cvae_grid(args, bundle, args.out, "oracle")
    if "all" in selected or "final_strict_analysis" in selected:
        run_final_strict_analysis(args, bundle, args.out)
    atomic_json(args.out / "RUN_COMPLETE.json", {"completed_at": now()})


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def parse_str_list(text: str) -> List[str]:
    return [str(x).strip() for x in str(text).split(",") if str(x).strip()]


def parse_landmarks(text: str) -> Any:
    text = str(text).strip()
    if text.lower() in {"auto", "metadata", "landmarks_27_35"}:
        return "auto"
    return parse_int_list(text)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path, default=Path("outputs/strict_sparse_dense_diagnostics"))
    p.add_argument("--experiments", nargs="+", default=["all"], choices=[
        "all",
        "preflight",
        "residual_heatmap",
        "oracle_region_routing",
        "basis_sweep",
        "cvae_capacity",
        "cvae_loss_matrix",
        "rbsr_learning_curve",
        "cvae_oracle_inputs",
        "final_strict_analysis",
    ])
    p.add_argument("--split", default="primary")
    p.add_argument("--chain", type=int, default=0)
    p.add_argument("--scale", type=int, default=676)
    p.add_argument("--lc-scales", type=parse_int_list, default=LC_SCALES)
    p.add_argument("--lc-train-pairs", type=parse_int_list, default=LC_TRAIN_PAIRS)
    p.add_argument("--lc-mode", choices=["pair_count", "identity_scale"], default="pair_count")
    p.add_argument("--landmarks", type=parse_landmarks, default="auto")
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument(
        "--eval-split-mode",
        choices=["manifest_identity", "identity_disjoint", "pair_level", "use_json_splits"],
        default="manifest_identity",
        help="manifest_identity matches the legacy CVAE/train.py identity-disjoint split; pair_level is diagnostic only.",
    )
    p.add_argument(
        "--max-manifest-test-pairs",
        type=int,
        default=0,
        help="Optional cap for manifest_identity main-test ordered pairs. 0 uses all main-test pairs.",
    )
    p.add_argument("--ridge-alpha", type=float, default=100.0)
    p.add_argument("--source-pca-dim", type=int, default=16)
    p.add_argument("--basis-grid", type=parse_int_list, default=BASIS_GRID)
    p.add_argument("--rbsr-predictor", choices=["ridge", "kernel_rbf", "extra_trees", "mlp", "mlp_supervised"], default="ridge")
    p.add_argument("--coeff-predictors", type=parse_str_list, default=["ridge", "kernel_rbf", "extra_trees"])
    p.add_argument("--rbsr-feature-mode", choices=["base", "f0", "deformation", "f1", "local_shape", "f2", "local_shape_deformation", "f3"], default="base")
    p.add_argument("--feature-ablation-modes", type=parse_str_list, default=["f0", "f1", "f2", "f3"])
    p.add_argument("--run-coeff-supervised-curve", action="store_true")
    p.add_argument("--run-vertex-gate", action="store_true")
    p.add_argument("--run-augmented-baselines", action="store_true")
    p.add_argument("--run-cvae-augmented", action="store_true")
    p.add_argument("--mlp-edge-weight", type=float, default=0.05)
    p.add_argument("--mlp-normal-weight", type=float, default=0.02)
    p.add_argument("--admission-val-fraction", type=float, default=0.25)
    p.add_argument("--staged-val-fraction", type=float, default=0.25)
    p.add_argument("--model-select-val-fraction", type=float, default=0.25)
    p.add_argument("--model-select-predictors", type=parse_str_list, default=["ridge", "extra_trees"])
    p.add_argument("--model-select-gate-modes", type=parse_str_list, default=["none", "global_soft", "subunit_soft"])
    p.add_argument("--latent-grid", type=parse_int_list, default=LATENT_GRID)
    p.add_argument("--width-grid", type=parse_int_list, default=[128, 256, 512])
    p.add_argument("--layer-grid", type=parse_int_list, default=[2, 3])
    p.add_argument("--default-latent", type=int, default=16)
    p.add_argument("--default-width", type=int, default=256)
    p.add_argument("--default-layers", type=int, default=2)
    p.add_argument("--default-basis", type=int, default=16)
    p.add_argument("--cvae-val-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--handle-weight", type=float, default=1000.0)
    p.add_argument("--system-ridge", type=float, default=1e-8)
    p.add_argument("--arap-iter", type=int, default=3)
    p.add_argument("--hybrid-alpha", type=float, default=0.3)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.root = args.root.resolve()
    args.out = (args.root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    log(f"Strict diagnostics root={args.root}")
    log(f"Output dir={args.out}")
    run_all(args)


if __name__ == "__main__":
    main()
