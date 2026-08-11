from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhinoform.data import edge_index, face_normals, load_rows, rmse_vertices
from rhinoform.repro import artifact_metadata, atomic_write_json, sha256_file


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def current_subunits(repo: Path) -> dict[str, np.ndarray]:
    _, by_id = load_rows(repo)
    return next(iter(by_id.values()))["subunits"]


def recompute_rows(by_id: dict, pairs: list[tuple[str, str]], pred_flat: np.ndarray, method: str, subunit_version: str) -> list[dict]:
    template = next(iter(by_id.values()))
    subunits = template["subunits"]
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)
    rows = []
    for i, (src_id, tgt_id) in enumerate(pairs):
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        true_delta = tgt - src
        pred_delta = pred_flat[i].reshape(-1, 3)
        pred_vertices = src + pred_delta
        e0 = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
        e1 = np.linalg.norm(pred_vertices[edges[:, 0]] - pred_vertices[edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)
        n0 = face_normals(src, faces)
        n1 = face_normals(pred_vertices, faces)
        row = {
            "pair_index": i,
            "source_id": src_id,
            "target_id": tgt_id,
            "method": method,
            "subunit_mask_version": subunit_version,
            "roi_rmse": rmse_vertices(pred_delta, true_delta),
            "landmark_rmse": rmse_vertices(pred_delta, true_delta, landmarks),
            "dorsum_rmse": rmse_vertices(pred_delta, true_delta, subunits["dorsum"]),
            "tip_rmse": rmse_vertices(pred_delta, true_delta, subunits["tip"]),
            "alar_left_rmse": rmse_vertices(pred_delta, true_delta, subunits["alar_left"]),
            "alar_right_rmse": rmse_vertices(pred_delta, true_delta, subunits["alar_right"]),
            "edge_strain_p95": float(np.percentile(strain, 95)),
            "normal_flip_pct": float(np.mean(np.sum(n0 * n1, axis=1) < 0.0) * 100.0),
        }
        rows.append(row)
    return rows


def method_arrays(npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    out = {}
    if "ridge_cond_pred" in npz:
        out["ridge"] = np.asarray(npz["ridge_cond_pred"])
    if "cvae_pred" in npz:
        out["cvae"] = np.asarray(npz["cvae_pred"])
    if "hybrid_selected_pred" in npz:
        out["hybrid"] = np.asarray(npz["hybrid_selected_pred"])
    for key in npz.files:
        if key.endswith("_pred") and key not in {"ridge_cond_pred", "cvae_pred", "hybrid_selected_pred"}:
            out[key.removesuffix("_pred")] = np.asarray(npz[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="data")
    parser.add_argument("--pred-npz", required=True)
    parser.add_argument("--subunits", default="roi/subunits.json")
    parser.add_argument("--out", default="frozen_artifacts/subunit_reversion")
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    repo = Path(args.repo)
    _, by_id = load_rows(repo)
    pred_path = Path(args.pred_npz)
    if not pred_path.exists():
        raise FileNotFoundError(f"missing cache: {pred_path}")
    pred = np.load(pred_path, allow_pickle=True)
    pairs = [(str(a), str(b)) for a, b in pred["test_pairs"].tolist()]
    subunit_version = sha256_file(Path(args.subunits))
    out_dir = Path(args.out)
    outputs = []
    for method, arr in method_arrays(pred).items():
        rows = recompute_rows(by_id, pairs, arr, method, subunit_version)
        path = out_dir / f"subunit_recomputed_{method}.csv"
        write_csv(path, rows)
        outputs.append({"method": method, "path": str(path), "sha256": sha256_file(path), "n_pairs": len(rows)})
    report = {
        "pred_npz": str(pred_path),
        "pred_npz_sha256": sha256_file(pred_path),
        "subunits": args.subunits,
        "subunit_mask_version": subunit_version,
        "outputs": outputs,
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            input_manifest_path=repo / "manifest.json",
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=None,
            data_root_identifier=repo.name,
        ),
    }
    atomic_write_json(out_dir / "subunit_reversion_manifest.json", report)
    print(json.dumps({"out": str(out_dir), "outputs": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

