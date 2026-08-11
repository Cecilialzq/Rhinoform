from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rhinoform.data import load_rows
from rhinoform.repro import SEED_REGISTRY, artifact_metadata, atomic_write_json


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    return path


def dense_path_for_run(plan: dict, run_id: str) -> Path:
    run = next((r for r in plan.get("runs", []) if r.get("run_id") == run_id), None)
    if run is None:
        raise FileNotFoundError(f"missing cache: execution-plan run {run_id}")
    out_dir = Path(run["out_dir"])
    summaries = sorted(out_dir.glob("neural_field_summary_*.json"))
    if not summaries:
        raise FileNotFoundError(f"missing cache: {out_dir}/neural_field_summary_*.json")
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    dense = summary.get("dense_prediction_npz")
    if not dense:
        raise FileNotFoundError(f"missing cache: dense_prediction_npz in {summaries[-1]}")
    return require(Path(dense))


def pair_lookup(test_pairs: np.ndarray) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in test_pairs.tolist()]


def compute_scale(
    plan: dict,
    by_id: dict[str, dict],
    scale: int,
    chains: int,
    out_dir: Path,
    method_key: str,
    chunk_size: int,
) -> dict:
    dense_paths = [dense_path_for_run(plan, f"required_primary_chain{chain}_scale{scale}") for chain in range(chains)]
    first = np.load(dense_paths[0], allow_pickle=True)
    pairs = pair_lookup(first["test_pairs"])
    n_pairs = len(pairs)
    n_vertices = first[method_key].shape[1] // 3
    std_sum = np.zeros(n_vertices, dtype=np.float64)
    counted_pairs = 0
    for start in range(0, n_pairs, chunk_size):
        stop = min(n_pairs, start + chunk_size)
        chain_errors = []
        true_chunk = np.empty((stop - start, n_vertices, 3), dtype=np.float32)
        for j, (src_id, tgt_id) in enumerate(pairs[start:stop]):
            true_chunk[j] = by_id[tgt_id]["vertices"] - by_id[src_id]["vertices"]
        for dense_path in dense_paths:
            data = np.load(dense_path, allow_pickle=True)
            if pair_lookup(data["test_pairs"]) != pairs:
                raise ValueError(f"dense test-pair order mismatch: {dense_path}")
            if method_key not in data.files:
                raise FileNotFoundError(f"missing cache: {method_key} in {dense_path}")
            pred = data[method_key][start:stop].reshape(stop - start, n_vertices, 3)
            chain_errors.append(np.linalg.norm(pred - true_chunk, axis=2).astype(np.float32))
        chunk_std = np.std(np.stack(chain_errors, axis=0), axis=0, ddof=1)
        std_sum += chunk_std.sum(axis=0)
        counted_pairs += stop - start
    per_vertex_std = std_sum / max(1, counted_pairs)

    template_id = pairs[0][0]
    vertices = by_id[template_id]["vertices"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"sensitivity_scale{scale}.npz"
    np.savez_compressed(
        npz,
        per_vertex_error_std=per_vertex_std.astype(np.float32),
        template_vertices=vertices.astype(np.float32),
        dense_prediction_npzs=np.asarray([str(p) for p in dense_paths]),
        scale=np.asarray([scale], dtype=np.int32),
        seed=np.asarray([SEED_REGISTRY["figure_case_seed"]], dtype=np.int64),
    )
    order = np.argsort(per_vertex_std)[::-1][:50]
    csv_path = out_dir / f"sensitivity_scale{scale}_top_vertices.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "local_vertex_index", "per_vertex_error_std"])
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow({"rank": rank, "local_vertex_index": int(idx), "per_vertex_error_std": float(per_vertex_std[idx])})

    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    sc = ax.scatter(vertices[:, 0], vertices[:, 1], c=per_vertex_std, s=2.0, cmap="magma")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="error std across chains")
    fig.tight_layout(pad=0)
    png = out_dir / f"fig_seed_subset_sensitivity_scale{scale}.png"
    pdf = out_dir / f"fig_seed_subset_sensitivity_scale{scale}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight", pad_inches=0)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
    return {
        "scale": scale,
        "dense_prediction_npzs": [str(p) for p in dense_paths],
        "npz": str(npz),
        "csv": str(csv_path),
        "png": str(png),
        "pdf": str(pdf),
        "n_pairs": n_pairs,
        "n_vertices": int(n_vertices),
        "method_key": method_key,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="execution_plan.json")
    parser.add_argument("--repo", default="data")
    parser.add_argument("--out", default="frozen_artifacts/sensitivity")
    parser.add_argument("--scales", default="30,676")
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--method-key", default="hybrid_selected_pred")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    plan = json.loads(require(Path(args.plan)).read_text(encoding="utf-8"))
    _, by_id = load_rows(Path(args.repo))
    out_dir = Path(args.out)
    outputs = [
        compute_scale(plan, by_id, int(scale), args.chains, out_dir, args.method_key, args.chunk_size)
        for scale in args.scales.split(",")
        if scale.strip()
    ]
    report = {
        "outputs": outputs,
        "description": "Seed/subset sensitivity map: per-vertex error standard deviation across Required-tier primary chains.",
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=SEED_REGISTRY["figure_case_seed"],
            data_root_identifier=str(args.repo),
        ),
    }
    atomic_write_json(out_dir / "seed_subset_sensitivity_manifest.json", report)
    print(json.dumps({"out": str(out_dir), "n_outputs": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
