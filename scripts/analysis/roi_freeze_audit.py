from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rhinoform.data import load_rows
from rhinoform.repro import SEED_REGISTRY, artifact_metadata, atomic_write_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="data")
    parser.add_argument("--roi", default="roi/vertices.json")
    parser.add_argument("--subunits", default="roi/subunits.json")
    parser.add_argument("--out", default="frozen_artifacts/roi_freeze_audit")
    parser.add_argument("--n-identities", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED_REGISTRY["global_seed"])
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    repo = Path(args.repo)
    rows, by_id = load_rows(repo)
    ids = sorted(by_id, key=lambda s: int(s) if s.isdigit() else 10**9)
    if len(ids) < args.n_identities:
        raise ValueError(f"Need at least {args.n_identities} identities for ROI audit, found {len(ids)}")
    rng = np.random.default_rng(args.seed)
    selected = [ids[int(i)] for i in sorted(rng.choice(len(ids), size=args.n_identities, replace=False))]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    fig, axes = plt.subplots(2, 5, figsize=(9.0, 3.8), sharex=True, sharey=True)
    for ax, sid in zip(axes.ravel(), selected):
        rec = by_id[sid]
        vertices = rec["vertices"]
        faces = rec["faces"]
        ax.scatter(vertices[:, 0], vertices[:, 1], s=0.55, c=vertices[:, 2], cmap="viridis")
        ax.set_title(str(sid), fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        selected_rows.append(
            {
                "subject_id": sid,
                "roi_vertex_count": int(vertices.shape[0]),
                "roi_face_count": int(faces.shape[0]),
                "landmark_count": int(rec["landmarks"].size),
                "subunit_counts": {name: int(vals.size) for name, vals in rec["subunits"].items()},
            }
        )
    fig.tight_layout(pad=0.3)
    overview = out_dir / "roi_freeze_audit_overview.png"
    fig.savefig(overview, dpi=220, bbox_inches="tight", pad_inches=0.02)

    report = {
        "decision": "manual_visual_confirmation_required",
        "protocol": "Seed-sample ten identities from the frozen ROI cache; render low-resolution XY nasal crop overviews; no OBJ/PLY crops are exported.",
        "seed": args.seed,
        "n_identities": args.n_identities,
        "selected": selected_rows,
        "overview_figure": str(overview),
        "roi_sha256": sha256_file(Path(args.roi)),
        "subunits_sha256": sha256_file(Path(args.subunits)),
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            input_manifest_path=repo / "manifest.json",
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=args.seed,
            data_root_identifier=str(repo),
        ),
    }
    atomic_write_json(out_dir / "roi_freeze_audit.json", report)
    print(json.dumps({"out": str(out_dir), "selected": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
