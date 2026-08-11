"""Export the frozen ridge linear operator for the browser demo engine.

The reported ridge anchor predicts a dense nasal-ROI displacement field from a
standardized condition vector ``cond = [control_displacement(27) | source_pca_code(16)]``:

    dense_delta = ((cond - cond_mean) / cond_std) @ W + b

For a *fixed* source face the source-PCA code is constant, so while a user drags
the nine nasal landmarks only the first 27 condition entries change. The visible
ROI response to an edit is therefore an exact linear function of the raw control
displacement (relative to the no-edit prediction):

    visible_delta(ctrl) = ctrl @ W_ctrl ,   W_ctrl = W[:27] / cond_std[:27]

This script writes ``W_ctrl`` (the genuine frozen operator) plus the training
control scale (for an out-of-distribution proxy) to ``demo/public/engine.json``
so the browser runs real ridge inference, not a re-fit approximation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model-package",
        type=Path,
        default=Path(
            "plan_a_clean_retrain/bases/primary_chain0_scale676/"
            "neural_field_model_package_cvae_ew0p1_lw0.pt"
        ),
    )
    parser.add_argument("--demo-case", type=Path, default=Path("demo/public/demo_case.json"))
    parser.add_argument("--out", type=Path, default=Path("demo/public/engine.json"))
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()

    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    w, b = base["ridge_cond"]
    w = np.asarray(w, dtype=np.float64)            # (43, 11802)
    b = np.asarray(b, dtype=np.float64).reshape(-1)  # (11802,)
    cond_std = np.asarray(base["cond_std"], dtype=np.float64).reshape(-1)  # (43,)

    n_ctrl = 27
    roi_dim = w.shape[1]
    n_vertices = roi_dim // 3

    # Genuine frozen operator: raw control displacement -> dense ROI delta.
    w_ctrl = w[:n_ctrl, :] / cond_std[:n_ctrl, None]          # (27, 11802)
    ctrl_std = cond_std[:n_ctrl]                              # training control scale

    demo_case = json.loads(args.demo_case.read_text(encoding="utf-8"))
    landmarks = demo_case["geometry"]["landmarks"]

    payload = {
        "meta": {
            "engine": "frozen ridge/PCA statistical anchor",
            "package": str(args.base_model_package),
            "source_id": demo_case["case"]["source_id"],
            "scale": "full23 clean split, scale676 chain0",
            "cond_dim": int(w.shape[0]),
            "n_control": n_ctrl,
            "n_vertices": int(n_vertices),
            "note": (
                "W_ctrl maps a raw 9-landmark (27-d) control displacement to the dense "
                "nasal-ROI displacement field via the exact frozen ridge weights. "
                "visible_delta(ctrl) = ctrl @ W_ctrl."
            ),
        },
        "landmarks": landmarks,
        "control_dim": n_ctrl,
        "vertex_dim": int(roi_dim),
        # row-major (27 x 11802) flattened
        "w_ctrl": np.round(w_ctrl, args.decimals).reshape(-1).tolist(),
        "ctrl_std": np.round(ctrl_std, args.decimals).tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(
        json.dumps(
            {
                "out": str(args.out),
                "w_ctrl_shape": [n_ctrl, roi_dim],
                "n_vertices": int(n_vertices),
                "file_mb": round(size_mb, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
