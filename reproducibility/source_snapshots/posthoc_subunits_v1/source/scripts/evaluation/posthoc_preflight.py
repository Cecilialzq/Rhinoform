"""Read-only preflight for the final-rerun post-hoc Colab workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Executing this file by path normally puts ``scripts/evaluation`` at
# sys.path[0].  An external checkout (notably official LAMM) may also expose a
# top-level ``scripts`` package through PYTHONPATH.  Force this repository to
# the front before importing any Rhinoform project packages below.
REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_string = str(REPO_ROOT)
while repo_root_string in sys.path:
    sys.path.remove(repo_root_string)
sys.path.insert(0, repo_root_string)


EXPECTED_MESHES = 846
EXPECTED_TEST_PAIRS = 9900
EXPECTED_VALIDATION_PAIRS = 4830
LAMM_COMMIT = "87354c05dec341c6d8dd319665dd52553fb03084"
LAMM_RUNTIME_MODULES = ("trimesh", "yaml", "einops", "timm")

# The original one-shot freeze remains immutable. These four files were
# subsequently extended while constructing the explicitly secondary post-hoc
# workflow. Bind that second implementation chain exactly instead of pretending
# it is byte-identical to the one-shot chain. Active inference is additionally
# required to pass the existing zero-noise/canonical replay before outputs are
# accepted; the two evaluation entry points below are not invoked by post-hoc.
KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES = {
    "rhinoform/rbsr_calibration.py":
        "74ca40f3b17ccdb7a671c0699060dd96d20703d010b6dfb4530557fdfba3bd0d",
    "rhinoform/train_rbsr_gate.py":
        "de0c08676d31d9a70416b32f5fe0e762d5b162b5db847c14528b979ece15111a",
    "scripts/evaluation/rbsr_gate.py":
        "55eed7e0b641b5d8ecf0ce5c366f614ec4b0776231bfda9e409ea2abd4a93b7d",
    "scripts/evaluation/rbsr_ridge_fold_projection.py":
        "004ed117c0d85ed6fa134ae6f75aa3901a12f925f267c003ccbd0ad828da5c1d",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_facescape_data_root(
    candidates: list[Path], *, expected_manifest_sha256: str
) -> Path:
    """Return the first complete licensed cache matching the public manifest."""
    inspected: list[str] = []
    for candidate in candidates:
        root = Path(candidate).expanduser()
        manifest = root / "manifest.json"
        meshes = root / "meshes"
        count = sum(1 for _ in meshes.glob("*.npz")) if meshes.is_dir() else 0
        manifest_ok = manifest.is_file() and _sha256(manifest) == expected_manifest_sha256
        inspected.append(f"{root}: meshes={count}, manifest_match={manifest_ok}")
        if count == EXPECTED_MESHES and manifest_ok:
            return root.resolve()
    raise RuntimeError(
        "Could not find 846 processed FaceScape meshes with the repository manifest. "
        "Set RHINOFORM_FACESCAPE_DATA_ROOT to your licensed data directory. Inspected: "
        + "; ".join(inspected)
    )


def validate_implementation_lineage(
    frozen_hashes: dict[str, str], actual_hashes: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Validate the immutable one-shot chain plus the exact post-hoc chain."""
    drifts: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    for relative, frozen_hash in frozen_hashes.items():
        actual_hash = actual_hashes.get(relative, "MISSING")
        if actual_hash == frozen_hash:
            continue
        approved_hash = KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES.get(relative)
        if approved_hash is None or actual_hash != approved_hash:
            failures.append(
                f"{relative}: frozen={frozen_hash}, approved={approved_hash}, actual={actual_hash}"
            )
            continue
        drifts[relative] = {
            "original_frozen_sha256": frozen_hash,
            "posthoc_approved_sha256": actual_hash,
        }
    if failures:
        raise RuntimeError("Unexpected implementation lineage: " + "; ".join(failures))
    return drifts


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_lamm_runtime_modules() -> dict[str, str]:
    """Import every dependency reached by the pinned LAMM inference graph."""
    loaded: dict[str, str] = {}
    failures: list[str] = []
    for name in LAMM_RUNTIME_MODULES:
        try:
            module = importlib.import_module(name)
            loaded[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # preserve the complete one-shot diagnosis
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "LAMM inference runtime is incomplete; rerun the notebook dependency "
            "bootstrap cell. Failures: " + "; ".join(failures)
        )
    return loaded


def _pair_keys(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or not {"source_id", "target_id"} <= set(rows.fieldnames):
            raise RuntimeError(f"Pair columns missing from canonical table: {path}")
        return [(str(row["source_id"]), str(row["target_id"])) for row in rows]


def _validate_table(path: Path, expected: list[tuple[str, str]], require_valid) -> None:
    require_valid(path, "canonical pair table")
    observed = _pair_keys(path)
    if observed != expected:
        raise RuntimeError(
            f"Canonical pair panel/order mismatch: {path} "
            f"({len(observed)} rows, expected {len(expected)})"
        )


def _validate_existing_resume(output: Path, valid_sha256_sidecar) -> tuple[int, int]:
    if not output.exists():
        return 0, 0
    protocol = output / "POSTHOC_SUBUNIT_ANALYSIS_PROTOCOL_FREEZE.json"
    if protocol.exists() and not valid_sha256_sidecar(protocol):
        raise RuntimeError(f"Existing post-hoc protocol or sidecar is invalid: {protocol}")
    resumable = partial = 0
    for path in output.rglob("*"):
        if not path.is_file() or path.name.endswith(".sha256.json"):
            continue
        if path.suffix.lower() not in {".csv", ".json", ".npz", ".pt", ".pdf", ".png"}:
            continue
        if valid_sha256_sidecar(path):
            resumable += 1
        else:
            partial += 1
    return resumable, partial


def run_preflight(args: argparse.Namespace) -> None:
    # Prevent imports of the pinned checkout from creating untracked pyc files.
    sys.dont_write_bytecode = True
    repo_root = args.repo_root.resolve()
    data_root = args.repo.resolve()
    results = args.results.resolve()
    output = args.out.resolve()
    lamm_root = args.lamm_root.resolve()
    expected_output = results / "posthoc_subunit_analysis_v1"
    if output != expected_output:
        raise RuntimeError(f"Unsafe output target: {output}; expected {expected_output}")
    if not os.access(results, os.R_OK | os.W_OK):
        raise RuntimeError(f"Results directory is not readable/writable: {results}")

    for module in ("numpy", "scipy", "torch", "sklearn", "pandas", "matplotlib", "pytest"):
        __import__(module)
    import torch
    lamm_runtime_versions = validate_lamm_runtime_modules()

    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before running the post-hoc notebook")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU smoke only"

    canonical_manifest = repo_root / "data/manifest.json"
    if not canonical_manifest.is_file():
        raise RuntimeError(f"Repository manifest missing: {canonical_manifest}")
    selected = discover_facescape_data_root(
        [data_root], expected_manifest_sha256=_sha256(canonical_manifest)
    )
    if selected != data_root:
        raise RuntimeError(f"Resolved data-root mismatch: {selected} != {data_root}")

    if not (lamm_root / ".git").is_dir():
        raise RuntimeError(f"Official LAMM checkout is missing: {lamm_root}")
    if _git(lamm_root, "rev-parse", "HEAD") != LAMM_COMMIT:
        raise RuntimeError("Official LAMM checkout is not at the pinned commit")
    if _git(lamm_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Official LAMM checkout is dirty")

    from rhinoform.data import load_rows
    from rhinoform.repro import sha256_file, valid_sha256_sidecar
    from scripts.evaluation.posthoc_strict_noise_robustness import (
        prepare_lamm,
        prepare_neural,
    )
    from scripts.evaluation.posthoc_subunit_analysis import Context, require_valid

    context_args = argparse.Namespace(
        repo_root=repo_root,
        repo=data_root,
        results=results,
        out=output,
        lamm_root=lamm_root,
        chunk_pairs=160,
        neural_batch_size=32,
        lamm_batch_size=32,
        seed=20260609,
        n_boot=10000,
    )
    context = Context(context_args)
    if len(context.pairs) != EXPECTED_TEST_PAIRS:
        raise RuntimeError(f"Expected 9900 frozen test pairs, got {len(context.pairs)}")
    validation_pairs = [(str(a), str(b)) for a, b in context.base["val_pairs"]]
    if len(validation_pairs) != EXPECTED_VALIDATION_PAIRS:
        raise RuntimeError(f"Expected 4830 frozen validation pairs, got {len(validation_pairs)}")
    if "ridge_ctrl" not in context.base:
        raise RuntimeError("Frozen base package lacks controls-only Ridge weights")

    # Preserve the immutable original chain and bind the exact, separately
    # versioned post-freeze evaluation chain. Any unrecognised fifth drift fails.
    actual_implementation_hashes = {}
    for relative in context.freeze["implementation_hashes"]:
        path = repo_root / relative
        actual_implementation_hashes[relative] = (
            sha256_file(path) if path.is_file() else "MISSING"
        )
    implementation_drifts = validate_implementation_lineage(
        context.freeze["implementation_hashes"], actual_implementation_hashes
    )
    for relative, hashes in implementation_drifts.items():
        print(
            "POSTHOC DUAL HASH CHAIN:", relative,
            hashes["original_frozen_sha256"], "->",
            hashes["posthoc_approved_sha256"],
            flush=True,
        )
    for name in (
        "RBSR_TEST_ACCESS_RECEIPT.json",
        "LAMM_TEST_ACCESS_RECEIPT.json",
        "CLASSICAL_TEST_ACCESS_RECEIPT.json",
    ):
        path = results / name
        require_valid(path, "completed one-shot receipt")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "COMPLETE" or receipt.get("test_access_count") != 1:
            raise RuntimeError(f"One-shot receipt is not COMPLETE exactly once: {path}")

    one_shot = results / "rbsr/seed20260609/one_shot_test"
    lamm_out = results / "lamm/seed20260609"
    test_tables = [
        one_shot / "pair_metrics_ridge_sourcepca_clean_test.csv",
        one_shot / "pair_metrics_rbsr_test.csv",
        one_shot / "pair_metrics_cvae_clean_test.csv",
        one_shot / "pair_metrics_hybrid_validation_selected_clean_test.csv",
        one_shot / "projection_diagnostics_test.csv",
        lamm_out / "identity_bootstrap_pair_metrics_lamm.csv",
        results / "classical/identity_bootstrap_pair_metrics_laplacian.csv",
        results / "classical/identity_bootstrap_pair_metrics_bilaplacian.csv",
        results / "classical/identity_bootstrap_pair_metrics_arap.csv",
    ]
    for path in test_tables:
        _validate_table(path, context.pairs, require_valid)

    validation_raw = results / "rbsr/seed20260609/validation_raw"
    projection_metrics = results / (
        "rbsr/seed20260609/certified_projection_validation/pair_metrics"
    )
    validation_tables = [
        validation_raw / "pair_metrics_ridge_sourcepca_clean_validation.csv",
        validation_raw / "pair_metrics_rbsr_validation.csv",
        validation_raw / "pair_metrics_cvae_clean_validation.csv",
        validation_raw / "pair_metrics_hybrid_validation_selected_clean_validation.csv",
        projection_metrics / "pair_metrics_ridge_validation.csv",
        projection_metrics / "pair_metrics_attenuation_0p75_validation.csv",
    ]
    for path in validation_tables:
        _validate_table(path, validation_pairs, require_valid)

    # Load the exact frozen neural and official-LAMM architectures on CPU now,
    # so incompatible checkpoints/configurations fail before long evaluation.
    required_ids = set(map(str, context.base["train_ids"]))
    required_ids.update(map(str, context.base["val_ids"]))
    required_ids.update(map(str, context.base["test_ids"]))
    _, all_by_id = load_rows(data_root, allowed_ids=required_ids)
    neural = prepare_neural(context, all_by_id, torch.device("cpu"))
    del neural
    lamm_model, _, _ = prepare_lamm(context, torch.device("cpu"))
    del lamm_model
    if _git(lamm_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("LAMM checkout became dirty during read-only preflight")

    resumable, partial = _validate_existing_resume(output, valid_sha256_sidecar)
    local_root = Path("/content") if Path("/content").is_dir() else repo_root
    free_gib = shutil.disk_usage(local_root).free / (1024 ** 3)
    if free_gib < 5:
        raise RuntimeError(f"Insufficient local free space: {free_gib:.2f} GiB; require at least 5 GiB")
    print(
        "POSTHOC PREFLIGHT PASS | "
        f"data={data_root} meshes={EXPECTED_MESHES} test_pairs=9900 "
        f"validation_pairs=4830 gpu={gpu!r} free_gib={free_gib:.1f} "
        f"lamm_runtime={lamm_runtime_versions!r} "
        f"valid_resume_artifacts={resumable} partial_recomputable={partial}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lamm-root", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    run_preflight(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
