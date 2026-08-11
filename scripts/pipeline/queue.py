from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from rhinoform.repro import atomic_write_json, sha256_file


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def status_path(run: dict) -> Path:
    return Path(run["out_dir"]) / "status.json"


def mark_status(run: dict, state: str, **extra) -> None:
    path = status_path(run)
    current = load_json(path, {})
    current.update(
        {
            "run_id": run["run_id"],
            "state": state,
            "updated_utc": now(),
            "out_dir": run["out_dir"],
            "seed": run["seed"],
            "pair_manifest": run["pair_manifest"],
            "pair_manifest_sha256": run.get("pair_manifest_sha256"),
        }
    )
    current.update(extra)
    atomic_write_json(path, current)


def completed(run: dict) -> bool:
    st = load_json(status_path(run), {})
    if st.get("state") != "done":
        return False
    out_dir = Path(run["out_dir"])
    has_summary = bool(list(out_dir.glob("neural_field_summary_*.json")))
    has_package = bool(list(out_dir.glob("neural_field_model_package_*.pt")))
    has_pairs = bool(list((out_dir / "pair_metrics").glob("identity_bootstrap_pair_metrics_*.csv")))
    if run.get("dense_prediction_npz_required") and not bool(list(out_dir.glob("neural_field_predictions_*.npz"))):
        return False
    return has_summary and has_package and has_pairs


def dataset_ready(plan: dict) -> tuple[bool, str]:
    manifest = Path(plan["dataset_manifest"])
    done_path = Path(plan.get("dataset_done", manifest.parent / "DATASET_DONE.json"))
    split_path = Path(plan.get("split_manifest", "splits/facescape_847/split_manifest.json"))
    if not manifest.exists():
        return False, "blocked_missing_dataset_manifest"
    if not done_path.exists():
        return False, "blocked_missing_dataset_done"
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "blocked_invalid_dataset_done"
    if done.get("decision") != "pass":
        return False, "blocked_dataset_done_not_pass"
    if split_path.exists() and done.get("split_manifest_sha256") != sha256_file(split_path):
        return False, "blocked_dataset_done_split_manifest_mismatch"
    try:
        manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False, "blocked_invalid_dataset_manifest"
    if int(done.get("n_rows", -1)) != int(manifest_obj.get("n_rows", -2)):
        return False, "blocked_dataset_done_manifest_row_mismatch"
    if done.get("failed"):
        return False, "blocked_dataset_done_contains_failures"
    return True, "pending"


def split_ready(plan: dict) -> tuple[bool, str]:
    split_path = Path(plan.get("split_manifest", "splits/facescape_847/split_manifest.json"))
    if not split_path.exists():
        return False, "blocked_missing_split_manifest"
    try:
        obj = json.loads(split_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "blocked_invalid_split_manifest"
    if int(obj.get("expected_usable", -1)) != 846 or int(obj.get("usable_count", -1)) != 846:
        return False, "blocked_split_manifest_usable_count_mismatch"
    sizes = obj.get("sizes", {})
    if {k: int(sizes.get(k, -1)) for k in ("test", "val", "train_pool")} != {"test": 100, "val": 70, "train_pool": 676}:
        return False, "blocked_split_manifest_size_mismatch"
    qc = obj.get("identity_qc", {})
    if int(qc.get("passed_count", -1)) != 846:
        return False, "blocked_split_manifest_qc_mismatch"
    if int(obj.get("pair_manifest_counts", {}).get("required", -1)) != int(plan.get("run_count", 23)):
        return False, "blocked_required_pair_manifest_count_mismatch"
    if len(obj.get("required_pair_manifests", [])) != int(plan.get("run_count", 23)):
        return False, "blocked_required_pair_manifest_list_mismatch"
    return True, "pending"


def refresh_plan_status(plan: dict) -> dict:
    ready, dataset_status = dataset_ready(plan)
    split_ok, split_status = split_ready(plan) if ready else (False, "pending")
    for run in plan["runs"]:
        if completed(run):
            run["status"] = "done"
        elif not ready:
            run["status"] = dataset_status
        elif not split_ok:
            run["status"] = split_status
        elif not Path(run["pair_manifest"]).exists():
            run["status"] = "blocked_missing_pair_manifest"
        else:
            st = load_json(status_path(run), {})
            run["status"] = st.get("state", "pending")
    plan["blocked_count"] = sum(1 for r in plan["runs"] if str(r["status"]).startswith("blocked"))
    plan["done_count"] = sum(1 for r in plan["runs"] if r["status"] == "done")
    plan["failed_count"] = sum(1 for r in plan["runs"] if r["status"] == "failed")
    return plan


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def run_one(run: dict, env: dict[str, str], dry_run: bool) -> int:
    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(x) for x in run["command"]]
    if dry_run:
        mark_status(run, "pending", dry_run_command=command)
        return 0
    mark_status(run, "running", started_utc=now(), command=command)
    log_path = out_dir / "log.txt"
    append_log(log_path, f"\n\n## {now()} START {' '.join(command)}\n")
    print(f"\n=== RUN {run['run_id']} START {now()} ===", flush=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(f"[{run['run_id']}] {line}")
            sys.stdout.flush()
        returncode = proc.wait()
    elapsed = time.time() - started
    print(f"=== RUN {run['run_id']} END rc={returncode} elapsed={elapsed/60:.1f} min {now()} ===", flush=True)
    if returncode == 0:
        artifacts = []
        for pattern in ("neural_field_summary_*.json", "neural_field_predictions_*.npz", "neural_field_metadata_*.json", "neural_field_model_package_*.pt", "pair_metrics/*.csv"):
            for path in sorted(out_dir.glob(pattern)):
                artifacts.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        mark_status(run, "done", completed_utc=now(), artifacts=artifacts)
    else:
        mark_status(run, "failed", completed_utc=now(), returncode=returncode, log=str(log_path))
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="execution_plan.json")
    parser.add_argument("--ledger", default="progress_ledger.json")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run until queue is exhausted.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan = refresh_plan_status(load_json(plan_path, {}))
    atomic_write_json(plan_path, plan)
    if plan.get("blocked_count", 0):
        ledger = {
            "state": "blocked",
            "updated_utc": now(),
            "reason": "one or more runs are blocked; inspect execution_plan.json",
            "blocked_count": plan["blocked_count"],
            "done_count": plan.get("done_count", 0),
            "failed_count": plan.get("failed_count", 0),
        }
        atomic_write_json(Path(args.ledger), ledger)
        print(json.dumps(ledger, indent=2))
        return 2

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "20260609")
    ran = 0
    failures = 0
    for run in plan["runs"]:
        if completed(run):
            continue
        if args.max_runs and ran >= args.max_runs:
            break
        try:
            rc = run_one(run, env, args.dry_run)
            ran += 1
            if rc != 0:
                failures += 1
                if not args.continue_on_failure:
                    break
        except Exception:
            failures += 1
            mark_status(run, "failed", traceback=traceback.format_exc())
            if not args.continue_on_failure:
                break
        plan = refresh_plan_status(plan)
        atomic_write_json(plan_path, plan)
        atomic_write_json(
            Path(args.ledger),
            {
                "state": "running" if failures == 0 else "running_with_failures",
                "updated_utc": now(),
                "ran_this_invocation": ran,
                "done_count": plan.get("done_count", 0),
                "failed_count": plan.get("failed_count", 0),
                "blocked_count": plan.get("blocked_count", 0),
            },
        )

    final_plan = refresh_plan_status(load_json(plan_path, plan))
    state = "done" if final_plan.get("done_count") == final_plan.get("run_count") else "incomplete"
    ledger = {
        "state": state if failures == 0 else "incomplete_with_failures",
        "updated_utc": now(),
        "ran_this_invocation": ran,
        "done_count": final_plan.get("done_count", 0),
        "failed_count": final_plan.get("failed_count", 0),
        "blocked_count": final_plan.get("blocked_count", 0),
    }
    atomic_write_json(Path(args.ledger), ledger)
    print(json.dumps(ledger, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
