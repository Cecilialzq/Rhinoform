from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    command: list[str]
    outputs: tuple[str, ...]


class Reporter:
    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._file.close()

    def line(self, message: str, *, timestamp: bool = True) -> None:
        prefix = f"[{utc_now()}] " if timestamp else ""
        text = prefix + message
        with self._lock:
            print(text, flush=True)
            self._file.write(text + "\n")

    def child_line(self, line: str) -> None:
        text = line.rstrip("\n")
        with self._lock:
            print(text, flush=True)
            self._file.write(text + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_run_paths(plan: dict, root: Path) -> tuple[Path, Path, Path, Path]:
    final_id = plan["final_model_run_id"]
    run = next(r for r in plan["runs"] if r["run_id"] == final_id)
    out_dir = root / run["out_dir"]
    packages = sorted(out_dir.glob("neural_field_model_package_*.pt"))
    hybrids = sorted((out_dir / "pair_metrics").glob("identity_bootstrap_pair_metrics_hybrid_alpha_*.csv"))
    if not packages:
        raise FileNotFoundError(f"No final model package found in {out_dir}")
    if not hybrids:
        raise FileNotFoundError(f"No hybrid pair-metrics CSV found in {out_dir / 'pair_metrics'}")
    ridge = out_dir / "pair_metrics" / "identity_bootstrap_pair_metrics_ridge_sourcepca.csv"
    if not ridge.exists():
        raise FileNotFoundError(f"Missing ridge pair-metrics CSV: {ridge}")
    return out_dir, packages[-1], ridge, hybrids[-1]


def build_stages(args: argparse.Namespace, plan: dict, root: Path) -> list[Stage]:
    py = sys.executable
    final_out_dir, package, ridge_csv, hybrid_csv = final_run_paths(plan, root)
    summary_paths = sorted(final_out_dir.glob("neural_field_summary_*.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No final neural-field summary found in {final_out_dir}")
    summary = json.loads(summary_paths[-1].read_text(encoding="utf-8"))
    op_candidates = summary["validation_operating_point_candidates"]

    stages: list[Stage] = []
    if not args.skip_geometric:
        stages.append(Stage(
            "geometric", "Tune geometric baselines",
            [py, "scripts/training/tune_baselines.py", "--repo", args.repo, "--out", "results/geometric_tuning", "--split-manifest", args.split_manifest],
            ("results/geometric_tuning",),
        ))
    stages.append(Stage(
        "learning_curve", "Aggregate learning curves and bootstrap intervals",
        [py, "scripts/evaluation/learning_curve.py", "--plan", args.plan, "--out", "frozen_artifacts/learning_curve", "--n-boot", str(args.n_boot), "--split-manifest", args.split_manifest],
        ("frozen_artifacts/learning_curve",),
    ))
    if not args.skip_noise:
        stages.append(Stage(
            "noise", "Evaluate noise robustness",
            [py, "scripts/evaluation/noise_robustness.py", "--repo", args.repo, "--model-package", str(package), "--out", "results/noise_robustness", "--geometric-tuning", "results/geometric_tuning/geometric_validation_tuning.json", "--split-manifest", args.split_manifest],
            ("results/noise_robustness",),
        ))
    stages.extend([
        Stage(
            "dependence", "Run dependence-aware bootstrap checks",
            [py, "scripts/evaluation/dependence.py", "--baseline-csv", str(ridge_csv), "--method-csv", str(hybrid_csv), "--baseline-name", "ridge_sourcepca", "--method-name", "hybrid_topscale_chain0", "--out", "results/dependence", "--n-boot", str(args.n_boot), "--split-manifest", args.split_manifest],
            ("results/dependence",),
        ),
        Stage(
            "operating_points", "Freeze operating points",
            [py, "scripts/analysis/operating_points.py", "--candidates", op_candidates, "--out", "frozen_artifacts/operating_points", "--split-manifest", args.split_manifest],
            ("frozen_artifacts/operating_points",),
        ),
        Stage(
            "qualitative_cases", "Freeze qualitative cases",
            [py, "scripts/analysis/qualitative_cases.py", "--pair-metrics", str(hybrid_csv), "--out", "frozen_artifacts/qualitative_cases", "--split-manifest", args.split_manifest],
            ("frozen_artifacts/qualitative_cases",),
        ),
        Stage(
            "roi_audit", "Audit the frozen ROI",
            [py, "scripts/analysis/roi_freeze_audit.py", "--repo", args.repo, "--out", "frozen_artifacts/roi_freeze_audit", "--split-manifest", args.split_manifest],
            ("frozen_artifacts/roi_freeze_audit",),
        ),
        Stage(
            "sensitivity", "Run seed and subset sensitivity checks",
            [py, "scripts/analysis/seed_sensitivity.py", "--plan", args.plan, "--repo", args.repo, "--out", "frozen_artifacts/sensitivity", "--split-manifest", args.split_manifest],
            ("frozen_artifacts/sensitivity",),
        ),
        Stage(
            "manifest", "Write reproducibility manifest",
            [py, "tools/manifest.py", "--root", ".", "--out", "reproducibility_manifest.json", "--split-manifest", args.split_manifest],
            ("reproducibility_manifest.json",),
        ),
        Stage(
            "reproducibility", "Run frozen-statistics reproducibility checks",
            [py, "legacy_pre_strict/tools/repro_check.py", "--mode", "frozen-statistics", "--root", ".", "--out", "outputs/reproducibility"],
            ("outputs/reproducibility",),
        ),
        Stage(
            "freeze_audit", "Run final required-tier freeze audit",
            [py, "tools/freeze_gate.py", "--root", ".", "--out", "REQUIRED_TIER_FREEZE_AUDIT.json", "--write-complete"],
            ("REQUIRED_TIER_FREEZE_AUDIT.json",),
        ),
    ])
    return stages


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def detect_backup_root(root: Path, explicit: str, disabled: bool) -> Path | None:
    if disabled:
        return None
    configured = explicit or os.environ.get("POSTPROCESS_BACKUP_DIR", "")
    if configured:
        backup_root = Path(configured).expanduser().resolve()
        return None if backup_root == root.resolve() else backup_root
    drive_roots = [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]
    for drive_root in drive_roots:
        if not drive_root.exists():
            continue
        if path_is_within(root, drive_root):
            return None
        return drive_root / root.name
    return None


def output_exists(root: Path, output: str) -> bool:
    path = root / output
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def sync_to_backup(
    root: Path,
    backup_root: Path | None,
    paths: list[Path],
    reporter: Reporter,
) -> None:
    if backup_root is None:
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    unique: dict[str, Path] = {}
    for path in paths:
        if path.exists():
            unique[str(path.resolve())] = path
    if not unique:
        return
    reporter.line(f"BACKUP start -> {backup_root} ({len(unique)} path(s))")
    for source in unique.values():
        try:
            relative = source.resolve().relative_to(root.resolve())
        except ValueError:
            relative = Path("external_artifacts") / source.name
        copy_path(source, backup_root / relative)
    reporter.line(f"BACKUP complete -> {backup_root}")


def new_state(signature: str, stages: list[Stage]) -> dict:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_signature": signature,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "pending",
        "stages": {
            stage.key: {
                "label": stage.label,
                "command": stage.command,
                "outputs": list(stage.outputs),
                "status": "pending",
            }
            for stage in stages
        },
    }


def load_state(path: Path, signature: str, stages: list[Stage]) -> dict:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("schema_version") == STATE_SCHEMA_VERSION and state.get("run_signature") == signature:
                for stage in stages:
                    state.setdefault("stages", {}).setdefault(stage.key, new_state(signature, [stage])["stages"][stage.key])
                return state
        except (OSError, json.JSONDecodeError):
            pass
    return new_state(signature, stages)


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def estimate_remaining(state: dict, remaining_stages: int) -> float | None:
    durations = [
        float(item["duration_seconds"])
        for item in state.get("stages", {}).values()
        if item.get("status") == "complete" and item.get("duration_seconds") is not None
    ]
    if not durations:
        return None
    return (sum(durations) / len(durations)) * remaining_stages


def run_stage(
    stage: Stage,
    root: Path,
    reporter: Reporter,
    heartbeat_seconds: float,
    progress_prefix: str,
    eta_seconds: float | None,
) -> float:
    reporter.line(f"{progress_prefix} START {stage.label} | ETA {format_duration(eta_seconds)}")
    reporter.line("COMMAND " + " ".join(stage.command))
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "20260609")
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    process = subprocess.Popen(
        stage.command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    def pump_output() -> None:
        for line in process.stdout:
            reporter.child_line(line)

    pump = threading.Thread(target=pump_output, daemon=True)
    pump.start()
    try:
        while True:
            try:
                return_code = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                reporter.line(f"{progress_prefix} RUNNING {stage.label} | stage elapsed {format_duration(elapsed)}")
    except KeyboardInterrupt:
        reporter.line(f"{progress_prefix} INTERRUPT requested; stopping child process")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        pump.join(timeout=5)
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, stage.command)
    reporter.line(f"{progress_prefix} COMPLETE {stage.label} | stage elapsed {format_duration(elapsed)}")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-safe required-tier post-processing with progress and Drive backups.")
    parser.add_argument("--plan", default="execution_plan.json")
    parser.add_argument("--repo", default="data")
    parser.add_argument("--split-manifest", default="splits/facescape_847/split_manifest.json")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-geometric", action="store_true")
    parser.add_argument("--state-file", default="outputs/postprocess_required_state.json")
    parser.add_argument("--log-file", default="outputs/postprocess_required.log")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--backup-dir", default="", help="Drive project root used for autosave; defaults to /content/drive/MyDrive/<project> on Colab.")
    parser.add_argument("--no-auto-backup", action="store_true", help="Disable automatic Drive backup detection.")
    parser.add_argument("--rerun-completed", action="store_true", help="Run every stage even when the state file marks it complete.")
    args = parser.parse_args()
    if args.n_boot <= 0:
        parser.error("--n-boot must be positive")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")

    plan_path = Path(args.plan).expanduser().resolve()
    root = plan_path.parent
    args.plan = os.path.relpath(plan_path, root)
    repo_path = Path(args.repo).expanduser()
    split_path = Path(args.split_manifest).expanduser()
    args.repo = str(repo_path if repo_path.is_absolute() else Path(args.repo))
    args.split_manifest = str(split_path if split_path.is_absolute() else Path(args.split_manifest))
    state_path = root / args.state_file
    log_path = root / args.log_file
    reporter = Reporter(log_path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        stages = build_stages(args, plan, root)
        signature_payload = {
            "plan_sha256": file_hash(plan_path),
            "split_sha256": file_hash(root / args.split_manifest),
            "n_boot": args.n_boot,
            "skip_noise": args.skip_noise,
            "skip_geometric": args.skip_geometric,
            "stage_commands": [
                [part.replace(str(root), "<PROJECT_ROOT>") for part in stage.command]
                for stage in stages
            ],
        }
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
        state = load_state(state_path, signature, stages)
        backup_root = detect_backup_root(root, args.backup_dir, args.no_auto_backup)

        reporter.line("=" * 72, timestamp=False)
        reporter.line(f"POSTPROCESS start | root={root} | stages={len(stages)} | n_boot={args.n_boot}")
        if backup_root is None:
            if path_is_within(root, Path("/content/drive")):
                reporter.line("PERSISTENCE project is already inside mounted Drive; outputs are written directly to Drive")
            else:
                reporter.line("PERSISTENCE no mounted Colab Drive detected; state and outputs remain in the project directory")
        else:
            reporter.line(f"PERSISTENCE completed stages will autosave to {backup_root}")
            sync_to_backup(root, backup_root, [Path(__file__).resolve(), state_path, log_path], reporter)

        total = len(stages)
        completed_before = 0
        for stage in stages:
            item = state["stages"][stage.key]
            outputs_valid = all(output_exists(root, output) for output in stage.outputs)
            if not args.rerun_completed and item.get("status") == "complete" and outputs_valid:
                completed_before += 1
                percent = 100.0 * completed_before / total
                reporter.line(f"[{completed_before:02d}/{total:02d} {percent:5.1f}%] SKIP {stage.label} | restored from state")
        reporter.line(f"RESUME {completed_before}/{total} stages already complete")

        state["status"] = "running"
        state["last_started_at"] = utc_now()
        save_state(state_path, state)
        run_started = time.monotonic()
        completed = 0
        for index, stage in enumerate(stages, start=1):
            item = state["stages"][stage.key]
            outputs_valid = all(output_exists(root, output) for output in stage.outputs)
            if not args.rerun_completed and item.get("status") == "complete" and outputs_valid:
                completed += 1
                continue

            remaining = total - completed
            eta = estimate_remaining(state, remaining)
            percent_before = 100.0 * completed / total
            prefix = f"[{index:02d}/{total:02d} {percent_before:5.1f}%]"
            item.update({"status": "running", "started_at": utc_now(), "error": None})
            state["current_stage"] = stage.key
            save_state(state_path, state)
            sync_to_backup(root, backup_root, [state_path, log_path], reporter)
            try:
                duration = run_stage(stage, root, reporter, args.heartbeat_seconds, prefix, eta)
            except KeyboardInterrupt:
                item.update({"status": "interrupted", "interrupted_at": utc_now()})
                state["status"] = "interrupted"
                save_state(state_path, state)
                sync_to_backup(root, backup_root, [state_path, log_path] + [root / output for output in stage.outputs], reporter)
                reporter.line("POSTPROCESS interrupted. Re-run the same command to resume from completed stages.")
                return 130
            except Exception as exc:
                item.update({"status": "failed", "failed_at": utc_now(), "error": repr(exc)})
                state["status"] = "failed"
                save_state(state_path, state)
                sync_to_backup(root, backup_root, [state_path, log_path] + [root / output for output in stage.outputs], reporter)
                reporter.line(f"POSTPROCESS failed in stage '{stage.label}': {exc}")
                return 1

            item.update({"status": "complete", "completed_at": utc_now(), "duration_seconds": round(duration, 3), "error": None})
            completed += 1
            state["completed_stages"] = completed
            save_state(state_path, state)
            sync_to_backup(
                root,
                backup_root,
                [state_path, log_path, Path(__file__).resolve()] + [root / output for output in stage.outputs],
                reporter,
            )
            percent_after = 100.0 * completed / total
            reporter.line(f"PROGRESS {completed}/{total} stages complete ({percent_after:.1f}%)")

        state["status"] = "complete"
        state["current_stage"] = None
        state["completed_at"] = utc_now()
        state["duration_seconds"] = round(time.monotonic() - run_started, 3)
        save_state(state_path, state)
        sync_to_backup(root, backup_root, [state_path, log_path, root / "REQUIRED_TIER_FREEZE_AUDIT.json"], reporter)
        reporter.line(f"POSTPROCESS complete | elapsed this invocation {format_duration(state['duration_seconds'])}")
        return 0
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
