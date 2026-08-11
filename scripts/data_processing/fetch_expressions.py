from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rhinoform.repro import atomic_write_json, sha256_file


DEFAULT_EXPRESSIONS = {
    "smile": "2_smile.obj",
    "anger": "4_anger.obj",
    "lip_puckerer": "12_lip_puckerer.obj",
    "lip_funneler": "13_lip_funneler.obj",
    "cheek_blowing": "17_cheek_blowing.obj",
    "brow_raiser": "19_brow_raiser.obj",
}


def parse_expression_args(values: list[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_EXPRESSIONS)
    expressions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expression must use label=filename syntax: {value!r}")
        label, filename = value.split("=", 1)
        label = label.strip()
        filename = filename.strip()
        if not label or not filename.endswith(".obj"):
            raise ValueError(f"Invalid expression mapping: {value!r}")
        expressions[label] = filename
    return expressions


def member_subject_id(member_name: str) -> str | None:
    parts = member_name.replace("\\", "/").split("/")
    if len(parts) < 3 or parts[-2] != "models_reg":
        return None
    return parts[-3]


def crop_obj_vertices(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    roi_indices: np.ndarray,
    expected_vertex_count: int,
) -> np.ndarray:
    lookup = {int(global_index): local_index for local_index, global_index in enumerate(roi_indices)}
    cropped = np.empty((len(roi_indices), 3), dtype=np.float32)
    seen = np.zeros(len(roi_indices), dtype=bool)
    vertex_index = 0
    with archive.open(member) as source:
        for raw_line in source:
            if not raw_line.startswith(b"v "):
                continue
            local_index = lookup.get(vertex_index)
            if local_index is not None:
                fields = raw_line.split()
                cropped[local_index] = (float(fields[1]), float(fields[2]), float(fields[3]))
                seen[local_index] = True
            vertex_index += 1
    if vertex_index != expected_vertex_count:
        raise ValueError(f"{member.filename}: {vertex_index} vertices, expected {expected_vertex_count}")
    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise ValueError(f"{member.filename}: failed to read {len(missing)} ROI vertices")
    if not np.isfinite(cropped).all():
        raise ValueError(f"{member.filename}: non-finite ROI coordinates")
    return cropped


def resolve_neutral_npz(repo: Path, row: dict) -> Path:
    path = Path(row["npz_path"])
    return path if path.is_absolute() else repo / path


def extract_archive(
    archive_index: int,
    archive_count: int,
    zip_path: Path,
    members: list[tuple[str, str, zipfile.ZipInfo]],
    output_meshes: Path,
    output_root: Path,
    rows_by_id: dict[str, dict],
    roi_indices: np.ndarray,
    expected_vertex_count: int,
    faces: np.ndarray,
    local_landmarks: np.ndarray,
    landmark_ids: np.ndarray,
    subunits: dict[str, np.ndarray],
    force: bool,
) -> tuple[list[dict], list[dict]]:
    print(f"[{archive_index}/{archive_count}] {zip_path.name}: {len(members)} ROI meshes", flush=True)
    rows: list[dict] = []
    failures: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive:
        for subject_id, label, indexed_member in members:
            destination = output_meshes / f"{subject_id}_{label}.npz"
            if destination.exists() and not force:
                rows.append(
                    {
                        "npz_path": str(destination.relative_to(output_root)),
                        "subject_id": subject_id,
                        "expression": label,
                        "split": str(rows_by_id[subject_id]["split"]),
                        "roi_count": int(len(roi_indices)),
                        "vertex_count": expected_vertex_count,
                        "sha256": sha256_file(destination),
                        "source_zip": zip_path.name,
                        "source_member": indexed_member.filename,
                        "source_crc": int(indexed_member.CRC),
                    }
                )
                continue
            try:
                member = archive.getinfo(indexed_member.filename)
                vertices = crop_obj_vertices(archive, member, roi_indices, expected_vertex_count)
                np.savez_compressed(
                    destination,
                    vertices=vertices,
                    faces=faces,
                    local_landmarks=local_landmarks,
                    landmark_ids=landmark_ids,
                    roi_indices_full=roi_indices.astype(np.int32),
                    subunit_root=subunits["root"],
                    subunit_dorsum=subunits["dorsum"],
                    subunit_tip=subunits["tip"],
                    subunit_alar_left=subunits["alar_left"],
                    subunit_alar_right=subunits["alar_right"],
                )
                rows.append(
                    {
                        "npz_path": str(destination.relative_to(output_root)),
                        "subject_id": subject_id,
                        "expression": label,
                        "split": str(rows_by_id[subject_id]["split"]),
                        "roi_count": int(len(roi_indices)),
                        "vertex_count": expected_vertex_count,
                        "sha256": sha256_file(destination),
                        "source_zip": zip_path.name,
                        "source_member": member.filename,
                        "source_crc": int(member.CRC),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "subject_id": subject_id,
                        "expression": label,
                        "source_zip": zip_path.name,
                        "source_member": indexed_member.filename,
                        "reason": repr(exc),
                    }
                )
    return rows, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream selected registered FaceScape expressions from the original zip files and "
            "save only the frozen nasal ROI. Full OBJ files are never staged on disk."
        )
    )
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--neutral-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["clean-prior validation", "main test"],
        help="Neutral-manifest split labels to extract.",
    )
    parser.add_argument(
        "--expression",
        action="append",
        default=[],
        help="Repeat label=registered_obj_filename. Defaults to six predeclared expressions.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Archives to extract concurrently.")
    parser.add_argument(
        "--archive-name",
        action="append",
        default=[],
        help="Optional exact archive filename filter. Repeat to process multiple archives.",
    )
    args = parser.parse_args()

    expressions = parse_expression_args(args.expression)
    neutral_manifest_path = args.neutral_repo / "manifest.json"
    neutral_manifest = json.loads(neutral_manifest_path.read_text(encoding="utf-8"))
    selected_rows = [row for row in neutral_manifest["rows"] if str(row["split"]) in set(args.splits)]
    if not selected_rows:
        raise ValueError(f"No neutral rows matched splits: {args.splits}")
    rows_by_id = {str(row["subject_id"]): row for row in selected_rows}
    expected_ids = set(rows_by_id)

    template_path = resolve_neutral_npz(args.neutral_repo, selected_rows[0])
    template = np.load(template_path, allow_pickle=False)
    roi_indices = np.asarray(template["roi_indices_full"], dtype=np.int64)
    faces = np.asarray(template["faces"], dtype=np.int32)
    local_landmarks = np.asarray(template["local_landmarks"], dtype=np.int32)
    landmark_ids = np.asarray(template["landmark_ids"], dtype=np.int32)
    subunits = {
        name: np.asarray(template[f"subunit_{name}"], dtype=np.int32)
        for name in ("root", "dorsum", "tip", "alar_left", "alar_right")
    }
    expected_vertex_count = int(neutral_manifest["topology_vertex_count"])

    zip_paths = sorted(args.zip_dir.glob("facescape_trainset_*.zip"))
    if args.archive_name:
        selected_archives = set(args.archive_name)
        zip_paths = [path for path in zip_paths if path.name in selected_archives]
    if not zip_paths:
        raise FileNotFoundError(f"No FaceScape trainset zip files found in {args.zip_dir}")

    filename_to_label = {filename: label for label, filename in expressions.items()}
    indexed: dict[Path, list[tuple[str, str, zipfile.ZipInfo]]] = defaultdict(list)
    available: dict[str, set[str]] = {label: set() for label in expressions}
    print(f"Indexing {len(zip_paths)} archives for {len(expressions)} expressions", flush=True)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                filename = Path(member.filename.replace("\\", "/")).name
                label = filename_to_label.get(filename)
                if label is None:
                    continue
                subject_id = member_subject_id(member.filename)
                if subject_id not in expected_ids:
                    continue
                indexed[zip_path].append((subject_id, label, member))
                available[label].add(subject_id)

    missing = {
        label: sorted(expected_ids - ids, key=lambda value: int(value))
        for label, ids in available.items()
        if expected_ids - ids
    }
    if missing and not args.allow_missing:
        details = "; ".join(f"{label}: {ids[:20]}" for label, ids in missing.items())
        raise RuntimeError(f"Missing requested expression meshes. Use --allow-missing to continue. {details}")

    output_meshes = args.out / "meshes"
    output_meshes.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    failures: list[dict] = []
    archive_jobs = [
        (archive_index, zip_path, indexed.get(zip_path, []))
        for archive_index, zip_path in enumerate(zip_paths, 1)
        if indexed.get(zip_path, [])
    ]
    worker_count = max(1, min(args.workers, len(archive_jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                extract_archive,
                archive_index,
                len(zip_paths),
                zip_path,
                members,
                output_meshes,
                args.out,
                rows_by_id,
                roi_indices,
                expected_vertex_count,
                faces,
                local_landmarks,
                landmark_ids,
                subunits,
                args.force,
            ): zip_path
            for archive_index, zip_path, members in archive_jobs
        }
        for future in as_completed(futures):
            rows, archive_failures = future.result()
            output_rows.extend(rows)
            failures.extend(archive_failures)

    output_rows.sort(key=lambda row: (int(str(row["subject_id"])), str(row["expression"])))
    manifest = {
        "source": "fetch_expressions.py",
        "neutral_manifest": str(neutral_manifest_path),
        "neutral_manifest_sha256": sha256_file(neutral_manifest_path),
        "requested_splits": args.splits,
        "expressions": expressions,
        "expected_subject_count": len(expected_ids),
        "missing": missing,
        "failures": failures,
        "topology_vertex_count": expected_vertex_count,
        "roi_vertex_count": int(len(roi_indices)),
        "n_rows": len(output_rows),
        "rows": output_rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out / "manifest.json", manifest)
    done = {
        "decision": "fail" if failures or (missing and not args.allow_missing) else "pass",
        "manifest_sha256": sha256_file(args.out / "manifest.json"),
        "expected_subject_count": len(expected_ids),
        "expression_counts": {
            label: sum(1 for row in output_rows if row["expression"] == label)
            for label in expressions
        },
        "missing": missing,
        "failures": failures,
    }
    atomic_write_json(args.out / "EXPRESSIONS_DONE.json", done)
    print(json.dumps(done, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
