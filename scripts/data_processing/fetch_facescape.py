from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
import os

from rhinoform.repro import atomic_write_json, sha256_file


NEUTRAL_SUFFIX = "models_reg/1_neutral.obj"


def has_unzipped_layout(root: Path) -> bool:
    for sid in ("1", "001", "847"):
        if (root / sid / "models_reg" / "1_neutral.obj").exists():
            return True
    return any(root.glob("*/models_reg/1_neutral.obj"))


def count_neutrals(root: Path) -> int:
    return sum(1 for _ in root.glob("*/models_reg/1_neutral.obj"))


def zip_files(root: Path) -> list[Path]:
    # Prefer the 9 FaceScape trainset zips; ignore extras (e.g. publishable_nomasaic_tex.zip).
    trainset = sorted(root.glob("facescape_trainset_*.zip"))
    return trainset if trainset else sorted(root.glob("*.zip"))


def resolve_tu_model_dir(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value))
    if os.environ.get("TU_MODEL_DIR"):
        candidates.append(Path(os.environ["TU_MODEL_DIR"]))
    candidates.append(Path("/content/drive/MyDrive/TU model"))
    candidates.append(Path("/content/drive/MyDrive/TU-Model"))
    candidates.append(Path("/content/drive/MyDrive/Copy of TU-Model"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not resolve TU model. Set TU_MODEL_DIR or create a My Drive shortcut at /content/drive/MyDrive/TU-Model."
    )


def neutral_id(member_name: str) -> str | None:
    name = member_name.replace("\\", "/")
    if not name.endswith(NEUTRAL_SUFFIX):
        return None
    parts = name.split("/")
    try:
        idx = parts.index("models_reg")
    except ValueError:
        return None
    if idx == 0:
        return None
    return parts[idx - 1]


def extract_neutrals(zip_path: Path, staged_root: Path) -> int:
    """Extract only <id>/models_reg/1_neutral.obj from a zip into a normalised layout."""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            sid = neutral_id(info.filename)
            if sid is None:
                continue
            target = staged_root / sid / "models_reg" / "1_neutral.obj"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".obj.tmp")
            with zf.open(info) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            os.replace(tmp, target)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tu-model-dir", default="")
    parser.add_argument("--scratch", default="/content/facescape_scratch")
    parser.add_argument("--out", default="data/facescape_stage.json")
    parser.add_argument("--raw-zip-manifest", default="data/raw_zip_manifest.json")
    parser.add_argument("--expected-zip-count", type=int, default=9)
    parser.add_argument("--force-unzip", action="store_true")
    parser.add_argument("--allow-unzipped-source", action="store_true")
    parser.add_argument(
        "--copy-zips-first",
        action="store_true",
        help="Copy each ~14GB zip to local scratch before extracting (slower, more disk). Default reads neutral OBJs directly from Drive.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="When --copy-zips-first, keep the copied zip instead of deleting it after extraction.",
    )
    args = parser.parse_args()

    source = resolve_tu_model_dir(args.tu_model_dir or None)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    staged_root = scratch / "tu_model_unzipped"
    staged_root.mkdir(parents=True, exist_ok=True)

    source_zips = zip_files(source)
    raw_rows: list[dict] = []

    if source_zips:
        if args.expected_zip_count and len(source_zips) != args.expected_zip_count:
            raise SystemExit(
                f"Expected {args.expected_zip_count} FaceScape trainset zip files, found {len(source_zips)} in {source}: "
                + ", ".join(z.name for z in source_zips)
            )

        existing = count_neutrals(staged_root)
        if existing >= args.expected_zip_count and has_unzipped_layout(staged_root) and not args.force_unzip:
            print(f"Found {existing} neutral OBJs already staged at {staged_root}; skipping extraction (use --force-unzip to redo).", flush=True)
        else:
            zip_scratch = scratch / "zips"
            zip_scratch.mkdir(parents=True, exist_ok=True)
            total_neutrals = 0
            for i, zp in enumerate(source_zips, 1):
                size_gb = zp.stat().st_size / 1e9
                if args.copy_zips_first:
                    local = zip_scratch / zp.name
                    print(f"[{i}/{len(source_zips)}] {zp.name} ({size_gb:.1f} GB): copying to scratch ...", flush=True)
                    if not (local.exists() and local.stat().st_size == zp.stat().st_size):
                        shutil.copy2(zp, local)
                    read_path = local
                else:
                    print(f"[{i}/{len(source_zips)}] {zp.name} ({size_gb:.1f} GB): extracting neutral OBJs directly from Drive ...", flush=True)
                    read_path = zp
                n = extract_neutrals(read_path, staged_root)
                total_neutrals += n
                print(f"[{i}/{len(source_zips)}] {zp.name}: extracted {n} neutral OBJs (running total {total_neutrals})", flush=True)
                raw_rows.append(
                    {
                        "source": str(zp),
                        "size_bytes": zp.stat().st_size,
                        "neutral_objs": n,
                        "read_mode": "copied_then_extract" if args.copy_zips_first else "direct_from_drive",
                    }
                )
                if args.copy_zips_first and not args.keep_zips:
                    local.unlink(missing_ok=True)
                    print(f"[{i}/{len(source_zips)}] {zp.name}: removed local zip copy to free disk", flush=True)
            atomic_write_json(Path(args.raw_zip_manifest), {"source": str(source), "zips": raw_rows, "total_neutral_objs": total_neutrals})
            print(f"Done: {total_neutrals} neutral OBJs staged at {staged_root}", flush=True)
        obj_root = staged_root
    elif args.allow_unzipped_source and has_unzipped_layout(source):
        obj_root = source
    else:
        raise FileNotFoundError(
            "TU_MODEL_DIR must contain the FaceScape trainset zip files. For a deliberate pre-unzipped scratch source, rerun with --allow-unzipped-source."
        )

    report = {
        "source": str(source),
        "scratch": str(scratch),
        "staged_obj_root": str(obj_root),
        "zips": raw_rows,
        "expected_zip_count": args.expected_zip_count,
        "neutral_obj_count": count_neutrals(obj_root),
        "allow_unzipped_source": bool(args.allow_unzipped_source),
        "has_unzipped_layout": has_unzipped_layout(obj_root),
    }
    atomic_write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
