"""Content-level mapping from the Drive code archive to the reorganised repo."""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import re
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(path: Path) -> set[tuple[str, str]]:
    if path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set()
    return {(type(node).__name__, node.name) for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def normalised(path: Path) -> str:
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "import ", "from ")):
            continue
        lines.append(re.sub(r"\s+", "", line).replace("code.", ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-code", type=Path, required=True)
    parser.add_argument("--rhinoform2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    drive = [path for path in args.drive_code.rglob("*") if path.is_file()
             and path.suffix.lower() in {".py", ".js"} and "__pycache__" not in path.parts]
    roots = [args.rhinoform2 / name for name in ("rhinoform", "scripts", "tools", "demo")]
    local = [path for root in roots for path in root.rglob("*") if path.is_file()
             and path.suffix.lower() in {".py", ".js"}]
    local_meta = [(path, digest(path), definitions(path), normalised(path)) for path in local]
    rows = []
    for source in sorted(drive):
        source_hash, source_defs, source_text = digest(source), definitions(source), normalised(source)
        exact = [item for item in local_meta if item[1] == source_hash]
        if exact:
            target, _, target_defs, _ = exact[0]
            status, similarity = "exact_sha256", 1.0
        else:
            # A lone generic ``main`` occurs in many unrelated entry points and
            # is not a useful structural fingerprint. Require at least two
            # definitions before using equality as the candidate pool.
            same_defs = [item for item in local_meta if len(source_defs) >= 2 and item[2] == source_defs]
            same_name = [item for item in local_meta if item[0].name == source.name]
            if same_defs:
                candidates = same_defs
            elif same_name:
                candidates = same_name
            else:
                stem_scores = sorted(
                    ((difflib.SequenceMatcher(None, source.stem, item[0].stem).ratio(), item)
                     for item in local_meta), reverse=True, key=lambda value: value[0]
                )
                candidates = [item for _, item in stem_scores[:3]]
            if candidates:
                scored = [(difflib.SequenceMatcher(None, source_text, item[3], autojunk=True).quick_ratio(), item)
                          for item in candidates]
                similarity, (target, _, target_defs, _) = max(scored, key=lambda value: value[0])
                status = "same_definitions_repackaged" if same_defs and similarity >= 0.85 else "modified_candidate"
            else:
                target, target_defs, similarity, status = None, set(), 0.0, "drive_only_no_match"
        title = source.name.lower()
        if status.startswith("drive_only") or status == "modified_candidate":
            if title.startswith(("make_", "plot_", "render_")) or any(x in title for x in ("visual_audit", "figure_atlas", "docx_figure")):
                disposition = "figure_or_visual_audit_review"
            elif title in {"metrics.py", "linear.py", "arap.py", "infer.py", "export_results.py"}:
                disposition = "legacy_flat_module_functionality_folded"
            else:
                disposition = "manual_review_not_release_evidence"
        else:
            disposition = "core_equivalent_keep_reorganised"
        rows.append({
            "drive_path": str(source.relative_to(args.drive_code)),
            "rhinoform2_path": "" if target is None else str(target.relative_to(args.rhinoform2)),
            "status": status,
            "normalised_similarity": f"{similarity:.4f}",
            "drive_definition_count": len(source_defs),
            "local_definition_count": len(target_defs),
            "disposition": disposition,
            "drive_sha256": source_hash,
            "local_sha256": "" if target is None else digest(target),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
