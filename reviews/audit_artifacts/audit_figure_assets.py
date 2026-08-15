#!/usr/bin/env python3
"""Inventory every image asset referenced by the final report.

The audit records the exact compiled asset, its digest and whether an identical
binary is present in the repository.  Evidence/source-level mappings for
empirical figures are maintained separately because a composite PDF need not be
byte-identical to the CSV/JSON/NPZ data from which it was drawn.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("figure_asset_inventory.csv"))
    args = parser.parse_args()
    report_root = args.report_root.resolve()
    repo = args.repo.resolve()
    out = args.out.resolve()
    tex = report_root / "main.tex"

    source = tex.read_text(encoding="utf-8")
    # A leading percent comments out the whole line; trailing comments do not
    # affect include paths captured before them.
    active = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    references = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", active)
    repo_files = [path for path in repo.rglob("*") if path.is_file() and ".git" not in path.parts]
    repo_by_size: dict[int, list[Path]] = {}
    for path in repo_files:
        repo_by_size.setdefault(path.stat().st_size, []).append(path)

    rows = []
    for index, reference in enumerate(references, start=1):
        asset = report_root / reference
        exists = asset.is_file()
        digest = sha256(asset) if exists else ""
        exact_matches = []
        if exists:
            for candidate in repo_by_size.get(asset.stat().st_size, []):
                if sha256(candidate) == digest:
                    exact_matches.append(str(candidate.relative_to(repo)))
        if reference.startswith("figures/classic/"):
            role = "external_prior_work_illustration"
        elif reference.startswith(("figures/final/", "figures/eval_review_v3/")):
            role = "project_empirical_or_method_composite"
        elif reference.startswith("figures/chapter3_review_v2/"):
            role = "project_system_design_diagram"
        elif reference.startswith("figures/redrawn/"):
            role = "project_redrawn_diagram_or_appendix_composite"
        else:
            role = "template_or_other"
        rows.append(
            {
                "include_index": index,
                "report_relative_path": reference,
                "role": role,
                "asset_exists": exists,
                "bytes": asset.stat().st_size if exists else "",
                "sha256": digest,
                "identical_repo_binary": ";".join(exact_matches),
            }
        )

    fields = [
        "include_index",
        "report_relative_path",
        "role",
        "asset_exists",
        "bytes",
        "sha256",
        "identical_repo_binary",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    exact = sum(bool(row["identical_repo_binary"]) for row in rows)
    missing = sum(not row["asset_exists"] for row in rows)
    print(f"includes={len(rows)} missing={missing} exact_repo_binaries={exact}")


if __name__ == "__main__":
    main()
