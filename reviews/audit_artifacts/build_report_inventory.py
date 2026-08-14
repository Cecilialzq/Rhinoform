#!/usr/bin/env python3
"""Build traceable inventories for the final Rhinoform report audit.

This is intentionally a review artifact, not project runtime code.  It keeps
the original TeX line number and searches current repository text evidence for
literal numeric candidates.  Candidate hits are leads, not proof: the audit
report adjudicates semantic agreement against controlling manifests/tables.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


REPO = Path("/Users/lizequan/Downloads/Rhinoform-main")
TEX = Path("/Users/lizequan/Desktop/FYP_revised/最终版report.tex")
OUT = REPO / "reviews" / "audit_artifacts"

TEXT_SUFFIXES = {
    ".py", ".js", ".jsx", ".mjs", ".json", ".csv", ".md", ".txt",
    ".toml", ".yml", ".yaml", ".cff",
}
EXCLUDED_PARTS = {".git", "node_modules", "dist", "__pycache__", "reviews", "tmp"}

NUM_RE = re.compile(
    r"(?<![A-Za-z_])(?:\\approx\s*)?[+-]?(?:\d{1,3}(?:[,{]\\,?[}\d]{1,4})+|\d+)(?:\.\d+)?(?:\\%|%|\\,?s|\\,?ms|\\times\s*10\^\{?-?\d+\}?)?"
)


def strip_comment(line: str) -> str:
    """Drop an unescaped TeX comment while preserving escaped percent signs."""
    for i, ch in enumerate(line):
        if ch == "%" and (i == 0 or line[i - 1] != "\\"):
            return line[:i]
    return line


def plain_tex(text: str) -> str:
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:Cref|cref|ref|label)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:textbf|emph|textsc|texttt|mathrm|mathbf|bm)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("~", " ").replace("&", " ")
    text = re.sub(r"[{}$]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_number(token: str) -> str:
    token = token.replace("\\approx", "").replace("\\,", "")
    token = token.replace("{", "").replace("}", "").replace(",", "")
    token = token.replace("\\%", "%").strip()
    return token


def repository_corpus() -> list[tuple[str, list[str]]]:
    corpus: list[tuple[str, list[str]]] = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(REPO)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        try:
            corpus.append((str(rel), path.read_text(encoding="utf-8", errors="replace").splitlines()))
        except OSError:
            continue
    return corpus


def numeric_inventory(lines: list[str], corpus: list[tuple[str, list[str]]]) -> None:
    repo_number_index: dict[str, list[str]] = {}
    broad_number_re = re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?%?")
    for rel, repo_lines in corpus:
        for repo_line_no, repo_line in enumerate(repo_lines, 1):
            for found in broad_number_re.findall(repo_line):
                key = found.replace(",", "").lstrip("+")
                bucket = repo_number_index.setdefault(key, [])
                if len(bucket) < 20:
                    bucket.append(f"{rel}:{repo_line_no}")
    rows: list[dict[str, object]] = []
    for line_no, raw in enumerate(lines, 1):
        clean = strip_comment(raw)
        for match in NUM_RE.finditer(clean):
            token = match.group(0)
            norm = normalise_number(token)
            if not re.search(r"\d", norm):
                continue
            candidates: list[str] = []
            search_forms = (norm, norm.replace("%", ""), norm.lstrip("+"))
            for form in search_forms:
                for candidate in repo_number_index.get(form, []):
                    if candidate not in candidates:
                        candidates.append(candidate)
                    if len(candidates) >= 20:
                        break
                if len(candidates) >= 20:
                    break
            rows.append({
                "tex_line": line_no,
                "token": token,
                "normalised": norm,
                "context": plain_tex(clean)[:600],
                "candidate_hit_count_capped": len(candidates),
                "candidate_hits": "; ".join(candidates),
            })
    with (OUT / "report_numeric_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sentence_inventory(lines: list[str]) -> None:
    rows: list[dict[str, object]] = []
    paragraph: list[tuple[int, str]] = []

    def flush() -> None:
        if not paragraph:
            return
        start = paragraph[0][0]
        text = plain_tex(" ".join(part for _, part in paragraph))
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", text):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            kind = "narrative"
            lower = sentence.lower()
            if re.search(r"\d", sentence):
                kind = "quantitative"
            if any(k in lower for k in ("improv", "lower", "higher", "best", "significant", "domin", "attains", "reduces", "increases")):
                kind = "comparative_claim"
            if any(k in lower for k in ("implemented", "browser", "runtime", "pipeline", "stores", "computes", "maps", "re-impose", "certificate")):
                kind = "implementation_claim" if kind == "narrative" else kind + "+implementation"
            rows.append({"paragraph_start_line": start, "kind": kind, "sentence": sentence})
        paragraph.clear()

    in_bibliography = False
    for line_no, raw in enumerate(lines, 1):
        clean = strip_comment(raw).strip()
        if "\\begin{thebibliography}" in clean:
            in_bibliography = True
        if in_bibliography:
            continue
        if not clean or re.match(r"^\\(?:chapter|section|subsection|subsubsection|paragraph|begin|end|label|caption|toprule|midrule|bottomrule)", clean):
            flush()
            continue
        paragraph.append((line_no, clean))
    flush()
    with (OUT / "report_sentence_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = TEX.read_text(encoding="utf-8", errors="replace").splitlines()
    corpus = repository_corpus()
    numeric_inventory(lines, corpus)
    sentence_inventory(lines)
    print(f"corpus_files={len(corpus)}")
    for name in ("report_numeric_inventory.csv", "report_sentence_inventory.csv"):
        path = OUT / name
        print(f"{name}: {sum(1 for _ in path.open(encoding='utf-8')) - 1} rows")


if __name__ == "__main__":
    main()
