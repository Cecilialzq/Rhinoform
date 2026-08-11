from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def identity_sort_key(s: str) -> tuple[int, str]:
    try:
        return (0, f"{int(s):08d}")
    except ValueError:
        return (1, s)


def all_ordered_pairs(ids: list[str]) -> list[tuple[str, str]]:
    ids = [str(i) for i in ids]
    return [(a, b) for a in ids for b in ids if a != b]


def _derangement(ids: list[str], rng: np.random.Generator) -> list[tuple[str, str]]:
    ids = list(ids)
    if len(ids) < 2:
        return []
    targets = ids.copy()
    for _ in range(1000):
        rng.shuffle(targets)
        if all(a != b for a, b in zip(ids, targets)):
            return list(zip(ids, targets))
    return [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]


def coverage_balanced_ordered_pairs(ids: list[str], budget: int, seed: int) -> list[tuple[str, str]]:
    """Fixed ordered-pair sampler required by GOAL §A.4/U2.

    For N=30 and budget >= N(N-1), this returns all ordered pairs. Otherwise it
    seeds the set with two derangement-style matchings so every identity appears
    at least once as source and at least once as target when budget >= N, then
    fills the remaining slots uniformly without replacement.
    """
    ids = sorted([str(i) for i in ids], key=identity_sort_key)
    all_pairs = all_ordered_pairs(ids)
    if int(budget) <= 0 or int(budget) >= len(all_pairs):
        return all_pairs
    budget = int(budget)
    rng = np.random.default_rng(seed)
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for pair in _derangement(ids, rng):
        if len(selected) >= budget:
            break
        if pair not in seen:
            selected.append(pair)
            seen.add(pair)

    # A reverse-direction pass improves role coverage when the first matching
    # happens to duplicate many target roles in small pools.
    if len(selected) < budget:
        for a, b in _derangement(ids[::-1], rng):
            pair = (b, a)
            if len(selected) >= budget:
                break
            if pair not in seen and pair[0] != pair[1]:
                selected.append(pair)
                seen.add(pair)

    remaining = [p for p in all_pairs if p not in seen]
    need = budget - len(selected)
    if need > 0:
        idx = rng.choice(len(remaining), size=need, replace=False)
        selected.extend(remaining[int(i)] for i in idx)

    return selected


def write_pair_manifest(path: Path, *, ids: list[str], budget: int, seed: int, label: str) -> dict:
    pairs = coverage_balanced_ordered_pairs(ids, budget, seed)
    sources = {s for s, _ in pairs}
    targets = {t for _, t in pairs}
    manifest = {
        "label": label,
        "seed": int(seed),
        "budget": int(budget),
        "n_identities": len(ids),
        "n_pairs": len(pairs),
        "source_coverage": len(sources),
        "target_coverage": len(targets),
        "ids": [str(i) for i in ids],
        "pairs": [{"source_id": s, "target_id": t} for s, t in pairs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_pairs(path: Path) -> list[tuple[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    pairs = obj.get("pairs", obj)
    return [(str(r["source_id"]), str(r["target_id"])) for r in pairs]


def read_ids(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids = obj.get("ids", obj)
    return [str(i) for i in ids]

