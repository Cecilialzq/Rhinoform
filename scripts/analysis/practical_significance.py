"""Practical-significance / worst-case analysis as a substitute for a user study.

Reads frozen per-pair metric CSVs (no training, no dense caches, no torch) and
reports, for RBSR vs ridge and vs global-hybrid on the fixed 9900 test pairs:
  - dominance: fraction of pairs where RBSR is strictly better (lower) per metric
  - tail percentiles (P50/P90/P95/P99/max) of flip% / edge-strain / ROI
  - catastrophic-flip pair counts (flip > 2% and > 3%)
  - worst-case fix: behaviour on the global-hybrid worst-decile flip pairs

All metrics are lower-is-better. Coordinates are non-metric FaceScape units.
Run on the authoritative cloud package.
"""
from __future__ import annotations
import argparse, csv, statistics as st


def load(path):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d[(str(r["source_id"]), str(r["target_id"]))] = r
    return d


def percentile(x, q):
    x = sorted(x)
    i = min(len(x) - 1, int(q / 100 * (len(x) - 1) + 0.5))
    return x[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbsr", required=True)
    ap.add_argument("--ridge", required=True)
    ap.add_argument("--global", dest="glob", required=True)
    args = ap.parse_args()
    rbsr, ridge, glob = load(args.rbsr), load(args.ridge), load(args.glob)
    keys = sorted(set(rbsr) & set(ridge) & set(glob))
    metrics = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]
    col = lambda d, m: [float(d[k][m]) for k in keys]
    print(f"aligned pairs: {len(keys)}")

    for name, bl in [("ridge", ridge), ("global", glob)]:
        print(f"\nRBSR vs {name}  (win = fraction of pairs RBSR strictly lower)")
        for m in metrics:
            a, b = col(rbsr, m), col(bl, m)
            win = sum(1 for x, y in zip(a, b) if x < y) / len(a)
            md = st.mean(x - y for x, y in zip(a, b))
            print(f"  {m:16s} win={win*100:5.1f}%  mean_diff={md:+.5f}")

    print("\nTail percentiles                P50    P90    P95    P99    MAX")
    for m in ["normal_flip_pct", "edge_strain_p95", "roi_rmse"]:
        print(f"[{m}]")
        for nm, d in [("ridge ", ridge), ("global", glob), ("RBSR  ", rbsr)]:
            x = col(d, m)
            print(f"  {nm} " + "  ".join(f"{percentile(x, q):.3f}" for q in (50, 90, 95, 99, 100)))

    print("\nCatastrophic-flip pairs (count / 9900)")
    for thr in (2.0, 3.0):
        print("  flip>%.0f%%: " % thr + "  ".join(
            f"{nm}={sum(1 for v in col(d, 'normal_flip_pct') if v > thr)}"
            for nm, d in [("ridge", ridge), ("global", glob), ("RBSR", rbsr)]))

    gf = col(glob, "normal_flip_pct")
    thr = percentile(gf, 90)
    idx = [i for i, v in enumerate(gf) if v >= thr]
    rb, rd, gr, rr = col(rbsr, "normal_flip_pct"), col(ridge, "normal_flip_pct"), col(glob, "roi_rmse"), col(rbsr, "roi_rmse")
    print(f"\nWorst-case fix on global worst-decile flip pairs (n={len(idx)}, flip>= {thr:.3f}%)")
    print(f"  mean flip  global={st.mean(gf[i] for i in idx):.3f}%  RBSR={st.mean(rb[i] for i in idx):.3f}%  ridge={st.mean(rd[i] for i in idx):.3f}%")
    print(f"  mean ROI   global={st.mean(gr[i] for i in idx):.4f}   RBSR={st.mean(rr[i] for i in idx):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
