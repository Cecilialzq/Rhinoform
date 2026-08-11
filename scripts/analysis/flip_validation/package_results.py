"""Assemble gate23a1_results.json from all gate outputs + provenance + unit tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.analysis.flip_validation.common as common

OUT = common.ensure_out()
WS = Path(__file__).resolve().parents[1]


def run_unit_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(WS / "tests" / "test_safe_fusion.py"), "-q"],
        capture_output=True, text=True, cwd=str(WS),
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return {"returncode": proc.returncode, "passed": proc.returncode == 0, "summary": tail}


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    g2a = load("g2a_signed_fold.json")
    g2b = load("g2b_self_intersection.json")
    g3 = load("g3_envelope.json")
    a1a = load("a1a_truth_line.json")
    a1b = load("a1b_synthetic.json")

    tests = run_unit_tests()

    a1_pulse = bool((a1a and a1a.get("nonlinear_has_pulse")) or (a1b and a1b.get("nonlinear_has_pulse")))
    a1_decision = "PULSE (re-verify after clean retrain)" if a1_pulse else \
        "NO PULSE: gain/divergence does not grow with magnitude/alpha -> nonlinear-linkage claim unsupported; reposition contribution (re-verify after clean retrain)."

    master = {
        "protocol": "GATE_2_3_A1_PROTOCOL_2026-06-15",
        "provenance": common.env_provenance(),
        "unit_tests_signed_fold_and_self_intersection": tests,
        "gate2A_signed_foldover": {
            "decision": g2a["decision"] if g2a else "MISSING",
            "synthetic_battery_100pct_correct": g2a["synthetic_battery"]["all_classes_100pct_correct"] if g2a else None,
            "real_legacy_normal_flip_pct": g2a["real_pair_comparison"]["legacy_normal_flip_pct"] if g2a else None,
            "real_signed_foldover_pct": g2a["real_pair_comparison"]["signed_foldover_pct"] if g2a else None,
            "sign_equivalent_on_real_data": g2a["real_pair_comparison"]["sign_equivalent_on_real_data"] if g2a else None,
            "note": g2a["real_pair_comparison"]["relationship_note"] if g2a else None,
        },
        "gate2B_self_intersection": {
            "decision": g2b["decision"] if g2b else "MISSING",
            "synthetic_battery_all_correct": g2b["synthetic_battery"]["all_correct"] if g2b else None,
            "n_src": g2b["real_diagnosis"]["n_src"] if g2b else None,
            "n_edit": g2b["real_diagnosis"]["n_edit"] if g2b else None,
            "n_new": g2b["real_diagnosis"]["n_new"] if g2b else None,
            "n_new_interior": g2b["real_diagnosis"]["n_new_interior"] if g2b else None,
            "boundary_share_of_new": g2b["real_diagnosis"]["boundary_share_of_new"] if g2b else None,
            "root_cause": g2b["real_diagnosis"]["root_cause"] if g2b else None,
        },
        "gate3_population_envelope": {
            "decision": g3["decision"] if g3 else "MISSING",
            "heldout_human_acceptance_rate": g3["heldout_human_acceptance_rate"] if g3 else None,
            "heldout_human_acceptance_ci95": g3["heldout_human_acceptance_ci95"] if g3 else None,
            "hard_negative_detection": g3["hard_negative_detection"] if g3 else None,
            "tau_bootstrap_ci95": g3["tau_bootstrap_ci95"] if g3 else None,
        },
        "A1_nonlinear_vs_linear": {
            "caveat": "Exploratory gate-1 weights -> DIRECTIONAL SIGNAL ONLY; re-verify after clean retrain.",
            "A1a_truth_line": {
                "overall_delta_ridge_minus_rbsr": a1a.get("overall_delta_ridge_minus_rbsr") if a1a else None,
                "overall_delta_ci95": a1a.get("overall_delta_ci95") if a1a else None,
                "gain_grows_with_magnitude": a1a.get("gain_grows_with_magnitude") if a1a else None,
                "buckets": a1a.get("buckets") if a1a else None,
                "conclusion": a1a.get("conclusion") if a1a else None,
            },
            "A1b_synthetic_line": {
                "growth_by_region": a1b.get("growth_by_region") if a1b else None,
                "alar_tip_divergence_grows": a1b.get("alar_tip_divergence_grows") if a1b else None,
                "max_divergence_observed": a1b.get("max_divergence_observed") if a1b else None,
                "conclusion": a1b.get("conclusion") if a1b else None,
            },
            "A1_overall_decision": a1_decision,
        },
        "GO_NO_GO_SUMMARY": {
            "gate2A": g2a["decision"] if g2a else "MISSING",
            "gate2B": g2b["decision"] if g2b else "MISSING",
            "gate3": g3["decision"] if g3 else "MISSING",
            "A1_nonlinear_pulse": a1_pulse,
        },
    }
    path = OUT / "gate23a1_results.json"
    path.write_text(json.dumps(master, indent=2), encoding="utf-8")
    print(json.dumps(master["GO_NO_GO_SUMMARY"], indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
