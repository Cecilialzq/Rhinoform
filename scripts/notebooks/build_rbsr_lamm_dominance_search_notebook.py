"""Build the Colab notebook for the targeted RB-SR/LAMM dominance search."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/Rhinoform_RBSR_LAMM_dominance_search_colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("""# Rhinoform — targeted RB-SR dominance search against frozen LAMM

This is an isolated **supplementary post-hoc development experiment**. It does not overwrite the completed final rerun. The exact success condition is predeclared as lower matched strict free-ROI RMSE, lower target-relative new-flip percentage, and lower edge-strain p95 than frozen LAMM.

The search fixes the earlier regularisation bottleneck rather than blindly increasing PCA dimension: it preserves the exact frozen 870-pair RB-SR training manifest, selects PCA dimension and Ridge regularisation on validation only, trains a deterministic neural-field residual proposer for more optimisation steps, and calibrates the spatial gate against a frozen LAMM **validation** reference. LAMM is evaluated on validation but is never retrained. The already-observed internal test is used only for the final post-hoc matched comparison.

All new artifacts are written under `results/supplemental_rbsr_lamm_dominance_search_v1/` inside the GitHub-ready Rhinoform Drive folder. Checkpoints and evaluation chunks are atomic, hash-chained, resumable, and live-printed."""),
    code("""from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import csv, importlib, json, os, shutil, subprocess, sys

LIVE_ENV = {**os.environ, 'PYTHONUNBUFFERED': '1'}
def run_live(cmd):
    command = [str(value) for value in cmd]
    print('RUN:', ' '.join(command), flush=True)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=LIVE_ENV,
    )
    try:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate(); process.wait(); raise
    finally:
        if process.stdout is not None: process.stdout.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return subprocess.CompletedProcess(command, return_code)

DRIVE_FYP = Path('/content/drive/MyDrive/FYP final')
REPO = Path('/content/drive/MyDrive/Rhinoform_GitHub_Ready_20260808')
FINAL_REFERENCE = REPO/'results/rbsr_final_rerun_holdout_v1'
SPLIT = FINAL_REFERENCE/'protocol/final_rerun_holdout_split_manifest.json'
ORIGINAL_TRAIN_PAIRS = FINAL_REFERENCE/'protocol/final_rerun_holdout_train_pairs.json'
ALL_MODELS_FREEZE = FINAL_REFERENCE/'ALL_MODELS_FROZEN_BEFORE_TEST.json'
LAMM_DIR = FINAL_REFERENCE/'lamm/seed20260609'
LAMM_TEST_CSV = LAMM_DIR/'identity_bootstrap_pair_metrics_lamm.csv'
LAMM_TEST_SUMMARY = LAMM_DIR/'lamm_test_summary.json'
LAMM_CONFIG_ERRATUM = LAMM_DIR/'LAMM_PRE_INFERENCE_CONFIG_CANONICALISATION_ERRATUM.json'
SOURCE_PCA_PACKAGE = REPO/'results/supplemental_rbsr_lamm_dominance_search_v1/frozen_inputs/pca384_lambda300/neural_field_model_package_cvae_ew0p1_lw0.pt'

EXPERIMENT_ROOT = REPO/'results/supplemental_rbsr_lamm_dominance_search_v1'
RAW = EXPERIMENT_ROOT/'raw_runs'
EVIDENCE = EXPERIMENT_ROOT/'release_evidence'
LOCAL_FYP = Path('/content/rhinoform_rbsr_dominance')
LOCAL_DATA = LOCAL_FYP/'data'
LAMM_ROOT = Path('/content/LAMM_official_rbsr_dominance')
for path in (DRIVE_FYP, REPO, SPLIT, ORIGINAL_TRAIN_PAIRS, ALL_MODELS_FREEZE, LAMM_TEST_CSV, LAMM_TEST_SUMMARY, LAMM_CONFIG_ERRATUM, SOURCE_PCA_PACKAGE):
    assert path.exists(), path
RAW.mkdir(parents=True, exist_ok=True)
repo_string = str(REPO)
if repo_string not in sys.path: sys.path.insert(0, repo_string)
prior_pythonpath = LIVE_ENV.get('PYTHONPATH', '')
LIVE_ENV['PYTHONPATH'] = repo_string + (os.pathsep + prior_pythonpath if prior_pythonpath else '')
os.environ['PYTHONPATH'] = LIVE_ENV['PYTHONPATH']
print('Isolated experiment:', EXPERIMENT_ROOT)
print('Existing final rerun remains read-only:', FINAL_REFERENCE)"""),
    markdown("""## 1. Install the package and stage licensed meshes to local Colab storage

This cell repairs the import path before any `rhinoform` import, copies all mesh data to `/content` for stable throughput, and leaves every persistent checkpoint/result on Drive."""),
    code("""run_live([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', REPO])
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
importlib.invalidate_caches()
import rhinoform
from rhinoform.repro import (
    atomic_copy_file, atomic_write_json, sha256_file, sha256_json,
    valid_sha256_sidecar, validate_torch_artifact, write_sha256_sidecar,
)
print('rhinoform loaded from:', rhinoform.__file__)

assert valid_sha256_sidecar(ALL_MODELS_FREEZE), ALL_MODELS_FREEZE
all_models_freeze = json.loads(ALL_MODELS_FREEZE.read_text())
strict_scorer = REPO/'rhinoform/strict_protocol_patch.py'
expected_scorer_sha256 = all_models_freeze['implementation_hashes']['rhinoform/strict_protocol_patch.py']
assert sha256_file(strict_scorer) == expected_scorer_sha256, {
    'error': 'Strict scorer differs from the implementation used for frozen LAMM test metrics',
    'expected': expected_scorer_sha256,
    'actual': sha256_file(strict_scorer),
}
print('Strict scorer matches frozen LAMM:', expected_scorer_sha256)

LOCAL_DATA.mkdir(parents=True, exist_ok=True)
shutil.copy2(DRIVE_FYP/'data/manifest.json', LOCAL_DATA/'manifest.json')
(LOCAL_FYP/'roi').mkdir(parents=True, exist_ok=True)
run_live(['rsync', '-a', '--info=progress2', f'{DRIVE_FYP / "data/meshes"}/', f'{LOCAL_DATA / "meshes"}/'])
run_live(['rsync', '-a', f'{DRIVE_FYP / "roi"}/', f'{LOCAL_FYP / "roi"}/'])
print('Local mesh staging complete:', LOCAL_FYP)"""),
    markdown("""## 2. Pin official LAMM and run implementation preflight

LAMM is not trained. A clean official checkout at the frozen commit is used only to load the existing checkpoint for validation inference. Every invoked Rhinoform implementation is compiled and the unit tests run before expensive work."""),
    code("""LAMM_COMMIT = '87354c05dec341c6d8dd319665dd52553fb03084'
run_live([sys.executable, '-m', 'pip', 'install', '-q', 'einops==0.6.1', 'timm==0.9.2', 'pyyaml==6.0.2', 'trimesh==3.22.3'])
if not LAMM_ROOT.exists():
    run_live(['git', 'clone', 'https://github.com/michaeltrs/LAMM.git', LAMM_ROOT])
head = subprocess.check_output(['git', '-C', str(LAMM_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
if head != LAMM_COMMIT:
    run_live(['git', '-C', LAMM_ROOT, 'checkout', '--detach', LAMM_COMMIT])
assert subprocess.check_output(['git', '-C', str(LAMM_ROOT), 'rev-parse', 'HEAD'], text=True).strip() == LAMM_COMMIT
assert not subprocess.check_output(['git', '-C', str(LAMM_ROOT), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()

required_sources = [
    REPO/'rhinoform/train.py', REPO/'rhinoform/train_rbsr_gate.py',
    REPO/'rhinoform/rbsr_calibration.py', REPO/'scripts/evaluation/rbsr_gate.py',
    REPO/'scripts/evaluation/rbsr_gate_calibration.py',
    REPO/'scripts/evaluation/lamm_validation_reference.py',
    REPO/'scripts/analysis/rbsr_anchor_validation_sweep.py',
    REPO/'scripts/analysis/select_rbsr_reference_dominating_calibration.py',
]
posthoc_analysis_sources = [REPO/'scripts/evaluation/rbsr_lamm_paired_statistics.py']
for path in required_sources + posthoc_analysis_sources:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
assert valid_sha256_sidecar(LAMM_CONFIG_ERRATUM), LAMM_CONFIG_ERRATUM
lamm_config_erratum = json.loads(LAMM_CONFIG_ERRATUM.read_text())
assert lamm_config_erratum['status'] == 'FROZEN_BEFORE_FIRST_LAMM_INFERENCE'
assert lamm_config_erratum['derived_defaults'] == {
    'Dinput': 3, 'Dlms': 8, 'depth': None, 'scale_dim_token': 0.5,
}
lamm_reference_source = (REPO/'scripts/evaluation/lamm_validation_reference.py').read_text(encoding='utf-8')
assert 'LAMM_CONSTRUCTOR_DERIVED_DEFAULTS' in lamm_reference_source
assert 'result.setdefault(key, value)' in lamm_reference_source
run_live([sys.executable, '-m', 'unittest', 'discover', '-s', REPO/'tests', '-v'])
print('Preflight PASS')"""),
    markdown("""## 3. Evaluate the frozen LAMM checkpoint on validation only

This produces the matched validation thresholds for RMSE, new flip, and edge strain. It neither retrains LAMM nor loads the 100 test identities."""),
    code("""# Rebind the erratum here as well so a refreshed live runtime can resume
# directly from this previously failed section without replaying data staging.
LAMM_CONFIG_ERRATUM = LAMM_DIR/'LAMM_PRE_INFERENCE_CONFIG_CANONICALISATION_ERRATUM.json'
assert valid_sha256_sidecar(LAMM_CONFIG_ERRATUM), LAMM_CONFIG_ERRATUM
LAMM_VALIDATION = RAW/'lamm_validation_reference'
LAMM_VALIDATION_SUMMARY = LAMM_VALIDATION/'lamm_validation_summary.json'
if not valid_sha256_sidecar(LAMM_VALIDATION_SUMMARY):
    run_live([
        sys.executable, '-u', REPO/'scripts/evaluation/lamm_validation_reference.py',
        '--fyp-root', LOCAL_FYP, '--lamm-root', LAMM_ROOT, '--lamm-out', LAMM_DIR,
        '--split-manifest', SPLIT, '--out', LAMM_VALIDATION,
        '--batch-size', '16', '--chunk-pairs', '320', '--device', 'cuda',
    ])
lamm_validation = json.loads(LAMM_VALIDATION_SUMMARY.read_text())
assert lamm_validation['status'] == 'FROZEN_LAMM_VALIDATION_REFERENCE_COMPLETE'
assert lamm_validation['split'] == 'validation' and lamm_validation['n_pairs'] == 4830
print('Frozen LAMM validation thresholds:', lamm_validation['means'])"""),
    markdown("""## 4. Reuse the exact frozen 870 training pairs and freeze the search protocol

The identity split and RB-SR training-pair manifest are unchanged from the completed final rerun. Validation still uses all 4,830 ordered pairs and the matched post-hoc test still uses the exact same 9,900 ordered pairs as LAMM."""),
    code("""SEED = 20260609
PAIR_SEED = 20260809
TRAIN_PAIR_BUDGET = 870
DIMENSIONS = (32, 64, 96, 128)
RIDGE_LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
STRICT_TOP_K = 12
VALIDATION_RELATIVE_MARGIN = 0.02
RUN_POSTHOC_TEST = True

split = json.loads(SPLIT.read_text())
train_ids = list(map(str, split['train_pool_ids']))
PAIR_MANIFEST = ORIGINAL_TRAIN_PAIRS
assert valid_sha256_sidecar(PAIR_MANIFEST), PAIR_MANIFEST
pair_payload = json.loads(PAIR_MANIFEST.read_text())
assert pair_payload['n_pairs'] == pair_payload['budget'] == TRAIN_PAIR_BUDGET
assert pair_payload['source_coverage'] == pair_payload['target_coverage'] == len(train_ids)
assert pair_payload['split_manifest_sha256'] == sha256_file(SPLIT)

PROTOCOL = RAW/'protocol/RBSR_LAMM_DOMINANCE_SEARCH_PROTOCOL_FREEZE.json'
CALIBRATION_HANDLE_ERRATUM = RAW/'protocol/RBSR_CALIBRATION_EXACT_CONTROL_ERRATUM.json'
protocol_payload = {
    'status': 'FROZEN_POSTHOC_RBSR_LAMM_DOMINANCE_SEARCH',
    'claim_boundary': 'supplementary post-hoc development; not blind confirmatory evidence',
    'success_metrics': ['roi_rmse', 'normal_flip_pct', 'edge_strain_p95'],
    'success_rule': 'all three RB-SR means strictly lower than frozen matched LAMM',
    'dimensions': list(DIMENSIONS), 'ridge_lambdas': list(RIDGE_LAMBDAS),
    'strict_top_k': STRICT_TOP_K, 'train_pair_budget': TRAIN_PAIR_BUDGET,
    'validation_relative_margin': VALIDATION_RELATIVE_MARGIN,
    'base_model': {'kind': 'field', 'hidden': 256, 'epochs': 500, 'batch_size': 16},
    'gate_model': {'mode': 'unconstrained', 'hidden': 128, 'epochs': 80, 'initial_gate': 0.5},
    'split_manifest_sha256': sha256_file(SPLIT),
    'train_pair_manifest_sha256': sha256_file(PAIR_MANIFEST),
    'source_pca_package_sha256': sha256_file(SOURCE_PCA_PACKAGE),
    'lamm_config_canonicalisation_erratum_sha256': sha256_file(LAMM_CONFIG_ERRATUM),
    'lamm_validation_summary_sha256': sha256_file(LAMM_VALIDATION_SUMMARY),
    'lamm_test_pair_metrics_sha256': sha256_file(LAMM_TEST_CSV),
    'implementation_sha256': {str(path.relative_to(REPO)): sha256_file(path) for path in required_sources},
}
if PROTOCOL.exists():
    assert valid_sha256_sidecar(PROTOCOL), PROTOCOL
    frozen_protocol = json.loads(PROTOCOL.read_text())
    if frozen_protocol != protocol_payload:
        assert valid_sha256_sidecar(CALIBRATION_HANDLE_ERRATUM), CALIBRATION_HANDLE_ERRATUM
        erratum = json.loads(CALIBRATION_HANDLE_ERRATUM.read_text())
        assert erratum['status'] == 'FROZEN_BEFORE_CALIBRATION_SELECTION_AND_TEST_ACCESS'
        assert erratum['parent_protocol_sha256'] == sha256_file(PROTOCOL)
        frozen_nonimplementation = {k: v for k, v in frozen_protocol.items() if k != 'implementation_sha256'}
        live_nonimplementation = {k: v for k, v in protocol_payload.items() if k != 'implementation_sha256'}
        assert frozen_nonimplementation == live_nonimplementation, 'Non-implementation protocol field changed.'
        for relative_path, old_hash in erratum['old_implementation_sha256'].items():
            assert frozen_protocol['implementation_sha256'][relative_path] == old_hash
        for relative_path, corrected_hash in erratum['corrected_implementation_sha256'].items():
            assert protocol_payload['implementation_sha256'][relative_path] == corrected_hash
        unchanged = set(frozen_protocol['implementation_sha256']) - set(erratum['old_implementation_sha256'])
        assert all(
            frozen_protocol['implementation_sha256'][path] == protocol_payload['implementation_sha256'][path]
            for path in unchanged
        )
        print('Protocol implementation correction validated through exact-control erratum.')
else:
    atomic_write_json(PROTOCOL, protocol_payload); write_sha256_sidecar(PROTOCOL)
print(json.dumps(protocol_payload, indent=2))"""),
    markdown("""## 5. Validation-only Ridge anchor sweep

The complete dimension/lambda grid is ranked by strict free-ROI RMSE. The predeclared top 12 are then fully scored for RMSE, new flip, and edge strain. Among candidates at least 2% below LAMM on all three validation metrics, selection maximises the weakest relative improvement, avoiding a fragile pure-RMSE optimum."""),
    code("""ANCHOR_SWEEP = RAW/'anchor_validation_sweep'
ANCHOR_FREEZE = ANCHOR_SWEEP/'RBSR_ANCHOR_SELECTION_FREEZE.json'
if not valid_sha256_sidecar(ANCHOR_FREEZE):
    run_live([
        sys.executable, '-u', REPO/'scripts/analysis/rbsr_anchor_validation_sweep.py',
        '--repo', LOCAL_DATA, '--split-manifest', SPLIT, '--train-pairs', PAIR_MANIFEST,
        '--source-pca-package', SOURCE_PCA_PACKAGE,
        '--reference-validation-summary', LAMM_VALIDATION_SUMMARY,
        '--dimensions', ','.join(map(str, DIMENSIONS)),
        '--lambdas', ','.join(f'{value:g}' for value in RIDGE_LAMBDAS),
        '--strict-top-k', str(STRICT_TOP_K), '--relative-margin', str(VALIDATION_RELATIVE_MARGIN),
        '--score-chunk-pairs', '320', '--out', ANCHOR_SWEEP,
    ])
anchor_freeze = json.loads(ANCHOR_FREEZE.read_text())
assert anchor_freeze['status'] == 'FROZEN_VALIDATION_SELECTED_RBSR_ANCHOR', anchor_freeze
selected_anchor = anchor_freeze['selected']
SOURCE_DIM = int(selected_anchor['source_pca_dim'])
RIDGE_LAMBDA = float(selected_anchor['ridge_lambda'])
print('Selected validation-dominating anchor:', selected_anchor)"""),
    markdown("""## 6. Train the deterministic residual proposer on the frozen 870 pairs

The proposer uses semantic subunit features and more optimisation steps while preserving the exact training pairs. It is selected on validation only. Checkpoints are written atomically every epoch and resume only if the complete configuration/data signature matches."""),
    code("""import torch
BASE_DIR = RAW/f'optimized_pca{SOURCE_DIM}_lambda{RIDGE_LAMBDA:g}'/'base'
BASE_DIR.mkdir(parents=True, exist_ok=True)
BASE = BASE_DIR/'neural_field_model_package_field_ew0p05_lw0.pt'
if not validate_torch_artifact(BASE, required_keys=('args','field_state_dict','feature_template_sha256','ridge_cond'), repair_sidecar=False):
    command = [
        sys.executable, '-u', '-m', 'rhinoform.train', '--repo', LOCAL_DATA, '--out', BASE_DIR,
        '--epochs', '500', '--batch-size', '16', '--num-workers', '0', '--hidden', '256',
        '--latent-dim', '8', '--source-pca-dim', str(SOURCE_DIM), '--delta-pca-dim', '16',
        '--ridge-lambda', str(RIDGE_LAMBDA), '--lr', '0.001', '--beta', '0.0001',
        '--dense-weight', '1', '--ctrl-weight', '0', '--edge-weight', '0.05',
        '--lap-weight', '0', '--strain-weight', '0', '--model-kind', 'field',
        '--eval-every', '5', '--patience', '40', '--seed', str(SEED),
        '--pair-seed', str(PAIR_SEED), '--device', 'cuda', '--use-subunit-features', 'true',
        '--split-manifest', SPLIT, '--train-pairs-json', PAIR_MANIFEST,
        '--defer-test-evaluation', '--checkpoint-dir', BASE_DIR/'checkpoints',
        '--checkpoint-every', '1', '--no-save-dense-predictions',
    ]
    last = BASE_DIR/'checkpoints/field_last.pt'
    if valid_sha256_sidecar(last): command += ['--resume-checkpoint', last]
    run_live(command)
assert validate_torch_artifact(BASE, required_keys=('args','field_state_dict','feature_template_sha256','ridge_cond'), repair_sidecar=False)
base_package = torch.load(BASE, map_location='cpu', weights_only=False)
assert base_package['split_manifest_sha256'] == sha256_file(SPLIT)
assert base_package['train_pair_manifest_sha256'] == sha256_file(PAIR_MANIFEST)
assert int(base_package['args']['source_pca_dim']) == SOURCE_DIM
assert float(base_package['args']['ridge_lambda']) == RIDGE_LAMBDA
print('Base/proposer complete:', BASE, sha256_file(BASE))"""),
    markdown("""## 7. Train an accuracy-first spatial residual gate

The gate is trained without the old Ridge-relative surrogate budget, because that budget discarded most of the large safety margin to LAMM. Geometry is enforced by the next validation-only strict calibration, using actual new-flip and edge-strain metrics rather than a surrogate."""),
    code("""GATE_DIR = BASE_DIR.parent/'gate_field_unconstrained'
GATE_DIR.mkdir(parents=True, exist_ok=True)
GATE = GATE_DIR/'rbsr_gate_model.pt'
if not validate_torch_artifact(GATE, required_keys=('gate_state_dict','base_model_package_sha256','best_validation','residual_proposer'), repair_sidecar=False):
    command = [
        sys.executable, '-u', REPO/'scripts/training/run_rbsr_gate_with_package_metadata.py',
        '--repo', LOCAL_DATA, '--base-model-package', BASE, '--out', GATE_DIR,
        '--residual-proposer', 'field', '--initial-gate', '0.5',
        '--epochs', '80', '--batch-size', '16', '--hidden', '128', '--lr', '0.0005',
        '--tv-weight', '0.002', '--gate-mean-weight', '0', '--surrogate-baseline', 'target',
        '--orientation-budget-multiplier', '1', '--strain-budget-multiplier', '1',
        '--projection', 'none', '--mode', 'unconstrained', '--eval-every', '5',
        '--seed', str(SEED), '--device', 'cuda', '--checkpoint-every', '1',
    ]
    last = GATE_DIR/'rbsr_gate_last.pt'
    if valid_sha256_sidecar(last): command += ['--resume-checkpoint', last]
    run_live(command)
assert validate_torch_artifact(GATE, required_keys=('gate_state_dict','base_model_package_sha256','best_validation','residual_proposer'), repair_sidecar=False)
gate_package = torch.load(GATE, map_location='cpu', weights_only=False)
assert gate_package['base_model_package_sha256'] == sha256_file(BASE)
assert gate_package['residual_proposer'] == 'field'
print('Gate complete:', GATE, sha256_file(GATE))"""),
    markdown("""## 8. Strict validation evaluation and gate calibration

All 4,830 validation pairs are cached once. Logit calibration only reduces the learned residual where necessary. Feasible points must be at least 2% below frozen LAMM validation on RMSE, new flip, and edge-strain p95; selection then maximises the weakest relative improvement across the three metrics rather than over-optimising RMSE."""),
    code("""# Rebind the correction artifact so a refreshed live runtime can resume here.
CALIBRATION_HANDLE_ERRATUM = RAW/'protocol/RBSR_CALIBRATION_EXACT_CONTROL_ERRATUM.json'
VALIDATION_RAW = BASE_DIR.parent/'validation_raw'
run_live([
    sys.executable, '-u', REPO/'scripts/evaluation/rbsr_gate.py', '--repo', LOCAL_DATA,
    '--base-model-package', BASE, '--rbsr-package', GATE, '--split', 'validation',
    '--out', VALIDATION_RAW, '--batch-size', '16', '--chunk-pairs', '320',
    '--device', 'cuda', '--save-base-comparators',
])
validation_report_path = VALIDATION_RAW/'rbsr_evaluation_validation.json'
assert valid_sha256_sidecar(validation_report_path)
validation_report = json.loads(validation_report_path.read_text())
assert valid_sha256_sidecar(CALIBRATION_HANDLE_ERRATUM), CALIBRATION_HANDLE_ERRATUM
calibration_handle_erratum = json.loads(CALIBRATION_HANDLE_ERRATUM.read_text())
assert calibration_handle_erratum['raw_validation_report_sha256'] == sha256_file(validation_report_path)
assert calibration_handle_erratum['base_model_package_sha256'] == sha256_file(BASE)
assert calibration_handle_erratum['rbsr_package_sha256'] == sha256_file(GATE)
for relative_path, expected_hash in calibration_handle_erratum['corrected_implementation_sha256'].items():
    assert sha256_file(REPO/relative_path) == expected_hash, relative_path
chunk_root = Path(validation_report['chunking']['chunk_root'])
if not chunk_root.is_absolute(): chunk_root = validation_report_path.parent/chunk_root

CALIBRATION_DIR = BASE_DIR.parent/'lamm_reference_calibration_validation'
run_live([
    sys.executable, '-u', REPO/'scripts/evaluation/rbsr_gate_calibration.py',
    '--repo', LOCAL_DATA, '--base-model-package', BASE, '--rbsr-package', GATE,
    '--source-validation-report', validation_report_path, '--source-chunk-root', chunk_root,
    '--out', CALIBRATION_DIR,
    '--logit-offsets', '0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8',
])
CALIBRATION_REPORT = CALIBRATION_DIR/'rbsr_logit_calibration_validation.json'
CALIBRATION_FREEZE = CALIBRATION_DIR/'RBSR_LAMM_REFERENCE_CALIBRATION_FREEZE.json'
run_live([
    sys.executable, '-u', REPO/'scripts/analysis/select_rbsr_reference_dominating_calibration.py',
    '--calibration-report', CALIBRATION_REPORT,
    '--reference-validation-summary', LAMM_VALIDATION_SUMMARY,
    '--relative-margin', str(VALIDATION_RELATIVE_MARGIN), '--out', CALIBRATION_FREEZE,
])
calibration_freeze = json.loads(CALIBRATION_FREEZE.read_text())
assert calibration_freeze['status'] == 'FROZEN_REFERENCE_DOMINATING_VALIDATION_CALIBRATION', calibration_freeze
print('Frozen validation operating point:', calibration_freeze['selected'])"""),
    markdown("""## 9. Matched post-hoc test, clustered inference, and exact verdict

This test is explicitly post-hoc because the internal test was observed in earlier development. It uses the exact frozen validation-selected base/gate/calibration chain and the same 9,900 ordered identity pairs as LAMM. Statistical inference does not treat those pairs as independent: differences are aggregated separately by the 100 source and 100 target identities, the more conservative two-sided Wilcoxon p-value is used, and Holm correction covers the three core metrics. Both clustered bootstrap confidence intervals must also lie below zero for a statistical-improvement judgement."""),
    code("""TEST_DIR = BASE_DIR.parent/'posthoc_matched_test'
if RUN_POSTHOC_TEST:
    run_live([
        sys.executable, '-u', REPO/'scripts/evaluation/rbsr_gate.py', '--repo', LOCAL_DATA,
        '--base-model-package', BASE, '--rbsr-package', GATE,
        '--calibration-freeze', CALIBRATION_FREEZE, '--split', 'test', '--out', TEST_DIR,
        '--batch-size', '16', '--chunk-pairs', '320', '--device', 'cuda', '--save-base-comparators',
    ])
    RBSR_TEST_REPORT = TEST_DIR/'rbsr_evaluation_test.json'
    RBSR_TEST_CSV = TEST_DIR/'pair_metrics_rbsr_test.csv'
    assert valid_sha256_sidecar(RBSR_TEST_REPORT) and valid_sha256_sidecar(RBSR_TEST_CSV)
    rbsr_test = json.loads(RBSR_TEST_REPORT.read_text())
    lamm_test = json.loads(LAMM_TEST_SUMMARY.read_text())
    with RBSR_TEST_CSV.open(newline='') as handle:
        rbsr_rows = list(csv.DictReader(handle))
    with LAMM_TEST_CSV.open(newline='') as handle:
        lamm_rows = list(csv.DictReader(handle))
    rbsr_pairs = [(int(float(row['source_id'])), int(float(row['target_id']))) for row in rbsr_rows]
    lamm_pairs = [(int(float(row['source_id'])), int(float(row['target_id']))) for row in lamm_rows]
    assert len(rbsr_pairs) == len(lamm_pairs) == 9900, (len(rbsr_pairs), len(lamm_pairs))
    assert rbsr_pairs == lamm_pairs, 'RB-SR and frozen LAMM test pair order differs'
    STATS_STEM = TEST_DIR/'rbsr_vs_lamm_core_paired_statistics'
    run_live([
        sys.executable, '-u', REPO/'scripts/evaluation/rbsr_lamm_paired_statistics.py',
        '--lamm-pairs', LAMM_TEST_CSV, '--rbsr-pairs', RBSR_TEST_CSV,
        '--out', STATS_STEM, '--seed', str(SEED), '--n-boot', '20000',
    ])
    STATS_JSON = STATS_STEM.with_suffix('.json')
    STATS_CSV = STATS_STEM.with_suffix('.csv')
    assert valid_sha256_sidecar(STATS_JSON) and valid_sha256_sidecar(STATS_CSV)
    paired_statistics = json.loads(STATS_JSON.read_text())
    assert paired_statistics['lamm_pair_metrics_sha256'] == sha256_file(LAMM_TEST_CSV)
    assert paired_statistics['rbsr_pair_metrics_sha256'] == sha256_file(RBSR_TEST_CSV)
    r = rbsr_test['summary']; l = lamm_test['means']
    metrics = ('roi_rmse', 'normal_flip_pct', 'edge_strain_p95')
    mean_dominance = all(float(r[m]) < float(l[m]) for m in metrics)
    statistical_dominance = paired_statistics['status'] == 'ALL_THREE_CORE_METRICS_SIGNIFICANTLY_IMPROVED'
    verdict = {
        'status': 'PASS' if mean_dominance else 'FAIL_CONTINUE_OPTIMISATION',
        'mean_dominance_all_three': mean_dominance,
        'statistical_dominance_all_three': statistical_dominance,
        'statistical_status': paired_statistics['status'],
        'claim_boundary': 'supplementary post-hoc matched comparison; not blind confirmatory headline',
        'matched_ordered_test_pairs': len(rbsr_pairs),
        'strict_scorer_sha256': sha256_file(strict_scorer),
        'metrics': {m: {'rbsr': float(r[m]), 'lamm': float(l[m]), 'delta': float(r[m])-float(l[m]), 'pass': float(r[m]) < float(l[m])} for m in metrics},
        'rbsr_pair_metrics_sha256': sha256_file(RBSR_TEST_CSV),
        'lamm_pair_metrics_sha256': sha256_file(LAMM_TEST_CSV),
        'base_model_package_sha256': sha256_file(BASE),
        'rbsr_package_sha256': sha256_file(GATE),
        'calibration_freeze_sha256': sha256_file(CALIBRATION_FREEZE),
        'calibration_exact_control_erratum_sha256': sha256_file(CALIBRATION_HANDLE_ERRATUM),
        'paired_statistics_json_sha256': sha256_file(STATS_JSON),
        'paired_statistics_csv_sha256': sha256_file(STATS_CSV),
    }
    VERDICT = TEST_DIR/'RBSR_VS_LAMM_CORE_DOMINANCE_VERDICT.json'
    atomic_write_json(VERDICT, verdict); write_sha256_sidecar(VERDICT)
    print(json.dumps(verdict, indent=2))
    if verdict['status'] != 'PASS':
        print('The target is not yet achieved. Do not claim dominance; retain all checkpoints for the next iteration.')
else:
    print('RUN_POSTHOC_TEST=False: validation-selected model is frozen; test not run.')"""),
    markdown("""## 10. Curate compact evidence and freeze the new result tree

Only when the exact three-metric verdict passes are the protocol, validation references/freezes, pair metrics, summaries, and verdict copied into `release_evidence`. Raw resumable checkpoints remain alongside them under `raw_runs`."""),
    code("""if RUN_POSTHOC_TEST:
    verdict = json.loads(VERDICT.read_text())
    if verdict['status'] == 'PASS':
        copies = {
            PROTOCOL: EVIDENCE/'protocol/RBSR_LAMM_DOMINANCE_SEARCH_PROTOCOL_FREEZE.json',
            CALIBRATION_HANDLE_ERRATUM: EVIDENCE/'protocol/RBSR_CALIBRATION_EXACT_CONTROL_ERRATUM.json',
            PAIR_MANIFEST: EVIDENCE/'protocol/final_rerun_holdout_train_pairs.json',
            ANCHOR_FREEZE: EVIDENCE/'validation/RBSR_ANCHOR_SELECTION_FREEZE.json',
            LAMM_VALIDATION_SUMMARY: EVIDENCE/'validation/lamm_validation_summary.json',
            CALIBRATION_FREEZE: EVIDENCE/'validation/RBSR_LAMM_REFERENCE_CALIBRATION_FREEZE.json',
            LAMM_TEST_SUMMARY: EVIDENCE/'frozen_lamm/lamm_test_summary.json',
            LAMM_TEST_CSV: EVIDENCE/'pair_metrics/identity_bootstrap_pair_metrics_lamm.csv',
            RBSR_TEST_REPORT: EVIDENCE/'rbsr/rbsr_evaluation_test.json',
            RBSR_TEST_CSV: EVIDENCE/'pair_metrics/pair_metrics_rbsr_optimized_test.csv',
            STATS_JSON: EVIDENCE/'analysis/rbsr_vs_lamm_core_paired_statistics.json',
            STATS_CSV: EVIDENCE/'analysis/rbsr_vs_lamm_core_paired_statistics.csv',
            VERDICT: EVIDENCE/'analysis/RBSR_VS_LAMM_CORE_DOMINANCE_VERDICT.json',
        }
        for source, destination in copies.items():
            assert valid_sha256_sidecar(source), source
            atomic_copy_file(source, destination)
        artifacts = [
            {'path': path.relative_to(EVIDENCE).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
            for path in sorted(EVIDENCE.rglob('*'))
            if path.is_file() and not path.name.endswith('.sha256.json') and path.name != 'RELEASE_MANIFEST.json'
        ]
        manifest = {
            'status': 'PASS', 'role': 'supplementary_posthoc_rbsr_lamm_core_dominance',
            'claim_boundary': 'not blind confirmatory evidence', 'artifact_count': len(artifacts),
            'artifacts': artifacts,
        }
        RELEASE = EVIDENCE/'RELEASE_MANIFEST.json'
        atomic_write_json(RELEASE, manifest); write_sha256_sidecar(RELEASE)
        print('CURATED PASS:', len(artifacts), 'artifacts at', EVIDENCE)
    else:
        print('No release package was created because the three-metric target did not pass.')"""),
]


payload = json.dumps(
    {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    },
    ensure_ascii=False,
    separators=(",", ":"),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
for attempt in range(1, 6):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUT.name}.", suffix=".tmp", dir=str(OUT.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, OUT)
        break
    except OSError:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        if attempt == 5:
            raise
        time.sleep(attempt)
print(OUT)
