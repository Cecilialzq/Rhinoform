"""Build the all-method final-rerun post-hoc subunit/spatial/noise Colab notebook."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb"


def md(value): return {"cell_type":"markdown","metadata":{},"source":value.splitlines(keepends=True)}
def code(value): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":value.splitlines(keepends=True)}


cells=[
md("""# Rhinoform — five nasal subunits × all metrics × all frozen methods

This notebook is a **secondary post-hoc** analysis of the completed internal final rerun. It never overwrites the original model freeze, receipts or `FINAL_CONFIRMATION_EVIDENCE.json`, and never trains a model. All newly generated artifacts are written inside the final GitHub-ready repository under `results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1`.

The five frozen vertex masks are unchanged. Reportable faces reproduce the old script's **legacy three-vertex majority vote** and frozen tie order; unique 2-of-3 majority is a sensitivity audit only. An edge belongs to a region only when **both endpoints** belong to it; cross-region edges are boundary.

All eight methods are included: Ridge, certified RB-SR, CVAE, Hybrid, official frozen LAMM, Laplacian, bi-Laplacian and ARAP. Every dense prediction needed by this analysis is recomputed from repository-frozen checkpoints or frozen classical configurations; no private archival prediction chunks are read. Every recomputed chunk must reproduce the repository's original global pair metrics before new regional rows are accepted."""),
md("""## Frozen analysis contract

Five-region metrics are free-vertex RMSE, hard-fixed landmark RMSE, intra-region `edge_strain_p95`, target-relative `normal_flip_pct`, absolute/missed/target flip transparency. All seven are reported. Hard-fixed landmark RMSE and target flip are deterministic descriptive sanity quantities rather than method effects, so they are excluded from hypothesis testing. The method-dependent families are 7×5×5 = **175** all-vs-Ridge Holm tests and 1×5×5 = **25** certified-RB-SR-vs-LAMM tests.

Spatial diagnostics contain every method's mean/worst per-vertex error, per-face new/absolute/missed/target **flip-frequency**, mean/worst per-edge strain, p50/p90/p95/p99/max and top-decile spatial mass.

Noise robustness uses all **4,830** frozen validation pairs for Certified RB-SR, Ridge, LAMM and ARAP, at 0/0.25/0.5/**1.0 mm** with three fixed nonzero-noise seeds. Every method receives identical hashed noise for each pair/seed. Seeds are averaged within pair before source-identity inference; a target-identity bootstrap CI is the cluster-sensitivity gate. Scoring hard-fixes unnoised true controls and reports ROI RMSE, **target-relative new flip** and edge strain. The old absolute-flip/nonzero-landmark noise results are not reused, and no operating point is reselected. Noisy-vs-clean family: 4×3×3 = **36** Holm tests.

Six qualitative cases are frozen by deterministic rules before dense re-inference: median, P10/P90 control magnitude, high certificate iterations, largest RB-SR gain over Ridge, and the **worst real failure** against the better of Ridge/LAMM. The three method error maps use a shared error scale and all target-relative new-flip faces are overlaid."""),
md("""## Personalization and interpretation boundary

The personalization ablation uses all 4,830 frozen validation pairs and identical landmark controls across four variants: independently fitted **controls-only Ridge**, frozen **source-PCA Ridge**, frozen full Ridge with **mean-source-code**, and frozen full Ridge with an identity-level **shuffled-source-code** cyclic derangement. The primary family is three source-PCA-vs-alternative free-ROI RMSE tests; flip and strain form a separate six-test geometry family. No test result or operating point is used.

The qualitative panels show source, target, the sparse **landmark edit**, Ridge, Certified RB-SR and LAMM, with error, target-relative new-flip and **strain overlay** evidence for both successes and the mandatory failure. Without a clinical observer study these outputs may only be described as **geometrically natural/regular** and **must not be called clinically preferred**."""),
code("""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import importlib, json, os, shutil, subprocess, sys

RUN_POSTHOC_SUBUNITS = True
RUN_NOISE_ROBUSTNESS = True
RUN_QUALITATIVE_CASES = True
RUN_PERSONALIZATION_ABLATION = True
LAMM_COMMIT='87354c05dec341c6d8dd319665dd52553fb03084'
CANONICAL_REPO=Path('/content/drive/MyDrive/Rhinoform_GitHub_Ready_20260808')
# Licensed FaceScape meshes remain external to the submission repository.
# Evaluators may set this environment variable to their independently obtained
# processed FaceScape directory. The private candidate below is only a
# convenience for the author's Drive and is never used for models/results.
FACESCAPE_OVERRIDE=os.environ.get('RHINOFORM_FACESCAPE_DATA_ROOT')
# Work from a disposable runtime overlay. The canonical Drive repository is
# copied read-only and is never patched by this notebook.
REPO=Path('/content/Rhinoform_GitHub_Ready_20260808_posthoc')
RESULTS=CANONICAL_REPO/'results/rbsr_final_rerun_holdout_v1'
OUT=RESULTS/'posthoc_subunit_analysis_v1'
EVIDENCE=OUT/'POSTHOC_SUBUNIT_ANALYSIS_EVIDENCE.json'
LOCAL_DATA=Path('/content/rhinoform_final_rerun_holdout/data')
LAMM_ROOT=Path('/content/LAMM_official')
LAMM_RUNTIME=Path('/content/rhinoform_lamm_inference_runtime')
LEGACY_LAMM_ARTIFACTS=Path('/content/drive/MyDrive/FYP final/results/rbsr_final_rerun_holdout_v1/lamm/seed20260609')
# Prefer the exact mount used by the successful final rerun. A repo-only
# evaluator falls back to the copied canonical artifacts; both sources are
# checked against the GitHub result-tree manifest before inference.
LAMM_ARTIFACTS=(
    LEGACY_LAMM_ARTIFACTS
    if (LEGACY_LAMM_ARTIFACTS/'manipulation_best.pt').is_file()
    else RESULTS/'lamm/seed20260609'
)
LIVE_ENV={
    **os.environ,
    'PYTHONUNBUFFERED':'1',
    'PYTHONDONTWRITEBYTECODE':'1',
    'RHINOFORM_LAMM_ARTIFACT_ROOT':str(LAMM_ARTIFACTS),
}
os.environ['RHINOFORM_LAMM_ARTIFACT_ROOT']=str(LAMM_ARTIFACTS)

def run_live(cmd, *, env=None):
    command=[str(v) for v in cmd]; print('RUN:',' '.join(command),flush=True)
    process_env=LIVE_ENV if env is None else env
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=process_env)
    try:
        assert p.stdout is not None
        for line in iter(p.stdout.readline,''): print(line,end='',flush=True)
        rc=p.wait()
    except KeyboardInterrupt: p.terminate(); p.wait(); raise
    finally:
        if p.stdout: p.stdout.close()
    if rc: raise subprocess.CalledProcessError(rc,command)

assert CANONICAL_REPO.is_dir() and RESULTS.is_dir()
REPO.mkdir(parents=True,exist_ok=True)
# Always refresh the small code/test overlay. This is idempotent and repairs a
# partially staged runtime without touching data or canonical results.
for directory in ('rhinoform','scripts','experiments','roi','tests','tools','notebooks'):
    source=CANONICAL_REPO/directory; assert source.is_dir(), source
    run_live(['rsync','-a',f'{source}/',f'{REPO/directory}/'])
for name in ('pyproject.toml','requirements.txt','requirements-lamm-inference.txt','seed_registry.json'):
    source=CANONICAL_REPO/name
    if source.is_file(): shutil.copy2(source,REPO/name)
# The public manifest is part of the repository contract; licensed meshes are
# deliberately not copied into the runtime code overlay.
(REPO/'data').mkdir(parents=True,exist_ok=True)
shutil.copy2(CANONICAL_REPO/'data/manifest.json',REPO/'data/manifest.json')
required_overlay_files=[
    REPO/'data/manifest.json',
    REPO/'requirements-lamm-inference.txt',
    REPO/'notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb',
    REPO/'rhinoform/regional_analysis.py',
    REPO/'scripts/evaluation/posthoc_subunit_analysis.py',
    REPO/'scripts/evaluation/posthoc_strict_noise_robustness.py',
    REPO/'scripts/evaluation/posthoc_strict_noise_resume_erratum.py',
    REPO/'scripts/evaluation/posthoc_qualitative_cases.py',
    REPO/'scripts/evaluation/posthoc_personalization_ablation.py',
    REPO/'scripts/evaluation/posthoc_personalization_resume_erratum.py',
    REPO/'scripts/evaluation/posthoc_preflight.py',
    REPO/'tests/test_regional_analysis.py',
    REPO/'tests/test_posthoc_subunit_notebook_contract.py',
    REPO/'tests/test_posthoc_strict_noise_contract.py',
    REPO/'tests/test_posthoc_qualitative_contract.py',
    REPO/'tests/test_posthoc_personalization_contract.py',
    REPO/'tests/test_posthoc_preflight.py',
]
for path in required_overlay_files: assert path.is_file(), path
print('PASS: repository-only runtime copy ready; canonical repo unchanged:',REPO)
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
LIVE_ENV['PYTHONPATH']=str(REPO)+(os.pathsep+LIVE_ENV['PYTHONPATH'] if LIVE_ENV.get('PYTHONPATH') else '')
os.environ['PYTHONPATH']=LIVE_ENV['PYTHONPATH']; importlib.invalidate_caches()
import rhinoform
expected_init=(REPO/'rhinoform/__init__.py').resolve()
assert Path(rhinoform.__file__).resolve()==expected_init, (
    'Kernel imported rhinoform from the wrong repository: '
    f'{rhinoform.__file__} != {expected_init}'
)
probe=subprocess.run(
    [sys.executable,'-c','import pathlib,rhinoform; print(pathlib.Path(rhinoform.__file__).resolve())'],
    check=True,capture_output=True,text=True,env=LIVE_ENV,
)
assert Path(probe.stdout.strip()).resolve()==expected_init, (
    'Subprocess imported rhinoform from the wrong repository: '+probe.stdout
)
print('PASS: kernel and subprocess import directly from runtime overlay:',expected_init)
from rhinoform.repro import sha256_file, valid_sha256_sidecar
from scripts.evaluation.posthoc_preflight import discover_facescape_data_root
facescape_candidates=[]
if FACESCAPE_OVERRIDE: facescape_candidates.append(Path(FACESCAPE_OVERRIDE))
facescape_candidates.extend([
    CANONICAL_REPO/'data',
    Path('/content/drive/MyDrive/FYP final/data'),
])
DATA_ROOT=discover_facescape_data_root(
    facescape_candidates,
    expected_manifest_sha256=sha256_file(CANONICAL_REPO/'data/manifest.json'),
)
print('Licensed FaceScape data root:',DATA_ROOT)
print('LAMM artifact source:',LAMM_ARTIFACTS)
ANALYSIS_COMPLETE=False
if valid_sha256_sidecar(EVIDENCE):
    report=json.loads(EVIDENCE.read_text())
    essentials=all(valid_sha256_sidecar(OUT/n) and sha256_file(OUT/n)==h for n,h in report.get('outputs',{}).items())
    spatial=all(valid_sha256_sidecar(OUT/'spatial'/n) and sha256_file(OUT/'spatial'/n)==h for n,h in report.get('spatial_outputs',{}).items())
    noise=OUT/'noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json'
    qualitative=OUT/'qualitative/QUALITATIVE_CASES_EVIDENCE.json'
    personalization=OUT/'personalization/PERSONALIZATION_ABLATION_EVIDENCE.json'
    ANALYSIS_COMPLETE=(report.get('status')=='COMPLETE_POSTHOC_SUBUNIT_ANALYSIS' and essentials and len(report.get('spatial_outputs',{}))==8 and spatial and valid_sha256_sidecar(noise) and sha256_file(noise)==report.get('strict_noise_evidence_sha256') and valid_sha256_sidecar(qualitative) and sha256_file(qualitative)==report.get('qualitative_evidence_sha256') and valid_sha256_sidecar(personalization) and sha256_file(personalization)==report.get('personalization_evidence_sha256'))
print('Complete fast path:',ANALYSIS_COMPLETE); print('Output:',OUT)"""),
md("""## 1. Stage licensed mesh cache and run the full read-only preflight

This cell deliberately checks the complete dependency chain before any long computation: licensed data and manifest, GPU, frozen receipts and hash sidecars, all canonical pair tables and ordering, every frozen implementation hash, Base/Gate/LAMM checkpoint loading, the clean pinned official LAMM checkout, output safety and resumable artifacts. Do not continue unless it ends with `POSTHOC PREFLIGHT PASS`. A pass cannot prevent an external Colab/network interruption, but it eliminates predictable setup and artifact-contract failures up front."""),
code("""if RUN_POSTHOC_SUBUNITS and not ANALYSIS_COMPLETE:
    mesh_root=DATA_ROOT/'meshes'
    assert len(list(mesh_root.glob('*.npz')))==846, (
        'FaceScape is licensed and is not distributed by this repository. '
        'Place the 846 processed meshes under repo/data/meshes, or set DATA_ROOT '
        'to an independently obtained licensed dataset directory.'
    )
    LOCAL_DATA.mkdir(parents=True,exist_ok=True); shutil.copy2(DATA_ROOT/'manifest.json',LOCAL_DATA/'manifest.json')
    if len(list((LOCAL_DATA/'meshes').glob('*.npz'))) != 846:
        run_live(['rsync','-a','--info=progress2',f'{mesh_root}/',f'{LOCAL_DATA/"meshes"}/'])
    assert len(list((LOCAL_DATA/'meshes').glob('*.npz')))==846
    def lamm_checkout_valid():
        if not (LAMM_ROOT/'.git').is_dir(): return False
        try:
            head=subprocess.check_output(['git','-C',str(LAMM_ROOT),'rev-parse','HEAD'],text=True).strip()
            dirty=subprocess.check_output(['git','-C',str(LAMM_ROOT),'status','--porcelain','--untracked-files=all'],text=True).strip()
            return head==LAMM_COMMIT and not dirty
        except subprocess.CalledProcessError: return False
    if LAMM_ROOT.exists() and not lamm_checkout_valid():
        # This is a disposable /content checkout, never the Drive repository.
        shutil.rmtree(LAMM_ROOT)
    if not LAMM_ROOT.exists():
        # Fetch only the pinned snapshot, not the repository's full history.
        LAMM_ROOT.mkdir(parents=True)
        run_live(['git','-C',LAMM_ROOT,'init','-q'])
        run_live(['git','-C',LAMM_ROOT,'remote','add','origin','https://github.com/michaeltrs/LAMM.git'])
        run_live(['git','-C',LAMM_ROOT,'fetch','--depth','1','--filter=blob:none','origin',LAMM_COMMIT])
        run_live(['git','-C',LAMM_ROOT,'checkout','-q','--detach','FETCH_HEAD'])
    assert lamm_checkout_valid(), 'Unable to establish the clean pinned official LAMM checkout'

    # The official commit's complete requirements file pins an old Torch/NumPy
    # stack.  Do not replace Colab's working GPU stack.  Install only the four
    # packages reached by the frozen LAMM inference import graph into a
    # disposable, isolated /content directory.  A content-hash marker makes
    # the operation idempotent while still rebuilding after a requirements edit.
    lamm_requirements=REPO/'requirements-lamm-inference.txt'
    lamm_requirements_sha=sha256_file(lamm_requirements)
    lamm_runtime_marker=LAMM_RUNTIME/'.requirements.sha256'
    lamm_runtime_ready=(
        lamm_runtime_marker.is_file()
        and lamm_runtime_marker.read_text(encoding='utf-8').strip()==lamm_requirements_sha
    )
    if not lamm_runtime_ready:
        if LAMM_RUNTIME.exists():
            # Disposable runtime dependency cache only; never Drive/results.
            shutil.rmtree(LAMM_RUNTIME)
        LAMM_RUNTIME.mkdir(parents=True)
        run_live([
            sys.executable,'-m','pip','install','--disable-pip-version-check',
            '--no-input','--no-deps','--target',LAMM_RUNTIME,
            '-r',lamm_requirements,
        ])
        lamm_runtime_marker.write_text(lamm_requirements_sha+'\\n',encoding='utf-8')
    lamm_runtime_str=str(LAMM_RUNTIME)
    if lamm_runtime_str not in sys.path: sys.path.insert(0,lamm_runtime_str)
    LIVE_ENV['PYTHONPATH']=(
        lamm_runtime_str+os.pathsep+LIVE_ENV['PYTHONPATH']
        if LIVE_ENV.get('PYTHONPATH') else lamm_runtime_str
    )
    # Import the exact official entry point in a fresh process.  This catches
    # trimesh/PyYAML/einops/timm and torchvision compatibility before preflight.
    lamm_probe_env={**LIVE_ENV}
    lamm_probe_env['PYTHONPATH']=str(LAMM_ROOT)+os.pathsep+lamm_probe_env['PYTHONPATH']
    run_live([
        sys.executable,'-c',
        'from models import LAMM; import trimesh, yaml, einops, timm; '
        'print("LAMM dependency bootstrap PASS")',
    ],env=lamm_probe_env)
    run_live([
        sys.executable,'-u',REPO/'scripts/evaluation/posthoc_preflight.py',
        '--repo-root',REPO,'--repo',LOCAL_DATA,'--results',RESULTS,'--out',OUT,
        '--lamm-root',LAMM_ROOT,'--require-gpu',
    ])
else: print('Complete fast path or run flag false: staging skipped')"""),
md("""## 2. Fail-fast tests"""),
code("""if RUN_POSTHOC_SUBUNITS and not ANALYSIS_COMPLETE:
    for path in [REPO/'rhinoform/regional_analysis.py',REPO/'scripts/evaluation/posthoc_subunit_analysis.py',REPO/'scripts/evaluation/posthoc_strict_noise_robustness.py',REPO/'scripts/evaluation/posthoc_strict_noise_resume_erratum.py',REPO/'scripts/evaluation/posthoc_qualitative_cases.py',REPO/'scripts/evaluation/posthoc_personalization_ablation.py',REPO/'scripts/evaluation/posthoc_personalization_resume_erratum.py']:
        compile(path.read_text(),str(path),'exec')
    run_live([sys.executable,'-m','pytest','-q',REPO/'tests/test_regional_analysis.py',REPO/'tests/test_posthoc_subunit_notebook_contract.py',REPO/'tests/test_posthoc_strict_noise_contract.py',REPO/'tests/test_posthoc_qualitative_contract.py',REPO/'tests/test_posthoc_personalization_contract.py',REPO/'tests/test_posthoc_preflight.py'])"""),
md("""## 3. Freeze mapping, formulas, spatial diagnostics and statistical families"""),
code("""SCRIPT=REPO/'scripts/evaluation/posthoc_subunit_analysis.py'
def run_stage(stage):
    if not RUN_POSTHOC_SUBUNITS or ANALYSIS_COMPLETE: print('skip',stage); return
    run_live([sys.executable,'-u',SCRIPT,'--stage',stage,'--repo-root',REPO,'--repo',LOCAL_DATA,'--results',RESULTS,'--out',OUT,'--lamm-root',LAMM_ROOT,'--neural-batch-size','32','--lamm-batch-size','32','--chunk-pairs','160','--seed','20260609','--n-boot','10000'])
run_stage('audit')"""),
md("""## 4. Frozen-validation source-personalization ablation"""),
code("""if RUN_POSTHOC_SUBUNITS and RUN_PERSONALIZATION_ABLATION and not ANALYSIS_COMPLETE:
    run_live([sys.executable,'-u',REPO/'scripts/evaluation/posthoc_personalization_resume_erratum.py','--repo-root',REPO,'--repo',LOCAL_DATA,'--results',RESULTS,'--out',OUT,'--chunk-pairs','200','--seed','20260810','--n-boot','10000'])
if RUN_POSTHOC_SUBUNITS or ANALYSIS_COMPLETE:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    personalization=pd.read_csv(OUT/'personalization/personalization_ablation_summary.csv')
    display(personalization); display(pd.read_csv(OUT/'personalization/paired_primary_personalization_rmse.csv'))
    demo=OUT/'personalization/fixed_control_personalization_demo.npz'; assert valid_sha256_sidecar(demo)
    FIGURES=OUT/'figures'; FIGURES.mkdir(parents=True,exist_ok=True)
    with np.load(demo) as d:
        vmax=float(np.max(d['response_difference'])); fig,axes=plt.subplots(4,4,figsize=(12,11),constrained_layout=True)
        for column in range(4):
            source=d['source_vertices'][column]; lm=d['landmarks']; edited_lm=source[lm]+d['fixed_control_delta']
            axes[0,column].scatter(source[:,0],source[:,1],s=1,c='#a0a0a0'); axes[0,column].scatter(edited_lm[:,0],edited_lm[:,1],s=16,c='#D55E00')
            for a,b in zip(source[lm],edited_lm): axes[0,column].plot([a[0],b[0]],[a[1],b[1]],c='#333333',lw=.7)
            axes[0,column].set_title(f"source {d['source_ids'][column]}, PC1={d['source_pc1'][column]:.2f}")
            axes[1,column].scatter(d['controls_only_vertices'][column,:,0],d['controls_only_vertices'][column,:,1],s=1,c='#0072B2')
            axes[2,column].scatter(d['source_pca_vertices'][column,:,0],d['source_pca_vertices'][column,:,1],s=1,c='#D55E00')
            artist=axes[3,column].scatter(d['source_pca_vertices'][column,:,0],d['source_pca_vertices'][column,:,1],c=d['response_difference'][column],s=1.5,cmap='magma',vmin=0,vmax=vmax)
            for row in range(4): axes[row,column].set_aspect('equal'); axes[row,column].axis('off')
        for row,label in enumerate(('Same landmark edit','Controls-only Ridge','Source-PCA Ridge','Response difference')): axes[row,0].text(-.08,.5,label,rotation=90,transform=axes[row,0].transAxes,ha='right',va='center')
        fig.colorbar(artist,ax=axes[3,:],label='|source-PCA response − controls-only response| (mm)',fraction=.02)
        fig.suptitle('Fixed-control personalization demo — same edit, four deterministic source faces; no ground-truth accuracy claim')
        fig.savefig(FIGURES/'fixed_control_personalization_demo.pdf',bbox_inches='tight'); fig.savefig(FIGURES/'fixed_control_personalization_demo.png',dpi=300,bbox_inches='tight'); plt.show(); plt.close(fig)"""),
md("""## 5. Recompute Ridge/CVAE/Hybrid/certified RB-SR and accumulate spatial diagnostics"""),
code("""run_stage('neural')"""),
md("""## 6. Frozen classical solve-only regional/spatial extraction"""),
code("""run_stage('classical')"""),
md("""## 7. Official frozen LAMM inference only"""),
code("""if RUN_POSTHOC_SUBUNITS and not ANALYSIS_COMPLETE:
    head=subprocess.check_output(['git','-C',str(LAMM_ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(LAMM_ROOT),'status','--porcelain','--untracked-files=all'],text=True).strip()
    assert head==LAMM_COMMIT and not dirty
run_stage('lamm')"""),
md("""## 8. Deterministic qualitative comparison, including the worst real failure"""),
code("""if RUN_POSTHOC_SUBUNITS and RUN_QUALITATIVE_CASES and not ANALYSIS_COMPLETE:
    run_live([sys.executable,'-u',REPO/'scripts/evaluation/posthoc_qualitative_cases.py','--repo-root',REPO,'--repo',LOCAL_DATA,'--results',RESULTS,'--out',OUT,'--lamm-root',LAMM_ROOT,'--batch-size','16'])"""),
md("""## 9. Render landmark edits, shared-scale error/strain and target-relative new-flip overlays"""),
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
FIGURES=OUT/'figures'; FIGURES.mkdir(parents=True,exist_ok=True)
case_paths=sorted((OUT/'qualitative/cases').glob('case_*.npz'))
if RUN_POSTHOC_SUBUNITS or ANALYSIS_COMPLETE:
    assert len(case_paths)==6
    global_error_max=0.0; global_strain_max=0.0
    for path in case_paths:
        with np.load(path) as d:
            for method in ('ridge','certified_rbsr','lamm'):
                global_error_max=max(global_error_max,float(np.max(d[f'{method}_vertex_error'])))
                global_strain_max=max(global_strain_max,float(np.max(d[f'{method}_edge_strain'])))
    for path in case_paths:
        assert valid_sha256_sidecar(path)
        with np.load(path) as d:
            fig,axes=plt.subplots(3,6,figsize=(18,9.5),constrained_layout=True)
            panels=[('source_vertices','Source'),(None,'Landmark edit'),('target_vertices','Target'),('ridge_vertices','Ridge'),('certified_rbsr_vertices','Certified RB-SR'),('lamm_vertices','LAMM')]
            for ax,(key,title) in zip(axes[0],panels):
                if key is None:
                    vertices=d['source_vertices']; lm=d['landmarks']; target_lm=d['target_vertices'][lm]
                    ax.scatter(vertices[:,0],vertices[:,1],s=1.0,c='#b0b0b0'); ax.scatter(vertices[lm,0],vertices[lm,1],s=20,c='#0072B2',label='source control'); ax.scatter(target_lm[:,0],target_lm[:,1],s=20,c='#D55E00',label='edited control')
                    for a,b in zip(vertices[lm],target_lm): ax.plot([a[0],b[0]],[a[1],b[1]],c='#333333',lw=.8)
                    ax.legend(fontsize=6,frameon=False); ax.set_title(title); ax.set_aspect('equal'); ax.axis('off'); continue
                vertices=d[key]; ax.scatter(vertices[:,0],vertices[:,1],s=1.0,c='#8c8c8c')
                method=key.removesuffix('_vertices')
                if method in ('ridge','certified_rbsr','lamm'):
                    centres=vertices[d['faces']].mean(1); mask=d[f'{method}_new_flip_faces'].astype(bool)
                    ax.scatter(centres[mask,0],centres[mask,1],s=5,c='#D62728',label='new-flip')
                ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
            for ax in axes[1,:3]: ax.axis('off')
            axes[1,0].text(0,1,f"role: {d['role'][0]}\\npair: {d['source_id'][0]}→{d['target_id'][0]}\\nretention: {float(d['certificate_retention'][0]):.4f}\\niterations: {int(d['certificate_iterations'][0])}",va='top')
            error_artist=None
            for ax,method,title in zip(axes[1,3:],('ridge','certified_rbsr','lamm'),('Ridge error','Certified RB-SR error','LAMM error')):
                vertices=d[f'{method}_vertices']; error_artist=ax.scatter(vertices[:,0],vertices[:,1],c=d[f'{method}_vertex_error'],s=1.5,cmap='magma',vmin=0,vmax=global_error_max)
                ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
            fig.colorbar(error_artist,ax=axes[1,3:],label='Per-vertex error (mm), shared globally',fraction=.02)
            for ax in axes[2,:3]: ax.axis('off')
            strain_artist=None
            for ax,method,title in zip(axes[2,3:],('ridge','certified_rbsr','lamm'),('Ridge strain','Certified RB-SR strain','LAMM strain')):
                vertices=d[f'{method}_vertices']; midpoint=vertices[d['edges']].mean(1); strain_artist=ax.scatter(midpoint[:,0],midpoint[:,1],c=d[f'{method}_edge_strain'],s=1.2,cmap='viridis',vmin=0,vmax=global_strain_max)
                ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
            fig.colorbar(strain_artist,ax=axes[2,3:],label='Relative edge strain, shared globally',fraction=.02)
            fig.suptitle(f"{d['role'][0]} — red = target-relative new-flip; shared error and strain overlay scales")
            stem=FIGURES/path.stem; fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight'); plt.show(); plt.close(fig)"""),
md("""## 10. New STRICT all-method noise robustness

The completed final-rerun validation tables are the authoritative **0 mm** baseline. A later live GPU replay agrees in continuous metrics within the frozen numerical tolerance, but one face at pair 3623 lies on the target-relative orientation boundary and is not bitwise stable across re-inference. The earlier batch-size-16 hypothesis was falsified by an identical rerun and is explicitly superseded. This entry point restores the actual batch size of 32, records a hashed replay audit, reuses the frozen clean rows without re-estimation, and still performs every nonzero-noise inference normally."""),
code("""if RUN_POSTHOC_SUBUNITS and RUN_NOISE_ROBUSTNESS and not ANALYSIS_COMPLETE:
    run_live([sys.executable,'-u',REPO/'scripts/evaluation/posthoc_strict_noise_resume_erratum.py','--repo-root',REPO,'--repo',LOCAL_DATA,'--results',RESULTS,'--out',OUT,'--lamm-root',LAMM_ROOT,'--chunk-pairs','200','--batch-size','32','--seed','20260810','--n-boot','10000'])"""),
md("""## 11. Aggregate regions, tails, spatial maps and all Holm families"""),
code("""if RUN_POSTHOC_SUBUNITS and not ANALYSIS_COMPLETE:
    assert RUN_NOISE_ROBUSTNESS, 'Aggregation requires the frozen strict-noise evidence; enable RUN_NOISE_ROBUSTNESS.'
    assert RUN_QUALITATIVE_CASES, 'Aggregation requires deterministic qualitative evidence; enable RUN_QUALITATIVE_CASES.'
    assert RUN_PERSONALIZATION_ABLATION, 'Aggregation requires frozen personalization evidence; enable RUN_PERSONALIZATION_ABLATION.'
run_stage('aggregate')
if RUN_POSTHOC_SUBUNITS or ANALYSIS_COMPLETE:
    assert valid_sha256_sidecar(EVIDENCE)
    print(json.dumps(json.loads(EVIDENCE.read_text()),indent=2))"""),
md("""## 12. Publication figures (PDF + 300-dpi PNG)"""),
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
means=pd.read_csv(OUT/'subunit_means_all_methods.csv'); display(means)
regions=['root','dorsum','tip','alar_left','alar_right']; methods=['ridge','certified_rbsr','cvae','hybrid','lamm','laplacian','bilaplacian','arap']
metrics=['free_rmse','landmark_rmse','edge_strain_p95','normal_flip_pct','abs_flip_pct','missed_flip_pct','target_flip_pct']
colors=['#0072B2','#D55E00','#56B4E9','#009E73','#CC79A7','#E69F00','#000000','#999999']; markers=['o','D','s','^','P','v','X','*']; styles=['-','-','--','-.','-',':','--','-.']
FIGURES=OUT/'figures'; FIGURES.mkdir(parents=True,exist_ok=True)
for metric in metrics:
    fig,ax=plt.subplots(figsize=(7,4.2),constrained_layout=True)
    for i,method in enumerate(methods):
        block=means[means.method==method].set_index('region').loc[regions]
        ax.plot(regions,block[metric],label=method,color=colors[i],marker=markers[i],linestyle=styles[i],linewidth=2.8 if method=='certified_rbsr' else 1.25)
    ax.set_xlabel('Frozen nasal subunit'); ax.set_ylabel(metric); ax.grid(axis='y',alpha=.25); ax.spines[['top','right']].set_visible(False); ax.legend(ncol=2,fontsize=8,frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.savefig(FIGURES/f'subunits_{metric}.pdf',bbox_inches='tight'); fig.savefig(FIGURES/f'subunits_{metric}.png',dpi=300,bbox_inches='tight'); plt.show(); plt.close(fig)"""),
md("""## 13. Per-method error / flip-frequency / strain spatial maps and tails"""),
code("""display(pd.read_csv(OUT/'spatial_tail_diagnostics_all_methods.csv')); display(pd.read_csv(OUT/'subunit_pair_tail_diagnostics.csv'))
for method in methods:
    path=OUT/'spatial'/f'{method}_spatial_diagnostics.npz'; assert valid_sha256_sidecar(path)
    with np.load(path) as d:
        v,f,e=d['display_vertices'],d['faces'],d['edges']; maps=[(v,d['vertex_error_mean'],'Mean vertex error'),(v[f].mean(1),d['face_new_flip_frequency_pct'],'New-flip frequency (%)'),(v[e].mean(1),d['edge_strain_mean'],'Mean edge strain')]
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),constrained_layout=True)
    for ax,(xy,c,title) in zip(axes,maps):
        artist=ax.scatter(xy[:,0],xy[:,1],c=c,s=2.2,cmap='viridis'); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title); fig.colorbar(artist,ax=ax,fraction=.046)
    fig.suptitle(method); fig.savefig(FIGURES/f'spatial_diagnostics_{method}.pdf',bbox_inches='tight'); fig.savefig(FIGURES/f'spatial_diagnostics_{method}.png',dpi=300,bbox_inches='tight'); plt.show(); plt.close(fig)"""),
md("""## 14. STRICT noise curves and paired statistics"""),
code("""noise=pd.read_csv(OUT/'noise/noise_robustness_summary.csv'); display(noise); display(pd.read_csv(OUT/'noise/noise_vs_clean_paired_statistics.csv'))
for metric in ['roi_rmse','edge_strain_p95','normal_flip_pct']:
    fig,ax=plt.subplots(figsize=(6.8,4.1),constrained_layout=True)
    for i,method in enumerate(methods):
        block=noise[noise.method==method].sort_values('noise_mm'); ax.plot(block.noise_mm,block[f'{metric}_mean'],label=method,color=colors[i],marker=markers[i],linestyle=styles[i],linewidth=2.8 if method=='certified_rbsr' else 1.25)
    ax.set_xlabel('Landmark-control noise σ (mm)'); ax.set_ylabel(metric); ax.grid(alpha=.25); ax.spines[['top','right']].set_visible(False); ax.legend(ncol=2,fontsize=8,frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.savefig(FIGURES/f'noise_{metric}.pdf',bbox_inches='tight'); fig.savefig(FIGURES/f'noise_{metric}.png',dpi=300,bbox_inches='tight'); plt.show(); plt.close(fig)"""),
md("""## 15. Hash all derived publication figures"""),
code("""from rhinoform.repro import atomic_write_json, write_sha256_sidecar
figure_hashes={}
for path in sorted(FIGURES.iterdir()):
    if path.suffix.lower() in {'.pdf','.png'}:
        write_sha256_sidecar(path); assert valid_sha256_sidecar(path); figure_hashes[path.name]=sha256_file(path)
FIGURE_EVIDENCE=FIGURES/'FIGURE_EVIDENCE.json'
atomic_write_json(FIGURE_EVIDENCE,{'status':'COMPLETE_HASHED_PUBLICATION_FIGURES','dpi_png':300,'qualitative_error_scale':'one shared global maximum across all cases and methods','outputs':figure_hashes})
write_sha256_sidecar(FIGURE_EVIDENCE); print('Hashed figures:',len(figure_hashes))"""),
md("""## Interpretation boundary

Reportable rows use the legacy face vote. Unique-majority results are sensitivity only. Spatial/noise results are secondary post-hoc evidence and must not be described as a new blind confirmation."""),
]
for i,c in enumerate(cells): c["id"]=f"posthoc-subunit-{i:02d}"
notebook={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"name":OUT.name,"provenance":[]},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.x"}},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n"); print(OUT)
