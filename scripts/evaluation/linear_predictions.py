"""Compute linear-method predictions only (no metrics). Fast. Save npz."""
import time
from pathlib import Path
import numpy as np
from rhinoform.data import (load_rows, split_ids, ordered_pairs, pair_arrays,
    fit_pca, pca_coeff, ridge_fit, ridge_predict)
from rhinoform.geometry import uniform_laplacian, solve_linear_handle_baseline

REPO=Path("data"); OUT=Path("large/classical"); OUT.mkdir(parents=True,exist_ok=True)
SEED=2026; HW=1000.0; RIDGE_SYS=1e-8; PCADIM=16; LAM=100.0
t0=time.time()
_,by_id=load_rows(REPO)
tr=split_ids(by_id,"clean-prior train"); te=split_ids(by_id,"main test")
trp=ordered_pairs(tr,None,SEED); tep=ordered_pairs(te,None,SEED)
tmpl=next(iter(by_id.values())); lms=tmpl["landmarks"]; faces=tmpl["faces"]; n=tmpl["vertices"].shape[0]
controls=np.stack([(by_id[t]["vertices"]-by_id[s]["vertices"])[lms] for s,t in tep],axis=0)
lap=uniform_laplacian(n,faces)
print("setup",round(time.time()-t0,1)); t0=time.time()
lap_pred=solve_linear_handle_baseline(controls,lms,n,lap,HW,RIDGE_SYS)
print("lap",round(time.time()-t0,1)); t0=time.time()
bilap_pred=solve_linear_handle_baseline(controls,lms,n,lap@lap,HW,RIDGE_SYS)
print("bilap",round(time.time()-t0,1)); t0=time.time()
src_pca=fit_pca(np.stack([by_id[i]["vertices"].reshape(-1) for i in tr]),PCADIM)
def cond(pairs):
    ctrl=[];sf=[]
    for s,t in pairs:
        ctrl.append((by_id[t]["vertices"]-by_id[s]["vertices"])[lms].reshape(-1)); sf.append(by_id[s]["vertices"].reshape(-1))
    return np.concatenate([np.stack(ctrl),pca_coeff(np.stack(sf),src_pca)],axis=1)
xtr,xte=cond(trp),cond(tep)
mu,sd=xtr.mean(0,keepdims=True),np.maximum(xtr.std(0,keepdims=True),1e-6)
xtr,xte=(xtr-mu)/sd,(xte-mu)/sd
_,ytr=pair_arrays(by_id,trp)
ridge_pred=ridge_predict(xte,ridge_fit(xtr,ytr,LAM))
print("ridge",round(time.time()-t0,1))
np.savez_compressed(OUT/"linear_preds.npz",
    test_pairs=np.asarray(tep,dtype=object),
    laplacian_handles_pred=lap_pred.astype(np.float32),
    bilaplacian_handles_pred=bilap_pred.astype(np.float32),
    ridge_sourcepca_pred=ridge_pred.astype(np.float32),
    laplacian_init=lap_pred.astype(np.float32))  # ARAP init
print("saved", OUT/"linear_preds.npz")
