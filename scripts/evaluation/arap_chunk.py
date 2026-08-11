"""Chunked ARAP runner. Usage: python scripts/evaluation/arap_chunk.py START END
Loads linear_preds.npz for test_pairs + laplacian init; computes vectorised ARAP
(3 iters) over pairs[START:END]; saves large/classical/arap_part_{START}.npy."""
import sys, time
from pathlib import Path
import numpy as np
from rhinoform.data import load_rows, split_ids, ordered_pairs
from rhinoform.geometry import uniform_laplacian
from rhinoform.baselines import arap_predict_vectorised

START=int(sys.argv[1]); END=int(sys.argv[2])
REPO=Path("data"); OUT=Path("large/classical")
HW=1000.0; RIDGE_SYS=1e-8; ITER=3; SEED=2026
t0=time.time()
_,by_id=load_rows(REPO)
te=split_ids(by_id,"main test"); tep=ordered_pairs(te,None,SEED)
tmpl=next(iter(by_id.values())); lms=tmpl["landmarks"]; faces=tmpl["faces"]; n=tmpl["vertices"].shape[0]
d=np.load(OUT/"linear_preds.npz",allow_pickle=True)
lap_init=d["laplacian_init"]
lap=uniform_laplacian(n,faces)
sl=tep[START:END]
sources=[by_id[s]["vertices"] for s,_ in sl]
controls=np.stack([(by_id[t]["vertices"]-by_id[s]["vertices"])[lms] for s,t in sl],axis=0)
init=lap_init[START:END]
print(f"chunk [{START}:{END}] n={len(sl)} setup={round(time.time()-t0,1)}"); t0=time.time()
pred=arap_predict_vectorised(sources,controls,lms,faces,lap,init,HW,RIDGE_SYS,ITER)
np.save(OUT/f"arap_part_{START}.npy", pred.astype(np.float32))
print(f"done [{START}:{END}] {round(time.time()-t0,1)}s -> arap_part_{START}.npy")
