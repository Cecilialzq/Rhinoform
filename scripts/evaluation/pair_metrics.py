"""Per-pair metrics for one classical method. Usage: python scripts/evaluation/pair_metrics.py METHOD
METHOD in {laplacian_handles,bilaplacian_handles,ridge_sourcepca,arap_handles_iter3}.
Writes identity_bootstrap_pair_metrics_{METHOD}.csv into large/classical."""
import sys
from pathlib import Path
import numpy as np
from rhinoform.data import load_rows, split_ids, ordered_pairs, edge_index
from rhinoform.baselines import per_pair_metrics, write_pair_csv, aggregate
import json

METHOD=sys.argv[1]
REPO=Path("data"); OUT=Path("large/classical"); SEED=2026
_,by_id=load_rows(REPO)
te=split_ids(by_id,"main test"); tep=ordered_pairs(te,None,SEED)
tmpl=next(iter(by_id.values())); lms=tmpl["landmarks"]; faces=tmpl["faces"]
edges=edge_index(faces); subunits=tmpl["subunits"]

d=np.load(OUT/"linear_preds.npz",allow_pickle=True)
if METHOD=="arap_handles_iter3":
    parts=[np.load(OUT/f"arap_part_{s}.npy") for s in (0,500,1000,1500,2000)]
    pred=np.concatenate(parts,axis=0)
    assert pred.shape[0]==len(tep), f"{pred.shape[0]} != {len(tep)}"
else:
    pred=d[f"{METHOD}_pred"]
rows=per_pair_metrics(METHOD,by_id,tep,pred,faces,edges,subunits,lms)
write_pair_csv(OUT/f"identity_bootstrap_pair_metrics_{METHOD}.csv",rows)
print(METHOD, json.dumps(aggregate(METHOD,rows)))
