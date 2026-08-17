from __future__ import annotations
import numpy as np
import pytest
from rhinoform.regional_analysis import (
    REGIONS, build_region_topology, empty_spatial_accumulator, regional_metric_row_sets,
)


def fixture():
    sub={name:np.asarray([2*i,2*i+1]) for i,name in enumerate(REGIONS)}
    faces=np.asarray([[0,1,2],[2,3,4],[0,2,4],[6,7,8]])
    landmarks=np.asarray([0,2,4,6,8])
    return sub,faces,landmarks


def test_mapping_and_boundary_rules():
    sub,faces,lm=fixture(); top=build_region_topology(sub,faces,lm,10)
    assert top.face_region_legacy.tolist()==[0,1,0,3]
    assert top.face_region_unique_majority.tolist()==[0,1,-1,3]
    assert top.n_face_ties==1
    assert top.edge_region[top.edges.tolist().index([0,1])]==0
    assert top.edge_region[top.edges.tolist().index([0,2])]==-1


def test_masks_must_be_exact_partition():
    sub,faces,lm=fixture(); sub["dorsum"]=np.asarray([1,2,3])
    with pytest.raises(ValueError,match="partition"):
        build_region_topology(sub,faces,lm,10)


def test_strict_metrics_and_spatial_accumulator_hard_fix_controls():
    sub,faces,lm=fixture(); top=build_region_topology(sub,faces,lm,10)
    source=np.asarray([[float(i),float(i%3),.1*(i%2)] for i in range(10)])
    target=source.copy(); target[:,2]+=.25
    prediction=np.zeros((1,10,3)); prediction[0,lm]=100
    acc=empty_spatial_accumulator(top)
    result=regional_metric_row_sets({"s":{"vertices":source},"t":{"vertices":target}},[("s","t")],prediction,top,method="m",spatial_accumulator=acc)
    assert set(result)=={"legacy_majority_vote","unique_majority_sensitivity"}
    assert acc["n_pairs"]==1 and np.allclose(acc["vertex_error_sum"][lm],0)
    root=result["legacy_majority_vote"][0]
    assert root["landmark_rmse"]==pytest.approx(0,abs=1e-15)
    assert {"free_rmse","edge_strain_p95","normal_flip_pct","abs_flip_pct","missed_flip_pct","target_flip_pct"} <= set(root)
