"""A1 model loading + unified inference path (exploratory gate-1 weights).

Replicates prepare_safe_fusion_cache's load/inference path. The base package
predates the feature_template_sha256 fix, so we rebuild the train-only feature
template here with build_static_vertex_features (use_subunit_features=False).

CRITICAL (A1 tightening clause): every method's prediction (ridge / rbsr /
optional 3dmm / ARAP base) is pushed through the SAME control-point projector
(RBF basis via project_handles) so control error is equalised across methods.
"""
from __future__ import annotations

import numpy as np
import torch

import scripts.analysis.flip_validation.common as common
from rhinoform.data import ridge_predict
from rhinoform.train import NeuralFieldCVAE, build_static_vertex_features, pair_conditions, pca_transform, predict_field
from rhinoform.train_rbsr_gate import (
    SpatialRiskGate,
    build_rbf_projection,
    predict_gate_batches,
    project_handles,
    resolve_device,
)


class A1Models:
    def __init__(self, device: str = "cpu"):
        self.device = resolve_device(device)
        self.base = torch.load(common.CVAE_PKG, map_location="cpu", weights_only=False)
        self.rbsr = torch.load(common.RBSR_PKG, map_location="cpu", weights_only=False)
        self.splits = common.load_split_ids()
        self.train_ids = [str(x) for x in self.base["train_ids"]]
        self.alpha = float(self.base["selected_alpha"])

    def prepare(self, extra_ids: list[str]):
        ids = sorted(set(self.train_ids) | set(extra_ids), key=int)
        self.by_id = common.load_meshes(ids)
        self.top = common.template_topology(self.by_id)
        self.faces = self.top["faces"]
        self.landmarks = self.top["landmarks"]
        self.n = self.top["n_vertices"]
        self.vertex_features, static = build_static_vertex_features(
            self.by_id, self.train_ids, use_subunit_features=bool(self.base["use_subunit_features"]))
        self.template_vertices = np.asarray(static["template_vertices"], dtype=np.float64)
        self.rbf_basis = build_rbf_projection(self.template_vertices, self.landmarks)
        self.rbf_t = torch.as_tensor(self.rbf_basis, dtype=torch.float32, device=self.device)
        self.lm_t = torch.as_tensor(self.landmarks, dtype=torch.long, device=self.device)
        # models
        obs_dim = int(np.asarray(self.base["obs_train_mean"]).shape[1])
        cond_dim = int(np.asarray(self.base["cond_mean"]).shape[1])
        a = self.base["args"]
        self.cvae = NeuralFieldCVAE(self.vertex_features.shape[1], cond_dim, obs_dim,
                                    latent_dim=int(a["latent_dim"]), hidden=int(a["hidden"])).to(self.device)
        self.cvae.load_state_dict(self.base["cvae_state_dict"]); self.cvae.eval()
        ra = self.rbsr["args"]
        self.gate = SpatialRiskGate(int(self.rbsr["gate_vertex_dim"]), int(self.rbsr["gate_cond_dim"]),
                                    int(ra["hidden"]), self.alpha).to(self.device)
        self.gate.load_state_dict(self.rbsr["gate_state_dict"]); self.gate.eval()
        self.center_t = torch.as_tensor(self.rbsr["center"], dtype=torch.float32, device=self.device)
        self.scale = float(self.rbsr["scale"])
        self.vfeat_t = torch.as_tensor(self.vertex_features, dtype=torch.float32, device=self.device)

    def conditions(self, pairs):
        cond, source, target_delta, controls, _, _, _ = pair_conditions(
            self.by_id, pairs, self.base["source_pca"], self.base["cond_mean"], self.base["cond_std"])
        return cond, source, target_delta, controls

    def project(self, pred_flat: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Push an arbitrary (B, V*3) prediction through the SAME RBF projector."""
        B = pred_flat.shape[0]
        pred = torch.as_tensor(pred_flat, dtype=torch.float32, device=self.device).reshape(B, self.n, 3)
        ctrl = torch.as_tensor(controls, dtype=torch.float32, device=self.device).reshape(B, -1, 3)
        out = project_handles(pred, ctrl, self.lm_t, self.rbf_t)
        return out.cpu().numpy().reshape(B, -1)

    def synthetic_conditions(self, source_ids, control_deltas):
        """Build standardized conditions for synthetic edits (no real target).

        control_deltas: (B, K, 3) landmark displacements. Mirrors pair_conditions
        but with arbitrary controls so any edit direction can be driven through
        the same ridge/cvae/gate path.
        """
        ctrl = np.asarray(control_deltas, dtype=np.float64).reshape(len(source_ids), -1)  # (B,27)
        src_flat = np.stack([self.by_id[s]["vertices"].reshape(-1) for s in source_ids], axis=0)
        source_code = pca_transform(src_flat, self.base["source_pca"])
        cond_raw = np.concatenate([ctrl, source_code], axis=1)
        cond = (cond_raw - self.base["cond_mean"]) / self.base["cond_std"]
        source = np.stack([self.by_id[s]["vertices"] for s in source_ids], axis=0).astype(np.float32)
        return cond.astype(np.float32), source, ctrl.astype(np.float32)

    def predict_synthetic(self, source_ids, control_deltas, batch_size: int = 16):
        cond, source, controls = self.synthetic_conditions(source_ids, control_deltas)
        ridge = ridge_predict(cond, self.base["ridge_cond"]).astype(np.float32)
        cvae = predict_field(self.cvae, self.vertex_features, cond, is_cvae=True).astype(np.float32)
        rbsr, _ = predict_gate_batches(
            self.gate, self.vfeat_t, cond, source, ridge, cvae,
            self.center_t, self.scale, self.lm_t, controls, self.rbf_t, batch_size)
        ridge_proj = self.project(ridge, controls)
        return {"ridge": ridge_proj, "rbsr": rbsr, "controls": controls, "source": source}

    def predict(self, pairs, batch_size: int = 16):
        """Return dict of projected predictions + target for a list of pairs."""
        cond, source, target_delta, controls = self.conditions(pairs)
        ridge = ridge_predict(cond, self.base["ridge_cond"]).astype(np.float32)
        cvae = predict_field(self.cvae, self.vertex_features, cond, is_cvae=True).astype(np.float32)
        rbsr, gate_vals = predict_gate_batches(
            self.gate, self.vfeat_t, cond, source, ridge, cvae,
            self.center_t, self.scale, self.lm_t, controls, self.rbf_t, batch_size)
        ridge_proj = self.project(ridge, controls)
        return {
            "ridge": ridge_proj,           # projected
            "rbsr": rbsr,                  # already projected inside predict_gate_batches
            "cvae_raw": cvae,
            "target": target_delta.reshape(len(pairs), -1),
            "source": source,
            "controls": controls,
        }
