# Split provenance

`facescape_847` is the immutable dataset-construction lineage, not the final
experiment split. The source audit scanned 847 FaceScape identities, rejected
one identity during quality control, and produced the 846-mesh processed data
universe bound by `data/manifest.json`.

The parent manifest also retains its historical chain and pair records because
its exact SHA-256 is stored in the processed-data and model provenance. Those
records must not be used as the report evaluation protocol.

The only report-facing final-rerun split is:

`results/rbsr_final_rerun_holdout_v1/protocol/final_rerun_holdout_split_manifest.json`

It declares 576 training, 70 validation and 100 test identities. All primary
methods use the same 9,900 ordered non-self test pairs. The supplementary and
post-hoc notebooks reuse this frozen identity and pair contract.
