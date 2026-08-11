# Browser demo status

The live deformation operator is ridge-only: `visible_delta = ctrl @ W_ctrl`.
The browser safety state computes OOD/flip/strain and may accept, shrink or reject
an edit. It does not run the CVAE or learned RB-SR gate for arbitrary slider edits.

`head.json`, `demo_case.json`, ROI PLY previews and rendered posters are deliberately
absent because they contain or may contain FaceScape-derived identity geometry.
Supply a redistribution-cleared or synthetic head before publishing a runnable
public demo. The included HTML/latency harness is therefore code evidence, not a
standalone hosted demo bundle.

