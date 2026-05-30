Context

 In saga_gui.py, segmentation produces a per-Gaussian similarity score combined_pts
 (shape (N,), ~[0,1]) from the user's click/brush prompts. The user then hand-tunes
 the ScoreThres slider to decide which Gaussians are kept (combined_pts > ScoreThres,
 saga_gui.py:1015). Finding a clean cutoff by hand is fiddly.

 This change adds a button that auto-picks the best ScoreThres using the existing YOLO
 detector as weak supervision for the current view. Intuition: Gaussians that project inside
 an allowed YOLO box (and sit near the view axis) are evidence for the object and should be
 kept; Gaussians that are part of the selection but project outside every box are evidence
 against and should be dropped. The button finds the threshold that best balances these,
 updates the ScoreThres slider (non-destructive — the user still clicks segment3d to
 carve), and the live 2D preview refreshes automatically.

 User-confirmed design choices:
 - "Scale factor" driving the angular falloff = the existing global _Scale slider.
 - "View angle" = angle from the camera's optical axis (dead-center = full reward).
 - Behavior = set the slider only (do not auto-run the destructive segment3d).

 Approach (the algorithm)

 Run inside the per-frame fetch_data() when the button's flag is set and prompts exist, so
 gated_chosen / gated_neg / neg_weight / self.gates are in scope and self.yolo_boxes
 holds the current view's boxes (computed at the end of the previous frame; the camera is
 static while clicking).

 For every Gaussian i:
 1. Score s_i = combined_pts[i] — reuse the exact metric segment3d uses.
 2. Project its center into the current view → pixel (px,py) + camera-space depth, using
 view_camera.world_view_transform / full_proj_transform (row-vector convention) and the
 rasterizer's ndc2Pix so pixels align with the YOLO boxes. Mark in_view =
 in-front (depth>0) and inside the image.
 3. Box membership via the existing yolo_prompt_masks() → allowed H×W bool mask;
 inside_i = allowed[py,px].
 4. Angular weight ang_w_i = exp(-angle_i² / (2σ²)), where angle_i is the angle from
 the optical axis (cos = depth/‖p_cam‖) and σ = σ_min + (σ_max-σ_min)·scale with
 scale = _Scale (larger scale → gentler, wider falloff).
 5. Weight: w_i = +ang_w_i if in_view & inside; w_i = −penalty if
 in_view & ~inside (penalty from a new _AutoPenalty slider); else 0 (can't judge
 off-screen Gaussians from this single view).
 6. Pick threshold: t* = argmax_t Σ_{s_i > t} w_i. Computed exactly in one pass — sort by
 score descending, take cumsum(w_sorted), argmax; t* = midpoint between the kept and
 first-dropped score. This directly encodes the request: lowering t to include an in-box
 Gaussian adds +reward; lowering it to include an outside-box selected Gaussian adds
 −penalty; the argmax balances them. If the best sum ≤ 0, set t* above the max score
 (select ~nothing).
 7. dpg.set_value('_ScoreThres', t*) and print a summary.

 Outside-box penalties only count when a Gaussian is actually selected (score > t), so the
 huge background (low scores) is naturally ignored and the penalty bites only on genuine
 false positives — robust to the in-box/out-of-box count imbalance.
