# Implement cubical PH on EfficientHCCRNet stage 2

Version 4, source-checked and reconciled 2026-10-19. This document replaces the previous handoff in full.

## Authorization and destination

Work in the original repository:

`C:\Users\Root\Documents\Swinburne\hccr-deep-learning`

The user explicitly requested an updated handoff that includes implementation. Inspect the destination, write a short implementation plan, then implement and verify it in the same task. Do not stop after the plan to request routine confirmation. The earlier planning-only restriction is superseded.

The implementation contract is: change the source of the TDA descriptors to a single-channel projection of stage 2, and preserve the verified raw-image experiment's fusion mechanism. Both feature blocks use train-fitted z-scores, the auxiliary block has a validation-selected weight, and their concatenation feeds CosFace. The projection-training scheme, vertex filtration convention, and 14-dimensional descriptor below are explicit pilot choices. They do not reproduce every detail of the historical raw-image extractor.

Implement the configurable experiment, tests, documentation, and bounded smoke checks. Full dataset extraction, full training sweeps, remote jobs, commits, and pushes are outside this handoff's execution scope. Supply runnable commands for those larger experiments. If data or weights are unavailable, complete implementation and synthetic verification and identify the missing inputs. Do not invent measurements.

Do not implement in `hccr-tda`. This document lives outside both repositories. No destination code or experiment was executed by this update. No chat has been created or messaged.

## Requirements and decisions

The supervisor requested TDA at stage 2, said one point is one vector, specified sublevel filtration, and then suggested a 1x1 convolution because stage 2 has multiple channels.

The user identified stage 2 as the complete second backbone stage: input, 3x3 stem, stage 1 with two blocks, stage 2 with three blocks, stage 3 with three blocks, GAP, CosFace, logits. It is not a second training phase.

The user delegated remaining methodological choices. Use this primary design:

`stage 2 [B,C,H,W] -> Conv2d(C,1,1) -> scalar map [B,1,H,W] -> cubical sublevel PH -> fixed features -> fusion classifier`

Each spatial position initially has a C-dimensional vector. The projection maps it to one scalar and retains its grid location. Each image has its own map and diagrams. The original stage 2 tensor continues to stage 3 through the unchanged main backbone.

This studies spatial topology of a projected feature map. Do not describe it as point-cloud topology in the original C-dimensional embedding space or equate its holes with ink loops.

One output channel, the grid-vertex convention, projection training, and the controls below are agent decisions. They were not specified by the supervisor. The previous embedding-neighbor complex and RMS vector filtration are superseded. Do not implement that earlier design, add a Rips complex, or interpret the channel axis as a third spatial dimension.

## Source facts and destination checks

The original repo's `src/hccr/models/efficient_hccr.py` exposes `EfficientHCCRNet.forward_features(inputs, return_stages=True)`, returning final features and a dictionary containing stage1, stage2, and stage3. Read the current source again. `_forward_embedding` applies final pooling, dropout, and embedding projection before the classifier.

`configs/model/baseline.yaml` specifies width 80, stage depths [2,3,3], stem stride 2, final backbone channels 320, embedding dimension 320, CosFace, scale 32, and margin 0.1. The paper protocol is `configs/experiment/thesis_model_comparison.yaml`. Derive actual settings from the selected checkpoint metadata rather than constructor defaults.

For 96x96 input under that preset, the derived stage 2 shape is [B,160,24,24]. Its 160-to-1 projection yields [B,1,24,24]. Verify these shapes by execution. No model execution occurred in the source chat.

Use GitNexus repo `hccr-deep-learning` explicitly. `hccr-tda` is separately registered despite copied AGENTS.md text. Previous graph context identifies `_forward_embedding` as a caller of `forward_features`.

Source-chat git rev-parse reported `853f63a9ce54f87098d2233dc8ca81aeca0ef46f`. Git status failed with `this operation must be run in a work tree`, including when explicitly targeting the Git directory and work tree. The target's clean or dirty state was not established. Inspect the destination's actual Git setup. Do not reset changes or rewrite Git configuration to suppress this error. Recheck graph freshness.

## Reproduce the verified raw-image CNN-TDA fusion

Current source in hccr-tda was read directly after GitNexus navigation. That repo's index reports one commit behind, so graph call lists are navigation aids rather than proof of complete current coverage. The code below, not an assumed fusion design, defines the reference behavior:

- `C:\Users\Root\Documents\Swinburne\hccr-tda\scripts\run_unseen_writer_sweep.py`, run_feature_variant, lines 516-719: freeze/cache CNN embeddings, fit separate train-only standardizers, construct weighted blocks, train a head for each weight, choose the best validation top-1, and only then construct test caches and evaluate.
- `C:\Users\Root\Documents\Swinburne\hccr-tda\src\hccr\tda\full_class.py`, Standardizer and fit_standardizer, lines 65-71 and 574-594: per-coordinate z-score using population variance. Replace scales below 1e-8 with 1.0.
- The same file, materialize_blocks, lines 634-649: select columns, standardize each block, multiply by its block weight, then concatenate on the feature axis.
- The same file, _build_head, lines 921-932: use AngularMarginClassifier for CosFace, without an intervening fusion MLP. The softmax option uses a linear classifier.
- `C:\Users\Root\Documents\Swinburne\hccr-tda\configs\experiment\tda_unseen_writer_2gpu.yaml`, lines 52-89: train_zscore_per_block, concat_linear_cosface, TDA weights [0.25,0.5,1.0,2.0], validation selection, CosFace scale 32 and margin 0.1.

The matching fusion formula is:

`fused = concatenate(z_cnn(final_embedding), alpha * z_aux(auxiliary_features))`.

Both z-score transforms are fitted on TRAIN only and reused unchanged on validation and test. CNN weight is 1.0. Select alpha from [0.25,0.5,1.0,2.0] by validation top-1, with deterministic first-candidate tie resolution. Select each head checkpoint by validation top-1 as in the reference runner. Apply the selected alpha and the frozen transforms at test time. Do not concatenate an unstandardized CNN embedding with only a standardized auxiliary block.

Reference head-training settings are 8 epochs, batch size 2048, AdamW learning rate 0.001, weight decay 0.0001, and configurable BF16 execution. CPU smoke checks can use float32 and smaller explicit budgets. Keep projection pretraining separate from this fusion-head budget.

The existing raw-image variants use all 340 TDA features or a train-selected 63-feature ANOVA subset. This handoff's 14 PH summaries and dimension-matched controls are a new descriptor protocol. They preserve the fusion mechanism, not the historical descriptor dimension or extractor. Do not present them as reproducing raw-image experiment results.

The current original repository has no src/hccr/tda directory and no matching FeatureBlock, train_feature_head, or concat_linear_cosface symbols in its checked source. The original repo's graph still returns historical TDA entries; do not assume those modules exist on disk. Bring over or recreate only the minimal block standardization, weighting, concatenation, and head behavior needed for the new experiment. Do not import the sibling repo at runtime or copy its whole TDA pipeline.

The similarly named hccr-tda/src/hccr/tda/analysis/fusion.py is a raw-pixel fusion proxy. It is not the configured full-class frozen CNN-embedding fusion reference above.

## Projection and training

Use a standard pointwise convolution with C input channels, one output channel, kernel size 1, stride 1, padding 0, groups 1, and bias disabled:

`S(h,w) = sum_c w[c] * F2(c,h,w)`.

Initialize every weight to 1/C, making the initial map the channel mean. Keep the output signed. Do not add ReLU, sigmoid, per-image min-max scaling, or spatial pooling in the primary map path.

Learn the projection before offline PH extraction:

1. Freeze baseline parameters and BN statistics in eval mode.
2. Attach the projection and a temporary spatial classifier, `S.flatten(1) -> Linear(H*W, selected_class_count)`.
3. Train only the projection and temporary classifier with cross-entropy on training data. The loss must depend on the projected map.
4. Select the projection checkpoint by validation loss within a declared budget. Validation has no backward passes. Save the checkpoint, seed, label mapping, budget, and selection rule.
5. Freeze that projection and discard the temporary classifier from the PH inference path. Extract maps, PH features, and final CNN embeddings.
6. Fit the common fusion heads on training features, selecting their checkpoints on validation.

Default projection-training settings are AdamW, learning rate 3e-4, weight decay 1e-4, batch size 256, and a 20-epoch ceiling, all configurable. These are proposed pilot settings, not recovered paper evidence. Smoke mode must impose smaller explicit limits.

For frozen-backbone computation feeding trainable layers, use no_grad or reload cached normal tensors. Do not feed inference-only tensors into a trainable convolution that needs to save them for backward.

The projection is supervised by classification, not PH. State this in reports. NumPy or GUDHI after detach does not propagate gradients through PH. Differentiable PH and joint topology training are separate later experiments, not required placeholders for this implementation.

Also implement a fixed channel-mean projection control that keeps weights at 1/C and bypasses projection training. Demonstrate learned projection weight updates on a deterministic spatial classification fixture. An untrained or unchanged projection must not be reported as learned.

## Exact cubical filtration

Interpret each `S[h,w]` as a grid VERTEX value. Use `gudhi.CubicalComplex(vertices=map_values)` with a 2D array. Do not flatten into a 1D signal or substitute top_dimensional_cells, which defines a different construction.

For every cubical cell q, its filtration value is the maximum of its vertex values:

`K_t = {q : max_vertex_value(q) <= t}`.

This is a vertex lower-star sublevel filtration. The grid has horizontal and vertical edges and elementary square faces. Do not insert diagonal edges or use periodic boundaries. Edges enter once both endpoints enter; squares enter once all four vertices enter. Equal values enter simultaneously without artificial perturbation.

After freezing a projection, fit one positive scale s as the RMS of its map values over a deterministic TRAIN-only calibration sample. Record the exact sample IDs. Use `map_values = S / s`. If s is exactly zero, use s=1 and record the degenerate map. Do not clip values or normalize independently per image. Store s with the projection checkpoint's extraction metadata.

For extraction, detach maps and convert each to a finite float64 2D CPU array. Invalid shapes or nonfinite values must fail at this boundary. Compute ordinary H0 and H1 over F2 on the full finite rectangular grid.

For each dimension, summarize strictly positive finite lifetimes with count, total, mean, population standard deviation, maximum, and entropy. Entropy uses natural logarithms of lifetime proportions. Return zeros if there are no positive finite intervals. Append that dimension's essential-interval count, yielding 14 finite scalar features for H0+H1 or seven for H0 alone.

Preserve raw diagrams for selected inspection samples. Never replace infinite deaths with an arbitrary finite cap or include them in lifetime statistics. A nonempty full finite rectangle has one essential H0 interval and no essential H1 interval. Use these facts as extraction checks and handle constant feature columns explicitly.

## Comparison protocol

Freeze the same baseline checkpoint for every pilot variant. Use identical samples, preprocessing, splits, label mappings, seeds, common-head family, and head-training budgets. Reuse the SAME learned projection for comparisons that consume the same learned map.

Implement these selectable variants:

1. Original checkpoint logits, as a separate reference without retraining.
2. Final embedding with a refitted common head and zero auxiliary features.
3. Final embedding plus stage 2 GAP features.
4. Final embedding plus flattened learned scalar-map features.
5. Final embedding plus nonspatial statistics of the same learned map.
6. Final embedding plus learned-map H0 features.
7. Final embedding plus learned-map H0+H1 features.
8. Final embedding plus fixed channel-mean-map H0+H1 features.

The flattened-map control tests the learned projection without PH. The scalar-statistics control tests the value distribution without spatial connectivity. The fixed projection tests whether learning the channel combination helps. An improvement over the unchanged checkpoint alone cannot isolate PH from head retraining.

Use 14 auxiliary dimensions for the main comparison. Fit training-only PCA for GAP and flattened-map controls. If tiny smoke data cannot support 14 components, retain available components, zero-pad, and report actual rank. Pad the seven H0-only features with seven zeros. Baseline auxiliary features are 14 zeros.

The 14 nonspatial map statistics are mean, population standard deviation, mean absolute value, skewness, excess kurtosis, and quantiles at 0, .05, .10, .25, .50, .75, .90, .95, and 1. Declare quantile interpolation. For constant maps, use zero skewness and excess kurtosis. After auxiliary construction and any PCA or padding, fit its standardizer using training statistics only, with the same scale floor as the reference. Fit a separate train-only standardizer for the CNN embedding. Constant training columns become zero for training rows after centering. For held-out rows, apply the same stored mean and scale=1; do not force a genuinely changed held-out value to zero.

Use the repository's CosFace family for refitted heads, including its target-margin training and inference paths. Concatenate the standardized final embedding with alpha times the standardized auxiliary vector, following the verified fusion definition above. Use the same alpha-search budget for nonzero auxiliary controls. Train the zero-auxiliary baseline once because its alpha has no effect. Report head parameters separately from projection and temporary-classifier parameters. Include projection training and alpha search in cost reports.

Use existing subset facilities for a tractable pilot. Configure seeds 7,17,29 without automatically launching the sweep. Fix subset identities and budgets before comparing validation results. Preserve writer-disjoint splits where available and verify writer IDs before claiming unseen-writer performance. Reserve held-out test evaluation for a finalized protocol.

Persist the mapping from original checkpoint class IDs to pilot class IDs. Evaluate baseline and refitted heads under the same declared candidate-class policy. If the pilot restricts candidate classes, apply that restriction consistently to baseline logits and remap labels before computing metrics. Do not report a delta between a full-class baseline and a subset-class head as if they solved the same classification problem.

Report top-1 accuracy, changes against non-PH controls, class errors, and projection training status. Distinguish backbone, projection, PH extraction, and classifier costs if measuring runtime. Cached-head latency is not complete inference latency. Stage 1 and stage 3 ablations are later work.

## Implementation deliverables

Briefly record actual file choices after inspecting the repo, then implement:

- An extractor or experiment wrapper that reuses stage outputs and preserves baseline behavior.
- A serializable scalar projection and temporary projection-training head.
- A pure cubical PH extractor with a fixed feature schema.
- Deterministic feature caching and compatibility checks.
- Common-head training and comparison reports.
- An experiment config and rerunnable commands for projection training, extraction, comparison, and bounded smoke execution.
- Behavioral tests and concise usage documentation.

Reuse existing checkpoint, manifest, preprocessing, and training facilities where suitable. Keep GUDHI scoped, for example in a TDA optional dependency group. Avoid adding a second ML framework solely for PH. Implement a complete runnable path rather than disconnected modules or TODOs.

Cache identity must cover baseline checkpoint digest, projection checkpoint or fixed-projection mode, manifest digest, sample IDs, label mapping, preprocessing, stage, grid-vertex convention, scalar calibration, coefficient field, feature schema, and relevant implementation versions. A projection weight change must invalidate maps and PH. Fit all calibration, scaling, and PCA on training data. Record both block standardizers, PCA state where used, alpha grid, selected alpha, and validation-selection rule in fusion artifacts and training signatures. Reject incompatible caches with actionable errors.

Store branch state separately or in a versioned wrapper checkpoint. Preserve original checkpoint loading and inference without the new branch. Reuse forward_features rather than permanent hooks where possible. Avoid a second full backbone traversal solely to obtain final embeddings alongside stage 2. Verify baseline-logit parity.

Baseline-logit parity means the unchanged final embedding followed by the ORIGINAL classifier, with no fusion standardization or new features. Refitted fusion heads intentionally produce different logits. Test these paths separately.

## Repository rules and execution limits

Read actual destination AGENTS.md and any more specific instructions. Use GitNexus query and context first. Run upstream impact before editing existing functions, classes, or methods. Report callers, processes, and risk. Warn on HIGH or CRITICAL risk. Resolve UNKNOWN with source evidence. Refresh stale indexes through the documented workflow.

Preserve user changes, datasets, weights, and historical experiments. Do not copy the whole hccr-tda implementation. Do not commit or push.

After implementation, run focused tests, Black, Ruff, and complete GitNexus detect_changes. Partial or truncated graph output is not a clean check. Graph analysis remains mandatory before any separately authorized commit.

Run deterministic synthetic smoke checks and a bounded real-data check when local inputs exist. Use temporary experiment output. Bound samples, batches, and epochs explicitly. Do not automatically download data, extract all CASIA features, run full training or GPU sweeps, or start remote services. Supply commands for those experiments.

Follow the repo's nonblocking policy for longer operations. Dispatch them asynchronously and report the job identifier. Do not sleep or poll in the foreground.

## Required verification

Test the ACTUAL vertex-based cubical extractor against literal outcomes:

- A constant 3x3 zero vertex map has one essential H0 interval born at zero and no positive finite H0 or H1 intervals.
- A 3x3 vertex map with boundary zero and center two has exactly one positive H1 interval [0,2).
- The map [[0,2,0],[2,2,2],[2,2,2]] has one finite positive H0 interval [0,2) and one essential H0 interval born at zero.
- Equal values are unperturbed, negative finite values are valid, empty finite diagrams produce finite zero summaries, and essential intervals remain separate.
- Known projection weights produce a literal expected map. Mean initialization equals the channel mean and preserves H and W.
- A small deterministic spatial task updates projection weights through the temporary classifier while frozen backbone parameters and BN buffers remain unchanged.
- Saving and loading the projection reproduces maps. Discarding the temporary classifier does not change them.
- Baseline-only eval logits match the original model through the experiment wrapper.
- Repeated extraction is deterministic. Changed projection weights or incompatible schemas invalidate caches.
- Altering validation or test data does not change training-derived calibration, scaling, or PCA.
- The block fusion has literal expected values. For CNN training rows [[1,2],[3,6]], auxiliary training rows [[10],[14]], and alpha=0.5, the fused rows are [[-1,-1,-0.5],[1,1,0.5]]. Test the actual block implementation, including both standardizers and the weight.
- Fusion weights and fusion-head checkpoints are selected only by validation top-1. The separate projection checkpoint uses validation loss as specified above. Test-cache construction and held-out evaluation do not occur during either selection phase.
- Pilot label remapping and candidate-class restriction produce consistent metrics for the baseline and the refitted fusion heads.

Exercise the complete synthetic path with a compatible tiny baseline: bounded projection training, projection freeze, map and PH extraction, common-head fitting, and report writing. Label synthetic metrics as smoke evidence. If actual weights and images exist, also check real stage shapes and extract a few maps.

Completion means the supplied commands connect the full implementation, relevant tests and style checks pass, and graph analysis is reported. State any unavailable check and its reason. Code completion and smoke tests do not establish a research accuracy gain.

## Final report from the destination agent

Report implemented files and runnable commands, exact projection-training and PH definitions, checks actually run, smoke artifact locations, and remaining environment or data limits. Distinguish completed code from full experiments that have not run. Do not request implementation approval again merely because the short plan is finished.

## References

- PyTorch Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- GUDHI cubical complexes: https://gudhi.inria.fr/python/latest/cubical_complex_user.html
- GUDHI vertex and top-dimensional-cell conventions: https://gudhi.inria.fr/python/latest/cubical_complex_ref.html
