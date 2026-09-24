# Near-optimal multi-route training split

This version contains training data only. Validation/test layouts were not read or changed.
Existing single-route fields retain their v1 meaning; new candidates are in mode_* arrays.
For layout i, flat candidate rows are mode_offsets[i]:mode_offsets[i+1].
mode_orders are zero-based indices into targets[i]; they are labels, not network inputs.
num_modes is the number of eligible visit orders, not necessarily the number of geometrically distinct paths.
Use multiroute_sampler.sample_training_batch(..., sampling='multi') to sample layout-balanced batches.
Use sampling='single' to reproduce the original expert sampling.
Do not uniformly sample rows of mode_routes: that would overweight layouts with more routes.
There are no model weights or measured success rates in this dataset version.
Do not change optimizer update count or model-selection rule merely because route count increased.
