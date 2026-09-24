"""Layout-balanced SFP samples from a prepared multi-route training split."""
import numpy as np

def sample_training_batch(split, batch_size, rng, config, sampling='multi', return_info=False):
    if sampling not in ('single', 'multi'):
        raise ValueError("sampling must be 'single' or 'multi'")
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    sigma_0 = float(config['sigma_0'])
    gain = float(config['stabilizing_gain'])
    if not np.isfinite([sigma_0, gain]).all() or sigma_0 <= 0 or gain < 0:
        raise ValueError('Require finite sigma_0 > 0 and stabilizing_gain >= 0')
    n = len(split['targets'])
    layout_indices = rng.integers(0, n, size=batch_size)
    if sampling == 'multi':
        # Each draw has its own upper bound, so every route within the chosen layout is equiprobable.
        mode_local_indices = rng.integers(0, split['num_modes'][layout_indices], size=batch_size)
        route_indices = split['mode_offsets'][layout_indices] + mode_local_indices
        routes = split['mode_routes'][route_indices]
        knots = split['mode_knot_tau'][route_indices]
    else:
        route_indices = np.full(batch_size, -1, dtype=np.int64)
        routes = split['routes'][layout_indices]
        knots = split['knot_tau'][layout_indices]
    tau = rng.uniform(0.0, 1.0, size=batch_size)
    segments = np.clip((tau[:, None] >= knots[:, 1:]).sum(axis=1), 0, 3)
    row = np.arange(batch_size)
    p0 = routes[row, segments]
    p1 = routes[row, segments + 1]
    t0 = knots[row, segments]
    duration = knots[row, segments + 1] - t0
    if np.any(duration <= 0):
        raise ValueError('Expert route contains a non-positive-duration segment')
    positions = p0 + ((tau - t0) / duration)[:, None] * (p1 - p0)
    derivatives = (p1 - p0) / duration[:, None]
    offsets = (sigma_0 * np.exp(-gain * tau))[:, None] * rng.standard_normal((batch_size, 2))
    actions = positions + offsets
    velocity_targets = derivatives - gain * offsets
    inputs = np.concatenate([actions, tau[:, None], split['conditions'][layout_indices]], axis=1)
    inputs = inputs.astype(np.float32)
    velocity_targets = velocity_targets.astype(np.float32)
    if return_info:
        return inputs, velocity_targets, {
            'layout_indices': layout_indices, 'route_indices': route_indices,
            'tau': tau, 'expert_positions': positions, 'expert_derivatives': derivatives,
            'offsets': offsets,
        }
    return inputs, velocity_targets

