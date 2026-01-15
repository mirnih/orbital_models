from __future__ import annotations
import numpy as np


def metropolis_sample_positions(
    logp_func,
    n_samples: int,
    *,
    burnin: float = 0.15,
    step_size: float = 1.0,
    init_box: float = 0.5,
    thin: int = 1,
    rng: np.random.Generator | None = None,
):
    """
    Metropolis-Hastings sampler for 3D positions.

    We sample from p(r) proportional to exp(logp_func(r)).
    For orbitals we typically use logp = log(|psi(r)|^2 + eps).
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not (0.0 <= burnin < 1.0):
        raise ValueError("burnin must be in [0, 1)")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if init_box <= 0:
        raise ValueError("init_box must be > 0")
    if thin < 1:
        raise ValueError("thin must be >= 1")

    kept_total = n_samples * thin
    total_steps = int(np.ceil(kept_total / (1.0 - burnin)))
    burnin_steps = int(np.floor(total_steps * burnin))

    current = rng.uniform(low=-init_box, high=init_box, size=3)
    current_logp = float(logp_func(current))

    samples = np.zeros((n_samples, 3), dtype=np.float64)
    accepted = 0
    collected = 0
    kept_counter = 0

    for step in range(total_steps):
        proposal = current + rng.uniform(low=-step_size, high=step_size, size=3)
        prop_logp = float(logp_func(proposal))

        if prop_logp >= current_logp:
            accept = True
        else:
            u = rng.random()
            accept = (np.log(u) < (prop_logp - current_logp))

        if accept:
            current = proposal
            current_logp = prop_logp
            accepted += 1

        if step >= burnin_steps:
            kept_counter += 1
            if kept_counter % thin == 0:
                if collected < n_samples:
                    samples[collected] = current
                    collected += 1
                else:
                    break

    acc_rate = 100.0 * accepted / max(1, total_steps)
    return samples, acc_rate
