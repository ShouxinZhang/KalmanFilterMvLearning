#!/usr/bin/env python3

"""2D study of transition, proposal, and posterior in particle filtering."""

from __future__ import annotations

import math


def gaussian_diag_pdf(x: tuple[float, float], mu: tuple[float, float], var: float) -> float:
    d = 2
    coeff = (2 * math.pi * var) ** (-d / 2)
    expo = -sum((xi - mi) ** 2 for xi, mi in zip(x, mu)) / (2 * var)
    return coeff * math.exp(expo)


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [w / total for w in weights]


def ess(weights: list[float]) -> float:
    return 1.0 / sum(w * w for w in weights)


def main() -> None:
    x_prev = (0.0, 0.0)
    mu_transition = (1.0, 1.0)
    var_transition = 1.0

    y = (2.8, 2.6)
    var_obs = 0.04

    mu_guided = (2.7, 2.5)
    var_guided = 0.09

    bootstrap_particles = [(0.2, 1.4), (1.1, 0.3), (2.0, 1.8), (-0.1, 0.7)]
    guided_particles = [(2.5, 2.3), (2.8, 2.6), (2.6, 2.7), (3.0, 2.4)]

    print("Previous state x_{t-1} =", x_prev)
    print("Transition p(x_t | x_{t-1}) = N((1,1), I)")
    print("Observation y =", y)
    print("Likelihood p(y | x_t) = N(x_t, 0.04 I)")
    print()

    posterior_mean = tuple((1 + 25 * yi) / 26 for yi in y)
    posterior_var = 1 / 26
    print("Analytic posterior p(x_t | y_t, x_{t-1}) = N(mu_post, Sigma_post)")
    print("mu_post   =", posterior_mean)
    print("Sigma_post= {:.6f} * I".format(posterior_var))
    print()

    print("Bootstrap proposal q_boot = transition")
    boot_weights = [gaussian_diag_pdf(y, x, var_obs) for x in bootstrap_particles]
    boot_norm = normalize(boot_weights)
    for x, w in zip(bootstrap_particles, boot_norm):
        print("  particle =", x, "normalized weight =", round(w, 6))
    print("  ESS =", round(ess(boot_norm), 6))
    print()

    print("Guided proposal q_guided = N((2.7,2.5), 0.09 I)")
    guided_weights = []
    for x in guided_particles:
        likelihood = gaussian_diag_pdf(y, x, var_obs)
        transition = gaussian_diag_pdf(x, mu_transition, var_transition)
        proposal = gaussian_diag_pdf(x, mu_guided, var_guided)
        guided_weights.append(likelihood * transition / proposal)
    guided_norm = normalize(guided_weights)
    for x, w in zip(guided_particles, guided_norm):
        print("  particle =", x, "normalized weight =", round(w, 6))
    print("  ESS =", round(ess(guided_norm), 6))
    print()

    print("Takeaway:")
    print("  transition = how the system tends to move")
    print("  proposal   = where the algorithm actually samples")
    print("  posterior  = what we truly want to approximate after seeing the observation")


if __name__ == "__main__":
    main()
