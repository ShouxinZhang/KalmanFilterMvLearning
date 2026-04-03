#!/usr/bin/env python3

"""A minimal particle-filter style distribution demo.

This script illustrates three ideas from the note:
1. a weighted particle cloud approximates a posterior distribution,
2. expectations and interval probabilities become weighted sums,
3. one multinomial resampling step turns weighted particles into equal-weight particles.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable


def weighted_expectation(
    particles: Iterable[float], weights: Iterable[float], phi: Callable[[float], float]
) -> float:
    return sum(w * phi(x) for x, w in zip(particles, weights))


def interval_mass(
    particles: Iterable[float], weights: Iterable[float], left: float, right: float
) -> float:
    return sum(w for x, w in zip(particles, weights) if left <= x <= right)


def multinomial_resample(
    particles: list[float], weights: list[float], seed: int = 0
) -> list[float]:
    rng = random.Random(seed)
    return rng.choices(particles, weights=weights, k=len(particles))


def main() -> None:
    particles = [0.0, 2.0, 3.0]
    weights = [0.2, 0.5, 0.3]

    print("Particles:", particles)
    print("Weights:  ", weights)
    print()

    mean_est = weighted_expectation(particles, weights, lambda x: x)
    second_moment_est = weighted_expectation(particles, weights, lambda x: x * x)
    interval_est = interval_mass(particles, weights, 1.5, 3.5)

    print("Weighted mean estimate E[x]       =", mean_est)
    print("Weighted second moment E[x^2]     =", second_moment_est)
    print("Estimated posterior mass [1.5,3.5] =", interval_est)
    print()

    resampled = multinomial_resample(particles, weights, seed=7)
    equal_weight = 1.0 / len(resampled)
    resampled_mean = sum(resampled) * equal_weight

    print("One multinomial resampling result =", resampled)
    print("Equal weight after resampling     =", equal_weight)
    print("Mean of this single resample      =", resampled_mean)
    print()
    print(
        "Remark: the resampled mean need not equal the weighted mean in one run;"
    )
    print(
        "only its conditional expectation matches the pre-resampling weighted mean."
    )


if __name__ == "__main__":
    main()
