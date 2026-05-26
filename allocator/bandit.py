"""Online bandit policies: UCB1, EXP3.P, Thompson (Gaussian), Random.

These pick *one arm per round* given the running per-arm reward history. Each round
the caller observes a reward (the new best-fitness from advancing the chosen greedy
arm by one generation) and feeds it back via ``update``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

BANDITS = ("ucb", "exp3p", "thompson", "random")


@dataclass
class BanditState:
    """Per-arm bookkeeping shared by all algorithms."""

    k: int
    n_pulls: list[int] = field(init=False)
    sum_reward: list[float] = field(init=False)
    last_reward: list[float] = field(init=False)
    total_rounds: int = 0

    def __post_init__(self) -> None:
        self.n_pulls = [0] * self.k
        self.sum_reward = [0.0] * self.k
        self.last_reward = [0.0] * self.k

    def update(self, arm: int, reward: float) -> None:
        self.n_pulls[arm] += 1
        self.sum_reward[arm] += reward
        self.last_reward[arm] = reward
        self.total_rounds += 1


class Bandit:
    """Base class. Subclasses implement ``select`` and (optionally) ``update``."""

    name: str = "bandit"

    def __init__(self, k: int) -> None:
        self.state = BanditState(k=k)

    def select(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.state.update(arm, reward)


class UCB1(Bandit):
    name = "ucb"

    def __init__(self, k: int, *, exploration_coef: float = math.sqrt(2.0)) -> None:
        super().__init__(k)
        self.c = exploration_coef

    def select(self) -> int:
        for i, n in enumerate(self.state.n_pulls):
            if n == 0:
                return i
        t = self.state.total_rounds
        best_arm, best_val = 0, float("-inf")
        log_t = math.log(t)
        for i in range(self.state.k):
            n = self.state.n_pulls[i]
            mean = self.state.sum_reward[i] / n
            bonus = self.c * math.sqrt(log_t / n)
            val = mean + bonus
            if val > best_val:
                best_val, best_arm = val, i
        return best_arm


class EXP3P(Bandit):
    """EXP3 with implicit exploration (EXP3-IX style)."""

    name = "exp3p"

    def __init__(
        self, k: int, *, eta: float = 0.07, gamma: float = 0.1,
        rng: random.Random | None = None,
        reward_min: float = 0.0, reward_max: float = 1.0,
    ) -> None:
        super().__init__(k)
        self.eta = eta
        self.gamma = gamma
        self.weights = [1.0] * k
        self.rng = rng or random.Random()
        self.reward_min = reward_min
        self.reward_max = reward_max
        self._last_probs: list[float] | None = None

    def _probs(self) -> list[float]:
        k = self.state.k
        w_sum = sum(self.weights)
        return [(1.0 - self.gamma) * w / w_sum + self.gamma / k for w in self.weights]

    def select(self) -> int:
        probs = self._probs()
        self._last_probs = probs
        u = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u <= acc:
                return i
        return self.state.k - 1

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        probs = self._last_probs or self._probs()
        span = max(self.reward_max - self.reward_min, 1e-12)
        g = (reward - self.reward_min) / span
        g = min(1.0, max(0.0, g))
        implicit = self.gamma / (2.0 * self.state.k)
        g_hat = g / (probs[arm] + implicit)
        self.weights[arm] *= math.exp(self.eta * g_hat)


class ThompsonGaussian(Bandit):
    """Thompson sampling on the *last-observed* reward of each arm (online, non-stationary)."""

    name = "thompson"

    def __init__(
        self, k: int, *, rng: random.Random | None = None,
        sigma: float = 0.1, prior_mean: float = 0.5, prior_strength: float = 1.0,
    ) -> None:
        super().__init__(k)
        self.rng = rng or random.Random()
        self.sigma = sigma
        self.prior_mean = prior_mean
        self.prior_strength = prior_strength

    def select(self) -> int:
        best_arm, best_sample = 0, float("-inf")
        for i in range(self.state.k):
            n = self.state.n_pulls[i]
            if n == 0:
                mu = self.prior_mean
                std = self.sigma / math.sqrt(self.prior_strength)
            else:
                obs_mean = self.state.sum_reward[i] / n
                mu = (self.prior_strength * self.prior_mean + n * obs_mean) / (
                    self.prior_strength + n
                )
                std = self.sigma / math.sqrt(self.prior_strength + n)
            theta = self.rng.gauss(mu, std)
            if theta > best_sample:
                best_sample, best_arm = theta, i
        return best_arm


class RandomBandit(Bandit):
    name = "random"

    def __init__(self, k: int, *, rng: random.Random | None = None) -> None:
        super().__init__(k)
        self.rng = rng or random.Random()

    def select(self) -> int:
        return self.rng.randrange(self.state.k)


def make_bandit(algo: str, k: int, *, bandit_seed: int = 0, **kwargs) -> Bandit:
    rng = random.Random(bandit_seed)
    if algo == "ucb":
        return UCB1(k, **kwargs)
    if algo == "exp3p":
        return EXP3P(k, rng=rng, **kwargs)
    if algo == "thompson":
        return ThompsonGaussian(k, rng=rng, **kwargs)
    if algo == "random":
        return RandomBandit(k, rng=rng)
    raise ValueError(f"Unknown bandit {algo!r}. Available: {BANDITS}")
