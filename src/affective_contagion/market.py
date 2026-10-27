"""Stylized endogenous price-impact process."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from typing import Protocol


class NormalRandomSource(Protocol):
    """Minimal interface for an independently seeded market-noise stream."""

    def normal(self) -> float:
        """Draw one standard-normal variate."""
        ...


@dataclass(frozen=True, slots=True)
class MarketParameters:
    """Coefficients for the stylized market process."""

    fundamental_value: float = 100.0
    price_impact: float = 0.02
    mean_reversion: float = 0.01
    noise_sigma: float = 0.003

    def __post_init__(self) -> None:
        """Validate process parameters."""
        if self.fundamental_value <= 0:
            msg = "fundamental_value must be positive"
            raise ValueError(msg)
        if self.price_impact < 0 or self.mean_reversion < 0 or self.noise_sigma < 0:
            msg = "market coefficients must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MarketStep:
    """One auditable market transition."""

    round_index: int
    previous_price: float
    price: float
    order_imbalance: float
    log_return: float
    noise_draw: float


@dataclass(slots=True)
class SimpleMarket:
    """Price-impact process with mean reversion and noise, not a limit-order book."""

    parameters: MarketParameters
    random_source: NormalRandomSource
    price: float = field(init=False)
    history: list[MarketStep] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Start at fundamental value."""
        self.price = self.parameters.fundamental_value

    def clear(self, signed_orders: list[float]) -> MarketStep:
        """Apply mean signed demand, reversion, and one independent noise draw."""
        if not signed_orders:
            msg = "at least one signed order is required"
            raise ValueError(msg)
        if any(not -1 <= order <= 1 for order in signed_orders):
            msg = "signed orders must be within [-1, 1]"
            raise ValueError(msg)

        imbalance = sum(signed_orders) / len(signed_orders)
        previous = self.price
        noise_draw = self.random_source.normal()
        drift = (
            self.parameters.price_impact * imbalance
            + self.parameters.mean_reversion * log(self.parameters.fundamental_value / previous)
            + self.parameters.noise_sigma * noise_draw
        )
        self.price = previous * exp(drift)
        step = MarketStep(
            round_index=len(self.history),
            previous_price=previous,
            price=self.price,
            order_imbalance=imbalance,
            log_return=drift,
            noise_draw=noise_draw,
        )
        self.history.append(step)
        return step

    def deviation_percent(self) -> float:
        """Return current price deviation from the known fundamental value."""
        return (self.price / self.parameters.fundamental_value - 1) * 100
