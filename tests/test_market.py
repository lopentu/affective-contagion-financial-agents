"""Tests for the stylized market process."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from affective_contagion.market import MarketParameters, SimpleMarket


@dataclass
class FixedNoise:
    """Deterministic market-noise source."""

    value: float

    def normal(self) -> float:
        """Return the fixed draw."""
        return self.value


def test_balanced_orders_and_zero_noise_keep_initial_price() -> None:
    """No demand, noise, or initial reversion implies no price move."""
    market = SimpleMarket(MarketParameters(), FixedNoise(0.0))
    step = market.clear([1.0, -1.0])
    assert step.order_imbalance == 0
    assert step.price == pytest.approx(100.0)
    assert market.deviation_percent() == pytest.approx(0.0)


def test_positive_imbalance_raises_price() -> None:
    """Signed demand should have endogenous price impact."""
    market = SimpleMarket(MarketParameters(noise_sigma=0), FixedNoise(0.0))
    step = market.clear([1.0, 0.5])
    assert step.price > step.previous_price
    assert step.round_index == 0


def test_empty_order_book_is_rejected() -> None:
    """A transition requires an explicitly defined order population."""
    market = SimpleMarket(MarketParameters(), FixedNoise(0.0))
    with pytest.raises(ValueError, match="order"):
        market.clear([])
