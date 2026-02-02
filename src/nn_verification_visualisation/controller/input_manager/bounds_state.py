from __future__ import annotations

from dataclasses import dataclass, field

from nn_verification_visualisation.model.data.input_bounds import InputBounds
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig


@dataclass
class BoundsState:
    saved_bounds: list[InputBounds] = field(default_factory=list)
    selected_bounds_index: int = -1
    draft_bounds: list[tuple[float, float]] = field(default_factory=list)


class BoundsStateRegistry:
    _states: dict[NetworkVerificationConfig, BoundsState] = {}

    @classmethod
    def get(cls, config: NetworkVerificationConfig) -> BoundsState:
        if config not in cls._states:
            cls._states[config] = BoundsState(draft_bounds=config.bounds.get_values())
        return cls._states[config]
