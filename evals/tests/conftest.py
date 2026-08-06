import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils_gateway  # noqa: E402


def make_gateway_model(
    name: str,
    family: str | None = "gpt",
    health: str = "healthy",
    supports_reasoning: bool = False,
) -> utils_gateway.GatewayModel:
    """Shared GatewayModel test fixture builder for test_utils_gateway.py / test_benchmark.py."""
    return utils_gateway.GatewayModel(
        name=name, family=family, health=health, supports_reasoning=supports_reasoning
    )
