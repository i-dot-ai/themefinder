"""Discover chat-capable models available on the LLM gateway.

Combines /model_group/info (which models exist and support chat) with
/health/latest (whether they're currently reachable) into the model list
the eval suite runs against, so it updates automatically as the gateway's
model list changes.
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

# Health checks are observed to run within ~48h; 72h gives margin before
# treating a check as stale.
STALE_AFTER = timedelta(hours=72)

_FAMILY_SUBSTRINGS = ("claude", "gemini", "locai")
_GPT_MARKERS = ("gpt", "o4-", "o1-", "o3-")

# Every family derive_family() can return (besides None).
KNOWN_FAMILIES = (*_FAMILY_SUBSTRINGS, "gpt")


@dataclass(frozen=True)
class GatewayModel:
    name: str
    family: str | None
    health: str  # "healthy" | "unhealthy" | "unknown"
    supports_reasoning: bool = False


def derive_family(name: str) -> str | None:
    """Bucket a model name into a known vendor family (claude/gemini/locai/gpt) by substring match."""
    lowered = name.lower()
    for family in _FAMILY_SUBSTRINGS:
        if family in lowered:
            return family
    if any(marker in lowered for marker in _GPT_MARKERS):
        return "gpt"
    return None


def filter_by_family(
    models: list[GatewayModel], families: list[str]
) -> list[GatewayModel]:
    """Return models whose family matches any of the given families."""
    family_set = set(families)
    return [m for m in models if m.family in family_set]


def split_unhealthy(
    models: list[GatewayModel],
) -> tuple[list[GatewayModel], list[GatewayModel]]:
    """Split models into (kept, unhealthy) by most recent, non-stale health check.

    Models with no recent health data ("unknown") are kept — absence of
    evidence isn't evidence of a problem.
    """
    kept = []
    unhealthy = []
    for m in models:
        if m.health == "unhealthy":
            unhealthy.append(m)
        else:
            kept.append(m)
    return kept, unhealthy


def select_by_name(
    models: list[GatewayModel], names: list[str]
) -> tuple[list[GatewayModel], list[str]]:
    """Split requested names into found models and names not on the gateway."""
    by_name = {m.name: m for m in models}
    found = []
    missing = []
    for name in names:
        if name in by_name:
            found.append(by_name[name])
        else:
            missing.append(name)
    return found, missing


def filter_chat_models(model_group_items: list[dict]) -> list[dict]:
    """Return model_group entries that support chat completions.

    Keeps the raw dicts, not just names, so callers can still read
    per-model fields like `supports_reasoning`.
    """
    return [item for item in model_group_items if item.get("mode") == "chat"]


def _parse_checked_at(value: str) -> datetime:
    """Parse a health-check timestamp, tolerating naive values and 'Z' suffixes."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def latest_health_by_model(
    health_checks: dict[str, dict],
    now: datetime | None = None,
    stale_after: timedelta = STALE_AFTER,
) -> dict[str, str]:
    """Reduce raw health-check rows to one status per model name.

    Keeps the most recent check per model name; a check older than
    `stale_after` is dropped (that row, not the model) rather than trusted
    as current status.
    """
    now = now or datetime.now(timezone.utc)
    latest: dict[str, tuple[datetime, str]] = {}  # name -> (checked_at, status)

    for check in health_checks.values():
        name = check["model_name"]
        checked_at = _parse_checked_at(check["checked_at"])
        if name in latest and checked_at <= latest[name][0]:
            continue
        latest[name] = (checked_at, check["status"])

    return {
        name: status
        for name, (checked_at, status) in latest.items()
        if now - checked_at <= stale_after
    }


def gateway_credentials() -> tuple[str, str]:
    """Read and validate the two required gateway env vars."""
    base_url = os.getenv("LLM_GATEWAY_URL")
    api_key = os.getenv("CONSULT_EVAL_LITELLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "LLM_GATEWAY_URL and CONSULT_EVAL_LITELLM_API_KEY must be set"
        )
    return base_url, api_key


def _gateway_client() -> httpx.AsyncClient:
    base_url, api_key = gateway_credentials()
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )


async def fetch_model_group_info(client: httpx.AsyncClient) -> list[dict]:
    response = await client.get("/model_group/info")
    response.raise_for_status()
    return response.json()["data"]


async def fetch_health_latest(client: httpx.AsyncClient) -> dict[str, dict]:
    response = await client.get("/health/latest")
    response.raise_for_status()
    return response.json()["latest_health_checks"]


async def discover_chat_models() -> list[GatewayModel]:
    """Fetch every chat-capable gateway model with its family and health resolved.

    Unfiltered by design — callers narrow the list themselves (split_unhealthy,
    filter_by_family, or an exact-name lookup) based on how they want to select.
    """
    async with _gateway_client() as client:
        model_group_items, health_checks = await asyncio.gather(
            fetch_model_group_info(client),
            fetch_health_latest(client),
        )

    chat_models = filter_chat_models(model_group_items)
    health_by_name = latest_health_by_model(health_checks)

    return [
        GatewayModel(
            name=item["model_group"],
            family=derive_family(item["model_group"]),
            health=health_by_name.get(item["model_group"], "unknown"),
            supports_reasoning=item["supports_reasoning"],
        )
        for item in chat_models
    ]
