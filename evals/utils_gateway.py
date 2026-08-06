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

# Health checks run in irregular batches (observed: most within ~48h, a long
# tail of abandoned models >7 days stale). 72h covers the normal cadence with
# margin while still treating truly stale entries as unknown, not trusted.
STALE_AFTER = timedelta(hours=72)

_FAMILY_SUBSTRINGS = ("claude", "gemini", "locai")
_GPT_MARKERS = ("gpt", "o4-", "o1-", "o3-")


@dataclass(frozen=True)
class GatewayModel:
    name: str
    family: str | None
    health: str  # "healthy" | "unhealthy" | "unknown"


def derive_family(name: str) -> str | None:
    """Bucket a model name into a known vendor family by substring match.

    Only the recognised families (claude/gemini/locai/gpt) are classified.
    Everything else returns None rather than a guessed catch-all bucket —
    the leftover models span unrelated architectures/vendors, so grouping
    them under one label would imply a relationship that isn't there.
    """
    lowered = name.lower()
    for family in _FAMILY_SUBSTRINGS:
        if family in lowered:
            return family
    if any(marker in lowered for marker in _GPT_MARKERS):
        return "gpt"
    return None


def filter_by_family(models: list[GatewayModel], families: list[str]) -> list[GatewayModel]:
    """Return models whose family matches any of the given families.

    Takes a list so callers can compare multiple families in one run
    (e.g. gemini vs. claude), not just select a single one.
    """
    family_set = set(families)
    return [m for m in models if m.family in family_set]


def exclude_unhealthy(models: list[GatewayModel]) -> list[GatewayModel]:
    """Drop models whose most recent, non-stale health check reports "unhealthy".

    Models with no recent health data ("unknown") are kept — absence of
    evidence isn't evidence of a problem. Kept separate from discovery so
    callers that want to see everything (e.g. explicit --models lookups,
    where "confirmed unhealthy" and "unknown model name" need to stay
    distinguishable) can skip this step.
    """
    return [m for m in models if m.health != "unhealthy"]


def select_by_name(
    models: list[GatewayModel], names: list[str]
) -> tuple[list[GatewayModel], list[str]]:
    """Split requested names into found models and names not on the gateway.

    Unlike --family (a fixed, argparse-validated choice set), model names
    come from whatever's currently on the gateway, so a typo can only be
    caught by looking it up at runtime. Returns plain data — callers decide
    how to report `missing` (error, warning, etc.), keeping that a CLI
    concern rather than something this module prints itself.
    """
    by_name = {m.name: m for m in models}
    found = []
    missing = []
    for name in names:
        if name in by_name:
            found.append(by_name[name])
        else:
            missing.append(name)
    return found, missing


def filter_chat_models(model_group_items: list[dict]) -> list[str]:
    """Return model_group names that support chat completions."""
    return [item["model_group"] for item in model_group_items if item.get("mode") == "chat"]


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


def _gateway_client() -> httpx.AsyncClient:
    base_url = os.getenv("LLM_GATEWAY_URL")
    api_key = os.getenv("CONSULT_EVAL_LITELLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("LLM_GATEWAY_URL and CONSULT_EVAL_LITELLM_API_KEY must be set")
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

    Unfiltered by design — callers decide whether/how to narrow the list
    (exclude_unhealthy, filter_by_family, or an exact-name lookup), since
    that decision depends on the selection mode (--family/--all/--models).
    """
    async with _gateway_client() as client:
        model_group_items, health_checks = await asyncio.gather(
            fetch_model_group_info(client),
            fetch_health_latest(client),
        )

    chat_names = filter_chat_models(model_group_items)
    health_by_name = latest_health_by_model(health_checks)

    return [
        GatewayModel(
            name=name,
            family=derive_family(name),
            health=health_by_name.get(name, "unknown"),
        )
        for name in chat_names
    ]
