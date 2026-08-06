from datetime import datetime, timedelta, timezone

import pytest

import utils_gateway
from conftest import make_gateway_model


def _health(model_name, status, hours_ago, check_id=None, now=None):
    now = now or datetime.now(timezone.utc)
    checked_at = (now - timedelta(hours=hours_ago)).isoformat()
    check_id = check_id or f"{model_name}-{hours_ago}-{status}"
    return check_id, {"model_name": model_name, "status": status, "checked_at": checked_at}


class TestFilterChatModels:
    def test_keeps_only_chat_mode(self):
        items = [
            {"model_group": "gpt-4o", "mode": "chat", "supports_reasoning": False},
            {"model_group": "dall-e-3", "mode": "image_generation"},
            {"model_group": "text-embedding-3", "mode": "embedding"},
        ]
        assert utils_gateway.filter_chat_models(items) == [items[0]]


class TestLatestHealthByModel:
    NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def test_healthy_fresh_included(self):
        checks = dict([_health("gpt-4o", "healthy", hours_ago=1, now=self.NOW)])
        assert utils_gateway.latest_health_by_model(checks, now=self.NOW) == {
            "gpt-4o": "healthy"
        }

    def test_unhealthy_fresh_included(self):
        checks = dict([_health("gpt-4o", "unhealthy", hours_ago=1, now=self.NOW)])
        assert utils_gateway.latest_health_by_model(checks, now=self.NOW) == {
            "gpt-4o": "unhealthy"
        }

    def test_stale_check_dropped(self):
        checks = dict([_health("gpt-4o", "unhealthy", hours_ago=200, now=self.NOW)])
        assert utils_gateway.latest_health_by_model(checks, now=self.NOW) == {}

    def test_most_recent_check_wins(self):
        checks = dict(
            [
                _health("gpt-4o", "healthy", hours_ago=48, check_id="old", now=self.NOW),
                _health("gpt-4o", "unhealthy", hours_ago=1, check_id="new", now=self.NOW),
            ]
        )
        assert utils_gateway.latest_health_by_model(checks, now=self.NOW) == {
            "gpt-4o": "unhealthy"
        }

    def test_most_recent_check_wins_regardless_of_dict_order(self):
        checks = dict(
            [
                _health("gpt-4o", "unhealthy", hours_ago=1, check_id="new", now=self.NOW),
                _health("gpt-4o", "healthy", hours_ago=48, check_id="old", now=self.NOW),
            ]
        )
        assert utils_gateway.latest_health_by_model(checks, now=self.NOW) == {
            "gpt-4o": "unhealthy"
        }


class TestDeriveFamily:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("gpt-4.1-sweden", "gpt"),
            ("o3-mini", "gpt"),
            ("claude-haiku-4.5", "claude"),
            ("gemini-2.5-flash", "gemini"),
            ("locailabs/locai-l1-large-2011", "locai"),
            ("mistral-large", None),
        ],
    )
    def test_family_bucket(self, name, expected):
        assert utils_gateway.derive_family(name) == expected


class TestFilterByFamily:
    MODELS = [
        make_gateway_model(name="gpt-4o", family="gpt"),
        make_gateway_model(name="claude-haiku", family="claude"),
        make_gateway_model(name="gemini-flash", family="gemini"),
        make_gateway_model(name="bedrock-qwen3", family=None),
    ]

    def test_single_family(self):
        result = utils_gateway.filter_by_family(self.MODELS, ["claude"])
        assert [m.name for m in result] == ["claude-haiku"]

    def test_multiple_families(self):
        result = utils_gateway.filter_by_family(self.MODELS, ["gemini", "claude"])
        assert {m.name for m in result} == {"gemini-flash", "claude-haiku"}

    def test_no_match_returns_empty(self):
        assert utils_gateway.filter_by_family(self.MODELS, ["locai"]) == []


class TestSelectByName:
    MODELS = [
        make_gateway_model(name="gpt-4o", family="gpt"),
        make_gateway_model(name="claude-haiku", family="claude", health="unhealthy"),
    ]

    def test_all_found(self):
        found, missing = utils_gateway.select_by_name(self.MODELS, ["gpt-4o", "claude-haiku"])
        assert {m.name for m in found} == {"gpt-4o", "claude-haiku"}
        assert missing == []

    def test_some_missing(self):
        found, missing = utils_gateway.select_by_name(self.MODELS, ["gpt-4o", "typo-model"])
        assert [m.name for m in found] == ["gpt-4o"]
        assert missing == ["typo-model"]

    def test_found_model_keeps_its_health_status(self):
        found, _ = utils_gateway.select_by_name(self.MODELS, ["claude-haiku"])
        assert found[0].health == "unhealthy"


class TestExcludeUnhealthy:
    MODELS = [
        make_gateway_model(name="gpt-4o", family="gpt"),
        make_gateway_model(name="claude-haiku", family="claude", health="unhealthy"),
        make_gateway_model(name="mystery-model", family=None, health="unknown"),
    ]

    def test_drops_unhealthy_keeps_healthy_and_unknown(self):
        result = utils_gateway.exclude_unhealthy(self.MODELS)
        assert {m.name for m in result} == {"gpt-4o", "mystery-model"}


class TestDiscoverChatModels:
    async def test_combines_and_resolves_unfiltered(self, monkeypatch):
        model_group_items = [
            {"model_group": "gpt-4o", "mode": "chat", "supports_reasoning": False},
            {"model_group": "claude-haiku", "mode": "chat", "supports_reasoning": False},
            {"model_group": "gemini-flash", "mode": "chat", "supports_reasoning": True},
            {"model_group": "mystery-model", "mode": "chat", "supports_reasoning": False},
            {"model_group": "text-embedding-3", "mode": "embedding"},
        ]
        health_checks = dict(
            [
                _health("gpt-4o", "healthy", hours_ago=1),
                _health("claude-haiku", "unhealthy", hours_ago=1),
                _health("gemini-flash", "unhealthy", hours_ago=200),  # stale
                # mystery-model: no health data at all
            ]
        )

        async def fake_model_group_info(client):
            return model_group_items

        async def fake_health_latest(client):
            return health_checks

        monkeypatch.setattr(utils_gateway, "fetch_model_group_info", fake_model_group_info)
        monkeypatch.setattr(utils_gateway, "fetch_health_latest", fake_health_latest)
        monkeypatch.setenv("LLM_GATEWAY_URL", "https://gateway.example.invalid")
        monkeypatch.setenv("CONSULT_EVAL_LITELLM_API_KEY", "test-key")

        result = await utils_gateway.discover_chat_models()

        by_name = {m.name: m for m in result}
        # unfiltered: non-chat excluded, but unhealthy/stale/unknown models all present
        assert set(by_name) == {"gpt-4o", "claude-haiku", "gemini-flash", "mystery-model"}
        assert by_name["gpt-4o"].health == "healthy"
        assert by_name["claude-haiku"].health == "unhealthy"
        assert by_name["gemini-flash"].health == "unknown"  # stale, not trusted
        assert by_name["mystery-model"].health == "unknown"  # no data at all
        assert by_name["gpt-4o"].family == "gpt"
        assert by_name["gemini-flash"].family == "gemini"
        assert by_name["mystery-model"].family is None
        assert by_name["gemini-flash"].supports_reasoning is True
        assert by_name["gpt-4o"].supports_reasoning is False
