from datetime import datetime, timedelta, timezone

import pytest

import utils_gateway


def _health(model_name, status, hours_ago, check_id=None, now=None):
    now = now or datetime.now(timezone.utc)
    checked_at = (now - timedelta(hours=hours_ago)).isoformat()
    check_id = check_id or f"{model_name}-{hours_ago}-{status}"
    return check_id, {"model_name": model_name, "status": status, "checked_at": checked_at}


class TestFilterChatModels:
    def test_keeps_only_chat_mode(self):
        items = [
            {"model_group": "gpt-4o", "mode": "chat"},
            {"model_group": "dall-e-3", "mode": "image_generation"},
            {"model_group": "text-embedding-3", "mode": "embedding"},
        ]
        assert utils_gateway.filter_chat_models(items) == ["gpt-4o"]


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


class TestDiscoverChatModels:
    async def test_combines_and_filters(self, monkeypatch):
        model_group_items = [
            {"model_group": "gpt-4o", "mode": "chat"},
            {"model_group": "claude-haiku", "mode": "chat"},
            {"model_group": "gemini-flash", "mode": "chat"},
            {"model_group": "mystery-model", "mode": "chat"},
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
        assert set(by_name) == {"gpt-4o", "gemini-flash", "mystery-model"}
        assert "claude-haiku" not in by_name  # unhealthy + fresh -> excluded
        assert by_name["gpt-4o"].health == "healthy"
        assert by_name["gemini-flash"].health == "unknown"  # stale, not trusted
        assert by_name["mystery-model"].health == "unknown"  # no data at all
        assert by_name["gpt-4o"].family == "gpt"
        assert by_name["gemini-flash"].family == "gemini"
        assert by_name["mystery-model"].family is None
