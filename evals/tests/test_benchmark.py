import argparse

import benchmark
from conftest import make_gateway_model


def _args(models=None, family=None, all_=False, **extra):
    return argparse.Namespace(models=models, family=family, all=all_, **extra)


class TestModelConfigTag:
    def test_plain_name_when_no_reasoning_effort(self):
        config = benchmark.ModelConfig(name="gpt-4o-uk")
        assert config.tag == "gpt-4o-uk"

    def test_appends_reasoning_effort(self):
        config = benchmark.ModelConfig(name="gpt-5-mini-sweden", reasoning_effort="low")
        assert config.tag == "gpt-5-mini-sweden_low"


class TestModelConfigCreateLlm:
    def _set_gateway_env(self, monkeypatch):
        monkeypatch.setenv("LLM_GATEWAY_URL", "https://gateway.example.invalid")
        monkeypatch.setenv("CONSULT_EVAL_LITELLM_API_KEY", "test-key")

    def test_sets_temperature_when_no_reasoning_effort(self, monkeypatch):
        self._set_gateway_env(monkeypatch)
        config = benchmark.ModelConfig(name="gpt-4o-uk", temperature=0.2)
        llm = config.create_llm()
        assert llm.model == "gpt-4o-uk"
        assert llm.request_kwargs == {"temperature": 0.2}

    def test_omits_temperature_when_reasoning_effort_set(self, monkeypatch):
        self._set_gateway_env(monkeypatch)
        config = benchmark.ModelConfig(name="gpt-5-mini-sweden", reasoning_effort="low")
        llm = config.create_llm()
        assert llm.model == "gpt-5-mini-sweden"
        assert llm.request_kwargs == {}


class TestToModelConfigs:
    def test_non_reasoning_model_ignores_effort_levels(self):
        gm = make_gateway_model(name="gpt-4o-uk", supports_reasoning=False)
        configs = benchmark._to_model_configs(gm, ["low", "medium", "high"])
        assert len(configs) == 1
        assert configs[0].name == "gpt-4o-uk"
        assert configs[0].reasoning_effort is None

    def test_reasoning_model_expands_per_effort_level(self):
        gm = make_gateway_model(name="gpt-5-mini-sweden", supports_reasoning=True)
        configs = benchmark._to_model_configs(gm, ["low", "medium", "high"])
        assert [c.tag for c in configs] == [
            "gpt-5-mini-sweden_low",
            "gpt-5-mini-sweden_medium",
            "gpt-5-mini-sweden_high",
        ]


class TestApplyQuickPreset:
    def test_noop_when_not_quick(self):
        args = _args(all_=True, quick=False, dataset="housing_S", runs=5)
        benchmark._apply_quick_preset(args)
        assert args.dataset == "housing_S"
        assert args.runs == 5
        assert args.all is True

    def test_overrides_other_selectors_when_quick(self):
        args = _args(
            models=["something"],
            family=["gpt"],
            quick=True,
            dataset="housing_S",
            runs=5,
        )
        benchmark._apply_quick_preset(args)
        assert args.dataset == "gambling_XS"
        assert args.runs == 1
        assert args.models == ["gpt-4.1-sweden-2025-03"]
        assert args.family is None
        assert args.all is False


class TestValidateSelectorArgs:
    def test_exactly_one_selector_is_valid(self):
        assert benchmark._validate_selector_args(_args(models=["gpt-4o-uk"])) is None

    def test_no_selector_is_invalid(self):
        assert benchmark._validate_selector_args(_args()) is not None

    def test_multiple_selectors_is_invalid(self):
        args = _args(models=["gpt-4o-uk"], family=["gpt"])
        assert benchmark._validate_selector_args(args) is not None


class TestResolveSelectedModels:
    MODELS = [
        make_gateway_model(name="gpt-4o-uk", family="gpt"),
        make_gateway_model(name="claude-haiku", family="claude", health="unhealthy"),
        make_gateway_model(name="gemini-flash", family="gemini"),
    ]

    def test_models_selector_returns_found_and_missing(self):
        args = _args(models=["gpt-4o-uk", "typo-model"])
        selected, missing, unhealthy = benchmark._resolve_selected_models(
            args, self.MODELS
        )
        assert [m.name for m in selected] == ["gpt-4o-uk"]
        assert missing == ["typo-model"]
        assert unhealthy == []

    def test_models_selector_reports_unhealthy_matches(self):
        # --models is explicit-by-name: unhealthy matches are still selected
        # (main() warns using `unhealthy`, doesn't silently drop them)
        args = _args(models=["claude-haiku"])
        selected, missing, unhealthy = benchmark._resolve_selected_models(
            args, self.MODELS
        )
        assert [m.name for m in selected] == ["claude-haiku"]
        assert missing == []
        assert [m.name for m in unhealthy] == ["claude-haiku"]

    def test_family_selector_excludes_unhealthy(self):
        args = _args(family=["claude"])
        selected, missing, unhealthy = benchmark._resolve_selected_models(
            args, self.MODELS
        )
        assert selected == []  # claude-haiku is unhealthy, dropped
        assert missing == []
        assert unhealthy == []

    def test_all_selector_excludes_unhealthy_and_ignores_family(self):
        args = _args(all_=True)
        selected, missing, unhealthy = benchmark._resolve_selected_models(
            args, self.MODELS
        )
        assert {m.name for m in selected} == {"gpt-4o-uk", "gemini-flash"}
        assert missing == []
        assert unhealthy == []
