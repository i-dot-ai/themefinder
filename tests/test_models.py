import pytest
from pydantic import ValidationError

from themefinder.models import (
    Position,
    EvidenceRich,
    ValidatedModel,
    Theme,
    ThemeGenerationResponses,
    CondensedTheme,
    ThemeCondensationResponses,
    RefinedTheme,
    ThemeRefinementResponses,
    DetailDetectionOutput,
    DetailDetectionResponses,
    ConstituentTheme,
    CrossCuttingTheme,
    CrossCuttingThemesResponse,
)


class TestValidatedModelAdditional:
    class NestedModel(ValidatedModel):
        attr: str

    class ContainerModel(ValidatedModel):
        items: list["TestValidatedModelAdditional.NestedModel"]
        other_list: list[str]

    class MockModel(ValidatedModel):
        field1: str
        field2: list[str] | None = None
        field3: list[str] | None = None

    def test_validate_unique_attribute_in_list(self):
        model = self.ContainerModel(
            items=[
                self.NestedModel(attr="value1"),
                self.NestedModel(attr="value2"),
            ],
            other_list=["a", "b"],
        )
        model.validate_unique_attribute_in_list("items", "attr")

        model = self.ContainerModel(
            items=[
                self.NestedModel(attr="same"),
                self.NestedModel(attr="same"),
            ],
            other_list=["a", "b"],
        )
        with pytest.raises(ValueError, match="must be unique across all items"):
            model.validate_unique_attribute_in_list("items", "attr")

    def test_validate_equal_lengths(self):
        model = self.MockModel(field1="test", field2=["a", "b"], field3=["x", "y"])
        model.validate_equal_lengths("field2", "field3")

        model = self.MockModel(field1="test", field2=["a", "b"], field3=["x"])
        with pytest.raises(ValueError, match="must all have the same length"):
            model.validate_equal_lengths("field2", "field3")


class TestTheme:
    def test_valid_theme(self):
        theme = Theme(
            topic_label="Healthcare",
            topic_description="Access to affordable healthcare services",
            position=Position.AGREEMENT,
        )
        assert theme.topic_label == "Healthcare"
        assert theme.position == Position.AGREEMENT

    def test_invalid_position(self):
        with pytest.raises(ValidationError):
            Theme(
                topic_label="Healthcare",
                topic_description="Access to affordable healthcare services",
                position="AGREE",
            )


class TestThemeGenerationResponses:
    def test_valid_themes(self):
        model = ThemeGenerationResponses(
            responses=[
                Theme(
                    topic_label="Healthcare",
                    topic_description="Access to affordable healthcare services",
                    position=Position.AGREEMENT,
                ),
                Theme(
                    topic_label="Education",
                    topic_description="Quality of public education system",
                    position=Position.DISAGREEMENT,
                ),
            ]
        )
        assert len(model.responses) == 2

    def test_duplicate_topics(self):
        with pytest.raises(ValueError, match="Duplicate topic labels detected"):
            ThemeGenerationResponses(
                responses=[
                    Theme(
                        topic_label="Healthcare",
                        topic_description="Description 1",
                        position=Position.AGREEMENT,
                    ),
                    Theme(
                        topic_label="healthcare",
                        topic_description="Description 2",
                        position=Position.DISAGREEMENT,
                    ),
                ]
            )


class TestCondensedTheme:
    def test_valid_condensed_theme(self):
        theme = CondensedTheme(
            topic_label="Combined Healthcare",
            topic_description="Healthcare accessibility and affordability",
            source_topic_count=3,
        )
        assert theme.source_topic_count == 3

    def test_invalid_source_count(self):
        with pytest.raises(ValidationError):
            CondensedTheme(
                topic_label="Combined Healthcare",
                topic_description="Healthcare accessibility and affordability",
                source_topic_count=0,  # Must be > 0
            )


class TestThemeCondensationResponses:
    def test_valid_condensed_themes(self):
        model = ThemeCondensationResponses(
            responses=[
                CondensedTheme(
                    topic_label="Combined Healthcare",
                    topic_description="Healthcare accessibility and affordability",
                    source_topic_count=3,
                ),
                CondensedTheme(
                    topic_label="Education Reform",
                    topic_description="Changes to education system",
                    source_topic_count=2,
                ),
            ]
        )
        assert len(model.responses) == 2

    def test_duplicate_topics(self):
        with pytest.raises(ValueError, match="Duplicate topic labels detected"):
            ThemeCondensationResponses(
                responses=[
                    CondensedTheme(
                        topic_label="Healthcare",
                        topic_description="Description 1",
                        source_topic_count=1,
                    ),
                    CondensedTheme(
                        topic_label="healthcare",
                        topic_description="Description 2",
                        source_topic_count=2,
                    ),
                ]
            )


class TestRefinedTheme:
    def test_valid_refined_theme(self):
        theme = RefinedTheme(
            topic_id="A",
            topic="Healthcare: Access to affordable healthcare services",
            source_topic_count=3,
        )
        assert theme.topic_id == "A"

    def test_invalid_topic_id_format(self):
        with pytest.raises(ValueError, match="must be uppercase letters only"):
            RefinedTheme(
                topic_id="1",
                topic="Healthcare: Description",
                source_topic_count=1,
            ).validate_topic_id_format()

    def test_invalid_topic_format(self):
        with pytest.raises(ValueError, match="must contain a label and description"):
            RefinedTheme(
                topic_id="A",
                topic="Healthcare without separator",
                source_topic_count=1,
            ).validate_topic_format()

    def test_topic_label_too_long(self):
        with pytest.raises(ValueError, match="must be under 10 words"):
            RefinedTheme(
                topic_id="A",
                topic="This is a very long topic label that exceeds the limit of ten words: Description",
                source_topic_count=1,
            ).validate_topic_format()


class TestThemeRefinementResponses:
    def test_valid_refined_themes(self):
        model = ThemeRefinementResponses(
            responses=[
                RefinedTheme(
                    topic_id="A",
                    topic="Healthcare: Access to affordable healthcare",
                    source_topic_count=3,
                ),
                RefinedTheme(
                    topic_id="B",
                    topic="Education: Quality of public education",
                    source_topic_count=2,
                ),
            ]
        )
        assert len(model.responses) == 2

    def test_duplicate_topic_ids(self):
        with pytest.raises(ValueError, match="Duplicate topic_ids detected"):
            ThemeRefinementResponses(
                responses=[
                    RefinedTheme(
                        topic_id="A",
                        topic="Healthcare: Description 1",
                        source_topic_count=1,
                    ),
                    RefinedTheme(
                        topic_id="A",
                        topic="Education: Description 2",
                        source_topic_count=2,
                    ),
                ]
            )

    def test_duplicate_topics(self):
        with pytest.raises(ValueError, match="Duplicate topics detected"):
            ThemeRefinementResponses(
                responses=[
                    RefinedTheme(
                        topic_id="A",
                        topic="Healthcare: Description",
                        source_topic_count=1,
                    ),
                    RefinedTheme(
                        topic_id="B",
                        topic="Healthcare: Description",
                        source_topic_count=2,
                    ),
                ]
            )


class TestDetailDetectionOutput:
    def test_valid_detail_detection(self):
        model = DetailDetectionOutput(response_id=1, evidence_rich=EvidenceRich.YES)
        assert model.evidence_rich == EvidenceRich.YES

    def test_invalid_evidence_rich(self):
        with pytest.raises(ValidationError):
            DetailDetectionOutput(
                response_id=1,
                evidence_rich="Maybe",
            )

    def test_invalid_response_id(self):
        with pytest.raises(ValidationError):
            DetailDetectionOutput(
                response_id=0,
                evidence_rich=EvidenceRich.NO,
            )


class TestDetailDetectionResponses:
    def test_valid_responses(self):
        model = DetailDetectionResponses(
            responses=[
                DetailDetectionOutput(response_id=1, evidence_rich=EvidenceRich.YES),
                DetailDetectionOutput(response_id=2, evidence_rich=EvidenceRich.NO),
            ]
        )
        assert len(model.responses) == 2

    def test_duplicate_response_ids(self):
        with pytest.raises(ValueError, match="Response IDs must be unique"):
            DetailDetectionResponses(
                responses=[
                    DetailDetectionOutput(
                        response_id=1, evidence_rich=EvidenceRich.YES
                    ),
                    DetailDetectionOutput(response_id=1, evidence_rich=EvidenceRich.NO),
                ]
            )


class TestConstituentTheme:
    def test_valid_theme(self):
        theme = ConstituentTheme(question_number=1, theme_key="A")
        assert theme.question_number == 1
        assert theme.theme_key == "A"

    def test_invalid_question_number(self):
        with pytest.raises(ValidationError):
            ConstituentTheme(question_number=0, theme_key="A")

    def test_empty_theme_key(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ConstituentTheme(question_number=1, theme_key="")


class TestCrossCuttingTheme:
    def test_valid_cross_cutting_theme(self):
        theme = CrossCuttingTheme(
            name="Digital Accessibility",
            description="Themes related to digital inclusion",
            themes=[
                ConstituentTheme(question_number=1, theme_key="A"),
                ConstituentTheme(question_number=2, theme_key="B"),
            ],
        )
        assert theme.name == "Digital Accessibility"
        assert len(theme.themes) == 2

    def test_minimum_themes_requirement(self):
        with pytest.raises(
            ValueError, match="must include at least 2 constituent themes"
        ):
            CrossCuttingTheme(
                name="Test Theme",
                description="Test description",
                themes=[ConstituentTheme(question_number=1, theme_key="A")],
            )

    def test_duplicate_theme_combinations(self):
        with pytest.raises(
            ValueError, match="Duplicate question_number and theme_key combinations"
        ):
            CrossCuttingTheme(
                name="Test Theme",
                description="Test description",
                themes=[
                    ConstituentTheme(question_number=1, theme_key="A"),
                    ConstituentTheme(question_number=1, theme_key="A"),
                ],
            )

    def test_empty_name(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            CrossCuttingTheme(
                name="",
                description="Test description",
                themes=[
                    ConstituentTheme(question_number=1, theme_key="A"),
                    ConstituentTheme(question_number=2, theme_key="B"),
                ],
            )


class TestCrossCuttingThemesResponse:
    def test_valid_response(self):
        response = CrossCuttingThemesResponse(
            cross_cutting_themes=[
                CrossCuttingTheme(
                    name="Theme 1",
                    description="Description 1",
                    themes=[
                        ConstituentTheme(question_number=1, theme_key="A"),
                        ConstituentTheme(question_number=2, theme_key="B"),
                    ],
                ),
                CrossCuttingTheme(
                    name="Theme 2",
                    description="Description 2",
                    themes=[
                        ConstituentTheme(question_number=3, theme_key="C"),
                        ConstituentTheme(question_number=4, theme_key="D"),
                    ],
                ),
            ]
        )
        assert len(response.cross_cutting_themes) == 2

    def test_empty_list_is_valid(self):
        response = CrossCuttingThemesResponse(cross_cutting_themes=[])
        assert len(response.cross_cutting_themes) == 0

    def test_duplicate_names(self):
        with pytest.raises(
            ValueError, match="Cross-cutting theme names must be unique"
        ):
            CrossCuttingThemesResponse(
                cross_cutting_themes=[
                    CrossCuttingTheme(
                        name="Same Name",
                        description="Description 1",
                        themes=[
                            ConstituentTheme(question_number=1, theme_key="A"),
                            ConstituentTheme(question_number=2, theme_key="B"),
                        ],
                    ),
                    CrossCuttingTheme(
                        name="Same Name",
                        description="Description 2",
                        themes=[
                            ConstituentTheme(question_number=3, theme_key="C"),
                            ConstituentTheme(question_number=4, theme_key="D"),
                        ],
                    ),
                ]
            )
