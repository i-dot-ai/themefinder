import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from themefinder import find_themes, theme_condensation


@pytest.mark.asyncio()
async def test_find_themes(mock_llm, sample_df):
    """Test the complete theme finding pipeline with mocked LLM responses."""
    mock_llm.ainvoke.side_effect = [
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
        MagicMock(content='{"responses": [{"themes": ["theme1", "theme2"]}]}'),
        MagicMock(content='{"responses": [{"themes": ["theme3", "theme4"]}]}'),
        MagicMock(
            content='{"responses": [{"condensed_themes": ["main_theme1", "main_theme2"]}]}'
        ),
        MagicMock(content='{"responses": [{"topic_id": "label1", "topic": "desc1"}]}'),
        MagicMock(
            content=json.dumps(
                {
                    "responses": [
                        {
                            "response_id": 1,
                            "reason": "reason1",
                            "label": ["label1"],
                        },
                        {
                            "response_id": 2,
                            "reason": "reason2",
                            "label": ["label2"],
                        },
                    ]
                }
            )
        ),
    ]
    result = await find_themes(sample_df, mock_llm, question="test question")
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in [
            "sentiment",
            "topics",
            "condensed_topics",
            "mapping",
            "question",
            "refined_topics",
        ]
    )
    assert mock_llm.ainvoke.call_count == 6


@pytest.mark.asyncio
async def test_theme_condensation(monkeypatch):
    """
    Test the while loop behavior in theme_condensation.

    The test creates an initial DataFrame with 5 rows and uses a batch size of 2.
    It patches batch_and_run with a dummy function that:
      - On the first call, returns a DataFrame with 3 rows (reducing the row count).
      - On the second call, returns a DataFrame with the same 3 rows (causing the loop to break).
      - On the final call (after the loop), returns a DataFrame with 2 rows.
    """
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 6)]})

    # First call: condensation reduces rows from 5 to 3.
    df_first = pd.DataFrame({"theme": ["A", "B", "C"]})
    # Second call: returns the same number of rows (3), so the while loop should break.
    df_second = pd.DataFrame({"theme": ["A", "B", "C"]})
    # Final call (outside the loop): returns the final condensed DataFrame with 2 rows.
    df_final = pd.DataFrame({"theme": ["A", "B"]})
    dummy_outputs = [df_first, df_second, df_final]

    call_count = 0

    async def dummy_batch_and_run(
        themes_df, prompt_template, llm, batch_size, question, system_prompt, **kwargs
    ):
        nonlocal call_count
        call_count += 1
        return dummy_outputs.pop(0)

    monkeypatch.setitem(
        theme_condensation.__globals__, "batch_and_run", dummy_batch_and_run
    )
    dummy_llm = MagicMock()

    await theme_condensation(
        themes_df=initial_df.copy(),
        llm=dummy_llm,
        question="test question",
        batch_size=2,
    )

    # --- Assertions ---
    # We expect:
    #   - The first call (reducing from 5 to 3 rows),
    #   - The second call (returns 3 rows, so while loop exits),
    #   - The final call (returns 2 rows) => total of 3 calls.
    assert call_count == 3, "batch_and_run should have been called 3 times"
