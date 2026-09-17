"""Run the ThemeFinder SystemOne hybrid pipeline on a file of responses.

The generative stages (theme generation, condensation, refinement) run on an
OpenAI model; the classification stages (theme mapping, detail detection) run
on TypeSafe's jev SystemOne model.

Usage:
    export OPENAI_API_KEY=...      # or the eval gateway variables
    export TYPESAFE_API_KEY=...
    uv run python examples/run_systemone.py \
        --question "How should the bus service be improved?" \
        --input examples/example_data.json \
        --llm-model gpt-4o-mini

Input: a JSON, JSONL or CSV file with 'response_id' and 'response' columns.
Output: themes.csv, mapping.csv and detailed_responses.csv in --output-dir.
"""

import argparse
import asyncio
from pathlib import Path

import dotenv
import pandas as pd

from themefinder import OpenAILLM, SystemOne, find_themes_hybrid


def load_responses(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run ThemeFinder with SystemOne classification stages"
    )
    parser.add_argument("--question", required=True, help="The survey question")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "example_data.json",
        help="JSON/JSONL/CSV file with response_id and response columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("themefinder_output"),
        help="Directory to write themes.csv, mapping.csv, detailed_responses.csv",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model for the generative stages",
    )
    parser.add_argument(
        "--responses-api",
        action="store_true",
        help="Use the OpenAI Responses API (required for gpt-5* models)",
    )
    parser.add_argument(
        "--model", default=None, help="SystemOne model override (default jev-latest)"
    )
    args = parser.parse_args()

    responses_df = load_responses(args.input)
    is_gpt5_family = args.llm_model.startswith("gpt-5")
    llm = OpenAILLM(
        model=args.llm_model,
        request_kwargs={} if is_gpt5_family else {"temperature": 0},
        use_responses_api=args.responses_api or is_gpt5_family,
    )
    systemone_client = SystemOne.from_env(model=args.model)

    result = await find_themes_hybrid(
        responses_df, llm, systemone_client, question=args.question
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("themes", "mapping", "detailed_responses", "unprocessables"):
        df = result[name]
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(args.output_dir / f"{name}.csv", index=False)

    print(f"\nFound {len(result['themes'])} themes:")
    for topic in result["themes"]["topic"]:
        print(f"  • {topic}")
    print(
        f"\nMapped {len(result['mapping'])} responses "
        f"({len(result['unprocessables'])} unprocessable)."
    )
    print(
        f"LLM usage: {llm.usage.input_tokens:,} in / {llm.usage.output_tokens:,} out; "
        f"SystemOne usage: {systemone_client.usage.input_tokens:,} in "
        f"across {systemone_client.usage.requests} requests."
    )
    print(f"Outputs written to {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
