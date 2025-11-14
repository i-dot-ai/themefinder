import asyncio
import os

import pandas as pd
from ecologits import EcoLogits
from langchain_openai import ChatOpenAI

import themefinder

EcoLogits.init(providers=["litellm"])

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_base=os.environ["LLM_GATEWAY_URL"],
    openai_api_key=os.environ["LITELLM_API_KEY"],
)

question = "What improvements would you most like to see in local public transportation?"

responses = pd.read_json("./examples/example_data.json")


async def a():
    results = await themefinder.find_themes(
        responses,
        llm=llm,
        question=question,
    )

    print(results)

asyncio.run(a())