"""Question generation using LLM for synthetic consultation datasets.

Generates UK government-style consultation questions following Cabinet Office
principles and the Gunning Principles for fair consultation.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field


class GeneratedQuestion(BaseModel):
    """A single generated consultation question."""

    question_text: str = Field(
        description="The main question text, written in plain English"
    )
    question_type: str = Field(
        description="One of: open_ended, agree_disagree, yes_no, multiple_choice"
    )
    scale_statement: str | None = Field(
        default=None,
        description="For agree_disagree type: the statement respondents rate on a 5-point scale",
    )
    multi_choice_options: list[str] | None = Field(
        default=None,
        description="For multiple_choice type: the options to select from",
    )
    rationale: str = Field(
        description="Brief explanation of why this question is valuable for the consultation"
    )


class QuestionSet(BaseModel):
    """Set of generated consultation questions."""

    questions: list[GeneratedQuestion]


SYSTEM_PROMPT = """You are an expert in UK government consultation design, following Cabinet Office
Consultation Principles and the Gunning Principles for fair public consultation.

## Question Types You Can Generate

1. **open_ended**: "What are your views on..." / "What measures should we consider..."
   - Used for gathering detailed qualitative feedback
   - Best for complex policy areas where diverse perspectives are needed

2. **agree_disagree**: Statement + 5-point Likert scale (Strongly Agree to Strongly Disagree)
   - Includes a `scale_statement` that respondents rate
   - Question asks "To what extent do you agree or disagree with the following statement:"
   - Always followed by "Please explain your reasoning"

3. **yes_no**: Binary question with "Please explain your answer"
   - Used for clear-cut policy decisions
   - Example: "Do you agree that [specific proposal] should be implemented?"

4. **multiple_choice**: "Which of the following... Please select all that apply"
   - Includes `multi_choice_options` array
   - Used when there are defined options to choose between

## UK Government Consultation Principles

- **Neutral framing**: Never use leading questions that suggest a preferred answer
- **Plain English**: Avoid jargon, acronyms, and complex policy language
- **One concept per question**: Don't combine multiple ideas
- **Formative stage**: Questions should gather input, not validate decisions already made
- **Balanced information**: Present different perspectives fairly

## Question Structure Patterns

1. Start with broader strategic questions
2. Progress to more specific/technical questions
3. Mix question types for variety
4. Each question should serve a clear purpose

## Common Phrasings

- "What are your views on..."
- "What measures should we consider to..."
- "To what extent do you agree or disagree that..."
- "Do you agree that [proposal] should..."
- "Which of the following [options] do you think..."
- "What would be the most appropriate way to..."
- "Are there other ways in which we can..."
- "What impact would [proposal] have on..."

Generate questions that would realistically appear in a UK government consultation."""


def _get_question_generation_llm(callbacks: list | None = None) -> AzureChatOpenAI:
    """Create LLM instance for question generation.

    Args:
        callbacks: LangChain callbacks for tracing.

    Returns:
        Configured AzureChatOpenAI instance.
    """
    return AzureChatOpenAI(
        azure_deployment="gpt-5-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        reasoning_effort="high",
        callbacks=callbacks or [],
    )


async def generate_questions(
    topic: str,
    n_questions: int,
    existing_questions: list[str] | None = None,
    feedback: str | None = None,
    callbacks: list | None = None,
) -> list[GeneratedQuestion]:
    """Generate consultation questions for a policy topic.

    Args:
        topic: The policy topic for the consultation.
        n_questions: Number of questions to generate.
        existing_questions: Previously approved questions (to avoid duplication).
        feedback: User feedback on previously rejected questions.
        callbacks: LangChain callbacks for tracing.

    Returns:
        List of GeneratedQuestion objects.
    """
    llm = _get_question_generation_llm(callbacks)
    structured_llm = llm.with_structured_output(QuestionSet)

    # Build context about existing questions
    existing_context = ""
    if existing_questions:
        existing_list = "\n".join(f"- {q}" for q in existing_questions)
        existing_context = f"""
## Already Approved Questions (do not duplicate these themes)
{existing_list}
"""

    # Build feedback context
    feedback_context = ""
    if feedback:
        feedback_context = f"""
## Feedback on Previous Suggestions
The user provided this feedback: {feedback}

Take this feedback into account when generating new questions.
"""

    human_prompt = f"""Generate {n_questions} consultation question(s) for the following UK government consultation topic.

## Consultation Topic
{topic}
{existing_context}{feedback_context}
## Requirements
1. Generate exactly {n_questions} question(s)
2. Use a variety of question types appropriate to the topic
3. Start with broader questions, then get more specific
4. Ensure questions are distinct and cover different aspects of the topic
5. Follow UK government consultation best practices
6. Use plain English accessible to the general public

For each question, provide:
- The question text
- The question type (open_ended, agree_disagree, yes_no, or multiple_choice)
- For agree_disagree: include the scale_statement
- For multiple_choice: include the options
- A brief rationale explaining why this question is valuable"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    result = await structured_llm.ainvoke(messages)
    return result.questions


async def regenerate_single_question(
    topic: str,
    rejected_question: str,
    feedback: str,
    existing_questions: list[str],
    callbacks: list | None = None,
) -> GeneratedQuestion:
    """Regenerate a single question based on user feedback.

    Args:
        topic: The policy topic for the consultation.
        rejected_question: The question that was rejected.
        feedback: User's feedback on why it was rejected.
        existing_questions: Other approved questions to avoid duplication.
        callbacks: LangChain callbacks for tracing.

    Returns:
        A new GeneratedQuestion.
    """
    llm = _get_question_generation_llm(callbacks)
    structured_llm = llm.with_structured_output(GeneratedQuestion)

    existing_list = (
        "\n".join(f"- {q}" for q in existing_questions)
        if existing_questions
        else "None yet"
    )

    human_prompt = f"""Generate a replacement consultation question for a UK government consultation.

## Consultation Topic
{topic}

## Rejected Question
{rejected_question}

## User Feedback
{feedback}

## Already Approved Questions (avoid duplicating)
{existing_list}

Generate ONE new question that:
1. Addresses the user's feedback
2. Covers a different angle or aspect than the rejected question
3. Doesn't duplicate existing approved questions
4. Follows UK government consultation best practices"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    result = await structured_llm.ainvoke(messages)
    return result
