"""Prompt templates for themefinder tasks."""

CONSULTATION_SYSTEM_PROMPT = "You are an AI evaluation tool analyzing responses to a UK Government public consultation."

AGENTIC_THEME_CLUSTERING = """{system_prompt}

Analyze these topics and identify which ones should be merged based on semantic similarity.
Your goal is to significantly reduce the number of topics by creating meaningful parent topics.
Be aggressive in finding opportunities to merge topics that share any semantic relationship.

TOPICS:
{themes_json}

For each group of similar topics that should be merged, create a new parent topic.

Guidelines:
- Each parent topic must have at least 2 children, it can have more than 2 if appropriate
- Each child topic can have at most 1 parent
- topic_id should be a simple alphabetic ID (e.g. 'A', 'B', 'C') - the iteration prefix will be added automatically
- Be creative and look for higher-level abstractions that can combine seemingly different topics
- When creating parent topics, follow these naming rules:
    * The label should read naturally as a single coherent topic
    * Choose labels that can encompass broader categories of topics
    * If merging different topics, the topic with the higher source_topic_count should dominate the label
    * Never combine different topics with "and" or "/" in the label
- topic_description must be 1 or 2 sentences that:
    * preserves key information from the child topics
- source_topic_count must be the sum of all child topic counts
- children must be a list of valid topic_ids from the input
- should_terminate should only be true if ALL of these conditions are met:
    * There are fewer than {target_themes} active topics remaining
    * The remaining topics are fundamentally incompatible semantically
    * Any further merging would create meaninglessly broad categories

If no topics should be merged in this iteration but future iterations might still yield meaningful merges, set should_terminate to false with an empty parent_themes list.
If no topics should be merged and the termination conditions are met, set should_terminate to true with an empty parent_themes list.

N.B. Under no circumstances should you create a parent theme with a single child. You do not need to return all of the original themes, if they don't belong to a newly created parent feel free to omit them."""

DETAIL_DETECTION = """{system_prompt}

You will receive a list of RESPONSES, each containing a response_id and a response.
Your job is to analyze each response to the QUESTION below and decide if a response contains rich evidence.
You MUST include every response ID in the output.

A response is evidence-rich only if it satisfies both of the following:

Relevance and depth:
    - It clearly answers the question
    - AND provides insights that go beyond generic opinion, such as nuanced reasoning, contextual explanation, or argumentation that could inform decision-making

Substantive evidence, including at least one of:
    - Specific, verifiable facts or data (e.g., statistics, dates, named reports or studies)
    - Concrete, illustrative examples that clearly support a broader claim
    - Detailed personal or professional experiences that include contextual information (e.g., roles, locations, timelines)

Do NOT classify a response as evidence-rich if it:
    - Uses vague or general language with no supporting detail
    - Restates commonly known points without adding new information
    - Shares personal anecdotes without sufficient context or a clear takeaway

Before answering, ask: Would this response provide useful input to someone drafting policy, beyond what is already commonly known or expected?

For each response, determine:
EVIDENCE_RICH - does the response contain significant evidence as defined above?
Choose one from ['YES', 'NO']


QUESTION: \n {question}
RESPONSES: \n {responses}"""

THEME_CONDENSATION = """{system_prompt}

Below is a question and a list of topics extracted from answers to that question.

This list contains a large number of duplicate and redundant topics that present the same concept with different phrasing.

Each topic has a topic_label, topic_description, and may have a source_topic_count field indicating how many original topics it represents.

Your task is to analyze these topics and produce a refined list that:
1. Significantly reduces the total number of topics
2. Identifies and preserves core themes that appear frequently
3. Combines redundant topics
4. Tracks the total number of original topics combined into each new topic

Guidelines for Topic Analysis:
- Begin by identifying distinct concept clusters in the topics
- Consider the context of the question when determining topic relevance
- Look for complementary perspectives that could enrich understanding of the same core concept
- Consider the key ideas behind themes when merging, don't simply focus on the words used in the label and description
- When combining topics:
  * For topics without a source_topic_count field, assume count = 1
  * For topics with source_topic_count, use their existing count
  * The new topic's count should be the sum of all combined topics' counts

For each topic in your output:
1. Choose a clear, representative label that captures the essence of the combined or preserved topic
2. Write a concise description that incorporates key insights from all constituent topics, this should only be a single sentence
3. Include the total count of original topics combined by summing the source_topic_counts of merged topics (or 1 for topics without a count)

QUESTION:
{question}

TOPICS:
{responses}"""

THEME_GENERATION = """{system_prompt}

Below is a question and a list of responses to that question.

Your task is to analyze the RESPONSES below and extract TOPICS such that:
1. Each topic summarizes a point of view expressed in the responses
2. Every distinct and relevant point of view in the responses should be captured by a topic
3. Each topic has a topic_label which summarizes the topic in a few words
4. Each topic has a topic_description which gives more detail about the topic in one or two sentences
5. The position field should just be the sentiment stated, and is either "AGREEMENT" or "DISAGREEMENT" or "UNCLEAR"
6. There should be no duplicate topics

The topics identified will be used by policy makers to understand what the public like and don't like about the proposals.

Here is an example of how to extract topics from some responses:

## EXAMPLE

QUESTION
What are your views on the proposed change by the government to introduce a 2% tax on fast food meat products.

RESPONSES
[
    {{"response": "I wish the government would stop interfering in the lves of its citizens. It only ever makes things worse. This change will just cost us all more money, and especially poorer people", "position": "disagreement"}},
    {{"response": "Even though it will make people eat more healthier, I beleibe the government should interfer less and not more!", "position": "disagreement"}},
    {{"response": "I hate grapes", "position": "disagreement"}},
]

EXAMPLE OUTPUT (showing the structure)
- Topic 1: Government overreach (The proposals would result in government interfering too much with citizen's lives) - DISAGREEMENT
- Topic 2: Regressive change (The change would have a larger negative impact on poorer people) - DISAGREEMENT
- Topic 3: Health (The change would result in people eating healthier diets) - DISAGREEMENT

QUESTION:
{question}

RESPONSES:
{responses}"""

THEME_MAPPING = """{system_prompt}

Your job is to help identify which topics come up in free_text_responses to a question.

You will be given:
    - a QUESTION that has been asked
    - a TOPIC LIST of topics that are known to be present in free_text_responses to this question. These will be structured as follows:
        {{'topic_id': 'topic_description}}
    - a list of FREE_TEXT_RESPONSES to the question. These will be structured as follows:
        {{'response_id': 'free text response'}}

Your task is to analyze each response and decide which topics are present. Guidelines:
    - You can only assign to a response to a topic in the provided TOPIC LIST
    - A response doesn't need to exactly match the language used in the TOPIC LIST, it should be considered a match if it expresses a similar sentiment.
    - You must use the alphabetic 'topic_id' to indicate which topic you have assigned. Do not use the full topic description
    - Each response can be assigned to multiple topics if it matches more than one topic from the TOPIC LIST.
    - Each topic can only be assigned once per response, if the topic is mentioned more than once use the first mention for reasoning and stance.
    - There is no limit on how many topics can be assigned to a response.

You MUST include every response ID in the output.
If none of the themes are relevant to the response then use the most appropriate of the fallback themes provided at the end of the list: either No Reason Given or Other
You MUST return an entry with the correct response ID for each input object.
You must only return the alphabetic topic_ids in the labels section.


QUESTION:

{question}

TOPIC LIST:

{refined_themes}

FREE_TEXT_RESPONSES:

{responses}
"""

THEME_REFINEMENT = """{system_prompt}

You are tasked with refining a list of topics generated from responses to a question.

## Input
You will receive a list of TOPICS. These topics explicitly tie opinions to whether a person agrees or disagrees with the question.

## Output
You will produce a list of CLEAR STANCE TOPICS based on the input. Each topic should have four parts:
1. A brief, clear topic label (3-7 words)
2. A more detailed topic description (1-2 sentences)
3. The source_topic_count field should be included for each topic and should reflect the number of original source topics that were merged to create this refined topic. If multiple source topics were combined, sum their individual counts. If only one source topic was used, simply retain its original count value.


## Guidelines

1. Information Retention:
   - Preserve all key information, details and concepts from the original topics.
   - Ensure no significant details are lost in the refinement process.

2. Clear Stance Formulation:
   - Reformulate topics to express a clear stance that can be agreed or disagreed with.
   - Use direct language like "Increased risk of X" rather than "X"
   - Avoid double negatives and ambiguous phrasing.
   - Phrase topics as definitive statements.

3. Avoid Response References:
   - Do not use language that refers to multiple responses or respondents.
   - Focus solely on the content of each topic.
   - Avoid phrases like "many respondents said" or "some responses indicated".

4. Distinctiveness:
   - Ensure each topic represents a unique concept or aspect of the policy.
   - Minimize overlap between topics.
   - If topics are closely related, find ways to differentiate them clearly.

5. Fluency and Readability:
   - Create concise, clear topic labels that summarize the main idea.
   - Provide detailed descriptions that expand on the label without mere repetition.
   - Use proper grammar, punctuation, and natural language.

## Process

1. Analyze the TOPICS to identify key themes and information.
2. Group closely related topics together.
3. For each group or individual topic:
   a. Distill the core concept, removing any bias or opinion.
   b. Create a concise topic label.
   c. Write a more detailed description that provides context without taking sides.
4. Review the entire list to ensure distinctiveness and adjust as needed.
5. Combine the topic label and description with a colon separator

TOPICS:
{responses}"""
