from typing import Any


def read_and_render(prompt_template: str, kwargs: Any = None) -> str:
    """Render a prompt template with provided variables.

    Args:
        prompt_template (str): The prompt template string
        kwargs (Any, optional): Dictionary of variables to render in the template

    Returns:
        str: The rendered prompt
    """
    if kwargs:
        return prompt_template.format(**kwargs)
    return prompt_template
