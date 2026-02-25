from pathlib import Path
from typing import Any


def load_prompt_from_file(file_path: str | Path) -> str:
    """Load a prompt template from a file in the 'prompts' directory.

    Args:
        file_path (str | Path): Path or string pointing to the file name within the prompts directory

    Returns:
        str: The contents of the prompt file
    """
    parent_dir = Path(__file__).parent
    with Path.open(parent_dir / "prompts" / f"{file_path}") as file:
        return file.read()


def read_and_render(prompt_path: str | Path, kwargs: Any = None) -> str:
    """Load a prompt template from a file and optionally render it with provided variables.

    Args:
        prompt_path (str | Path): Path or string pointing to the prompt template file
        kwargs (Any, optional): Dictionary of variables to render in the template

    Returns:
        str: The rendered prompt
    """
    prompt_content = load_prompt_from_file(prompt_path)
    if kwargs:
        return prompt_content.format(**kwargs)
    return prompt_content
