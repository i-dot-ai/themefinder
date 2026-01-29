"""Interactive CLI for synthetic consultation dataset generation."""

from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from synthetic.config import (
    GenerationConfig,
    NoiseLevel,
    QuestionConfig,
    QuestionType,
)
from synthetic.demographics import get_uk_demographic_presets
from synthetic.llm_generators.question_generator import (
    GeneratedQuestion,
    generate_questions,
    regenerate_single_question,
)

console = Console()

# ASCII art banner
BANNER = """
[bold cyan]
  _____ _                        _____ _           _
 |_   _| |__   ___ _ __ ___   ___|  ___(_)_ __   __| | ___ _ __
   | | | '_ \\ / _ \\ '_ ` _ \\ / _ \\ |_  | | '_ \\ / _` |/ _ \\ '__|
   | | | | | |  __/ | | | | |  __/  _| | | | | | (_| |  __/ |
   |_| |_| |_|\\___|_| |_| |_|\\___|_|   |_|_| |_|\\__,_|\\___|_|
[/bold cyan]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
[bold white]          Synthetic Consultation Dataset Generator[/bold white]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
"""

STEP_ICONS = {
    1: "📋",
    2: "🔢",
    3: "🤖",
    4: "📊",
    5: "👥",
    6: "⚙️",
}


class SmoothedTimeRemainingColumn(ProgressColumn):
    """Renders estimated time remaining using exponential moving average.

    More accurate than the default TimeRemainingColumn because it weights
    recent speed more heavily, accounting for:
    - Cold start delays (first few API calls are slower)
    - Variable task completion times
    - Bursty parallel task completion
    """

    def __init__(self, smoothing: float = 0.3) -> None:
        """Initialise with smoothing factor.

        Args:
            smoothing: EMA smoothing factor (0-1). Higher = more responsive
                      to recent speed changes. Default 0.3.
        """
        super().__init__()
        self.smoothing = smoothing
        self._ema_speed: dict[int, float] = {}  # task_id -> smoothed speed
        self._last_completed: dict[int, float] = {}  # task_id -> last completed count
        self._last_time: dict[int, float] = {}  # task_id -> last update time

    def render(self, task: Task) -> Text:
        """Render the smoothed time remaining."""
        import time

        if task.finished:
            return Text("0:00:00", style="green")

        if task.total is None or task.completed == 0:
            return Text("-:--:--", style="dim")

        task_id = id(task)
        current_time = time.monotonic()
        elapsed = task.elapsed or 0.001

        # Calculate instantaneous speed
        if task_id in self._last_completed and task_id in self._last_time:
            time_delta = current_time - self._last_time[task_id]
            completed_delta = task.completed - self._last_completed[task_id]

            if time_delta > 0.1 and completed_delta > 0:  # Minimum threshold
                instant_speed = completed_delta / time_delta

                # Update EMA
                if task_id in self._ema_speed:
                    self._ema_speed[task_id] = (
                        self.smoothing * instant_speed
                        + (1 - self.smoothing) * self._ema_speed[task_id]
                    )
                else:
                    self._ema_speed[task_id] = instant_speed

                self._last_completed[task_id] = task.completed
                self._last_time[task_id] = current_time
        else:
            # First update - use overall average
            self._last_completed[task_id] = task.completed
            self._last_time[task_id] = current_time
            self._ema_speed[task_id] = task.completed / elapsed

        # Calculate remaining time using smoothed speed
        speed = self._ema_speed.get(task_id, task.completed / elapsed)
        if speed <= 0:
            return Text("-:--:--", style="dim")

        remaining = (task.total - task.completed) / speed

        # Format as H:MM:SS or M:SS
        hours, remainder = divmod(int(remaining), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"{minutes}:{seconds:02d}"

        return Text(time_str, style="cyan")


def _print_step_header(step: int, title: str) -> None:
    """Print a formatted step header."""
    icon = STEP_ICONS.get(step, "•")
    console.print()
    console.print(
        Rule(
            f"[bold magenta]Step {step}[/bold magenta]  {icon}  [bold]{title}[/bold]",
            style="magenta",
        )
    )


async def run_interactive_cli() -> GenerationConfig:
    """Run interactive CLI to collect generation parameters.

    Returns:
        GenerationConfig with all user selections.

    Raises:
        SystemExit: If user cancels generation.
    """
    console.clear()
    console.print(BANNER)

    # Step 1: Policy topic
    _print_step_header(1, "Policy Topic")
    console.print("[dim]What is the consultation about?[/dim]\n")
    topic = Prompt.ask(
        "[bold yellow]Topic[/bold yellow]",
        default="",
    )
    if not topic:
        topic = "general policy consultation"

    # Step 2: Number of questions
    _print_step_header(2, "Number of Questions")
    console.print(
        "[dim]How many consultation questions should be generated? (1-25)[/dim]\n"
    )
    n_questions = IntPrompt.ask(
        "[bold yellow]Questions[/bold yellow]",
        default=3,
    )
    n_questions = max(1, min(25, n_questions))

    # Step 3: AI-generated questions with approval workflow
    _print_step_header(3, "Question Generation")
    console.print(
        Panel(
            "[bold]AI-Powered Question Design[/bold]\n\n"
            "Questions will be generated following UK Government consultation best practices.\n"
            "For each question, you can:\n"
            "  [green]approve[/green]  ─  Accept the question as-is\n"
            "  [yellow]edit[/yellow]     ─  Modify the question text\n"
            "  [red]reject[/red]   ─  Provide feedback and regenerate",
            box=ROUNDED,
            border_style="blue",
            padding=(1, 2),
        )
    )

    questions = await _question_approval_workflow(topic, n_questions)

    # Step 4: Number of responses
    _print_step_header(4, "Number of Responses")
    console.print("[dim]How many synthetic responses per question?[/dim]\n")

    size_table = Table(box=ROUNDED, border_style="dim", pad_edge=False)
    size_table.add_column("Preset", style="bold cyan", justify="center")
    size_table.add_column("Responses", style="green", justify="right")
    size_table.add_column("Use Case", style="white")
    size_table.add_row("100", "100", "Quick testing & debugging")
    size_table.add_row("500", "500", "Standard evaluation runs")
    size_table.add_row("1000", "1,000", "Comprehensive testing")
    size_table.add_row("5000", "5,000", "Full-scale stress testing")
    console.print(size_table)
    console.print("[dim]Or enter any custom number.[/dim]\n")

    n_responses = IntPrompt.ask(
        "[bold yellow]Responses per question[/bold yellow]",
        default=500,
    )
    n_responses = max(10, n_responses)  # Minimum 10 responses

    # Step 5: Demographic presets
    demographics = _configure_demographics()

    # Step 6: Advanced options (noise level)
    _print_step_header(6, "Advanced Options")
    noise_level = NoiseLevel.MEDIUM

    if Confirm.ask(
        "[bold yellow]Configure noise settings?[/bold yellow]", default=False
    ):
        console.print()
        noise_table = Table(box=ROUNDED, border_style="dim")
        noise_table.add_column("Level", style="bold")
        noise_table.add_column("Typos", justify="center")
        noise_table.add_column("Grammar", justify="center")
        noise_table.add_column("Emotional", justify="center")
        noise_table.add_row("low", "2%", "2%", "5%")
        noise_table.add_row("medium", "5%", "8%", "15%")
        noise_table.add_row("high", "15%", "20%", "30%")
        console.print(noise_table)
        console.print()

        noise = Prompt.ask(
            "[bold yellow]Noise level[/bold yellow]",
            choices=["low", "medium", "high"],
            default="medium",
        )
        noise_level = NoiseLevel(noise)

    # Generate dataset name
    safe_topic = topic.lower().replace(" ", "_").replace("-", "_")
    safe_topic = "".join(c for c in safe_topic if c.isalnum() or c == "_")
    dataset_name = f"{safe_topic}_{n_responses}"

    # Confirmation summary
    _show_confirmation_summary(
        dataset_name, topic, questions, n_responses, demographics, noise_level
    )

    if not Confirm.ask("\n[bold]Proceed with generation?[/bold]", default=True):
        raise SystemExit("Generation cancelled")

    return GenerationConfig(
        dataset_name=dataset_name,
        topic=topic,
        n_responses=n_responses,
        questions=questions,
        demographic_fields=demographics,
        noise_level=noise_level,
    )


async def _question_approval_workflow(
    topic: str, n_questions: int
) -> list[QuestionConfig]:
    """Interactive workflow for generating and approving questions.

    Args:
        topic: The consultation topic.
        n_questions: Target number of questions to approve.

    Returns:
        List of approved QuestionConfig objects.
    """
    approved_questions: list[QuestionConfig] = []
    approved_texts: list[str] = []

    while len(approved_questions) < n_questions:
        remaining = n_questions - len(approved_questions)

        with console.status(
            f"[bold magenta]🤖 Generating {remaining} question{'s' if remaining > 1 else ''}...[/bold magenta]",
            spinner="dots",
        ):
            generated = await generate_questions(
                topic=topic,
                n_questions=remaining,
                existing_questions=approved_texts if approved_texts else None,
            )

        for gen_q in generated:
            if len(approved_questions) >= n_questions:
                break

            question_config = await _review_single_question(
                gen_q,
                topic,
                len(approved_questions) + 1,
                n_questions,
                approved_texts,
            )

            if question_config:
                approved_questions.append(question_config)
                approved_texts.append(question_config.text)

    # Show completion message
    console.print()
    console.print(
        Panel(
            f"[bold green]✓ All {n_questions} questions approved![/bold green]",
            box=ROUNDED,
            border_style="green",
            padding=(0, 2),
        )
    )

    return approved_questions


async def _review_single_question(
    gen_q: GeneratedQuestion,
    topic: str,
    question_num: int,
    total_questions: int,
    approved_texts: list[str],
) -> QuestionConfig | None:
    """Review a single generated question with the user.

    Args:
        gen_q: The generated question to review.
        topic: The consultation topic.
        question_num: Current question number (1-indexed).
        total_questions: Total questions needed.
        approved_texts: Already approved question texts.

    Returns:
        QuestionConfig if approved, None if we need to regenerate.
    """
    type_colors = {
        "open_ended": "green",
        "agree_disagree": "yellow",
        "yes_no": "blue",
        "multiple_choice": "magenta",
    }

    while True:
        # Display the question with nice formatting
        console.print()
        console.print(
            Rule(
                f"[bold]Question {question_num} of {total_questions}[/bold]",
                style="cyan",
            )
        )

        q_type = gen_q.question_type
        color = type_colors.get(q_type, "white")
        type_label = q_type.replace("_", " ").upper()

        console.print(
            Panel(
                _format_question_display(gen_q),
                title=f"[bold {color}]◆ {type_label}[/bold {color}]",
                subtitle=f"[dim italic]{gen_q.rationale}[/dim italic]",
                box=DOUBLE,
                border_style=color,
                padding=(1, 2),
            )
        )

        # Action prompt with visual options
        console.print(
            "  [green]approve[/green] │ [yellow]edit[/yellow] │ [red]reject[/red]"
        )
        action = Prompt.ask(
            "[bold yellow]Action[/bold yellow]",
            choices=["approve", "reject", "edit"],
            default="approve",
        )

        if action == "approve":
            console.print("[green]✓ Question approved[/green]")
            return _question_to_config(gen_q, question_num)

        elif action == "edit":
            console.print()
            edited_text = Prompt.ask(
                "[bold yellow]Question text[/bold yellow]",
                default=gen_q.question_text,
            )
            gen_q.question_text = edited_text

            # For agree/disagree, allow editing the statement
            if gen_q.question_type == "agree_disagree" and gen_q.scale_statement:
                edited_statement = Prompt.ask(
                    "[bold yellow]Scale statement[/bold yellow]",
                    default=gen_q.scale_statement,
                )
                gen_q.scale_statement = edited_statement

            console.print("[green]✓ Question updated and approved[/green]")
            return _question_to_config(gen_q, question_num)

        else:  # reject
            console.print()
            feedback = Prompt.ask(
                "[bold yellow]Feedback[/bold yellow] [dim](what should be different?)[/dim]"
            )

            with console.status(
                "[bold magenta]🤖 Regenerating question...[/bold magenta]",
                spinner="dots",
            ):
                gen_q = await regenerate_single_question(
                    topic=topic,
                    rejected_question=gen_q.question_text,
                    feedback=feedback,
                    existing_questions=approved_texts,
                )
            # Loop back to display the new question


def _format_question_display(gen_q: GeneratedQuestion) -> str:
    """Format a generated question for display.

    Args:
        gen_q: The generated question.

    Returns:
        Formatted string for display.
    """
    lines = [f"[bold]{gen_q.question_text}[/bold]"]

    if gen_q.question_type == "agree_disagree" and gen_q.scale_statement:
        lines.append("")
        lines.append(f'[italic]Statement: "{gen_q.scale_statement}"[/italic]')
        lines.append(
            "[dim]Scale: Strongly Agree | Agree | Neither | Disagree | Strongly Disagree[/dim]"
        )

    elif gen_q.question_type == "yes_no":
        lines.append("[dim]Options: Yes | No[/dim]")
        lines.append("[dim]Follow-up: Please explain your answer[/dim]")

    elif gen_q.question_type == "multiple_choice" and gen_q.multi_choice_options:
        lines.append("")
        lines.append("[dim]Options (select all that apply):[/dim]")
        for opt in gen_q.multi_choice_options:
            lines.append(f"  • {opt}")

    return "\n".join(lines)


def _question_to_config(gen_q: GeneratedQuestion, number: int) -> QuestionConfig:
    """Convert a GeneratedQuestion to QuestionConfig.

    Args:
        gen_q: The generated question.
        number: Question number.

    Returns:
        QuestionConfig instance.
    """
    return QuestionConfig(
        number=number,
        text=gen_q.question_text,
        question_type=QuestionType(gen_q.question_type),
        multi_choice_options=gen_q.multi_choice_options,
        scale_statement=gen_q.scale_statement,
    )


def _configure_demographics():
    """Interactive demographic field configuration.

    Returns:
        List of DemographicField with user-selected enabled states.
    """
    _print_step_header(5, "Demographics")
    console.print(
        "[dim]Select demographic fields to include in respondent profiles.[/dim]\n"
    )

    presets = get_uk_demographic_presets()

    def _print_table():
        table = Table(box=ROUNDED, border_style="dim")
        table.add_column("#", style="bold white", width=3, justify="center")
        table.add_column("Field", style="cyan")
        table.add_column("Sample Values", style="dim")
        table.add_column("Status", justify="center")

        for i, field in enumerate(presets, 1):
            values_preview = ", ".join(field.values[:3])
            if len(field.values) > 3:
                values_preview += "…"
            status = (
                "[bold green]● ON[/bold green]" if field.enabled else "[dim]○ OFF[/dim]"
            )
            table.add_row(
                str(i),
                field.name.replace("_", " ").title(),
                values_preview,
                status,
            )
        console.print(table)

    _print_table()

    console.print(
        Panel(
            "[bold yellow]Toggle:[/bold yellow] Enter numbers (e.g., [cyan]4 5 6[/cyan])\n"
            "[bold yellow]Enable all:[/bold yellow] Type [cyan]all[/cyan]\n"
            "[bold yellow]Disable all:[/bold yellow] Type [cyan]none[/cyan]\n"
            "[bold yellow]Keep current:[/bold yellow] Press [cyan]Enter[/cyan]",
            box=ROUNDED,
            border_style="dim",
            padding=(0, 1),
        )
    )

    selection = Prompt.ask("[bold yellow]Selection[/bold yellow]", default="")

    if selection.strip():
        selection = selection.strip().lower()

        if selection == "all":
            for field in presets:
                field.enabled = True
            console.print("[green]All fields enabled.[/green]")
        elif selection == "none":
            for field in presets:
                field.enabled = False
            console.print("[yellow]All fields disabled.[/yellow]")
        else:
            # Parse numbers to toggle
            try:
                indices = [int(x) for x in selection.split()]
                for idx in indices:
                    if 1 <= idx <= len(presets):
                        presets[idx - 1].enabled = not presets[idx - 1].enabled
                console.print("\n[bold]Updated configuration:[/bold]")
                _print_table()
            except ValueError:
                console.print("[red]Invalid input, keeping defaults.[/red]")

    return presets


def _show_confirmation_summary(
    dataset_name: str,
    topic: str,
    questions: list[QuestionConfig],
    n_responses: int,
    demographics,
    noise_level: NoiseLevel,
) -> None:
    """Display configuration summary for confirmation.

    Args:
        dataset_name: Generated dataset name.
        topic: Consultation topic.
        questions: List of approved question configurations.
        n_responses: Number of responses per question.
        demographics: List of demographic fields.
        noise_level: Noise injection level.
    """
    console.print()
    console.print(Rule("[bold green]Configuration Summary[/bold green]", style="green"))
    console.print()

    # Main config table
    config_table = Table(box=None, show_header=False, padding=(0, 2))
    config_table.add_column("Key", style="bold white")
    config_table.add_column("Value", style="cyan")

    config_table.add_row("📁 Dataset", dataset_name)
    config_table.add_row("📋 Topic", topic)
    config_table.add_row("📊 Responses/Question", f"{n_responses:,}")
    config_table.add_row("📝 Total Responses", f"{n_responses * len(questions):,}")
    config_table.add_row("🎭 Noise Level", noise_level.value.title())

    enabled_demographics = [
        f.name.replace("_", " ").title() for f in demographics if f.enabled
    ]
    config_table.add_row("👥 Demographics", ", ".join(enabled_demographics))

    console.print(config_table)
    console.print()

    # Questions panel
    questions_content = []
    for q in questions:
        truncated = q.text[:55] + "..." if len(q.text) > 55 else q.text
        type_label = q.question_type.value.replace("_", " ").upper()
        type_colors = {
            "OPEN_ENDED": "green",
            "AGREE_DISAGREE": "yellow",
            "YES_NO": "blue",
            "MULTIPLE_CHOICE": "magenta",
        }
        color = type_colors.get(q.question_type.value.upper(), "white")
        questions_content.append(
            f"  [dim]{q.number}.[/dim] [{color}]{type_label:15}[/{color}] {truncated}"
        )

    console.print(
        Panel(
            "\n".join(questions_content),
            title="[bold]Questions[/bold]",
            box=ROUNDED,
            border_style="dim",
            padding=(1, 1),
        )
    )
    console.print()


def create_progress_bar() -> Progress:
    """Create a progress bar for generation tracking.

    Returns:
        Configured Rich Progress instance with smoothed ETA, elapsed time, and M/N display.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA[/dim]"),
        SmoothedTimeRemainingColumn(smoothing=0.3),
        console=console,
        refresh_per_second=4,
    )


def print_success(output_path: str, n_themes: int, n_responses: int) -> None:
    """Print success message after generation.

    Args:
        output_path: Path to generated dataset.
        n_themes: Number of themes generated.
        n_responses: Total responses generated.
    """
    success_art = """
[bold green]
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗  ██████╗ ███╗   ██╗███████╗██╗                  ║
    ║   ██╔══██╗██╔═══██╗████╗  ██║██╔════╝██║                  ║
    ║   ██║  ██║██║   ██║██╔██╗ ██║█████╗  ██║                  ║
    ║   ██║  ██║██║   ██║██║╚██╗██║██╔══╝  ╚═╝                  ║
    ║   ██████╔╝╚██████╔╝██║ ╚████║███████╗██╗                  ║
    ║   ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
[/bold green]"""

    console.print(success_art)

    results_table = Table(
        box=ROUNDED, border_style="green", title="[bold]Generation Results[/bold]"
    )
    results_table.add_column("Metric", style="bold")
    results_table.add_column("Value", style="cyan", justify="right")

    results_table.add_row("📁 Output Path", output_path)
    results_table.add_row("🏷️  Themes Generated", str(n_themes))
    results_table.add_row("💬 Responses Generated", f"{n_responses:,}")

    console.print(results_table)
    console.print()
    console.print(
        "[dim]Run ThemeFinder evaluation with this dataset to test theme extraction accuracy.[/dim]"
    )
    console.print()


def print_error(error: Exception) -> None:
    """Print error message.

    Args:
        error: The exception that occurred.
    """
    error_panel = Panel(
        f"[bold red]Generation Failed[/bold red]\n\n{error}",
        title="[red]❌ Error[/red]",
        box=HEAVY,
        border_style="red",
        padding=(1, 2),
    )
    console.print()
    console.print(error_panel)
    console.print()
