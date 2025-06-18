"""
Enhanced DSPy Event Analysis Pipeline with OpenAI
=================================================

A comprehensive pipeline for analyzing events through interpretation,
impact evaluation, and actionable recommendations using DSPy with OpenAI.
"""

import dspy
from typing import Optional
import typer
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize rich console for better output formatting
console = Console()
app = typer.Typer(help="DSPy Event Analysis Pipeline with OpenAI")


def configure_openai(
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 1000,
):
    """
    Configure DSPy to use OpenAI language models.

    Args:
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        model: OpenAI model to use (gpt-4, gpt-3.5-turbo, etc.)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        console.print("[bold red]❌ OpenAI API key not found![/bold red]")
        console.print(
            "Set OPENAI_API_KEY environment variable or pass --api-key parameter"
        )
        raise typer.Exit(1)

    # Configure DSPy with OpenAI using the correct method
    try:
        # Method 1: Try the newer dspy.LM approach
        try:
            lm = dspy.LM(
                model=f"openai/{model}",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            dspy.configure(lm=lm)
            console.print(
                f"[green]✅ Configured DSPy with OpenAI model: {model} (using dspy.LM)[/green]"
            )
            return True
        except:
            # Method 2: Try the legacy approach with dspy-ai
            import openai

            openai.api_key = api_key

            # Configure using environment or direct setting
            os.environ["OPENAI_API_KEY"] = api_key

            # Use the legacy dspy configuration
            lm = dspy.OpenAI(
                model=model, temperature=temperature, max_tokens=max_tokens
            )
            dspy.settings.configure(lm=lm)
            console.print(
                f"[green]✅ Configured DSPy with OpenAI model: {model} (using legacy method)[/green]"
            )
            return True

    except Exception as e:
        console.print(f"[bold red]❌ Failed to configure OpenAI: {str(e)}[/bold red]")
        console.print("Trying alternative configuration method...")

        # Method 3: Manual configuration approach
        try:
            os.environ["OPENAI_API_KEY"] = api_key

            # Configure directly

            # Simple configuration - let DSPy handle the rest
            dspy.settings.configure()
            console.print(
                f"[yellow]⚠️  Using basic DSPy configuration. Model: {model}[/yellow]"
            )
            console.print(
                "[yellow]Note: You may need to install the latest version of dspy-ai[/yellow]"
            )
            return True

        except Exception as e2:
            console.print(
                f"[bold red]❌ All configuration methods failed: {str(e2)}[/bold red]"
            )
            console.print("\n[bold yellow]Troubleshooting suggestions:[/bold yellow]")
            console.print("1. Install/update DSPy: pip install --upgrade dspy-ai")
            console.print("2. Install OpenAI: pip install --upgrade openai")
            console.print("3. Check your API key is valid")
            console.print("4. Try: pip install 'dspy-ai[openai]'")
            raise typer.Exit(1)


class EventInterpreter(dspy.Signature):
    """Interpret and analyze the context and meaning of an event."""

    event: str = dspy.InputField(desc="The event description to interpret")
    interpretation: str = dspy.OutputField(
        desc="Detailed interpretation of the event's context and meaning"
    )


class ImpactEvaluator(dspy.Signature):
    """Evaluate the potential impact and consequences of an interpreted event."""

    interpretation: str = dspy.InputField(desc="The interpreted event context")
    impact: str = dspy.OutputField(
        desc="Assessment of potential impacts and consequences"
    )


class ActionRecommender(dspy.Signature):
    """Generate reasoned recommendations based on impact analysis."""

    impact: str = dspy.InputField(desc="The evaluated impact of the event")
    recommendation: str = dspy.OutputField(
        desc="Specific, actionable recommendations with reasoning"
    )


class EventAnalysisPipeline:
    """
    A DSPy pipeline for comprehensive event analysis.

    This pipeline processes events through three stages:
    1. Context interpretation
    2. Impact evaluation
    3. Action recommendation
    """

    def __init__(self, model_config: dict = None):
        # Verify DSPy is configured
        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            console.print(
                "[yellow]⚠️  DSPy not configured with OpenAI. Using default configuration.[/yellow]"
            )
            configure_openai()

        # Initialize DSPy modules
        self.interpreter = dspy.Predict(EventInterpreter)
        self.evaluator = dspy.Predict(ImpactEvaluator)
        self.recommender = dspy.ChainOfThought(ActionRecommender)

        # Store model configuration for reference
        self.model_config = model_config or {}

    def analyze(self, event_text: str) -> dict:
        """
        Run the complete analysis pipeline on an event.

        Args:
            event_text: Description of the event to analyze

        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Step 1: Interpret the event context
            console.print("\n[bold blue]🔍 Step 1: Event Interpretation[/bold blue]")
            console.print(Panel(event_text, title="Input Event", border_style="dim"))

            interpretation_result = self.interpreter(event=event_text)
            console.print(
                Panel(
                    interpretation_result.interpretation,
                    title="Interpretation",
                    border_style="green",
                )
            )

            # Step 2: Evaluate impact
            console.print("\n[bold yellow]⚡ Step 2: Impact Evaluation[/bold yellow]")
            impact_result = self.evaluator(
                interpretation=interpretation_result.interpretation
            )
            console.print(
                Panel(
                    impact_result.impact,
                    title="Impact Assessment",
                    border_style="yellow",
                )
            )

            # Step 3: Generate recommendations
            console.print(
                "\n[bold magenta]💡 Step 3: Action Recommendations[/bold magenta]"
            )
            recommendation_result = self.recommender(impact=impact_result.impact)
            console.print(
                Panel(
                    recommendation_result.recommendation,
                    title="Recommended Actions",
                    border_style="magenta",
                )
            )

            return {
                "event": event_text,
                "interpretation": interpretation_result.interpretation,
                "impact": impact_result.impact,
                "recommendation": recommendation_result.recommendation,
            }

        except Exception as e:
            console.print(f"[bold red]❌ Error during analysis: {str(e)}[/bold red]")
            raise typer.Exit(1)


@app.command()
def analyze(
    event: str = typer.Argument(..., help="The event description to analyze"),
    output_format: str = typer.Option(
        "console", "--format", "-f", help="Output format: console, json, or text"
    ),
    save_to: Optional[str] = typer.Option(
        None, "--save", "-s", help="Save results to file"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    ),
    model: str = typer.Option(
        "gpt-4",
        "--model",
        "-m",
        help="OpenAI model to use (gpt-4, gpt-3.5-turbo, gpt-4-turbo)",
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Temperature for OpenAI model (0.0 to 1.0)"
    ),
):
    """
    Analyze an event through interpretation, impact evaluation, and recommendations.

    Example usage:
    python script.py analyze "Our main competitor just launched a new product"
    python script.py analyze "Supply chain disruption" --model gpt-4-turbo --temperature 0.3
    """
    # Configure OpenAI
    configure_openai(api_key=api_key, model=model, temperature=temperature)

    # Initialize pipeline
    pipeline = EventAnalysisPipeline({"model": model, "temperature": temperature})

    # Run analysis
    console.print(
        f"[bold green]🚀 Starting Event Analysis Pipeline with {model}...[/bold green]"
    )
    results = pipeline.analyze(event)

    # Handle output formatting
    if output_format == "json":
        import json

        output = json.dumps(results, indent=2, ensure_ascii=False)
        console.print("\n[bold]JSON Output:[/bold]")
        console.print(output)
    elif output_format == "text":
        output = f"""
EVENT ANALYSIS REPORT
=====================

Event: {results['event']}

Interpretation:
{results['interpretation']}

Impact Assessment:
{results['impact']}

Recommendations:
{results['recommendation']}
"""
        console.print(output)

    # Save to file if requested
    if save_to:
        try:
            with open(save_to, "w", encoding="utf-8") as f:
                if output_format == "json":
                    import json

                    json.dump(results, f, indent=2, ensure_ascii=False)
                else:
                    f.write(output if output_format == "text" else str(results))
            console.print(f"[green]✅ Results saved to {save_to}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save file: {str(e)}[/red]")


@app.command()
def interactive(
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    ),
    model: str = typer.Option("gpt-4", "--model", "-m", help="OpenAI model to use"),
):
    """
    Run the pipeline in interactive mode for multiple event analyses.
    """
    # Configure OpenAI
    configure_openai(api_key=api_key, model=model)

    pipeline = EventAnalysisPipeline({"model": model})
    console.print(
        f"[bold green]🎯 Interactive Event Analysis Mode with {model}[/bold green]"
    )
    console.print("Enter events to analyze (type 'quit' to exit):\n")

    while True:
        try:
            event_input = typer.prompt("Event to analyze")

            if event_input.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break

            pipeline.analyze(event_input)
            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")


@app.command()
def batch(
    input_file: str = typer.Argument(
        ..., help="Path to file containing events (one per line)"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    ),
    model: str = typer.Option("gpt-4", "--model", "-m", help="OpenAI model to use"),
):
    """
    Process multiple events from a file in batch mode.
    """
    # Configure OpenAI
    configure_openai(api_key=api_key, model=model)

    pipeline = EventAnalysisPipeline({"model": model})

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            events = [line.strip() for line in f if line.strip()]

        console.print(
            f"[green]📄 Processing {len(events)} events from {input_file}[/green]"
        )

        all_results = []
        for i, event in enumerate(events, 1):
            console.print(f"\n[bold]Processing event {i}/{len(events)}[/bold]")
            result = pipeline.analyze(event)
            all_results.append(result)

        if output_file:
            import json

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✅ Batch results saved to {output_file}[/green]")

    except FileNotFoundError:
        console.print(f"[red]❌ Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Batch processing error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def configure(
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="OpenAI API key to test"
    ),
    model: str = typer.Option("gpt-4", "--model", "-m", help="OpenAI model to test"),
):
    """
    Test OpenAI configuration and connection.
    """
    console.print("[bold blue]🔧 Testing OpenAI Configuration...[/bold blue]")

    # Configure OpenAI
    configure_openai(api_key=api_key, model=model)

    # Test with a simple query
    try:
        test_signature = dspy.Signature("question -> answer")
        test_module = dspy.Predict(test_signature)

        console.print("[yellow]Testing connection with simple query...[/yellow]")
        result = test_module(question="What is 2+2?")

        console.print(f"[green]✅ Connection successful![/green]")
        console.print(f"[dim]Test response: {result.answer}[/dim]")
        console.print(f"[green]✅ Ready to use {model} for event analysis![/green]")

    except Exception as e:
        console.print(f"[red]❌ Connection test failed: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
