import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .clients.supermemory import SuperMemoryClient
from .config import settings
from .runner import load_test_cases, run_benchmark

app = typer.Typer(name="vrb", help="Visual RAG Benchmark CLI")
console = Console()

_FREAK_DATA = Path("data/freak")


@app.command()
def run(
    test_cases_dir: Annotated[
        Path,
        typer.Option("--cases", help="Directory containing YAML test case files"),
    ] = Path("test_cases"),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve per query")] = 5,
    no_cleanup: Annotated[
        bool, typer.Option("--no-cleanup", help="Keep uploaded documents after testing")
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save JSON report to this path"),
    ] = None,
    platform: Annotated[
        str, typer.Option(help="Platform to test (currently: supermemory)")
    ] = "supermemory",
) -> None:
    """Run the visual RAG benchmark against a platform."""
    test_cases = load_test_cases(test_cases_dir)
    if not test_cases:
        console.print(f"[yellow]No test cases found in {test_cases_dir}[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Running {len(test_cases)} test case(s) against [cyan]{platform}[/cyan][/bold]")

    client = SuperMemoryClient()
    report = asyncio.run(
        run_benchmark(client, test_cases, top_k=top_k, cleanup=not no_cleanup)
    )

    _print_report(report)

    if output is None:
        output = Path("results") / f"{platform}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2))
    console.print(f"\n[dim]Report saved to {output}[/dim]")


@app.command()
def freak(
    predictions: Annotated[
        Path | None,
        typer.Option("--predictions", "-p", help="JSON file with model predictions (offline mode)"),
    ] = None,
    platform: Annotated[
        str | None,
        typer.Option(help="Run live against a RAG platform, e.g. supermemory"),
    ] = None,
    model: Annotated[
        str | None,
typer.Option(help="Run directly through a vision model: anthropic | openai"),
    ] = None,
    mcq_dataset: Annotated[
        Path,
        typer.Option("--mcq", help="Path to dataset.json (MCQ)"),
    ] = _FREAK_DATA / "dataset.json",
    qa_dataset: Annotated[
        Path,
        typer.Option("--qa", help="Path to dataset_qa.json (free-form QA)"),
    ] = _FREAK_DATA / "dataset_qa.json",
    mode: Annotated[
        str,
        typer.Option(help="Which split to evaluate: mcq | qa | both"),
    ] = "both",
    top_k: Annotated[
        int,
        typer.Option(help="Chunks to retrieve per query (platform mode only)"),
    ] = 5,
    no_cleanup: Annotated[
        bool,
        typer.Option("--no-cleanup", help="Keep uploaded images after testing (platform mode)"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save JSON results to this path"),
    ] = None,
) -> None:
    """Evaluate FREAK hallucination benchmark — offline or against a live model.

    \b
    Offline (pre-generated predictions file):
      vrb freak --predictions my_preds.json

    \b
    Live RAG platform (upload images → query → check retrieved text):
      vrb freak --platform supermemory

    \b
    Vision model (send image + question directly to the model):
      vrb freak --model anthropic
      vrb freak --model openai

    \b
    Predictions file format (JSON):
      {
        "mcq": [{"id": 502, "prediction": "3."}, ...],
        "qa":  [{"id": 1513, "prediction": "No plaque is visible."}, ...]
      }

    \b
    MCQ predictions: full option text ("3.") or letter ("A").
    Omit a split key to skip it.  Run without any flag to inspect the dataset.
    """
    from .freak_loader import load_mcq, load_qa
    from .freak_evaluator import evaluate_mcq, evaluate_qa, FREAKMCQResult, FREAKQAResult

    run_mcq = mode in ("mcq", "both")
    run_qa = mode in ("qa", "both")

    # Load datasets
    mcq_samples, qa_samples = [], []
    if run_mcq and mcq_dataset.exists():
        mcq_samples = load_mcq(mcq_dataset)
    elif run_mcq:
        console.print(f"[yellow]MCQ dataset not found: {mcq_dataset}[/yellow]")

    if run_qa and qa_dataset.exists():
        qa_samples = load_qa(qa_dataset)
    elif run_qa:
        console.print(f"[yellow]QA dataset not found: {qa_dataset}[/yellow]")

    if not mcq_samples and not qa_samples:
        console.print("[red]No dataset loaded. Exiting.[/red]")
        raise typer.Exit(1)

    # Dataset summary — no flags provided
    if predictions is None and platform is None and model is None:
        console.print("[bold]FREAK dataset summary[/bold]")
        if mcq_samples:
            console.print(f"  MCQ samples : [cyan]{len(mcq_samples)}[/cyan]")
        if qa_samples:
            console.print(f"  QA  samples : [cyan]{len(qa_samples)}[/cyan]")
        console.print(
            "\nEvaluate offline : [bold]vrb freak --predictions <file>[/bold]"
            "\nEvaluate live    : [bold]vrb freak --platform supermemory[/bold]"
            "\nVision model     : [bold]vrb freak --model anthropic|openai[/bold]"
        )
        return

    report: dict = {"timestamp": datetime.utcnow().isoformat()}

    # ------------------------------------------------------------------ #
    # Live platform mode                                                   #
    # ------------------------------------------------------------------ #
    if platform is not None:
        from .freak_runner import run_freak_benchmark

        if platform == "supermemory":
            client = SuperMemoryClient()
        else:
            console.print(f"[red]Unknown platform: {platform}[/red]")
            raise typer.Exit(1)

        console.print(
            f"[bold]Running FREAK live against [cyan]{platform}[/cyan] "
            f"({len(mcq_samples)} MCQ + {len(qa_samples)} QA samples)[/bold]"
        )

        run_report = asyncio.run(
            run_freak_benchmark(
                client,
                mcq_samples if run_mcq else [],
                qa_samples if run_qa else [],
                top_k=top_k,
                cleanup=not no_cleanup,
                log=console.print,
            )
        )

        if run_report.mcq:
            _print_platform_results("MCQ", run_report.mcq, run_report.mcq_accuracy)
            report["mcq"] = {
                "accuracy": run_report.mcq_accuracy,
                "by_category": run_report.accuracy_by_category(run_report.mcq),
                "items": [
                    {"id": r.id, "correct": r.correct, "ground_truth": r.ground_truth,
                     "retrieved_text": r.retrieved_text, "error": r.error}
                    for r in run_report.mcq
                ],
            }

        if run_report.qa:
            _print_platform_results(
                "QA", run_report.qa, run_report.qa_accuracy,
                hallucination_rate=run_report.qa_hallucination_rate,
            )
            report["qa"] = {
                "accuracy": run_report.qa_accuracy,
                "hallucination_rate": run_report.qa_hallucination_rate,
                "by_category": run_report.accuracy_by_category(run_report.qa),
                "items": [
                    {"id": r.id, "correct": r.correct, "hallucinated": r.hallucinated,
                     "ground_truth": r.ground_truth, "retrieved_text": r.retrieved_text,
                     "error": r.error}
                    for r in run_report.qa
                ],
            }

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        if output is None:
            output = Path("results") / f"freak_{platform}_{ts}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"\n[dim]Report saved to {output}[/dim]")
        return

    # ------------------------------------------------------------------ #
    # Vision model mode                                                    #
    # ------------------------------------------------------------------ #
    if model is not None:
        if model == "anthropic":
            from .freak_anthropic_runner import run_freak_anthropic
            if not settings.anthropic_api_key:
                console.print("[red]ANTHROPIC_API_KEY is not set in .env[/red]")
                raise typer.Exit(1)
            console.print(
                f"[bold]Running FREAK via [cyan]Claude (claude-opus-4-7)[/cyan] "
                f"({len(mcq_samples)} MCQ + {len(qa_samples)} QA samples)[/bold]"
            )
            mcq_result, qa_result = run_freak_anthropic(
                mcq_samples if run_mcq else [],
                qa_samples if run_qa else [],
                log=console.print,
            )
            model_tag = "anthropic"
        elif model == "openai":
            from .freak_openai_runner import run_freak_openai
            if not settings.openai_api_key:
                console.print("[red]OPENAI_API_KEY is not set in .env[/red]")
                raise typer.Exit(1)
            console.print(
                f"[bold]Running FREAK via [cyan]OpenAI GPT-5.5[/cyan] "
                f"({len(mcq_samples)} MCQ + {len(qa_samples)} QA samples)[/bold]"
            )
            mcq_result, qa_result = asyncio.run(
                run_freak_openai(
                    mcq_samples if run_mcq else [],
                    qa_samples if run_qa else [],
                    log=console.print,
                )
            )
            model_tag = "openai"
        else:
            console.print(f"[red]Unknown model: {model}. Supported: anthropic | openai[/red]")
            raise typer.Exit(1)

        if mcq_result:
            _print_mcq_result(mcq_result)
            report["mcq"] = {
                "accuracy": mcq_result.accuracy,
                "by_category": mcq_result.accuracy_by_category(),
                "items": [
                    {"id": r.id, "correct": r.correct, "prediction": r.prediction,
                     "ground_truth": r.ground_truth}
                    for r in mcq_result.items
                ],
            }

        if qa_result:
            _print_qa_result(qa_result)
            report["qa"] = {
                "accuracy": qa_result.accuracy,
                "hallucination_rate": qa_result.hallucination_rate,
                "by_category": qa_result.accuracy_by_category(),
                "items": [
                    {"id": r.id, "correct": r.correct, "hallucinated": r.hallucinated,
                     "prediction": r.prediction, "ground_truth": r.ground_truth}
                    for r in qa_result.items
                ],
            }

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        if output is None:
            output = Path("results") / f"freak_{model_tag}_{ts}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"\n[dim]Report saved to {output}[/dim]")
        return

    # ------------------------------------------------------------------ #
    # Offline predictions mode                                             #
    # ------------------------------------------------------------------ #
    raw_preds = json.loads(predictions.read_text(encoding="utf-8"))  # type: ignore[union-attr]

    mcq_result: FREAKMCQResult | None = None
    if run_mcq and mcq_samples and "mcq" in raw_preds:
        pred_map = {p["id"]: p["prediction"] for p in raw_preds["mcq"]}
        aligned = [pred_map.get(s.id, "") for s in mcq_samples]
        missing = sum(1 for v in aligned if v == "")
        if missing:
            console.print(f"[yellow]MCQ: {missing} sample(s) have no prediction — treated as empty.[/yellow]")
        mcq_result = evaluate_mcq(mcq_samples, aligned)
        _print_mcq_result(mcq_result)
        report["mcq"] = {
            "accuracy": mcq_result.accuracy,
            "by_category": mcq_result.accuracy_by_category(),
            "items": [
                {"id": r.id, "correct": r.correct, "prediction": r.prediction,
                 "ground_truth": r.ground_truth}
                for r in mcq_result.items
            ],
        }

    qa_result: FREAKQAResult | None = None
    if run_qa and qa_samples and "qa" in raw_preds:
        pred_map = {p["id"]: p["prediction"] for p in raw_preds["qa"]}
        aligned = [pred_map.get(s.id, "") for s in qa_samples]
        missing = sum(1 for v in aligned if v == "")
        if missing:
            console.print(f"[yellow]QA: {missing} sample(s) have no prediction — treated as empty.[/yellow]")
        qa_result = evaluate_qa(qa_samples, aligned)
        _print_qa_result(qa_result)
        report["qa"] = {
            "accuracy": qa_result.accuracy,
            "hallucination_rate": qa_result.hallucination_rate,
            "by_category": qa_result.accuracy_by_category(),
            "items": [
                {"id": r.id, "correct": r.correct, "hallucinated": r.hallucinated,
                 "prediction": r.prediction, "ground_truth": r.ground_truth}
                for r in qa_result.items
            ],
        }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    if output is None:
        output = Path("results") / f"freak_{ts}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[dim]Report saved to {output}[/dim]")


def _print_platform_results(
    label: str,
    results: list,
    accuracy: float,
    hallucination_rate: float | None = None,
) -> None:
    from .freak_runner import FREAKSampleResult
    errors = [r for r in results if r.error]
    table = Table(title=f"FREAK {label} — platform retrieval", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Accuracy", justify="right")

    from collections import defaultdict
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for r in results:
        if r.error:
            continue
        cats = r.category if isinstance(r.category, list) else [r.category]
        for c in cats:
            total[c] += 1
            correct[c] += int(r.correct)
    for cat in total:
        table.add_row(cat, f"{correct[cat] / total[cat]:.1%}")

    console.print(table)
    summary = f"[bold]Overall {label} accuracy: [cyan]{accuracy:.1%}[/cyan][/bold]"
    if hallucination_rate is not None:
        summary += f"  Hallucination rate: [red]{hallucination_rate:.1%}[/red]"
    if errors:
        summary += f"  [yellow]{len(errors)} error(s)[/yellow]"
    console.print(summary)


def _print_mcq_result(result: object) -> None:
    from .freak_evaluator import FREAKMCQResult
    assert isinstance(result, FREAKMCQResult)

    table = Table(title="FREAK — Multiple-Choice Results", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Accuracy", justify="right")

    by_cat = result.accuracy_by_category()
    for cat, acc in by_cat.items():
        if acc == acc:  # skip NaN
            table.add_row(cat, f"{acc:.1%}")

    console.print(table)
    console.print(f"[bold]Overall MCQ accuracy: [cyan]{result.accuracy:.1%}[/cyan][/bold]")


def _print_qa_result(result: object) -> None:
    from .freak_evaluator import FREAKQAResult
    assert isinstance(result, FREAKQAResult)

    table = Table(title="FREAK — Free-form QA Results", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Accuracy", justify="right")

    by_cat = result.accuracy_by_category()
    for cat, acc in by_cat.items():
        if acc == acc:
            table.add_row(cat, f"{acc:.1%}")

    console.print(table)
    console.print(
        f"[bold]Overall QA accuracy: [cyan]{result.accuracy:.1%}[/cyan]  "
        f"Hallucination rate: [red]{result.hallucination_rate:.1%}[/red][/bold]"
    )


_GQA_DATA = Path("data/gqa_sample")
_SEEDBENCH_DATA = Path("data/seedbench_dataset")


@app.command()
def seedbench(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to SEED-Bench dataset.json"),
    ] = _SEEDBENCH_DATA / "dataset.json",
    models: Annotated[
        str,
        typer.Option("--models", "-m", help="Comma-separated: anthropic,openai,gemini-pro,gemini-flash,kimi"),
    ] = "anthropic,openai,gemini-pro,gemini-flash,kimi",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Excel file path"),
    ] = None,
) -> None:
    """Run SEED-Bench MCQ benchmark against vision models and export results to Excel.
    Skips any model whose cached results already exist in results/cache/.
    """
    from .seedbench_loader import load_seedbench
    from .seedbench_anthropic_runner import run_seedbench_anthropic
    from .seedbench_openai_runner import run_seedbench_openai
    from .seedbench_openrouter_runner import run_seedbench_openrouter
    from .seedbench_evaluator import SEEDBenchResult, SEEDBenchItemResult
    from .seedbench_excel_export import export_seedbench_excel

    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    samples = load_seedbench(dataset)
    console.print(f"[bold]Loaded [cyan]{len(samples)}[/cyan] SEED-Bench samples[/bold]")

    cache_dir = Path("results/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(key: str) -> SEEDBenchResult | None:
        p = cache_dir / f"seedbench_{key}.json"
        if not p.exists():
            return None
        items = [SEEDBenchItemResult(**i) for i in json.loads(p.read_text())]
        r = SEEDBenchResult(); r.items = items
        return r

    def _save_cache(key: str, result: SEEDBenchResult) -> None:
        import dataclasses
        p = cache_dir / f"seedbench_{key}.json"
        p.write_text(json.dumps([dataclasses.asdict(i) for i in result.items], indent=2))

    def _run_or_load(key: str, label: str, runner_fn) -> SEEDBenchResult:
        cached = _load_cache(key)
        if cached is not None:
            console.print(f"\n[dim]Loaded cached results for [cyan]{label}[/cyan] ({cached.accuracy:.1%})[/dim]")
            return cached
        console.print(f"\n[bold]Running via [cyan]{label}[/cyan]...[/bold]")
        result = runner_fn()
        _save_cache(key, result)
        console.print(f"[bold]Accuracy: [cyan]{result.accuracy:.1%}[/cyan][/bold]")
        return result

    model_list = [m.strip() for m in models.split(",")]
    anthropic_result = openai_result = gemini_pro_result = gemini_flash_result = kimi_result = None

    if "anthropic" in model_list:
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        anthropic_result = _run_or_load("anthropic", "Claude Opus 4.7",
            lambda: run_seedbench_anthropic(samples, log=console.print))

    if "openai" in model_list:
        if not settings.openai_api_key:
            console.print("[red]OPENAI_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        openai_result = _run_or_load("openai", "GPT-5.5",
            lambda: run_seedbench_openai(samples, log=console.print))

    if "gemini-pro" in model_list:
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        gemini_pro_result = _run_or_load("gemini_pro", "Gemini 3.1 Pro",
            lambda: run_seedbench_openrouter(samples, model="~google/gemini-pro-latest", log=console.print))

    if "gemini-flash" in model_list:
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        gemini_flash_result = _run_or_load("gemini_flash", "Gemini 3.5 Flash",
            lambda: run_seedbench_openrouter(samples, model="~google/gemini-flash-latest", log=console.print))

    if "kimi" in model_list:
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        kimi_result = _run_or_load("kimi", "Kimi K2.6",
            lambda: run_seedbench_openrouter(samples, model="moonshotai/kimi-k2.6", log=console.print))

    if output is None:
        output = Path("results") / "seedbench_all_models.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Exporting Excel report to [cyan]{output}[/cyan]...[/bold]")
    export_seedbench_excel(output,
        anthropic_result=anthropic_result, openai_result=openai_result,
        gemini_pro_result=gemini_pro_result, gemini_flash_result=gemini_flash_result,
        kimi_result=kimi_result,
    )
    console.print(f"[green]Done![/green] Saved to {output}")


@app.command()
def gqa(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to GQA dataset.json"),
    ] = _GQA_DATA / "dataset.json",
    models: Annotated[
        str,
        typer.Option("--models", "-m", help="Comma-separated: anthropic,openai,gemini-pro,gemini-flash,kimi"),
    ] = "anthropic,openai,gemini-pro,gemini-flash,kimi",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Excel file path"),
    ] = None,
) -> None:
    """Run GQA benchmark against vision models and export results to Excel with images.
    Skips any model whose cached results already exist in results/cache/.
    """
    from .gqa_loader import load_gqa
    from .gqa_anthropic_runner import run_gqa_anthropic
    from .gqa_openai_runner import run_gqa_openai
    from .gqa_openrouter_runner import run_gqa_openrouter
    from .gqa_evaluator import GQAResult, GQAItemResult
    from .gqa_excel_export import export_gqa_excel

    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    samples = load_gqa(dataset)
    console.print(f"[bold]Loaded [cyan]{len(samples)}[/cyan] GQA samples[/bold]")

    cache_dir = Path("results/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(key: str) -> GQAResult | None:
        p = cache_dir / f"gqa_{key}.json"
        if not p.exists():
            return None
        items = [GQAItemResult(**i) for i in json.loads(p.read_text())]
        r = GQAResult(); r.items = items
        return r

    def _save_cache(key: str, result: GQAResult) -> None:
        import dataclasses
        p = cache_dir / f"gqa_{key}.json"
        p.write_text(json.dumps([dataclasses.asdict(i) for i in result.items], indent=2))

    def _run_or_load(key: str, label: str, runner_fn) -> GQAResult:
        cached = _load_cache(key)
        if cached is not None:
            console.print(f"\n[dim]Loaded cached results for [cyan]{label}[/cyan] ({cached.accuracy:.1%})[/dim]")
            return cached
        console.print(f"\n[bold]Running via [cyan]{label}[/cyan]...[/bold]")
        result = runner_fn()
        _save_cache(key, result)
        console.print(f"[bold]Accuracy: [cyan]{result.accuracy:.1%}[/cyan][/bold]")
        return result

    model_list = [m.strip() for m in models.split(",")]
    anthropic_result = openai_result = gemini_pro_result = gemini_flash_result = kimi_result = None

    if "anthropic" in model_list:
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        anthropic_result = _run_or_load("anthropic", "Claude Opus 4.7",
            lambda: run_gqa_anthropic(samples, log=console.print))

    if "openai" in model_list:
        if not settings.openai_api_key:
            console.print("[red]OPENAI_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        openai_result = _run_or_load("openai", "GPT-5.5",
            lambda: run_gqa_openai(samples, log=console.print))

    if "gemini-pro" in model_list:
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        gemini_pro_result = _run_or_load("gemini_pro", "Gemini 3.1 Pro",
            lambda: run_gqa_openrouter(samples, model="~google/gemini-pro-latest", log=console.print))

    if "gemini-flash" in model_list:
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        gemini_flash_result = _run_or_load("gemini_flash", "Gemini 3.5 Flash",
            lambda: run_gqa_openrouter(samples, model="~google/gemini-flash-latest", log=console.print))

    if "kimi" in model_list:
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is not set in .env[/red]"); raise typer.Exit(1)
        kimi_result = _run_or_load("kimi", "Kimi K2.6",
            lambda: run_gqa_openrouter(samples, model="moonshotai/kimi-k2.6", log=console.print))

    if output is None:
        output = Path("results") / "gqa_all_models.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Exporting Excel report to [cyan]{output}[/cyan]...[/bold]")
    export_gqa_excel(output,
        anthropic_result=anthropic_result, openai_result=openai_result,
        gemini_pro_result=gemini_pro_result, gemini_flash_result=gemini_flash_result,
        kimi_result=kimi_result,
    )
    console.print(f"[green]Done![/green] Saved to {output}")


def _print_report(report: object) -> None:
    from .models import BenchmarkReport

    assert isinstance(report, BenchmarkReport)

    table = Table(title=f"Results — {report.platform}", show_lines=True)
    table.add_column("Test Case", style="bold")
    table.add_column("File Type")
    table.add_column("Ingested", justify="center")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Hallucinations", justify="center")

    for r in report.file_results:
        ingested = "[green]Yes[/green]" if r.ingestion_success else f"[red]No[/red]"
        halluc = sum(len(e.hallucinations_detected) for e in r.query_evaluations)
        table.add_row(
            r.test_case_name,
            r.file_type,
            ingested,
            f"{r.avg_precision:.2f}",
            f"{r.avg_recall:.2f}",
            f"{r.avg_f1:.2f}",
            f"[red]{halluc}[/red]" if halluc else "[green]0[/green]",
        )

    console.print(table)
    console.print(
        f"\n[bold]Overall[/bold]  "
        f"Precision: {report.overall_precision:.2f}  "
        f"Recall: {report.overall_recall:.2f}  "
        f"F1: {report.overall_f1:.2f}  "
        f"Hallucination rate: {report.hallucination_rate:.1%}"
    )
