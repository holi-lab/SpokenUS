from pathlib import Path

import click
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
)
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from augmentation.batch.client import BatchClient
from augmentation.constants import DATASETS
from augmentation.disfluency import DisfluencyConfig
from augmentation.loaders import is_dataset_available
from augmentation.pipeline import AugmentationPipeline, PipelineConfig


def _append_usage_summary_rows(table: Table, pipeline) -> None:
    tracker = getattr(pipeline, "llm_usage_tracker", None)
    if tracker is None or not hasattr(tracker, "snapshot"):
        return

    snapshot = tracker.snapshot()
    if snapshot.total.requests == 0:
        return

    barge_in = snapshot.by_stage.get("barge-in")
    disfluency = snapshot.by_stage.get("disfluency")
    emotion = snapshot.by_stage.get("emotion")

    def _stage_value(stats, attr: str) -> str:
        if stats is None:
            return ""
        value = getattr(stats, attr)
        if attr == "estimated_cost_usd":
            return f"${value:.6f}"
        return str(value)

    def _format_breakdown(attr: str, default: str) -> str:
        return ", ".join(
            [
                f"barge-in: {_stage_value(barge_in, attr) or default}",
                f"disfluency: {_stage_value(disfluency, attr) or default}",
                f"emotion: {_stage_value(emotion, attr) or default}",
            ]
        )

    def _format_metric(total: str, attr: str, default: str) -> str:
        return f"{total} ({_format_breakdown(attr, default)})"

    table.add_row(
        "LLM Requests",
        _format_metric(str(snapshot.total.requests), "requests", "0"),
    )
    table.add_row(
        "Prompt Tokens",
        _format_metric(str(snapshot.total.prompt_tokens), "prompt_tokens", "0"),
    )
    table.add_row(
        "Completion Tokens",
        _format_metric(str(snapshot.total.completion_tokens), "completion_tokens", "0"),
    )
    table.add_row(
        "Total Tokens",
        _format_metric(str(snapshot.total.total_tokens), "total_tokens", "0"),
    )
    table.add_row(
        "Estimated Cost",
        _format_metric(
            f"${snapshot.total.estimated_cost_usd:.6f}",
            "estimated_cost_usd",
            "$0.000000",
        ),
        style="bold",
    )


@click.command(context_settings={"show_default": True})
@click.option(
    "--datasets",
    default=",".join(DATASETS),
    help="Comma-separated list of datasets to process",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets"),
    help="Base directory containing datasets",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets/SpokenTOD"),
    help="Output directory for augmented data",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100,
    help="Number of dialogues to process per chunk",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of datasets to process in parallel",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Limit samples per dataset (for testing)",
)
@click.option(
    "--splits",
    default="train,valid,test",
    help="Comma-separated list of splits to process",
)
@click.option(
    "--model",
    default="openrouter/qwen/qwen3-32b",
    help="Model name to use for emotion tagging and other LLM tasks",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only print what would be done",
)
@click.option(
    "--with_selected_speakers",
    "--with-selected-speakers",
    is_flag=True,
    help="Restrict user speaker sampling to selected_speakers.csv",
)
def main(
    datasets: str,
    data_dir: Path,
    output_dir: Path,
    chunk_size: int,
    workers: int,
    sample_size: int | None,
    splits: str,
    model: str,
    dry_run: bool,
    with_selected_speakers: bool,
):
    console = Console()
    console.clear()
    if not is_dataset_available("saa", data_dir):
        raise ValueError(
            f"SAA dataset not found in {data_dir}. Please run the download script to obtain it. make download DATASETS=saa"
        )

    datasets = [d.strip() for d in datasets.split(",")]
    splits = [s.strip() for s in splits.split(",")]

    for dataset in datasets:
        if not is_dataset_available(dataset, data_dir):
            console.print(f"[red]Error:[/red] Dataset not found for: {dataset} in {data_dir}")
            raise ValueError(f"Dataset not found for: {dataset} in {data_dir}")

    output_files = ", ".join(
        str(output_dir / f"{split}_text.jsonl") for split in splits
    )

    header = Table.grid(padding=(0, 1))
    header.add_column(justify="right", style="bold cyan", no_wrap=True)
    header.add_column(style="white")
    header.add_row("Datasets", ", ".join(datasets))
    header.add_row("Splits", ", ".join(splits))
    header.add_row("Output", output_files)
    header.add_row("Model", str(model))
    header.add_row("Chunk size", str(chunk_size))
    header.add_row("Workers", str(workers))
    header.add_row("Sample size", str(sample_size) if sample_size else "All")
    header.add_row(
        "Selected speakers",
        "Yes (selected_speakers.csv)" if with_selected_speakers else "No",
    )
    header.add_row(
        "Total Size",
        f"{sample_size * len(datasets) * len(splits)}" if sample_size else "All",
    )
    console.print()
    console.print(
        Panel(
            header,
            title="SpokenTOD - Text Augmentation Pipeline",
            box=box.ASCII,
            border_style="cyan",
        )
    )
    console.print(
        "[yellow]Note:[/yellow] This pipeline can take a long time. "
        "We recommend running it in a background session (tmux, nohup, screen) "
        "to avoid interruption."
    )
    console.print()

    if dry_run:
        console.print(
            f"[yellow][DRY RUN][/yellow] Would process {len(datasets)} datasets with {len(splits)} splits and save to {output_dir}",
            style="yellow",
        )
        return

    config = PipelineConfig(
        model=model,
        datasets=datasets,
        data_dir=data_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        workers=workers,
        sample_size=sample_size,
        with_selected_speakers=with_selected_speakers,
        disfluency_config=DisfluencyConfig(),
    )

    pipeline = AugmentationPipeline(config)

    try:
        stats = pipeline.run(splits=splits)
    except (APIConnectionError, AuthenticationError, BadRequestError, APIError) as exc:
        message = BatchClient(model=model)._format_request_error(exc)
        console.print(f"[red]Error:[/red] {message}")
        raise click.ClickException(message) from exc

    print()

    table = Table(
        title="Augmentation Results",
        box=box.ASCII,
        border_style="cyan",
        title_style="bold cyan",
        header_style="bold cyan",
    )
    table.add_column("Item", style="bold cyan")
    table.add_column("Value", style="white")
    for dataset, count in stats.items():
        table.add_row(dataset, str(count))

    total = sum(stats.values())
    table.add_row("Dataset Counts", str(total), style="bold", end_section=True)
    _append_usage_summary_rows(table, pipeline)
    console.print(table)
    console.print(f"Done! Output saved to: {output_dir}", style="green")


if __name__ == "__main__":
    main()
