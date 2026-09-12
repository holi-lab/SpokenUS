"""Main augmentation pipeline orchestrator."""

import asyncio
import concurrent.futures
import json
import re
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from augmentation.bargein import inject_bargein_dialogues_batch_async
from augmentation.batch.client import LLMUsageTracker
from augmentation.constants import (
    BARGEIN_EXCLUDED,
    CROSSTURN_EXCLUDED,
    DATASETS,
    DISFLUENCY_EXCLUDED,
    EMOTION_EXCLUDED,
    SEGMENTABLE_SLOTS,
)
from augmentation.disfluency import (
    DisfluencyConfig,
    inject_disfluency_dialogues_batch_async,
)
from augmentation.emotion.tagger import EmotionTagger
from augmentation.loaders.abcd import ABCDLoader
from augmentation.loaders.emowoz import EmoWOZLoader
from augmentation.loaders.sgd import SGDLoader
from augmentation.loaders.spokenwoz import SpokenWOZLoader
from augmentation.loaders.tm2 import TM2Loader
from augmentation.schema import AugmentedDialogue, Dialogue, Turn
from augmentation.segmentation.generator import (
    find_segmentable_slots,
    generate_crossturn_dialogue,
)
from demographic_sampler import DemographicSampler


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    model: str = "openrouter/qwen/qwen3-32b"
    datasets: list[str] = field(default_factory=lambda: DATASETS)
    data_dir: Path = field(default_factory=lambda: Path("datasets"))
    output_dir: Path = field(
        default_factory=lambda: Path("datasets/SpokenTOD")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    workers: int = 1
    chunk_size: int = 100
    disfluency_config: DisfluencyConfig = field(default_factory=DisfluencyConfig)
    sample_size: int | None = None
    with_selected_speakers: bool = False


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


@dataclass
class _RunProgressTracker:
    total_dialogues: int
    start_time: float = field(default_factory=time.time)
    completed_dialogues: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_completed(self, count: int = 1) -> None:
        with self._lock:
            self.completed_dialogues += count

    def adjust_total(self, delta: int) -> int:
        with self._lock:
            self.total_dialogues = max(
                self.completed_dialogues,
                self.total_dialogues + delta,
            )
            return self.total_dialogues

    def estimate_remaining_seconds(self) -> float:
        with self._lock:
            elapsed = time.time() - self.start_time
            avg_seconds = elapsed / max(self.completed_dialogues, 1)
            remaining = max(self.total_dialogues - self.completed_dialogues, 0)
        return remaining * avg_seconds


class _CurrentETAColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields.get("kind") == "overall":
            return Text("")
        total = int(task.total or 0)
        remaining = max(total - int(task.completed), 0)
        elapsed = task.elapsed or 0.0
        avg_seconds = elapsed / max(int(task.completed), 1)
        return Text(f"current {_format_eta(remaining * avg_seconds)}", style="yellow")


class _SplitBarColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self._bar = BarColumn()

    def render(self, task: Task) -> Text:
        if task.fields.get("kind") == "overall":
            return Text("")
        return self._bar.render(task)


class _SplitCountColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields.get("kind") == "overall":
            return Text("")
        total = int(task.total or 0)
        return Text(f"{int(task.completed)}/{total}")


class _SplitElapsedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields.get("kind") == "overall":
            return Text("")
        return Text(_format_eta(task.elapsed or 0.0))


class _OverallETAColumn(ProgressColumn):
    def __init__(self, tracker: _RunProgressTracker | None):
        super().__init__()
        self._tracker = tracker

    def render(self, task: Task) -> Text:
        if task.fields.get("kind") != "overall":
            return Text("")
        if self._tracker is None:
            return Text("overall 0:00:00", style="magenta")
        return Text(
            f"overall {_format_eta(self._tracker.estimate_remaining_seconds())}",
            style="magenta",
        )


class _ExpectedFinishColumn(ProgressColumn):
    def __init__(self, tracker: _RunProgressTracker | None):
        super().__init__()
        self._tracker = tracker

    def render(self, task: Task) -> Text:
        if task.fields.get("kind") != "overall":
            return Text("")
        if self._tracker is None:
            return Text("finish 00:00:00", style="cyan")
        finish_at = time.time() + self._tracker.estimate_remaining_seconds()
        return Text(
            f"finish {datetime.fromtimestamp(finish_at).strftime('%H:%M:%S')}",
            style="cyan",
        )


class _OverallTokensColumn(ProgressColumn):
    def __init__(self, usage_tracker: LLMUsageTracker | None):
        super().__init__()
        self._usage_tracker = usage_tracker

    def render(self, task: Task) -> Text:
        if task.fields.get("kind") != "overall":
            return Text("")
        if self._usage_tracker is None:
            return Text("tokens 0", style="green")
        snapshot = self._usage_tracker.snapshot()
        return Text(f"tokens {snapshot.total.total_tokens}", style="green")


class _OverallCostColumn(ProgressColumn):
    def __init__(self, usage_tracker: LLMUsageTracker | None):
        super().__init__()
        self._usage_tracker = usage_tracker

    def render(self, task: Task) -> Text:
        if task.fields.get("kind") != "overall":
            return Text("")
        if self._usage_tracker is None:
            return Text("cost $0.000000", style="bright_green")
        snapshot = self._usage_tracker.snapshot()
        return Text(
            f"cost ${snapshot.total.estimated_cost_usd:.6f}",
            style="bright_green",
        )


class AugmentationPipeline:
    """Main pipeline for voice dataset augmentation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm_usage_tracker = LLMUsageTracker()
        self.emotion_tagger = EmotionTagger(
            model=config.model,
            usage_tracker=self.llm_usage_tracker,
        )
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._progress_tracker: _RunProgressTracker | None = None
        self._shared_status_progress: Progress | None = None
        self._shared_split_progress: Progress | None = None
        self._shared_overall_progress: Progress | None = None
        self._status_task_id: int | None = None
        self._overall_task_id: int | None = None

        self.demographic_sampler = DemographicSampler(
            base_dir=str(config.data_dir / "SpeechAccentArchive"),
            balance_distribution=True,
            with_selected_speakers=config.with_selected_speakers,
        )

        # Timing statistics
        self.timing_stats = {
            "crossturn": 0.0,
            "emotion": 0.0,
            "barge-in": 0.0,
            "disfluency": 0.0,
            "demographic": 0.0,
            "total_batches": 0,
        }

    def _get_async_loop(self) -> asyncio.AbstractEventLoop:
        if self._async_loop is None or self._async_loop.is_closed():
            self._async_loop = asyncio.new_event_loop()
        return self._async_loop

    def _run_async(self, coro):
        loop = self._get_async_loop()
        return loop.run_until_complete(coro)

    def close(self) -> None:
        if self._async_loop is not None and not self._async_loop.is_closed():
            if getattr(GLOBAL_LOGGING_WORKER, "_bound_loop", None) is self._async_loop:
                self._async_loop.run_until_complete(GLOBAL_LOGGING_WORKER.flush())
                self._async_loop.run_until_complete(GLOBAL_LOGGING_WORKER.stop())
            self._async_loop.close()
        self._async_loop = None

    def _get_loader(self, dataset: str, split: str):
        """Get appropriate loader for dataset."""
        cfg = self.config

        if dataset == "emowoz":
            return EmoWOZLoader(
                cfg.data_dir / "EmoWOZ",
                split=split,
            )
        elif dataset == "sgd":
            return SGDLoader(cfg.data_dir / "dstc8-schema-guided-dialogue", split=split)
        elif dataset == "abcd":
            return ABCDLoader(cfg.data_dir / "abcd", split=split)
        elif dataset == "tm2":
            return TM2Loader(cfg.data_dir / "TM-2-2020", split=split)
        elif dataset == "spokenwoz":
            return SpokenWOZLoader(
                cfg.data_dir / "SpokenWOZ",
                split=split,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def process_dialogue(self, dialogue: Dialogue) -> AugmentedDialogue:
        """Process a single dialogue using the batch pipeline for identical behavior."""
        results = self._process_batch([dialogue])
        if not results:
            raise ValueError(f"No augmented result for dialogue: {dialogue.id}")
        return results[0]

    def process_dataset(
        self,
        dataset: str,
        split: str = "train",
        console: Console | None = None,
    ) -> Iterator[AugmentedDialogue]:
        """Process all dialogues in a dataset split."""
        if console is None:
            console = Console()

        if self._shared_split_progress is not None:
            planned_total = self._get_dataset_total(dataset, split)
            task_id = self._shared_split_progress.add_task(
                f"{dataset}/{split}",
                total=planned_total,
                kind="split",
            )
            completed = 0
            for augmented in self._iter_dataset_split(dataset, split):
                self._shared_split_progress.update(task_id, advance=1)
                completed += 1
                if self._progress_tracker is not None:
                    self._progress_tracker.mark_completed()
                if (
                    self._shared_overall_progress is not None
                    and self._overall_task_id is not None
                ):
                    self._shared_overall_progress.update(self._overall_task_id, advance=1)
                yield augmented
            self._finalize_progress_totals(
                progress=self._shared_split_progress,
                task_id=task_id,
                completed=completed,
                planned_total=planned_total,
            )
            return

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            _SplitBarColumn(),
            _SplitCountColumn(),
            _SplitElapsedColumn(),
            _CurrentETAColumn(),
            console=console,
        )

        with progress:
            planned_total = self._get_dataset_total(dataset, split)
            task_id = progress.add_task(
                f"{dataset}/{split}",
                total=planned_total,
            )
            completed = 0
            for augmented in self._iter_dataset_split(dataset, split):
                progress.update(task_id, advance=1)
                completed += 1
                if self._progress_tracker is not None:
                    self._progress_tracker.mark_completed()
                yield augmented
            self._finalize_progress_totals(
                progress=progress,
                task_id=task_id,
                completed=completed,
                planned_total=planned_total,
            )

    def _get_dataset_total(self, dataset: str, split: str) -> int:
        if self.config.sample_size is not None:
            return self.config.sample_size
        loader = self._get_loader(dataset, split)
        return len(loader)

    def _finalize_progress_totals(
        self,
        progress: Progress,
        task_id: int,
        completed: int,
        planned_total: int,
    ) -> None:
        adjusted_total = max(completed, 0)
        progress.update(task_id, total=adjusted_total, completed=completed)

        missing = planned_total - adjusted_total
        if missing <= 0:
            return

        if self._progress_tracker is not None:
            new_total = self._progress_tracker.adjust_total(-missing)
        else:
            new_total = None

        if self._shared_overall_progress is not None and self._overall_task_id is not None:
            self._shared_overall_progress.update(
                self._overall_task_id,
                total=new_total,
            )

    def _iter_dataset_split(self, dataset: str, split: str) -> Iterator[AugmentedDialogue]:
        loader = self._get_loader(dataset, split)
        iterator_src = loader.load()

        if self.config.sample_size is not None:
            iterator_src = islice(iterator_src, self.config.sample_size)

        chunk_size = self.config.chunk_size
        chunk = []

        for dialogue in iterator_src:
            chunk.append(dialogue)

            if len(chunk) >= chunk_size:
                yield from self._process_batch(chunk)
                chunk = []

        if chunk:
            yield from self._process_batch(chunk)

    def _process_batch(self, dialogues: list[Dialogue]) -> list[AugmentedDialogue]:
        """Process a batch of dialogues with concurrent LLM calls.

        Steps:
        1. Pre-process all dialogues (cross-turn segmentation)
        2. Collect all emotion tagging requests
        3. Run async emotion tagging for all utterances at once
        4. Apply emotion results and inject disfluency
        """
        self.timing_stats["total_batches"] += 1

        # Step 1: Pre-process all dialogues (cross-turn segmentation)
        t0 = time.time()
        preprocessed: list[tuple[Dialogue, list[Turn]]] = []
        for dialogue in dialogues:
            turns = list(dialogue.turns)
            dataset = dialogue.source

            # Cross-turn slot segmentation
            has_segmentable_slots = (
                dataset in SEGMENTABLE_SLOTS
                and find_segmentable_slots(dialogue.state, dataset)
            )
            if dataset not in CROSSTURN_EXCLUDED and has_segmentable_slots:
                pass  # logger.info is too verbose in batch processing
            turns = self._apply_crossturn(
                turns, dialogue.state, dataset, dialogue.metadata
            )
            preprocessed.append((dialogue, turns))
        self.timing_stats["crossturn"] += time.time() - t0

        # Step 2: Collect all emotion tagging requests
        emotion_requests = []
        for dlg_idx, (dialogue, turns) in enumerate(preprocessed):
            dataset = dialogue.source
            # Only tag emotions for datasets without existing labels
            if dataset not in EMOTION_EXCLUDED and not dialogue.emotion_labels:
                for turn_idx, turn in enumerate(turns):
                    if turn.get("role") == "user" and not turn.get("segment"):
                        emotion_requests.append((dlg_idx, turn_idx, turn["text"]))

        # Step 3: Run async emotion tagging for all utterances at once
        t0 = time.time()
        if emotion_requests:
            utterances = [req[2] for req in emotion_requests]
            emotions = self._run_async(self.emotion_tagger.tag_utterances_async(utterances))

            # Apply emotion results to preprocessed turns
            for (dlg_idx, turn_idx, _), emotion in zip(emotion_requests, emotions):
                dialogue, turns = preprocessed[dlg_idx]
                turns[turn_idx] = dict(turns[turn_idx])
                turns[turn_idx]["emotion"] = emotion
        self.timing_stats["emotion"] += time.time() - t0

        # Handle existing emotion labels (EmoWOZ)
        for dlg_idx, (dialogue, turns) in enumerate(preprocessed):
            if dialogue.emotion_labels:
                from augmentation.constants import EMOTION_LABELS

                for i, turn in enumerate(turns):
                    if turn.get("role") == "user" and i < len(dialogue.emotion_labels):
                        label = dialogue.emotion_labels[i]
                        if label >= 0:
                            turns[i] = dict(turns[i])
                            turns[i]["emotion"] = {
                                "label": label,
                                "name": EMOTION_LABELS.get(label, "neutral"),
                            }

        # Propagate emotion to cross-turn segments
        for dlg_idx, (dialogue, turns) in enumerate(preprocessed):
            last_emotion = None
            from augmentation.constants import EMOTION_LABELS

            for i, turn in enumerate(turns):
                if turn.get("role") == "user":
                    if turn.get("segment"):
                        if last_emotion is None:
                            last_emotion = {
                                "label": 0,
                                "name": EMOTION_LABELS.get(0, "neutral"),
                            }
                        turns[i] = dict(turns[i])
                        turns[i]["emotion"] = last_emotion
                    else:
                        last_emotion = turn.get("emotion")

        # Step 4: Apply barge-in augmentation
        dialogues_turns = [turns for _, turns in preprocessed]

        # Filter dialogues for barge-in (exclude SpokenWOZ)
        bargein_indices = []
        bargein_dialogues_turns = []
        for i, (dialogue, _) in enumerate(preprocessed):
            if dialogue.source not in BARGEIN_EXCLUDED and not dialogue.metadata.get(
                "skip_augmentation"
            ):
                bargein_indices.append(i)
                bargein_dialogues_turns.append(dialogues_turns[i])

        t0 = time.time()
        if bargein_indices:
            bargein_results = self._run_async(
                inject_bargein_dialogues_batch_async(
                    bargein_dialogues_turns,
                    model=self.config.model,
                    sample_rate=0.25,
                    max_concurrency=self.config.workers,
                    usage_tracker=self.llm_usage_tracker,
                )
            )
            # Update dialogues with barge-in results
            for idx, result in zip(bargein_indices, bargein_results):
                dialogues_turns[idx] = result
        self.timing_stats["barge-in"] += time.time() - t0

        # Step 5: Apply disfluency injection using async batch processing
        t0 = time.time()

        # Filter dialogues for disfluency (exclude SpokenWOZ)
        disfluency_indices = []
        disfluency_dialogues_turns = []
        for i, (dialogue, _) in enumerate(preprocessed):
            if (
                dialogue.source not in DISFLUENCY_EXCLUDED
                and not dialogue.metadata.get("skip_augmentation")
            ):
                disfluency_indices.append(i)
                disfluency_dialogues_turns.append(dialogues_turns[i])

        if disfluency_indices:
            disfluency_results = self._run_async(
                inject_disfluency_dialogues_batch_async(
                    disfluency_dialogues_turns,
                    self.config.disfluency_config,
                    model=self.config.model,
                    max_concurrency=100,
                    usage_tracker=self.llm_usage_tracker,
                )
            )
            # Update dialogues with disfluency results
            for idx, result in zip(disfluency_indices, disfluency_results):
                dialogues_turns[idx] = result
        self.timing_stats["disfluency"] += time.time() - t0

        # Build augmented dialogues
        results = []
        for (dialogue, _), turns in zip(preprocessed, dialogues_turns):
            # Apply <|endoftext|> token injection (for ALL datasets including SpokenWOZ)
            turns = inject_endoftext(turns)

            goal_dict = {
                "text": dialogue.goal.text if dialogue.goal else "",
                "structured": (
                    (
                        {
                        "domains": (
                            dialogue.goal.structured.domains if dialogue.goal else []
                        ),
                        "intents": (
                            dialogue.goal.structured.intents if dialogue.goal else []
                        ),
                        }
                    )
                    if dialogue.goal
                    else {}
                ),
            }
            if (
                dialogue.goal
                and dialogue.goal.structured.metadata
            ):
                goal_dict["structured"]["metadata"] = dialogue.goal.structured.metadata

            augmented_turns = []
            for t in turns:
                role = t.get("role", "user")
                if role == "assistant":
                    turn = Turn(
                        role=role,
                        text=t.get("text", ""),
                        bargein=t.get("bargein"),
                    )
                else:
                    # User turn
                    turn = Turn(
                        role=role,
                        text=t.get("text", ""),
                        tagged=t.get("tagged"),
                        emotion=t.get("emotion"),
                        disfluency=t.get("disfluency"),
                        segment=t.get("segment"),
                        state=t.get("state"),
                        bargein=t.get("bargein"),
                        audio_path=t.get("audio_path"),
                    )
                augmented_turns.append(turn)

            # augment demographic
            t0 = time.time()
            demo = self.demographic_sampler.sample_demographic()
            speaker = self.demographic_sampler.find_speaker(
                demo
            )  # Excludes assistant pool
            assistant_speaker = self.demographic_sampler.sample_assistant_speaker()
            self.timing_stats["demographic"] += time.time() - t0

            augmented = AugmentedDialogue(
                id=dialogue.id,
                source=dialogue.source,
                goal=goal_dict,
                turns=augmented_turns,
                state=dialogue.state,
                metadata=dialogue.metadata,
                speaker=speaker,
                assistant_speaker=assistant_speaker,
            )

            # Validate dialogue (turn alternation and speaker constraints)
            self._validate_dialogue(augmented)

            results.append(augmented)

        return results

    def run(self, splits: list[str] | None = None) -> dict[str, int]:
        if splits is None:
            splits = ["train", "valid", "test"]

        console = Console()
        self._progress_tracker = _RunProgressTracker(
            total_dialogues=self._get_total_dialogues(splits)
        )
        self._shared_overall_progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        self._shared_status_progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            TextColumn("[dim] | [/dim]"),
            _OverallTokensColumn(self.llm_usage_tracker),
            TextColumn("[dim] | [/dim]"),
            _OverallCostColumn(self.llm_usage_tracker),
            TextColumn("[dim] | [/dim]"),
            _OverallETAColumn(self._progress_tracker),
            TextColumn("[dim] | [/dim]"),
            _ExpectedFinishColumn(self._progress_tracker),
            console=console,
        )
        self._shared_split_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            _SplitBarColumn(),
            _SplitCountColumn(),
            _SplitElapsedColumn(),
            _CurrentETAColumn(),
            console=console,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        demographics_dir = self.config.output_dir / "demographics"
        demographics_dir.mkdir(parents=True, exist_ok=True)

        worker_count = max(1, min(self.config.workers, len(self.config.datasets)))
        dataset_label = "dataset" if len(self.config.datasets) == 1 else "datasets"
        worker_label = "worker" if worker_count == 1 else "workers"
        processing_status = (
            f"Processing {len(self.config.datasets)} {dataset_label} "
            f"with {worker_count} {worker_label}"
        )

        live_render = Group(
            self._shared_status_progress,
            self._shared_overall_progress,
            self._shared_split_progress,
        )
        with Live(live_render, console=console, refresh_per_second=10):
            self._status_task_id = self._shared_status_progress.add_task(
                processing_status,
                total=self._progress_tracker.total_dialogues,
                kind="overall",
            )
            self._overall_task_id = self._shared_overall_progress.add_task(
                "Overall",
                total=self._progress_tracker.total_dialogues,
                kind="overall",
            )
            if worker_count == 1:
                stats, split_demographics = self._run_sequential_datasets(splits, console)
            else:
                stats, split_demographics = self._run_parallel_datasets(splits)

        # Calculate overall statistics across all splits
        overall_demographics = {"sex": {}, "cohort": {}}
        for split_stats in split_demographics.values():
            for sex, count in split_stats["sex"].items():
                overall_demographics["sex"][sex] = (
                    overall_demographics["sex"].get(sex, 0) + count
                )
            for cohort, count in split_stats["cohort"].items():
                overall_demographics["cohort"][cohort] = (
                    overall_demographics["cohort"].get(cohort, 0) + count
                )

        for split in splits:
            split_stats = split_demographics[split]
            total_dialogues = sum(split_stats["sex"].values())

            if total_dialogues == 0:
                continue

            # Calculate percentages
            stats_with_pct = {
                "total_dialogues": total_dialogues,
                "sex": {},
                "cohort": {},
            }

            # Sex distribution
            for sex, count in split_stats["sex"].items():
                pct = (count / total_dialogues) * 100
                stats_with_pct["sex"][sex] = {
                    "count": count,
                    "percentage": round(pct, 2),
                }

            # Cohort distribution
            for cohort, count in split_stats["cohort"].items():
                pct = (count / total_dialogues) * 100
                stats_with_pct["cohort"][cohort] = {
                    "count": count,
                    "percentage": round(pct, 2),
                }

            # Save to JSON file
            stats_file = demographics_dir / f"{split}_demographics.json"
            with open(stats_file, "w") as f:
                json.dump(stats_with_pct, f, indent=2)

        # Save and print overall statistics
        total_overall = sum(overall_demographics["sex"].values())
        if total_overall > 0:
            overall_stats_with_pct = {
                "total_dialogues": total_overall,
                "sex": {},
                "cohort": {},
            }

            # Sex distribution
            for sex, count in overall_demographics["sex"].items():
                pct = (count / total_overall) * 100
                overall_stats_with_pct["sex"][sex] = {
                    "count": count,
                    "percentage": round(pct, 2),
                }

            # Cohort distribution
            for cohort, count in overall_demographics["cohort"].items():
                pct = (count / total_overall) * 100
                overall_stats_with_pct["cohort"][cohort] = {
                    "count": count,
                    "percentage": round(pct, 2),
                }

            # Save to JSON file
            overall_stats_file = demographics_dir / "overall_demographics.json"
            with open(overall_stats_file, "w") as f:
                json.dump(overall_stats_with_pct, f, indent=2)

        self._shared_status_progress = None
        self._shared_split_progress = None
        self._shared_overall_progress = None
        self._status_task_id = None
        self._overall_task_id = None
        return stats

    def _get_total_dialogues(self, splits: list[str]) -> int:
        return sum(
            self._get_dataset_total(dataset, split)
            for dataset in self.config.datasets
            for split in splits
        )

    def _make_worker_pipeline(self) -> "AugmentationPipeline":
        worker = AugmentationPipeline(replace(self.config))
        worker.llm_usage_tracker = self.llm_usage_tracker
        worker.emotion_tagger = EmotionTagger(
            model=worker.config.model,
            usage_tracker=worker.llm_usage_tracker,
        )
        worker._progress_tracker = self._progress_tracker
        worker._shared_split_progress = self._shared_split_progress
        worker._shared_overall_progress = self._shared_overall_progress
        worker._overall_task_id = self._overall_task_id
        overridden_process_dataset = self.__dict__.get("process_dataset")
        if overridden_process_dataset is not None:
            worker.process_dataset = overridden_process_dataset
        return worker

    def _run_single_dataset(self, dataset: str, splits: list[str]) -> dict:
        worker = self._make_worker_pipeline()
        dataset_dir = self.config.output_dir / ".tmp" / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        split_demographics = {split: {"sex": {}, "cohort": {}} for split in splits}
        total = 0
        split_paths: dict[str, Path] = {}

        try:
            for split in splits:
                split_path = dataset_dir / f"{split}.jsonl"
                split_paths[split] = split_path
                with split_path.open("w", encoding="utf-8") as f:
                    try:
                        for augmented in worker.process_dataset(dataset, split, console=None):
                            augmented_dict = augmented.to_dict()
                            original_id = augmented_dict.get("dialogue_id", "")
                            augmented_dict["dialogue_id"] = f"{dataset}_{original_id}"

                            speaker = augmented_dict.get("speaker", {})
                            if speaker:
                                sex = speaker.get("sex", "unknown")
                                cohort = speaker.get("cohort", "unknown")
                                split_demographics[split]["sex"][sex] = (
                                    split_demographics[split]["sex"].get(sex, 0) + 1
                                )
                                split_demographics[split]["cohort"][cohort] = (
                                    split_demographics[split]["cohort"].get(cohort, 0) + 1
                                )

                            f.write(json.dumps(augmented_dict) + "\n")
                            total += 1
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        raise RuntimeError(
                            f"Dataset '{dataset}' split '{split}' failed: {exc}"
                        ) from exc
        finally:
            worker.close()

        return {
            "dataset": dataset,
            "total": total,
            "split_paths": split_paths,
            "split_demographics": split_demographics,
            "temp_dir": dataset_dir,
        }

    def _merge_dataset_results(
        self,
        result: dict,
        split_files: dict[str, Any],
        split_demographics: dict[str, dict[str, dict[str, int]]],
    ) -> int:
        for split, split_path in result["split_paths"].items():
            if split not in split_files or not split_path.exists():
                continue
            contents = split_path.read_text(encoding="utf-8")
            if contents:
                split_files[split].write(contents)

            for sex, count in result["split_demographics"][split]["sex"].items():
                split_demographics[split]["sex"][sex] = (
                    split_demographics[split]["sex"].get(sex, 0) + count
                )
            for cohort, count in result["split_demographics"][split]["cohort"].items():
                split_demographics[split]["cohort"][cohort] = (
                    split_demographics[split]["cohort"].get(cohort, 0) + count
                )

        shutil.rmtree(result["temp_dir"], ignore_errors=True)
        temp_root = self.config.output_dir / ".tmp"
        try:
            temp_root.rmdir()
        except OSError:
            pass
        return result["total"]

    def _run_sequential_datasets(
        self,
        splits: list[str],
        console: Console,
    ) -> tuple[dict[str, int], dict[str, dict[str, dict[str, int]]]]:
        stats = {}
        split_demographics = {split: {"sex": {}, "cohort": {}} for split in splits}
        split_files = {
            split: (self.config.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in splits
        }

        try:
            for dataset in self.config.datasets:
                if self._shared_split_progress is None:
                    console.print()
                    console.rule(f"[bold cyan]{dataset}[/]")
                result = self._run_single_dataset(dataset, splits)
                stats[dataset] = self._merge_dataset_results(
                    result,
                    split_files,
                    split_demographics,
                )
        finally:
            for f in split_files.values():
                f.close()

        return stats, split_demographics

    def _run_parallel_datasets(
        self,
        splits: list[str],
    ) -> tuple[dict[str, int], dict[str, dict[str, dict[str, int]]]]:
        stats = {}
        split_demographics = {split: {"sex": {}, "cohort": {}} for split in splits}
        split_files = {
            split: (self.config.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in splits
        }

        max_workers = max(1, min(self.config.workers, len(self.config.datasets)))
        futures = {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for dataset in self.config.datasets:
                    futures[dataset] = executor.submit(self._run_single_dataset, dataset, splits)

                for dataset in self.config.datasets:
                    result = futures[dataset].result()
                    stats[dataset] = self._merge_dataset_results(
                        result,
                        split_files,
                        split_demographics,
                    )
        finally:
            for f in split_files.values():
                f.close()

        return stats, split_demographics

    def _apply_crossturn(
        self,
        turns: list[dict],
        state: dict,
        dataset: str,
        metadata: dict | None = None,
    ) -> list[dict]:
        """Apply cross-turn slot segmentation."""
        segment_state = self._get_segmentable_state(state, dataset, metadata or {})
        slots = find_segmentable_slots(segment_state, dataset)

        if not slots:
            return turns

        segmentable_values = {name: value for name, value, _ in slots}
        segmentable_types = {name: slot_type for name, _, slot_type in slots}
        segmentable_names = set(segmentable_values)

        new_turns = []
        prev_state = {}
        i = 0

        while i < len(turns):
            turn = turns[i]
            if turn.get("role") != "user":
                new_turns.append(turn)
                prev_state = turn.get("state", prev_state)
                i += 1
                continue

            spans = turn.get("slots") or []
            ordered_slots = []
            for span in sorted(
                spans,
                key=lambda s: s.start if hasattr(s, "start") else s.get("start", 0),
            ):
                slot_name = getattr(span, "slot", None) or span.get("slot")
                if not slot_name:
                    continue
                if slot_name in segmentable_names and slot_name not in ordered_slots:
                    ordered_slots.append(slot_name)

            if not ordered_slots:
                new_turns.append(turn)
                prev_state = turn.get("state", prev_state)
                i += 1
                continue

            original_asst = None
            if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                original_asst = turns[i + 1]

            prev_state = new_turns[-1].get("state", {}) if new_turns else prev_state

            for slot_idx, slot_name in enumerate(ordered_slots):
                slot_value = segmentable_values[slot_name]
                slot_type = segmentable_types[slot_name]
                crossturn_turns = generate_crossturn_dialogue(
                    slot_name, slot_value, slot_type
                )
                use_original_asst = (
                    slot_idx == len(ordered_slots) - 1 and original_asst is not None
                )

                cumulative_value = ""
                for ct_idx, ct in enumerate(crossturn_turns):
                    if (
                        ct.role == "user"
                        and ct.segment
                        and not ct.segment.get("is_correction")
                    ):
                        seg_val = ct.segment.get("value", "")
                        cumulative_value = (
                            (cumulative_value + " " + seg_val).strip()
                            if cumulative_value
                            else seg_val
                        )

                    turn_state = {
                        k: dict(v) if isinstance(v, dict) else v
                        for k, v in prev_state.items()
                    }

                    if (
                        ct.segment
                        and ct.segment.get("idx", 0) == ct.segment.get("total", 1) - 1
                    ):
                        self._set_slot_in_state(turn_state, slot_name, slot_value)
                    elif ct.role == "user" and cumulative_value:
                        self._set_slot_in_state(
                            turn_state, slot_name, f"{cumulative_value}..."
                        )

                    is_last_ct = ct_idx == len(crossturn_turns) - 1
                    if is_last_ct and ct.role == "assistant" and use_original_asst:
                        new_turns.append(
                            {
                                "role": "assistant",
                                "text": original_asst.get("text", ""),
                                "tagged": original_asst.get("tagged"),
                                "emotion": original_asst.get("emotion"),
                                "disfluency": original_asst.get("disfluency"),
                                "segment": None,
                                "state": turn_state,
                            }
                        )
                    else:
                        new_turns.append(
                            {
                                "role": ct.role,
                                "text": ct.text,
                                "segment": ct.segment,
                                "state": turn_state,
                            }
                        )

                    prev_state = turn_state

            i += 2 if original_asst is not None else 1

        return new_turns

    def _get_segmentable_state(
        self,
        state: dict,
        dataset: str,
        metadata: dict,
    ) -> dict:
        """Prefer clean scenario values for segmentation when available."""
        if dataset != "abcd":
            return state
        scenario = metadata.get("scenario")
        if not scenario:
            return state
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})
        merged = {"customer_service": {}}
        merged["customer_service"].update(personal)
        merged["customer_service"].update(order)
        return merged

    def _set_slot_in_state(self, state: dict, slot_name: str, value: str) -> None:
        if "." in slot_name:
            parts = slot_name.split(".", 1)
            domain = parts[0]
            slot = parts[1] if len(parts) > 1 else slot_name
        else:
            domain = "general"
            slot = slot_name

        if domain not in state:
            state[domain] = {}
        state[domain][slot] = value

    def _validate_dialogue(self, dialogue: AugmentedDialogue) -> None:
        """Validate dialogue structure and speaker constraints.

        Checks:
        1. Turns alternate between user and assistant
        2. Assistant speaker is from Native category
        3. Assistant and User are different speakers

        Raises:
            ValueError: If validation fails
        """
        is_spokenwoz = dialogue.source == "spokenwoz"
        prev_role = None
        for i, turn in enumerate(dialogue.turns):
            if turn.role == prev_role:
                raise ValueError(
                    f"Dialogue {dialogue.id}: consecutive {turn.role} turns at positions {i} and {i+1}. "
                    f"Turn {i}: '{turn.text[:50]}...', Turn {i+1}: '{dialogue.turns[i].text[:50]}...'"
                )
            prev_role = turn.role

            if turn.role == "user":
                if is_spokenwoz:
                    if (
                        turn.disfluency
                        or turn.segment is not None
                        or turn.bargein is not None
                        or turn.tagged is not None
                    ):
                        raise ValueError(
                            f"Dialogue {dialogue.id} from {dialogue.source}: user turn at position {i} "
                            "has forbidden augmentation fields (disfluency/segment/bargein/tagged)"
                        )
                if turn.text.strip() != "<|endoftext|>" and turn.emotion is None:
                    raise ValueError(
                        f"Dialogue {dialogue.id}: user turn at position {i} missing emotion"
                    )
                if turn.disfluency:
                    tagged_text = turn.tagged or turn.text or ""
                    for dis in turn.disfluency:
                        dtype = dis.get("type")
                        if not dtype:
                            continue
                        if dtype in {"FP", "DM", "EDIT"}:
                            pattern = rf"(?:^|\s)\[{dtype}\]\s+\S"
                            if not re.search(pattern, tagged_text):
                                raise ValueError(
                                    f"Dialogue {dialogue.id} from {dialogue.source}: {dtype} tag must precede inserted text in user turn {i}. "
                                    f"Tagged text: '{tagged_text}'"
                                )
                        elif dtype in {"REP", "COR", "RST"}:
                            pattern = rf"\S+\s+\[{dtype}\]\s+\S"
                            if not re.search(pattern, tagged_text):
                                raise ValueError(
                                    f"Dialogue {dialogue.id} from {dialogue.source}: {dtype} tag must be in the middle of user turn {i}. "
                                    f"Tagged text: '{tagged_text}'"
                                )
            elif turn.role == "assistant":
                if is_spokenwoz and turn.bargein is not None:
                    raise ValueError(
                        f"Dialogue {dialogue.id} from {dialogue.source}: assistant turn at position {i} "
                        "has forbidden bargein field"
                    )
                if (
                    turn.tagged is not None
                    or turn.emotion is not None
                    or turn.disfluency is not None
                    or turn.segment is not None
                    or turn.state is not None
                    or turn.audio_path is not None
                    or (turn.slots and len(turn.slots) > 0)
                ):
                    raise ValueError(
                        f"Dialogue {dialogue.id} from {dialogue.source}: assistant turn at position {i} has forbidden fields"
                    )

        # 2. Speaker Verification
        if dialogue.assistant_speaker:
            # Check 1: Assistant must be Native
            category = dialogue.assistant_speaker.get("category")
            if category != "Native":
                raise ValueError(
                    f"Dialogue {dialogue.id} from {dialogue.source}: Assistant speaker category must be 'Native', "
                    f"but got '{category}'"
                )

            # Check 2: IDs must differ
            if dialogue.speaker:
                user_id = dialogue.speaker.get("filename")
                asst_id = dialogue.assistant_speaker.get("filename")
                if user_id and asst_id and user_id == asst_id:
                    raise ValueError(
                        f"Dialogue {dialogue.id} from {dialogue.source}: User and Assistant share same speaker ID '{user_id}'"
                    )


def inject_endoftext(turns: list[dict]) -> list[dict]:
    """Append one end-of-text marker after the dialogue's final turn."""
    if not turns:
        return [{"role": "user", "text": "<|endoftext|>"}]

    last = turns[-1]
    if last.get("role") == "user":
        new_last = dict(last)
        text = new_last.get("text", "")
        if text.endswith("<|endoftext|>"):
            return turns
        new_last["text"] = f"{text}<|endoftext|>"
        return turns[:-1] + [new_last]

    return turns + [{"role": "user", "text": "<|endoftext|>"}]
