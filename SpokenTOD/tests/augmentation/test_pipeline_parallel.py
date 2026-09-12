import json
import threading
import time

from click.testing import CliRunner
from rich.progress import Progress

import augment
import augmentation.pipeline as pipeline_module
from augmentation.pipeline import AugmentationPipeline, PipelineConfig, inject_endoftext
from augmentation.schema import AugmentedDialogue, Dialogue, Goal, StructuredGoal, Turn


class DummyEmotionTagger:
    def __init__(self, *args, **kwargs):
        pass


class DummyDemographicSampler:
    def __init__(self, *args, **kwargs):
        pass


def test_inject_endoftext_adds_control_turn_when_dialogue_ends_with_assistant():
    turns = [
        {"role": "user", "text": "Thank you."},
        {"role": "assistant", "text": "You're welcome."},
    ]

    result = inject_endoftext(turns)

    assert result == [
        {"role": "user", "text": "Thank you."},
        {"role": "assistant", "text": "You're welcome."},
        {"role": "user", "text": "<|endoftext|>"},
    ]


def _make_augmented_dialogue(dataset: str, split: str) -> AugmentedDialogue:
    return AugmentedDialogue(
        id=f"{split}-dlg",
        source=dataset,
        goal={"text": "", "structured": {}},
        turns=[Turn(role="user", text=f"{dataset}-{split}")],
        state={},
        speaker={"sex": "female", "cohort": "20s"},
        assistant_speaker={"speaker_id": "assistant-1"},
        metadata={},
    )


def test_run_processes_datasets_in_parallel_and_merges_split_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(pipeline_module, "DemographicSampler", DummyDemographicSampler)
    monkeypatch.setattr(AugmentationPipeline, "_get_dataset_total", lambda self, dataset, split: 1)

    config = PipelineConfig(
        model="gpt-4.1-mini",
        datasets=["emowoz", "sgd"],
        data_dir=tmp_path / "datasets",
        output_dir=tmp_path / "out",
        workers=2,
        chunk_size=1,
    )
    pipeline = AugmentationPipeline(config)

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def fake_process_dataset(dataset: str, split: str = "train", console=None):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.1)
        try:
            yield _make_augmented_dialogue(dataset, split)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(pipeline, "process_dataset", fake_process_dataset)

    stats = pipeline.run(splits=["train", "test"])

    assert state["max_active"] >= 2
    assert stats == {"emowoz": 2, "sgd": 2}

    train_records = [
        json.loads(line)
        for line in (config.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_records = [
        json.loads(line)
        for line in (config.output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert {record["dialogue_id"] for record in train_records} == {
        "emowoz_train-dlg",
        "sgd_train-dlg",
    }
    assert {record["dialogue_id"] for record in test_records} == {
        "emowoz_test-dlg",
        "sgd_test-dlg",
    }
    assert not (config.output_dir / ".tmp").exists()


def test_progress_tracker_and_eta_columns_show_values_without_completed_samples(monkeypatch):
    tracker = pipeline_module._RunProgressTracker(total_dialogues=10)
    tracker.start_time = 100.0
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 105.0)
    usage_tracker = pipeline_module.LLMUsageTracker()
    usage_tracker.record_response(
        type(
            "Resp",
            (),
            {
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                }
            },
        )(),
        model="openai/gpt-4.1-mini",
        stage="emotion",
    )

    current_column = pipeline_module._CurrentETAColumn()
    tokens_column = pipeline_module._OverallTokensColumn(usage_tracker)
    cost_column = pipeline_module._OverallCostColumn(usage_tracker)
    overall_column = pipeline_module._OverallETAColumn(tracker)
    finish_column = pipeline_module._ExpectedFinishColumn(tracker)

    with Progress(current_column, tokens_column, cost_column, overall_column, finish_column) as progress:
        progress.add_task("Overall", total=10, kind="overall")
        progress.add_task("sgd/train", total=10, kind="split")
        overall_task = progress.tasks[0]
        split_task = progress.tasks[1]
        split_task.start_time = progress.get_time() - 5.0

        assert current_column.render(overall_task).plain == ""
        assert tokens_column.render(overall_task).plain == "tokens 20"
        assert cost_column.render(overall_task).plain == "cost $0.000000"
        assert overall_column.render(overall_task).plain == "overall 0:00:50"
        assert finish_column.render(overall_task).plain.startswith("finish ")
        assert finish_column.render(overall_task).plain.endswith("02:35")
        assert current_column.render(split_task).plain == "current 0:00:50"
        assert tokens_column.render(split_task).plain == ""
        assert cost_column.render(split_task).plain == ""
        assert overall_column.render(split_task).plain == ""
        assert finish_column.render(split_task).plain == ""


def test_process_dataset_updates_shared_overall_task(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(pipeline_module, "DemographicSampler", DummyDemographicSampler)

    config = PipelineConfig(
        model="gpt-4.1-mini",
        datasets=["sgd"],
        data_dir=tmp_path / "datasets",
        output_dir=tmp_path / "out",
        workers=1,
        chunk_size=1,
    )
    pipeline = AugmentationPipeline(config)

    monkeypatch.setattr(
        pipeline,
        "_iter_dataset_split",
        lambda dataset, split: iter(
            [_make_augmented_dialogue(dataset, split), _make_augmented_dialogue(dataset, split)]
        ),
    )
    monkeypatch.setattr(pipeline, "_get_dataset_total", lambda dataset, split: 2)

    with Progress() as split_progress, Progress() as overall_progress:
        overall_task_id = overall_progress.add_task("Overall", total=2, kind="overall")
        pipeline._shared_split_progress = split_progress
        pipeline._shared_overall_progress = overall_progress
        pipeline._overall_task_id = overall_task_id

        results = list(pipeline.process_dataset("sgd", "train"))

    split_task = next(task for task in split_progress.tasks if task.description == "sgd/train")
    overall_task = next(task for task in overall_progress.tasks if task.description == "Overall")

    assert len(results) == 2
    assert split_task.completed == 2
    assert overall_task.completed == 2


def test_get_dataset_total_skips_loader_len_when_sample_size_is_set(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(pipeline_module, "DemographicSampler", DummyDemographicSampler)

    config = PipelineConfig(
        model="gpt-4.1-mini",
        datasets=["sgd"],
        data_dir=tmp_path / "datasets",
        output_dir=tmp_path / "out",
        workers=1,
        chunk_size=1,
        sample_size=3,
    )
    pipeline = AugmentationPipeline(config)

    class LoaderWithoutLen:
        def __len__(self):
            raise AssertionError("len(loader) should not be called when sample_size is set")

    monkeypatch.setattr(pipeline, "_get_loader", lambda dataset, split: LoaderWithoutLen())

    assert pipeline._get_dataset_total("sgd", "train") == 3


def test_process_dataset_adjusts_progress_totals_when_sample_is_shorter(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(pipeline_module, "DemographicSampler", DummyDemographicSampler)

    config = PipelineConfig(
        model="gpt-4.1-mini",
        datasets=["sgd"],
        data_dir=tmp_path / "datasets",
        output_dir=tmp_path / "out",
        workers=1,
        chunk_size=1,
        sample_size=3,
    )
    pipeline = AugmentationPipeline(config)
    pipeline._progress_tracker = pipeline_module._RunProgressTracker(total_dialogues=3)

    monkeypatch.setattr(
        pipeline,
        "_iter_dataset_split",
        lambda dataset, split: iter([_make_augmented_dialogue(dataset, split)]),
    )

    with Progress() as split_progress, Progress() as overall_progress:
        overall_task_id = overall_progress.add_task("Overall", total=3, kind="overall")
        pipeline._shared_split_progress = split_progress
        pipeline._shared_overall_progress = overall_progress
        pipeline._overall_task_id = overall_task_id

        results = list(pipeline.process_dataset("sgd", "train"))

    split_task = next(task for task in split_progress.tasks if task.description == "sgd/train")
    overall_task = next(task for task in overall_progress.tasks if task.description == "Overall")

    assert len(results) == 1
    assert split_task.completed == 1
    assert split_task.total == 1
    assert overall_task.completed == 1
    assert overall_task.total == 1
    assert pipeline._progress_tracker.total_dialogues == 1


def test_augment_cli_clears_console_before_rendering(monkeypatch, tmp_path):
    captured = {"cleared": False}
    data_dir = tmp_path / "datasets"
    saa_root = data_dir / "SpeechAccentArchive"
    saa_root.mkdir(parents=True)
    (saa_root / "speakers_all.csv").write_text("speaker_id\n", encoding="utf-8")
    (saa_root / "recordings").mkdir()

    class FakeConsole:
        def clear(self):
            captured["cleared"] = True

        def print(self, *args, **kwargs):
            pass

    def fake_is_dataset_available(name, base_dir=None):
        return name == "saa"

    class FakePipeline:
        def __init__(self, config):
            pass

        def run(self, splits):
            return {"saa": 0}

    monkeypatch.setattr(augment, "Console", FakeConsole)
    monkeypatch.setattr(augment, "is_dataset_available", fake_is_dataset_available)
    monkeypatch.setattr(augment, "AugmentationPipeline", FakePipeline)

    result = CliRunner().invoke(
        augment.main,
        [
            "--datasets",
            "saa",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--chunk-size",
            "1",
            "--model",
            "gpt-4.1-mini",
        ],
    )

    assert result.exit_code == 0
    assert captured["cleared"] is True


def test_augment_cli_passes_workers_to_pipeline(monkeypatch, tmp_path):
    captured = {}
    data_dir = tmp_path / "datasets"
    saa_root = data_dir / "SpeechAccentArchive"
    saa_root.mkdir(parents=True)
    (saa_root / "speakers_all.csv").write_text("speaker_id\n", encoding="utf-8")
    (saa_root / "recordings").mkdir()

    def fake_is_dataset_available(name, base_dir=None):
        return name == "saa"

    class FakePipeline:
        def __init__(self, config):
            captured["workers"] = config.workers
            captured["chunk_size"] = config.chunk_size
            captured["with_selected_speakers"] = config.with_selected_speakers

        def run(self, splits):
            return {"saa": 0}

    monkeypatch.setattr(augment, "is_dataset_available", fake_is_dataset_available)
    monkeypatch.setattr(augment, "AugmentationPipeline", FakePipeline)

    result = CliRunner().invoke(
        augment.main,
        [
            "--datasets",
            "saa",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--chunk-size",
            "1",
            "--workers",
            "3",
            "--with_selected_speakers",
            "--model",
            "gpt-4.1-mini",
        ],
    )

    assert result.exit_code == 0
    assert captured["workers"] == 3
    assert captured["chunk_size"] == 1
    assert captured["with_selected_speakers"] is True


def test_pipeline_passes_selected_speakers_mode_and_data_directory(monkeypatch, tmp_path):
    captured = {}

    class CapturingDemographicSampler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(
        pipeline_module,
        "DemographicSampler",
        CapturingDemographicSampler,
    )

    data_dir = tmp_path / "datasets"
    AugmentationPipeline(
        PipelineConfig(
            data_dir=data_dir,
            with_selected_speakers=True,
        )
    )

    assert captured == {
        "base_dir": str(data_dir / "SpeechAccentArchive"),
        "balance_distribution": True,
        "with_selected_speakers": True,
    }


def test_process_batch_does_not_use_asyncio_run(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(pipeline_module, "DemographicSampler", DummyDemographicSampler)

    config = PipelineConfig(
        model="gpt-4.1-mini",
        datasets=["sgd"],
        data_dir=tmp_path / "datasets",
        output_dir=tmp_path / "out",
        workers=2,
        chunk_size=1,
    )
    pipeline = AugmentationPipeline(config)

    async def fake_tag_utterances_async(utterances, max_concurrency=50):
        return [{"label": 0, "name": "neutral"} for _ in utterances]

    async def fake_inject_bargein(dialogues_turns, **kwargs):
        return dialogues_turns

    async def fake_inject_disfluency(dialogues_turns, *args, **kwargs):
        return dialogues_turns

    monkeypatch.setattr(
        pipeline.emotion_tagger,
        "tag_utterances_async",
        fake_tag_utterances_async,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline.demographic_sampler,
        "sample_demographic",
        lambda: {"sex": "female", "cohort": "20s"},
        raising=False,
    )
    monkeypatch.setattr(
        pipeline.demographic_sampler,
        "find_speaker",
        lambda demo: {"sex": "female", "cohort": "20s"},
        raising=False,
    )
    monkeypatch.setattr(
        pipeline.demographic_sampler,
        "sample_assistant_speaker",
        lambda: {"speaker_id": "assistant-1", "category": "Native", "filename": "assistant-1.wav"},
        raising=False,
    )
    monkeypatch.setattr(pipeline_module, "inject_bargein_dialogues_batch_async", fake_inject_bargein)
    monkeypatch.setattr(pipeline_module, "inject_disfluency_dialogues_batch_async", fake_inject_disfluency)

    def fail_asyncio_run(*args, **kwargs):
        raise AssertionError("asyncio.run should not be used inside worker processing")

    monkeypatch.setattr(pipeline_module.asyncio, "run", fail_asyncio_run)

    dialogue = Dialogue(
        id="dlg-1",
        source="sgd",
        turns=[
            {"role": "user", "text": "hello", "slots": [], "state": {}},
            {"role": "assistant", "text": "hi", "slots": [], "state": {}},
        ],
        goal=Goal(text="", structured=StructuredGoal(domains=[], intents=[])),
        state={},
        metadata={},
    )

    results = pipeline._process_batch([dialogue])

    assert len(results) == 1


def test_close_flushes_and_stops_litellm_logging_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "EmotionTagger", DummyEmotionTagger)
    monkeypatch.setattr(pipeline_module, "DemographicSampler", DummyDemographicSampler)

    config = PipelineConfig(
        model="gpt-4.1-mini",
        datasets=["sgd"],
        data_dir=tmp_path / "datasets",
        output_dir=tmp_path / "out",
        workers=1,
        chunk_size=1,
    )
    pipeline = AugmentationPipeline(config)
    loop = pipeline._get_async_loop()
    events = []

    class FakeLoggingWorker:
        def __init__(self, bound_loop):
            self._bound_loop = bound_loop

        async def flush(self):
            events.append("flush")

        async def stop(self):
            events.append("stop")

    monkeypatch.setattr(
        pipeline_module,
        "GLOBAL_LOGGING_WORKER",
        FakeLoggingWorker(loop),
        raising=False,
    )

    pipeline.close()

    assert events == ["flush", "stop"]
