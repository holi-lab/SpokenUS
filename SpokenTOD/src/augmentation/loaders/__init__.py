"""Data loaders for each dataset."""

from pathlib import Path

from augmentation.constants import DATASETS
from augmentation.loaders.abcd import ABCDLoader
from augmentation.loaders.emowoz import EmoWOZLoader
from augmentation.loaders.sgd import SGDLoader
from augmentation.loaders.spokenwoz import SpokenWOZLoader
from augmentation.loaders.tm2 import TM2Loader

DATASET_DIRS = {
    "emowoz": "EmoWOZ",
    "sgd": "dstc8-schema-guided-dialogue",
    "abcd": "abcd",
    "tm2": "TM-2-2020",
    "spokenwoz": "SpokenWOZ",
    "saa": "SpeechAccentArchive",
}


def _resolve_dataset_root(name: str, base_dir: Path | None) -> Path | None:
    dataset_dir = DATASET_DIRS.get(name)
    if dataset_dir is None:
        if name not in DATASETS:
            return None
        dataset_dir = name
    if base_dir is None:
        return Path(dataset_dir)
    base_dir = Path(base_dir)
    # If caller already passed the dataset root, use it.
    if base_dir.name == dataset_dir:
        return base_dir
    # If caller passed datasets/<name> but dataset dir differs, use parent/<dataset_dir> if it exists.
    if base_dir.name == name and dataset_dir != name:
        parent_candidate = base_dir.parent / dataset_dir
        if parent_candidate.exists():
            return parent_candidate
    candidate = base_dir / dataset_dir
    if candidate.exists():
        return candidate
    return candidate


def is_dataset_available(name: str, base_dir: Path | None = None) -> bool:
    """Check dataset availability with per-dataset requirements."""
    dataset_root = _resolve_dataset_root(name, base_dir)
    if dataset_root is None:
        return False

    if name == "tm2":
        return (dataset_root / "data").is_dir()
    if name == "sgd":
        return all(
            (dataset_root / split).is_dir() for split in ("dev", "test", "train")
        )
    if name == "abcd":
        data_dir = dataset_root / "data"
        return data_dir.is_dir() and (
            (data_dir / "abcd_v1.1.json").is_file()
            or (data_dir / "abcd_v1.1json").is_file()
        )
    if name == "spokenwoz":
        return all(
            [
                (dataset_root / "audio").is_dir(),
                (dataset_root / "train").is_dir(),
                (dataset_root / "test").is_dir(),
                (dataset_root / "train.json").is_file(),
                (dataset_root / "test.json").is_file(),
            ]
        )
    if name == "emowoz":
        emowoz_file = dataset_root / "emowoz-multiwoz.json"
        multiwoz_root = dataset_root.parent / "MultiWOZ_2.1"
        multiwoz_file = multiwoz_root / "data.json"
        return emowoz_file.is_file() and multiwoz_file.is_file()
    if name == "saa":
        saa_file = dataset_root / "speakers_all.csv"
        recordings = dataset_root / "recordings"
        return saa_file.is_file() and recordings.is_dir()
    return dataset_root.is_dir()


def get_loader(name: str, base_dir: Path | None = None, split: str = "train"):
    """Return the dataset loader for a given dataset name."""
    dataset_root = _resolve_dataset_root(name, base_dir)
    if dataset_root is None:
        raise ValueError(f"Unknown dataset: {name}")

    if name == "emowoz":
        return EmoWOZLoader(dataset_root, split=split)
    if name == "sgd":
        return SGDLoader(dataset_root, split=split)
    if name == "abcd":
        return ABCDLoader(dataset_root, split=split)
    if name == "tm2":
        return TM2Loader(dataset_root, split=split)
    if name == "spokenwoz":
        return SpokenWOZLoader(dataset_root, split=split)
    raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "DATASET_DIRS",
    "get_loader",
    "is_dataset_available",
]
