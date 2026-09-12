"""Base loader interface for datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue


class BaseLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, data_dir: Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass

    @abstractmethod
    def load(self) -> Iterator[Dialogue]:
        """Yield dialogues from the dataset."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of dialogues in the split."""
        pass

    @abstractmethod
    def _extract_goal(self, raw: dict) -> dict:
        """Extract user goal from raw dialogue data.
        
        Returns:
            {"text": str, "structured": dict}
        """
        pass

    def _get_split_file(self, pattern: str) -> Path:
        """Get file path for current split."""
        return self.data_dir / pattern.format(split=self.split)
