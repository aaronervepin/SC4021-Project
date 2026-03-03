"""Handles saving crawled data to CSV and tracking progress."""

import json
import os

import pandas as pd

import config


class DataStore:
    """Stores crawled records in CSV with dedup and progress tracking."""

    def __init__(
        self,
        csv_path: str = config.RAW_CSV_PATH,
        progress_path: str = config.PROGRESS_PATH,
    ):
        self.csv_path = csv_path
        self.progress_path = progress_path
        self._seen_ids: set[str] = set()
        self._buffer: list[dict] = []
        self._buffer_flush_size = 100

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._load_existing_ids()

    def _load_existing_ids(self) -> None:
        """Load IDs from existing CSV for dedup on resume."""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path, usecols=["id"], dtype=str)
            self._seen_ids = set(df["id"].tolist())

    def has_id(self, record_id: str) -> bool:
        return record_id in self._seen_ids

    def add_record(self, record: dict) -> bool:
        """Add a record to the buffer. Returns True if added, False if duplicate."""
        rid = record["id"]
        if rid in self._seen_ids:
            return False
        self._seen_ids.add(rid)
        self._buffer.append(record)
        if len(self._buffer) >= self._buffer_flush_size:
            self.flush()
        return True

    def flush(self) -> None:
        """Write buffered records to CSV on disk."""
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer, columns=config.CSV_COLUMNS)
        write_header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=write_header, index=False)
        self._buffer.clear()

    @property
    def total_records(self) -> int:
        return len(self._seen_ids)

    # --- Progress tracking ---

    def load_progress(self) -> dict:
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r") as f:
                return json.load(f)
        return {"completed_tasks": []}

    def save_progress(self, progress: dict) -> None:
        os.makedirs(os.path.dirname(self.progress_path) or ".", exist_ok=True)
        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def load_all(self) -> pd.DataFrame:
        """Load the full dataset as a DataFrame."""
        self.flush()
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=config.CSV_COLUMNS)
