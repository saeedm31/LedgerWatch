"""Settings and paths. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MONTHS = [f"{y}-{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 11)]]
TRAIN_MONTHS = MONTHS[:9]  # Nov 2025 - Jul 2026: models learn here
TEST_MONTHS = MONTHS[9:]  # Aug - Oct 2026: never seen in training, used to measure
CURRENT_CLOSE = MONTHS[-1]  # the close the app opens on
FLAGS_PER_1000 = float(os.getenv("LW_FLAGS_PER_1000", "6"))  # review capacity: flags per 1,000 entries
INVESTIGATE_TOP = int(os.getenv("LW_INVESTIGATE_TOP", "25"))  # the agent investigates this many per close
MAX_AGENT_STEPS = int(os.getenv("LW_MAX_AGENT_STEPS", "6"))
PATTERN_DAYS = int(os.getenv("LW_PATTERN_DAYS", "180"))  # a known pattern expires after this many days


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("LW_DATA_DIR", ROOT / "data")))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))  # or "bedrock"
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-central-1"))
    agent_model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "claude-sonnet-4-6"))
    reader_model: str = field(default_factory=lambda: os.getenv("READER_MODEL", "claude-haiku-4-5"))
    app_url: str = field(default_factory=lambda: os.getenv("APP_URL", "").rstrip("/"))

    @property
    def app_db(self) -> Path:
        """Flags, decisions, memory: the app's own data, separate from the ledger."""
        return self.data_dir / "app.db"

    @property
    def audit_db(self) -> Path:
        return self.data_dir / "audit.db"

    @property
    def labels(self) -> Path:
        """Planted problems (the answer key). Only the evaluation reads it; no tool serves it."""
        return self.data_dir / "labels.json"

    @property
    def models(self) -> Path:
        return self.data_dir / "models"
