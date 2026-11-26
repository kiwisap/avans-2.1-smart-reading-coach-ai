"""Retraining while the service runs.

Training takes a few seconds and little memory, so it runs in a worker thread of this process.
The current model keeps answering until the new one is ready; then it is swapped in one step.
A failed training leaves the current model in place.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from reading_coach import config
from reading_coach.hybrid import HybridRecommender
from reading_coach.sources import MongoSettings
from reading_coach.train import train

log = logging.getLogger("reading_coach.training")


@dataclass
class Trainer:
    model_path: Path
    on_model: Callable[[HybridRecommender], None]
    dataset: Path = config.RAW_DATASET
    running: bool = False
    last_finished: datetime | None = None
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _task: asyncio.Task | None = None

    def start(self, reason: str) -> bool:
        """Starts a training in the background. False when one is already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
        log.info("Training a new model (%s)", reason)
        self._task = asyncio.get_running_loop().create_task(asyncio.to_thread(self._run))
        return True

    def _run(self) -> None:
        try:
            result = train(
                self.dataset,
                MongoSettings.from_env(),
                evaluate=False,
                model_path=self.model_path,
                log=log.info,
            )
            self.on_model(result.model)
            self.last_error = None
            log.info("New model %s in use", result.model.version)
        except Exception as error:  # the service keeps running with the model it has
            self.last_error = str(error)
            log.exception("Training failed, the current model stays in use")
        finally:
            self.last_finished = datetime.now(timezone.utc)
            with self._lock:
                self.running = False

    async def wait(self) -> None:
        """For tests and shutdown: wait until the running training is done."""
        if self._task is not None:
            await self._task
