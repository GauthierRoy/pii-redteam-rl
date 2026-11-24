"""Pause training between steps while the GPU runs hot (a workaround for the RTX 3090).

The roommate's RTX 3090 fell off the bus four times under sustained load. On 2026-11-22 a 2-s
log showed thermal slowdown flags on from the first minute while the core never passed 76 C,
so a sensor nvidia-smi does not report (most likely the GDDR6X memory junction) is at its
limit. Without root the power limit and clocks cannot be changed, but the job can idle:
a background thread samples `nvidia-smi` every `poll_seconds` during the step, and when any
sample was hot (a thermal clock-event flag, or the core at or above `max_temp_c`) the
callback pauses after that step for at least `min_pause_seconds` and until the core is at or
below `resume_temp_c`, at most `max_pause_seconds`. Nothing here needs root.

Config (all optional except enabling the block):
  gpu_cooldown: {max_temp_c: 70, resume_temp_c: 50, min_pause_seconds: 10,
                 max_pause_seconds: 120, poll_seconds: 1}
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable

from transformers import TrainerCallback

# clocks_event_reasons bits: 0x08 hardware slowdown, 0x20 software thermal, 0x40 hardware thermal
THERMAL_REASONS = 0x08 | 0x20 | 0x40
QUERY = "temperature.gpu,clocks_event_reasons.active"


def read_nvidia_smi() -> tuple[int, int] | None:
    """(core temperature in C, clocks_event_reasons bitmask), or None if the read fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        temp, reasons = (field.strip() for field in out.splitlines()[0].split(","))
        return int(temp), int(reasons, 16)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class GpuCooldown(TrainerCallback):
    def __init__(
        self,
        max_temp_c: int = 70,
        resume_temp_c: int = 50,
        min_pause_seconds: float = 10.0,
        max_pause_seconds: float = 120.0,
        poll_seconds: float = 1.0,
        reader: Callable[[], tuple[int, int] | None] = read_nvidia_smi,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        background: bool = True,
    ) -> None:
        self.max_temp_c = max_temp_c
        self.resume_temp_c = resume_temp_c
        self.min_pause_seconds = min_pause_seconds
        self.max_pause_seconds = max_pause_seconds
        self.poll_seconds = poll_seconds
        self.reader = reader
        self.sleep = sleep
        self.clock = clock
        self.hot_seen = threading.Event()
        self.pauses = 0
        self.paused_seconds = 0.0
        self.max_temp_seen = 0
        self.hot_samples = 0
        self.samples = 0
        self.read_failures = 0
        self._stop = threading.Event()
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def is_hot(self, reading: tuple[int, int]) -> bool:
        temp, reasons = reading
        return temp >= self.max_temp_c or bool(reasons & THERMAL_REASONS)

    def sample(self) -> None:
        reading = self.reader()
        if reading is None:
            self.read_failures += 1
            return
        self.samples += 1
        self.max_temp_seen = max(self.max_temp_seen, reading[0])
        if self.is_hot(reading):
            self.hot_samples += 1
            self.hot_seen.set()

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.sample()

    def cool_down(self) -> float:
        """Pause if the GPU ran hot since the last call; returns the seconds paused."""
        if not self.hot_seen.is_set():
            return 0.0
        started = self.clock()
        while True:
            waited = self.clock() - started
            if waited >= self.max_pause_seconds:
                break
            if waited >= self.min_pause_seconds:
                reading = self.reader()
                if reading is None or (
                    reading[0] <= self.resume_temp_c and not reading[1] & THERMAL_REASONS
                ):
                    break
            self.sleep(self.poll_seconds)
        self.hot_seen.clear()
        paused = self.clock() - started
        self.pauses += 1
        self.paused_seconds += paused
        return paused

    def on_step_end(self, args, state, control, **kwargs):
        self.cool_down()
        return control

    def close(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        return {
            "pauses": self.pauses,
            "paused_seconds": round(self.paused_seconds, 1),
            "samples": self.samples,
            "hot_samples": self.hot_samples,
            "max_core_temp_c": self.max_temp_seen,
            "read_failures": self.read_failures,
        }


def cooldown_from_config(config: dict) -> GpuCooldown | None:
    block = config.get("gpu_cooldown")
    if not block:
        return None
    return GpuCooldown(**{k: v for k, v in block.items() if k != "enabled"})
