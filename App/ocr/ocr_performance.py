"""Low-overhead per-job wall/CPU profiling for Torgsoft OCR.

No external packages; process CPU time accumulates across native ONNX/OpenCV
threads, thread CPU time covers only the invoking Python thread.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter_ns, process_time_ns, thread_time_ns
from typing import Any


VERSION = "ocr-profiling-fastpath-v1"


def _enabled(root: Path) -> bool:
    # Per-file .env is read once per OCR job; process env wins.
    from dotenv import dotenv_values
    values = dotenv_values(root / ".env", encoding="utf-8-sig") if (root / ".env").is_file() else {}
    flag = os.environ.get("OCR_PERF_PROFILE_ENABLED", values.get("OCR_PERF_PROFILE_ENABLED", "1"))
    return str(flag).strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class OCRPerformance:
    enabled: bool = True
    attempt: str = "normal"
    samples: list[dict[str, Any]] = field(default_factory=list)
    started_wall: int = field(default_factory=perf_counter_ns)
    started_process: int = field(default_factory=process_time_ns)
    outcome: str = "running"

    def save(self, directory: Path) -> None:
        if not self.enabled:
            return
        wall_ms = round((perf_counter_ns() - self.started_wall) / 1_000_000, 3)
        process_ms = round((process_time_ns() - self.started_process) / 1_000_000, 3)
        grouped: dict[str, dict] = {}
        for sample in self.samples:
            bucket = grouped.setdefault(sample["name"], {
                "calls": 0, "wall_ms": 0.0, "process_cpu_ms": 0.0, "thread_cpu_ms": 0.0
            })
            bucket["calls"] += 1
            for key in ("wall_ms", "process_cpu_ms", "thread_cpu_ms"):
                bucket[key] += sample[key]
        for bucket in grouped.values():
            for key in ("wall_ms", "process_cpu_ms", "thread_cpu_ms"):
                bucket[key] = round(bucket[key], 3)
            bucket["cpu_core_equivalents"] = round(
                bucket["process_cpu_ms"] / bucket["wall_ms"], 3
            ) if bucket["wall_ms"] else None
        native_threads = None
        cv_threads = None
        try:
            import cv2
            cv_threads = int(cv2.getNumThreads())
        except (ImportError, AttributeError):
            pass
        try:
            import psutil  # optional existing install, not a new dependency
            native_threads = psutil.Process().num_threads()
        except (ImportError, OSError):
            pass
        payload = {
            "opencv_threads_setting": cv_threads,
            "process_native_thread_count_at_end": native_threads,
            "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "outcome": self.outcome, "wall_ms": wall_ms,
            "process_cpu_ms": process_ms,
            "cpu_core_equivalents": round(process_ms / wall_ms, 3) if wall_ms else None,
            "cpu_count": os.cpu_count(),
            "note": "process_cpu_ms sums CPU usage across all native threads; nested timings overlap and must not be summed as total time",
            "grouped": grouped, "samples": self.samples,
        }
        tmp = Path(directory) / "performance_profile.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(Path(directory) / "performance_profile.json")


def begin_profile(job, root: Path) -> None:
    # Jobs instantiated by tests keep the same API.
    job._ocr_performance = OCRPerformance(enabled=_enabled(root))


def attempt_profile(job, attempt: str) -> None:
    profile = getattr(job, "_ocr_performance", None)
    if profile is not None:
        profile.attempt = attempt


def finish_profile(job, outcome: str) -> None:
    profile = getattr(job, "_ocr_performance", None)
    if profile is not None:
        profile.outcome = outcome
        try:
            profile.save(job.directory)
        except Exception:
            # Never turn successful OCR into an error just because profiling
            # could not be persisted or profiling metadata is unavailable.
            pass


@contextmanager
def measured(job, name: str, **attributes):
    profile = getattr(job, "_ocr_performance", None)
    if profile is None or not profile.enabled:
        yield
        return
    wall_start, cpu_start, thread_start = perf_counter_ns(), process_time_ns(), thread_time_ns()
    state = "ok"
    try:
        yield
    except BaseException:
        state = "exception"
        raise
    finally:
        wall_ms = (perf_counter_ns() - wall_start) / 1_000_000
        cpu_ms = (process_time_ns() - cpu_start) / 1_000_000
        thread_ms = (thread_time_ns() - thread_start) / 1_000_000
        profile.samples.append({
            "name": name, "attempt": profile.attempt,
            "status": state, "wall_ms": round(wall_ms, 3),
            "process_cpu_ms": round(cpu_ms, 3),
            "thread_cpu_ms": round(thread_ms, 3),
            "cpu_core_equivalents": round(cpu_ms / wall_ms, 3) if wall_ms else None,
            **attributes,
        })


def profiled_call(job, name, fn, *args, **kwargs):
    """Time one function without changing its signature or exception semantics."""
    with measured(job, name):
        return fn(*args, **kwargs)
