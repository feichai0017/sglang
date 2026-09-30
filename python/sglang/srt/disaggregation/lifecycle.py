"""Public, observational callbacks for decode-owned P/D handoffs.

Callbacks never authorize writes, request readiness, or memory reclamation.
The decode queue emits a terminal event only after its native release condition.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDTransferEvent:
    transfer_id: str
    request_id: str
    bootstrap_room: int
    source_endpoint: str
    logical_bytes: int
    fragment_count: int
    queue_depth: int
    started_ns: int
    phase: Literal["admitted", "completed", "failed", "cancelled", "quarantined"]
    elapsed_ns: int = 0


_observers: dict[str, Callable[[PDTransferEvent], None]] = {}


def register_pd_transfer_observer(
    name: str, observer: Callable[[PDTransferEvent], None]
) -> None:
    """Register a process-local observer during plugin initialization."""
    if name in _observers and _observers[name] is not observer:
        raise ValueError(f"P/D observer already registered: {name}")
    _observers[name] = observer


def _emit(event: PDTransferEvent) -> None:
    for name, observer in tuple(_observers.items()):
        try:
            observer(event)
        except Exception:
            logger.exception("Disabling failed P/D observer %s", name)
            _observers.pop(name, None)


def observe_admission(
    *,
    request_id: str,
    bootstrap_room: int,
    source_endpoint: str,
    logical_bytes: int,
    fragment_count: int,
    queue_depth: int,
) -> PDTransferEvent | None:
    if not _observers:
        return None
    event = PDTransferEvent(
        transfer_id=uuid.uuid4().hex,
        request_id=request_id,
        bootstrap_room=bootstrap_room,
        source_endpoint=source_endpoint,
        logical_bytes=logical_bytes,
        fragment_count=fragment_count,
        queue_depth=queue_depth,
        started_ns=time.monotonic_ns(),
        phase="admitted",
    )
    _emit(event)
    return event


def observe_transition(event: PDTransferEvent | None, phase: str) -> None:
    if event is not None:
        _emit(
            replace(
                event,
                phase=phase,
                elapsed_ns=max(1, time.monotonic_ns() - event.started_ns),
            )
        )
