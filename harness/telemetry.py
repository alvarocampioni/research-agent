from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any


@dataclass
class TelemetryEvent:
    name: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=monotonic)


class Telemetry:
    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def record(self, name: str, **details: Any) -> None:
        self.events.append(TelemetryEvent(name=name, details=details))

    def summary(self) -> dict[str, Any]:
        return {
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "details": e.details}
                for e in self.events
            ],
            "count": len(self.events),
        }
