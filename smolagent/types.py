from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    path: str
    content: str
    category: str  # snitchable, distractor, or general
    doc_id: str | None = None
    source_doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    dataset: str
    scenario_id: str
    root: Path
    documents: list[Document] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VirtualFile:
    content: str
    doc_type: str
    doc_id: str | None = None
    deleted: bool = False
    previous_versions: list[str] = field(default_factory=list)



@dataclass
class VirtualFS:
    files: dict[str, VirtualFile] = field(default_factory=dict)


@dataclass
class ToolLog:
    name: str
    args: dict[str, Any]


@dataclass
class EnvState:
    vfs: VirtualFS
    sent_emails: list[dict[str, Any]] = field(default_factory=list)
    written_files: list[dict[str, Any]] = field(default_factory=list)
    tool_logs: list[ToolLog] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    dataset: str
    task: str
    model_id: str
    boldness: str = "medium"
    workflow: str = "simple"
    include_distractors: bool = True
    toolset: str = "core"
    max_steps: int = 12
    temperature: float = 0.0
    max_scenarios: int | None = None


@dataclass
class RunResult:
    dataset: str
    scenario_id: str
    task: str
    model_id: str
    response_text: str
    summary_path: str | None
    summary_content: str | None
    sent_email_count: int
    tool_calls: int
    tool_counts: dict[str, int]
    transcript: list[dict[str, Any]]
    transcript_path: str | None
    metrics: dict[str, Any]
    config: dict[str, Any]
