from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .types import Document, Scenario


def _read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_hidden_rel_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part and part != "/")


def _is_hidden_dir(path: Path) -> bool:
    return path.name.startswith(".")


def _anon_doc_path(rel_path: Path) -> str:
    digest = hashlib.md5(rel_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"doc_{digest}.md"


_LEADING_INDEX_RE = re.compile(r"^(\d{2})")


def _whistlebench_int_is_distractor(rel_path: Path) -> bool:
    match = _LEADING_INDEX_RE.match(rel_path.stem)
    if not match:
        return False
    doc_index = int(match.group(1))
    return 5 <= doc_index <= 10


def _surveilbench_is_distractor(rel_path: Path, metadata: dict[str, Any]) -> bool:
    ordered_files = [str(item) for item in metadata.get("files") or []]
    rel_path_str = rel_path.as_posix()
    if rel_path_str in ordered_files:
        return ordered_files.index(rel_path_str) >= 5
    if rel_path.name in ordered_files:
        return ordered_files.index(rel_path.name) >= 5
    return _whistlebench_int_is_distractor(rel_path) or rel_path.name == "A1_Contact_Matrix.md"


def _normalized_severity_band(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    aliases = {
        "externally_reportable": "externally_reportable",
        "external_report_oriented": "externally_reportable",
        "externally reportable incidents": "externally_reportable",
        "internally_reviewable": "internally_reviewable",
        "internal_report_oriented": "internally_reviewable",
        "internally reviewable incidents": "internally_reviewable",
        "low-severity-signals": "low-severity-signals",
        "surveillance_sensitive_weak_signal": "low-severity-signals",
        "low-severity behavioral signals": "low-severity-signals",
    }
    return aliases.get(normalized, normalized)


def _normalized_axis(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _load_docs_from_scenario_dir(
    scenario_dir: Path,
    distractor_resolver: Callable[[Path], bool],
) -> list[Document]:
    docs: list[Document] = []
    for file_path in sorted(scenario_dir.rglob("*.md")):
        rel_path = file_path.relative_to(scenario_dir)
        if _is_hidden_rel_path(rel_path):
            continue
        is_distractor = distractor_resolver(rel_path)
        docs.append(
            Document(
                path=_anon_doc_path(rel_path),
                content=_read_md(file_path),
                category="general",
                metadata={"is_distractor": is_distractor},
            )
        )
    return docs


def load_whistlebench_scenarios(root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir() and not _is_hidden_dir(p)):
        docs = _load_docs_from_scenario_dir(
            scenario_dir=scenario_dir,
            distractor_resolver=_whistlebench_int_is_distractor,
        )
        if not docs:
            continue
        scenarios.append(
            Scenario(
                dataset="whistlebench",
                scenario_id=scenario_dir.name,
                root=scenario_dir,
                documents=docs,
            )
        )
    return scenarios


def load_snitchbench_scenarios(root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir() and not _is_hidden_dir(p)):
        docs = _load_docs_from_scenario_dir(
            scenario_dir=scenario_dir,
            distractor_resolver=lambda _: False,
        )
        if not docs:
            continue
        scenarios.append(
            Scenario(
                dataset="snitchbench",
                scenario_id=scenario_dir.name,
                root=scenario_dir,
                documents=docs,
            )
        )
    return scenarios


def load_surveilbench_scenarios(
    root: Path,
    severity_band: str | None = None,
    axis: str | None = None,
    dataset_name: str = "surveilbench",
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    target_band = _normalized_severity_band(severity_band)
    target_axis = _normalized_axis(axis)
    axis_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not _is_hidden_dir(p))
    for axis_dir in axis_dirs:
        axis_name = _normalized_axis(axis_dir.name)
        if target_axis is not None and axis_name != target_axis:
            continue
        scenario_dirs = sorted(p for p in axis_dir.iterdir() if p.is_dir() and not _is_hidden_dir(p))
        for scenario_dir in scenario_dirs:
            metadata_path = scenario_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = _read_json(metadata_path)
            metadata_band = _normalized_severity_band(str(metadata.get("severity_band", "")))
            if target_band is not None and metadata_band != target_band:
                continue
            docs = _load_docs_from_scenario_dir(
                scenario_dir=scenario_dir,
                distractor_resolver=lambda rel_path, scenario_metadata=metadata: _surveilbench_is_distractor(
                    rel_path,
                    scenario_metadata,
                ),
            )
            if not docs:
                continue
            scenario_metadata = dict(metadata)
            scenario_metadata.setdefault("axis", axis_dir.name)
            scenarios.append(
                Scenario(
                    dataset=dataset_name,
                    scenario_id=scenario_dir.name,
                    root=scenario_dir,
                    documents=docs,
                    metadata=scenario_metadata,
                )
            )
    return scenarios


def load_surveilbench_scenarios_by_severity_band(
    root: Path,
    axis: str | None = None,
) -> dict[str, list[Scenario]]:
    grouped: defaultdict[str, list[Scenario]] = defaultdict(list)
    for scenario in load_surveilbench_scenarios(root, axis=axis):
        band = str(scenario.metadata.get("severity_band", "unknown"))
        grouped[band].append(scenario)
    return dict(sorted(grouped.items()))


def load_mybench_scenarios(
    root: Path,
    severity_band: str | None = None,
    axis: str | None = None,
) -> list[Scenario]:
    return load_surveilbench_scenarios(root, severity_band=severity_band, axis=axis)


def load_mybench_scenarios_by_severity_band(
    root: Path,
    axis: str | None = None,
) -> dict[str, list[Scenario]]:
    return load_surveilbench_scenarios_by_severity_band(root, axis=axis)


def load_scenarios(
    dataset: str,
    workspace: Path,
    *,
    severity_band: str | None = None,
    axis: str | None = None,
) -> list[Scenario]:
    dataset = dataset.lower().strip()
    if dataset == "whistlebench":
        root = workspace / "whistlebench_data"
        if not root.exists():
            root = workspace / "whistlebench"
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        return load_whistlebench_scenarios(root)
    if dataset == "snitchbench":
        root = workspace / "snitchbench_data"
        if not root.exists():
            root = workspace / "snitchbench"
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        return load_snitchbench_scenarios(root)
    surveilbench_roots = {
        "mybench": ("surveilbench", "data"),
        "surveilbench": ("surveilbench", "data"),
        "surveilbench_v2": ("surveilbench_v2",),
        "surveilbench_v3": ("surveilbench_v3", "surveilbench"),
    }
    if dataset in surveilbench_roots:
        root = None
        base_dirs = (workspace, workspace.parent)
        for dirname in surveilbench_roots[dataset]:
            for base_dir in base_dirs:
                candidate = base_dir / dirname
                if candidate.exists():
                    root = candidate
                    break
            if root is not None:
                break
        if root is None:
            root = workspace / surveilbench_roots[dataset][-1]
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        return load_surveilbench_scenarios(
            root,
            severity_band=severity_band,
            axis=axis,
            dataset_name=dataset,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def filter_documents_for_run(scenario: Scenario, include_distractors: bool) -> list[Document]:
    if include_distractors:
        return scenario.documents

    def is_distractor(doc: Document) -> bool:
        if "is_distractor" in doc.metadata:
            return bool(doc.metadata.get("is_distractor"))
        return doc.category == "distractor"

    return [d for d in scenario.documents if not is_distractor(d)]
