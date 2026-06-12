from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

from .agent_runner import run_single_scenario
from .dataset import filter_documents_for_run, load_scenarios
from .io_utils import summarize_results, write_jsonl
from .types import ExperimentConfig, RunResult


def run_experiment(config: ExperimentConfig, workspace: Path, output: Path) -> tuple[list[RunResult], dict]:
    scenarios = load_scenarios(config.dataset, workspace)
    if config.max_scenarios is not None:
        scenarios = scenarios[: config.max_scenarios]

    results: list[RunResult] = []
    transcript_dir = output.parent / f"{output.stem}_transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        docs = filter_documents_for_run(
            scenario=scenario,
            include_distractors=config.include_distractors,
        )
        result = run_single_scenario(config=config, scenario=scenario, documents=docs)
        transcript_path = transcript_dir / f"{scenario.scenario_id}.json"
        transcript_path.write_text(
            json.dumps(result.transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result.transcript_path = str(transcript_path)
        results.append(result)

    write_jsonl(output, results)
    summary = summarize_results(results)

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results, summary


def run_sweep(
    dataset: str,
    model_id: str,
    workspace: Path,
    output_dir: Path,
    max_scenarios: int | None = None,
    temperature: float = 0.0,
) -> list[tuple[ExperimentConfig, dict, Path]]:
    tasks = ["summarization", "pii_detection", "ediscovery_review", "translation"]
    boldness_levels = ["low", "medium", "high"]
    workflows = ["simple", "step_by_step"]
    distractor_flags = [True, False]
    toolsets = ["core", "extended"]

    base = ExperimentConfig(
        dataset=dataset,
        task="summarization",
        model_id=model_id,
        max_scenarios=max_scenarios,
        temperature=temperature,
    )

    runs: list[tuple[ExperimentConfig, dict, Path]] = []
    for task, boldness, workflow, include_distractors, toolset in itertools.product(
        tasks, boldness_levels, workflows, distractor_flags, toolsets
    ):
        cfg = replace(
            base,
            task=task,
            boldness=boldness,
            workflow=workflow,
            include_distractors=include_distractors,
            toolset=toolset,
        )
        out_name = (
            f"{dataset}__{task}__{boldness}__{workflow}__"
            f"distractor-{int(include_distractors)}__{toolset}.jsonl"
        )
        out_path = output_dir / out_name
        _, summary = run_experiment(config=cfg, workspace=workspace, output=out_path)
        runs.append((cfg, summary, out_path))

    return runs
