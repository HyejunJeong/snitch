# Anonymous Artifact: Benchmark Code and Data

This directory contains the code and bundled data submitted for anonymous paper review. It provides the experiment harness used to run document-review agents over benchmark scenarios, record transcripts, and compute lightweight run metrics.

The artifact is intentionally self-contained at the `src/` level. It does not include author names, institution names, private API keys, or machine-specific setup files.

## Contents

```text
.
|-- README.md
|-- smolagent/                  # Experiment harness and virtual tool environment
|-- notebooks/                  # Analysis and visualization notebooks
|-- enron_candidate_audit/      # Candidate-selection audit artifacts
`-- surveilbench.zip            # Bundled benchmark scenarios and metadata
```

Important modules:

- `smolagent/dataset.py` loads supported benchmark datasets and filters distractor documents.
- `smolagent/prompts.py` constructs system and user prompts for each task condition.
- `smolagent/env.py` defines the virtual file system and tools available to the agent.
- `smolagent/agent_runner.py` runs one scenario, records tool calls, captures reasoning metadata when available, and returns a structured result.
- `smolagent/experiment.py` runs individual experiments or parameter sweeps.
- `smolagent/metrics.py` computes simple behavioral and output metrics from each run.
- `smolagent/build_hdo_per_run_per_scenario_with_contents.py` aggregates run outputs into a flat CSV for downstream analysis.

## Setup

The code is plain Python and can be run directly from this directory. A fresh virtual environment is recommended.

```bash
cd src
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install smolagents litellm openai anthropic google-generativeai together transformers torch jupyter pandas matplotlib seaborn
```

Only `smolagents` and its LiteLLM-backed model support are required for the core experiment path. The provider SDKs and notebook packages are needed only for the corresponding model providers or analysis notebooks.

Configure the API keys required by the model provider you plan to use, for example:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
```

Model identifiers are passed through to `smolagents.LiteLLMModel`, so use the provider/model naming convention expected by LiteLLM for your environment.

## Data

The submitted directory includes `surveilbench.zip`. Extract it from `src/` before running experiments on the bundled benchmark:

```bash
unzip surveilbench.zip
```

This creates a `surveilbench/` directory containing scenario folders grouped by axis, along with scenario metadata. The loader also supports these dataset names when corresponding data directories are present:

- `surveilbench`
- `surveilbench_v2`
- `surveilbench_v3`
- `whistlebench`
- `snitchbench`

For `surveilbench`, each scenario directory should contain markdown documents and a `metadata.json` file. The loader can filter by severity band or axis through `load_surveilbench_scenarios(...)` or `load_scenarios(...)`.

## Minimal Run

The following example runs one scenario from the bundled benchmark and writes results under `outputs/`.

```bash
python3 - <<'PY'
from pathlib import Path

from smolagent.experiment import run_experiment
from smolagent.types import ExperimentConfig

config = ExperimentConfig(
    dataset="surveilbench",
    task="summarization",
    model_id="openai/gpt-4o-mini",
    boldness="medium",
    workflow="simple",
    include_distractors=True,
    toolset="core",
    max_scenarios=1,
    temperature=0.0,
)

results, summary = run_experiment(
    config=config,
    workspace=Path("."),
    output=Path("outputs/example.jsonl"),
)

print(summary)
print(f"wrote {len(results)} result(s)")
PY
```

Each run writes:

- a JSONL record file, such as `outputs/example.jsonl`;
- a summary file next to it, such as `outputs/example.summary.json`;
- per-scenario transcripts in a sibling directory, such as `outputs/example_transcripts/`.

## Parameter Sweep

`run_sweep(...)` evaluates combinations of task, boldness level, workflow, distractor setting, and toolset:

```bash
python3 - <<'PY'
from pathlib import Path

from smolagent.experiment import run_sweep

run_sweep(
    dataset="surveilbench",
    model_id="openai/gpt-4o-mini",
    workspace=Path("."),
    output_dir=Path("outputs/sweep"),
    max_scenarios=1,
    temperature=0.0,
)
PY
```

The sweep covers:

- tasks: `summarization`, `pii_detection`, `ediscovery_review`, `translation`;
- boldness levels: `low`, `medium`, `high`;
- workflows: `simple`, `step_by_step`;
- distractor settings: included or excluded;
- toolsets: `core` and `extended`.

## Virtual Tool Environment

Agents interact with documents through an in-memory virtual file system. The core toolset includes:

- `write_file`, for writing the requested output;
- `send_email`, for recording attempted emails.

The extended toolset additionally includes document search, word counts, metadata extraction, document merging, indexing, deletion, hash computation, markdown-to-PDF path conversion, and a deliberately risky escalation tool. Tool calls are logged and included in the structured transcript for later analysis.

## Outputs and Metrics

Each `RunResult` includes:

- dataset, scenario, task, model, and configuration metadata;
- final model response;
- generated summary file content, when present;
- sent-email records;
- tool-call counts and per-tool counts;
- a transcript containing prompts, captured model reasoning metadata when available, tool calls, generated files, and final response;
- lightweight metrics such as summary-file rate, tool usage, email calls, source-token coverage, and template-field handling.

Use `smolagent/build_hdo_per_run_per_scenario_with_contents.py` to aggregate a `results/` directory into a flat CSV after running larger batches.

## Notebooks

The `notebooks/` directory contains exploratory notebooks for running repeated conditions, evaluating outputs, and generating visualizations. They are included for transparency, but the primary reproducible entry points are the Python modules in `smolagent/`.

## Anonymous Review Notes

- This README avoids author-identifying details.
- API keys and provider credentials are not included.
- Local outputs, virtual environments, checkpoints, and result folders are ignored by `.gitignore`.
- The code may still create provider-side logs depending on the model API used by the reviewer.
