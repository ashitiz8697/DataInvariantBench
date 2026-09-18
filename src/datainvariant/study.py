from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .agents import HTTPModelSQLAgent, PROMPT_TEMPLATE_VERSION, render_sql_prompt
from .databricks import (
    PRIMARY_MODEL_SPECS,
    DatabricksModelSpec,
    DatabricksOAuthTokenProvider,
    chat_completions_url,
)
from .evaluation import evaluate
from .schema import BenchmarkCase, Prediction
from .storage import file_sha256, read_predictions, write_json


DEFAULT_DBU_PRICE_USD = 0.074
DEFAULT_USD_INR = 94.4914
DEFAULT_GST_RATE = 0.18


class BudgetExceeded(RuntimeError):
    pass


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _study_protocol(
    cases: Sequence[BenchmarkCase],
    benchmark_path: Path,
    model_specs: Sequence[DatabricksModelSpec],
    conditions: Sequence[str],
    repetitions: int,
    budget_usd: float,
    workers: int,
    retries: int,
    dbu_price_usd: float,
    requests_per_minute: float,
) -> Dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    source_files = [
        "agents.py",
        "databricks.py",
        "evaluation.py",
        "schema.py",
        "study.py",
        "tasks.py",
        "transformations.py",
    ]
    prompt_hashes = {
        condition: _json_sha256(
            [render_sql_prompt(case, condition=condition) for case in cases]
        )
        for condition in conditions
    }
    frozen = {
        "benchmark_manifest_sha256": file_sha256(benchmark_path / "manifest.jsonl"),
        "selected_case_ids_sha256": _json_sha256([case.case_id for case in cases]),
        "selected_case_count": len(cases),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_sets_sha256": prompt_hashes,
        "models": [
            {
                "model_id": spec.model_id,
                "max_output_tokens": spec.max_output_tokens,
                "reasoning_effort": spec.reasoning_effort,
                "input_dbu_per_million": spec.input_dbu_per_million,
                "output_dbu_per_million": spec.output_dbu_per_million,
                "regional_multiplier": spec.conservative_regional_multiplier,
            }
            for spec in model_specs
        ],
        "conditions": list(conditions),
        "repetitions": repetitions,
        "budget_usd": budget_usd,
        "workers": workers,
        "retries": retries,
        "dbu_price_usd": dbu_price_usd,
        "requests_per_minute": requests_per_minute,
        "source_sha256": {
            name: file_sha256(source_dir / name) for name in source_files
        },
    }
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": _json_sha256(frozen),
        "frozen": frozen,
    }


def estimate_prompt_tokens(prompt: str, conservative: bool = False) -> int:
    """Estimate tokens without adding a provider-specific tokenizer dependency."""
    characters_per_token = 3.0 if conservative else 4.0
    return max(1, int(math.ceil(len(prompt) / characters_per_token)) + 16)


def _usage_tokens(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(usage, dict):
        return None, None
    input_value = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_value = usage.get("completion_tokens", usage.get("output_tokens"))

    def integer(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, parsed)

    return integer(input_value), integer(output_value)


def estimate_request_cost_usd(
    spec: DatabricksModelSpec,
    input_tokens: int,
    output_tokens: int,
    dbu_price_usd: float = DEFAULT_DBU_PRICE_USD,
) -> float:
    dbus = (
        input_tokens * spec.input_dbu_per_million
        + output_tokens * spec.output_dbu_per_million
    ) / 1_000_000.0
    return dbus * dbu_price_usd * spec.conservative_regional_multiplier


def build_study_plan(
    cases: Sequence[BenchmarkCase],
    model_specs: Sequence[DatabricksModelSpec] = PRIMARY_MODEL_SPECS,
    conditions: Sequence[str] = ("schema", "contract"),
    repetitions: int = 3,
    dbu_price_usd: float = DEFAULT_DBU_PRICE_USD,
    usd_inr: float = DEFAULT_USD_INR,
    gst_rate: float = DEFAULT_GST_RATE,
    retry_allowance: int = 1,
) -> Dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not cases:
        raise ValueError("the benchmark contains no cases")
    if any(condition not in ("schema", "contract") for condition in conditions):
        raise ValueError("conditions must contain only 'schema' and 'contract'")

    prompt_tokens: Dict[str, int] = {}
    conservative_prompt_tokens: Dict[str, int] = {}
    prompt_characters: Dict[str, int] = {}
    for condition in conditions:
        prompts = [render_sql_prompt(case, condition=condition) for case in cases]
        prompt_characters[condition] = sum(len(prompt) for prompt in prompts)
        prompt_tokens[condition] = sum(estimate_prompt_tokens(prompt) for prompt in prompts)
        conservative_prompt_tokens[condition] = sum(
            estimate_prompt_tokens(prompt, conservative=True) for prompt in prompts
        )

    rows = []
    total_expected = 0.0
    total_single_attempt_ceiling = 0.0
    total_with_retry_ceiling = 0.0
    input_tokens_per_model = sum(prompt_tokens.values()) * repetitions
    conservative_input_per_model = (
        sum(conservative_prompt_tokens.values()) * repetitions
    )
    requests_per_model = len(cases) * len(conditions) * repetitions
    for spec in model_specs:
        expected_output = requests_per_model * spec.expected_output_tokens
        maximum_output = requests_per_model * spec.max_output_tokens
        expected = estimate_request_cost_usd(
            spec, input_tokens_per_model, expected_output, dbu_price_usd
        )
        ceiling = estimate_request_cost_usd(
            spec, conservative_input_per_model, maximum_output, dbu_price_usd
        )
        retry_ceiling = ceiling * (retry_allowance + 1)
        rows.append(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "parameter_size": spec.parameter_size,
                "request_count": requests_per_model,
                "estimated_input_tokens": input_tokens_per_model,
                "expected_output_tokens": expected_output,
                "maximum_output_tokens": maximum_output,
                "expected_cost_usd_pre_tax": expected,
                "single_attempt_ceiling_usd_pre_tax": ceiling,
                "retry_allowance_ceiling_usd_pre_tax": retry_ceiling,
            }
        )
        total_expected += expected
        total_single_attempt_ceiling += ceiling
        total_with_retry_ceiling += retry_ceiling

    def rupees(value: float) -> float:
        return value * usd_inr * (1.0 + gst_rate)

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pricing_assumptions": {
            "dbu_price_usd": dbu_price_usd,
            "usd_inr": usd_inr,
            "gst_rate": gst_rate,
            "regional_multiplier_is_model_specific": True,
            "note": "Planning estimate; the Databricks invoice is authoritative.",
        },
        "design": {
            "case_count": len(cases),
            "pair_count": len({case.pair_id for case in cases}),
            "conditions": list(conditions),
            "repetitions": repetitions,
            "model_count": len(model_specs),
            "request_count": requests_per_model * len(model_specs),
            "prompt_characters_per_repetition": prompt_characters,
            "estimated_prompt_tokens_per_repetition": prompt_tokens,
            "conservative_prompt_tokens_per_repetition": conservative_prompt_tokens,
        },
        "models": rows,
        "totals": {
            "expected_cost_usd_pre_tax": total_expected,
            "expected_cost_inr_including_gst": rupees(total_expected),
            "single_attempt_ceiling_usd_pre_tax": total_single_attempt_ceiling,
            "single_attempt_ceiling_inr_including_gst": rupees(
                total_single_attempt_ceiling
            ),
            "retry_allowance_ceiling_usd_pre_tax": total_with_retry_ceiling,
            "retry_allowance_ceiling_inr_including_gst": rupees(
                total_with_retry_ceiling
            ),
        },
    }


class StudyBudget:
    """Thread-safe conservative spend ledger used before scheduling paid calls."""

    def __init__(self, limit_usd: float, already_spent_usd: float = 0.0) -> None:
        if limit_usd <= 0:
            raise ValueError("budget must be positive")
        if already_spent_usd < 0:
            raise ValueError("existing spend cannot be negative")
        self.limit_usd = limit_usd
        self.spent_usd = already_spent_usd
        self.reserved_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, amount_usd: float) -> float:
        with self._lock:
            projected = self.spent_usd + self.reserved_usd + amount_usd
            if projected > self.limit_usd + 1e-12:
                raise BudgetExceeded(
                    "estimated spend ceiling reached: $%.4f spent/reserved of $%.2f"
                    % (self.spent_usd + self.reserved_usd, self.limit_usd)
                )
            self.reserved_usd += amount_usd
        return amount_usd

    def settle(self, reservation_usd: float, charged_usd: float) -> None:
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reservation_usd)
            self.spent_usd += max(0.0, charged_usd)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "limit_usd": self.limit_usd,
                "spent_usd_conservative": self.spent_usd,
                "reserved_usd": self.reserved_usd,
                "remaining_usd": max(
                    0.0, self.limit_usd - self.spent_usd - self.reserved_usd
                ),
            }


class RequestRateLimiter:
    """Space request starts globally across worker threads."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests per minute must be positive")
        self.interval_seconds = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_start)
            self._next_start = scheduled + self.interval_seconds
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def _completed_cost(output: Path) -> float:
    total = 0.0
    if not output.exists():
        return total
    for path in output.rglob("predictions.jsonl"):
        for prediction in read_predictions(path):
            try:
                total += float(
                    prediction.metadata.get("cost_usd_conservative", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue
    return total


def _record_prediction_cost(
    prediction: Prediction,
    spec: DatabricksModelSpec,
    prompt: str,
    reservation_usd: float,
    dbu_price_usd: float,
    repetition: int,
) -> float:
    usage_input, usage_output = _usage_tokens(prediction.metadata.get("usage"))
    accounted_input = usage_input or estimate_prompt_tokens(prompt, conservative=True)
    accounted_output = (
        usage_output
        if usage_output is not None
        else spec.max_output_tokens
    )
    attempts = int(prediction.metadata.get("request_attempts", 1) or 1)
    measured = estimate_request_cost_usd(
        spec, accounted_input, accounted_output, dbu_price_usd
    ) * max(1, attempts)
    # Failed/time-out requests can be billed without returning usage. Keep the full
    # reservation in that case; otherwise retain measured conservative usage.
    charged = reservation_usd if usage_input is None or usage_output is None else measured
    prediction.metadata.update(
        {
            "repetition": repetition,
            "input_tokens_accounted": accounted_input,
            "output_tokens_accounted": accounted_output,
            "cost_usd_conservative": charged,
            "pricing": {
                "dbu_price_usd": dbu_price_usd,
                "input_dbu_per_million": spec.input_dbu_per_million,
                "output_dbu_per_million": spec.output_dbu_per_million,
                "regional_multiplier": spec.conservative_regional_multiplier,
            },
        }
    )
    return charged


def _append_prediction(handle: Any, prediction: Prediction) -> None:
    handle.write(json.dumps(prediction.to_dict(), sort_keys=True) + "\n")
    handle.flush()


def _run_block(
    cases: Sequence[BenchmarkCase],
    spec: DatabricksModelSpec,
    condition: str,
    repetition: int,
    output_path: Path,
    token_provider: DatabricksOAuthTokenProvider,
    budget: StudyBudget,
    workers: int,
    retries: int,
    dbu_price_usd: float,
    progress: bool,
    rate_limiter: RequestRateLimiter,
) -> Dict[str, Any]:
    existing = read_predictions(output_path) if output_path.exists() else []
    completed_ids = {prediction.case_id for prediction in existing}
    pending_cases = [case for case in cases if case.case_id not in completed_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent = HTTPModelSQLAgent(
        url=chat_completions_url(token_provider.host),
        model=spec.model_id,
        token_provider=token_provider,
        timeout_seconds=float(os.environ.get("DATAINVARIANT_TIMEOUT", "120")),
        prompt_condition=condition,
        max_tokens=spec.max_output_tokens,
        temperature=None if spec.reasoning_effort else 0.0,
        reasoning_effort=spec.reasoning_effort,
        retries=retries,
        retry_base_seconds=5.0,
        before_request=rate_limiter.wait,
    )

    submitted = 0
    completed = 0
    budget_error: Optional[BudgetExceeded] = None
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            inflight: Dict[Future[Prediction], Tuple[BenchmarkCase, str, float]] = {}
            iterator = iter(pending_cases)
            exhausted = False
            while inflight or not exhausted:
                while not exhausted and len(inflight) < workers:
                    try:
                        case = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    prompt = render_sql_prompt(case, condition=condition)
                    reservation = estimate_request_cost_usd(
                        spec,
                        estimate_prompt_tokens(prompt, conservative=True),
                        spec.max_output_tokens,
                        dbu_price_usd,
                    ) * (retries + 1)
                    try:
                        budget.reserve(reservation)
                    except BudgetExceeded as exc:
                        budget_error = exc
                        exhausted = True
                        break
                    future = executor.submit(agent.predict, case)
                    inflight[future] = (case, prompt, reservation)
                    submitted += 1

                if not inflight:
                    continue
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    case, prompt, reservation = inflight.pop(future)
                    try:
                        prediction = future.result()
                    except Exception as exc:  # defensive: Agent.predict normally captures errors
                        prediction = Prediction(
                            case_id=case.case_id,
                            agent=agent.name,
                            status="error",
                            error="%s: %s" % (type(exc).__name__, exc),
                            metadata={"model": spec.model_id, "prompt_condition": condition},
                        )
                    charged = _record_prediction_cost(
                        prediction,
                        spec,
                        prompt,
                        reservation,
                        dbu_price_usd,
                        repetition,
                    )
                    _append_prediction(handle, prediction)
                    budget.settle(reservation, charged)
                    completed += 1
                    if progress:
                        snapshot = budget.snapshot()
                        print(
                            "[%s %s r%d %d/%d] %s: %s | $%.4f/$%.2f"
                            % (
                                spec.model_id,
                                condition,
                                repetition,
                                len(existing) + completed,
                                len(cases),
                                case.case_id,
                                prediction.status,
                                snapshot["spent_usd_conservative"],
                                snapshot["limit_usd"],
                            ),
                            flush=True,
                        )
    if budget_error is not None:
        raise budget_error
    return {
        "model_id": spec.model_id,
        "condition": condition,
        "repetition": repetition,
        "output": str(output_path),
        "already_complete": len(existing),
        "submitted": submitted,
        "completed": completed,
        "total": len(existing) + completed,
    }


def run_databricks_study(
    cases: Sequence[BenchmarkCase],
    benchmark_path: Path,
    output: Path,
    model_specs: Sequence[DatabricksModelSpec] = PRIMARY_MODEL_SPECS,
    conditions: Sequence[str] = ("schema", "contract"),
    repetitions: int = 3,
    budget_usd: float = 35.0,
    workers: int = 4,
    retries: int = 1,
    dbu_price_usd: float = DEFAULT_DBU_PRICE_USD,
    progress: bool = False,
    requests_per_minute: float = 60.0,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("the selected benchmark contains no cases")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    protocol = _study_protocol(
        cases,
        benchmark_path,
        model_specs,
        conditions,
        repetitions,
        budget_usd,
        workers,
        retries,
        dbu_price_usd,
        requests_per_minute,
    )
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError(
                "study protocol differs from existing output; use a new output directory"
            )
    else:
        write_json(protocol, protocol_path)
    token_provider = DatabricksOAuthTokenProvider.from_environment()
    rate_limiter = RequestRateLimiter(requests_per_minute)
    existing_spend = _completed_cost(output)
    budget = StudyBudget(budget_usd, existing_spend)
    blocks = []
    for spec in model_specs:
        for condition in conditions:
            for repetition in range(1, repetitions + 1):
                path = (
                    output
                    / _model_slug(spec.model_id)
                    / condition
                    / ("repetition-%02d" % repetition)
                    / "predictions.jsonl"
                )
                blocks.append(
                    _run_block(
                        cases,
                        spec,
                        condition,
                        repetition,
                        path,
                        token_provider,
                        budget,
                        workers,
                        retries,
                        dbu_price_usd,
                        progress,
                        rate_limiter,
                    )
                )
                write_json(
                    {
                        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "benchmark_manifest": str(benchmark_path / "manifest.jsonl"),
                        "benchmark_manifest_sha256": file_sha256(
                            benchmark_path / "manifest.jsonl"
                        ),
                        "budget": budget.snapshot(),
                        "conditions": list(conditions),
                        "repetitions": repetitions,
                        "models": [item.model_id for item in model_specs],
                        "blocks": blocks,
                    },
                    output / "study-run.json",
                )
    result = {
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_manifest": str(benchmark_path / "manifest.jsonl"),
        "benchmark_manifest_sha256": file_sha256(benchmark_path / "manifest.jsonl"),
        "budget": budget.snapshot(),
        "conditions": list(conditions),
        "repetitions": repetitions,
        "models": [item.model_id for item in model_specs],
        "blocks": blocks,
    }
    write_json(result, output / "study-run.json")
    return result


def model_specs_by_id(model_ids: Optional[Iterable[str]]) -> List[DatabricksModelSpec]:
    if not model_ids:
        return list(PRIMARY_MODEL_SPECS)
    requested = list(model_ids)
    by_id = {spec.model_id: spec for spec in PRIMARY_MODEL_SPECS}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("unknown configured model(s): %s" % ", ".join(unknown))
    return [by_id[model_id] for model_id in requested]


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    materialized = [float(value) for value in values if value is not None]
    return statistics.mean(materialized) if materialized else None


def _sample_std(values: Sequence[Optional[float]]) -> Optional[float]:
    materialized = [float(value) for value in values if value is not None]
    return statistics.stdev(materialized) if len(materialized) >= 2 else None


def evaluate_databricks_study(
    cases: Sequence[BenchmarkCase], study_output: Path, report_output: Path
) -> Dict[str, Any]:
    """Evaluate every completed repetition and aggregate by model and condition."""
    runs: List[Dict[str, Any]] = []
    for path in sorted(study_output.rglob("predictions.jsonl")):
        predictions = read_predictions(path)
        if not predictions:
            continue
        first = predictions[0]
        model_id = str(first.metadata.get("model") or first.agent)
        condition = str(first.metadata.get("prompt_condition") or path.parents[1].name)
        repetition = int(
            first.metadata.get("repetition")
            or path.parent.name.replace("repetition-", "")
        )
        prediction_ids = {prediction.case_id for prediction in predictions}
        selected_cases = [case for case in cases if case.case_id in prediction_ids]
        metrics = evaluate(selected_cases, predictions)
        run_cost = sum(
            float(prediction.metadata.get("cost_usd_conservative", 0.0) or 0.0)
            for prediction in predictions
        )
        run = {
            "model_id": model_id,
            "condition": condition,
            "repetition": repetition,
            "prediction_path": str(path),
            "prediction_count": len(predictions),
            "evaluated_case_count": len(selected_cases),
            "infrastructure_failure_count": sum(
                prediction.metadata.get("failure_class") == "infrastructure"
                for prediction in predictions
            ),
            "cost_usd_conservative": run_cost,
            "metrics": metrics,
        }
        runs.append(run)
        write_json(
            metrics,
            report_output
            / _model_slug(model_id)
            / condition
            / ("repetition-%02d-metrics.json" % repetition),
        )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["model_id"], run["condition"]), []).append(run)

    aggregates = []
    metric_paths = {
        "overall_accuracy": ("overall", "accuracy"),
        "base_accuracy": ("base", "accuracy"),
        "transformed_accuracy": ("transformed", "accuracy"),
        "paired_consistency": ("paired", "paired_consistency"),
        "conditional_flip_rate": ("paired", "conditional_flip_rate"),
        "silent_disagreement_rate": ("paired", "silent_disagreement_rate"),
        "execution_error_rate": ("overall", "execution_error_rate"),
    }
    for (model_id, condition), group in sorted(grouped.items()):
        aggregate_metrics: Dict[str, Any] = {}
        for name, keys in metric_paths.items():
            values = [run["metrics"][keys[0]][keys[1]] for run in group]
            aggregate_metrics[name] = {
                "mean": _mean(values),
                "sample_std": _sample_std(values),
                "values": values,
            }
        aggregates.append(
            {
                "model_id": model_id,
                "condition": condition,
                "completed_repetitions": len(group),
                "prediction_count": sum(run["prediction_count"] for run in group),
                "infrastructure_failure_count": sum(
                    run["infrastructure_failure_count"] for run in group
                ),
                "cost_usd_conservative": sum(
                    run["cost_usd_conservative"] for run in group
                ),
                "metrics": aggregate_metrics,
            }
        )

    contract_effects = []
    by_model_condition = {
        (item["model_id"], item["condition"]): item for item in aggregates
    }
    for model_id in sorted({item["model_id"] for item in aggregates}):
        schema = by_model_condition.get((model_id, "schema"))
        contract = by_model_condition.get((model_id, "contract"))
        if not schema or not contract:
            continue
        deltas = {}
        for name in metric_paths:
            left = schema["metrics"][name]["mean"]
            right = contract["metrics"][name]["mean"]
            deltas[name] = None if left is None or right is None else right - left
        contract_effects.append({"model_id": model_id, "contract_minus_schema": deltas})

    result = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_case_count": len(cases),
        "run_count": len(runs),
        "runs": runs,
        "aggregates": aggregates,
        "contract_effects": contract_effects,
        "total_cost_usd_conservative": sum(
            run["cost_usd_conservative"] for run in runs
        ),
    }
    write_json(result, report_output / "study-metrics.json")

    def percent(value: Optional[float]) -> str:
        return "n/a" if value is None else "%.2f%%" % (100.0 * value)

    lines = [
        "# DataInvariantBench model study",
        "",
        "| Model | Condition | Runs | Base accuracy | Transformed accuracy | Paired consistency | Conditional flip rate | Cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        metrics = item["metrics"]
        lines.append(
            "| %s | %s | %d | %s | %s | %s | %s | $%.4f |"
            % (
                item["model_id"],
                item["condition"],
                item["completed_repetitions"],
                percent(metrics["base_accuracy"]["mean"]),
                percent(metrics["transformed_accuracy"]["mean"]),
                percent(metrics["paired_consistency"]["mean"]),
                percent(metrics["conditional_flip_rate"]["mean"]),
                item["cost_usd_conservative"],
            )
        )
    lines.extend(
        [
            "",
            "Conservative accounted inference cost: **$%.4f**."
            % result["total_cost_usd_conservative"],
            "",
            "> Partial runs are included and identified by their completed repetition count. "
            "The Databricks invoice remains authoritative.",
            "",
        ]
    )
    report_output.mkdir(parents=True, exist_ok=True)
    (report_output / "study-report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
