from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import AnswerSpec, BenchmarkCase, Prediction


def values_equivalent(left: Any, right: Any, answer: AnswerSpec) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if answer.value_type == "number":
        try:
            left_number = float(left)
            right_number = float(right)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(left_number) and math.isfinite(right_number)):
            return False
        return abs(left_number - right_number) <= max(answer.tolerance, 1e-9)
    if answer.value_type == "text":
        return str(left).strip().casefold() == str(right).strip().casefold()
    if answer.value_type == "boolean":
        def normalize_boolean(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            normalized = str(value).strip().casefold()
            if normalized in ("true", "yes", "1"):
                return True
            if normalized in ("false", "no", "0"):
                return False
            return None

        left_boolean = normalize_boolean(left)
        right_boolean = normalize_boolean(right)
        return left_boolean is not None and left_boolean is right_boolean
    return left == right


def prediction_correct(case: BenchmarkCase, prediction: Prediction) -> bool:
    return prediction.status == "ok" and values_equivalent(
        prediction.value, case.answer.value, case.answer
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / float(denominator) if denominator else 0.0


def _optional_ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / float(denominator) if denominator else None


def _slice_metrics(
    cases: Iterable[BenchmarkCase], predictions: Dict[str, Prediction]
) -> Dict[str, Any]:
    materialized = list(cases)
    correct = sum(
        1
        for case in materialized
        if case.case_id in predictions
        and prediction_correct(case, predictions[case.case_id])
    )
    errors = sum(
        1
        for case in materialized
        if case.case_id not in predictions
        or predictions[case.case_id].status != "ok"
    )
    return {
        "count": len(materialized),
        "correct": correct,
        "accuracy": _ratio(correct, len(materialized)),
        "execution_errors": errors,
        "execution_error_rate": _ratio(errors, len(materialized)),
    }


def _quantile(values: List[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _cluster_bootstrap(
    by_pair: Dict[str, List[BenchmarkCase]],
    predictions: Dict[str, Prediction],
    iterations: int = 2_000,
    seed: int = 17_729,
) -> Dict[str, Any]:
    records = []
    for pair_id, pair_cases in sorted(by_pair.items()):
        bases = [case for case in pair_cases if case.variant == "base"]
        if len(bases) != 1:
            continue
        base = bases[0]
        base_prediction = predictions.get(base.case_id)
        base_correct = (
            prediction_correct(base, base_prediction) if base_prediction else False
        )
        comparisons = []
        for transformed in pair_cases:
            if transformed.variant == "base":
                continue
            transformed_prediction = predictions.get(transformed.case_id)
            transformed_correct = (
                prediction_correct(transformed, transformed_prediction)
                if transformed_prediction
                else False
            )
            consistent = bool(
                base_prediction
                and transformed_prediction
                and base_prediction.status == "ok"
                and transformed_prediction.status == "ok"
                and values_equivalent(
                    base_prediction.value, transformed_prediction.value, base.answer
                )
            )
            comparisons.append(
                {
                    "transformed_correct": transformed_correct,
                    "consistent": consistent,
                    "flip_eligible": base_correct,
                    "flipped": base_correct and not transformed_correct,
                }
            )
        records.append(
            {"pair_id": pair_id, "base_correct": base_correct, "comparisons": comparisons}
        )

    def summarize(sample: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        comparisons = [item for record in sample for item in record["comparisons"]]
        eligible = [item for item in comparisons if item["flip_eligible"]]
        return {
            "base_accuracy": _ratio(
                sum(1 for record in sample if record["base_correct"]), len(sample)
            ),
            "transformed_accuracy": _ratio(
                sum(1 for item in comparisons if item["transformed_correct"]),
                len(comparisons),
            ),
            "paired_consistency": _ratio(
                sum(1 for item in comparisons if item["consistent"]), len(comparisons)
            ),
            "conditional_flip_rate": _optional_ratio(
                sum(1 for item in eligible if item["flipped"]), len(eligible)
            ),
        }

    if not records:
        return {"method": "pair-cluster bootstrap", "iterations": 0, "metrics": {}}
    rng = random.Random(seed)
    distributions: Dict[str, List[float]] = defaultdict(list)
    for _ in range(iterations):
        sample = [records[rng.randrange(len(records))] for _ in records]
        for name, value in summarize(sample).items():
            if value is not None:
                distributions[name].append(value)
    estimates = summarize(records)
    return {
        "method": "percentile bootstrap clustered by base task",
        "confidence": 0.95,
        "iterations": iterations,
        "seed": seed,
        "metrics": {
            name: {
                "estimate": estimates[name],
                "lower": _quantile(values, 0.025),
                "upper": _quantile(values, 0.975),
            }
            for name, values in sorted(distributions.items())
        },
    }


def _exact_mcnemar_pvalue(base_only_wrong: int, transformed_only_wrong: int) -> float:
    discordant = base_only_wrong + transformed_only_wrong
    if discordant == 0:
        return 1.0
    tail = min(base_only_wrong, transformed_only_wrong)
    probability = sum(
        math.comb(discordant, index) * (0.5 ** discordant)
        for index in range(tail + 1)
    )
    return min(1.0, 2.0 * probability)


def _paired_changes_by_variant(
    by_pair: Dict[str, List[BenchmarkCase]], predictions: Dict[str, Prediction]
) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "both_correct": 0,
            "base_correct_transformed_wrong": 0,
            "base_wrong_transformed_correct": 0,
            "both_wrong": 0,
        }
    )
    for pair_cases in by_pair.values():
        bases = [case for case in pair_cases if case.variant == "base"]
        if len(bases) != 1:
            continue
        base = bases[0]
        base_prediction = predictions.get(base.case_id)
        base_correct = (
            prediction_correct(base, base_prediction) if base_prediction else False
        )
        for transformed in pair_cases:
            if transformed.variant == "base":
                continue
            transformed_prediction = predictions.get(transformed.case_id)
            transformed_correct = (
                prediction_correct(transformed, transformed_prediction)
                if transformed_prediction
                else False
            )
            if base_correct and transformed_correct:
                key = "both_correct"
            elif base_correct:
                key = "base_correct_transformed_wrong"
            elif transformed_correct:
                key = "base_wrong_transformed_correct"
            else:
                key = "both_wrong"
            counts[transformed.variant][key] += 1

    result = {}
    for variant, values in sorted(counts.items()):
        result[variant] = dict(values)
        result[variant]["mcnemar_exact_pvalue"] = _exact_mcnemar_pvalue(
            values["base_correct_transformed_wrong"],
            values["base_wrong_transformed_correct"],
        )
    return result


def evaluate(
    cases: List[BenchmarkCase], prediction_list: List[Prediction]
) -> Dict[str, Any]:
    predictions = {prediction.case_id: prediction for prediction in prediction_list}
    duplicate_count = len(prediction_list) - len(predictions)
    case_ids = {case.case_id for case in cases}
    unknown_predictions = sorted(set(predictions) - case_ids)

    base_cases = [case for case in cases if case.variant == "base"]
    transformed_cases = [case for case in cases if case.variant != "base"]

    by_variant: Dict[str, List[BenchmarkCase]] = defaultdict(list)
    by_family: Dict[str, List[BenchmarkCase]] = defaultdict(list)
    by_pair: Dict[str, List[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        by_variant[case.variant].append(case)
        by_family[case.family].append(case)
        by_pair[case.pair_id].append(case)

    pair_total = 0
    pair_consistent = 0
    eligible_for_flip = 0
    conditional_flips = 0
    successful_pairs = 0
    silent_disagreements = 0

    for pair_cases in by_pair.values():
        bases = [case for case in pair_cases if case.variant == "base"]
        if len(bases) != 1:
            continue
        base = bases[0]
        base_prediction = predictions.get(base.case_id)
        base_is_correct = (
            prediction_correct(base, base_prediction) if base_prediction else False
        )
        for transformed in pair_cases:
            if transformed.variant == "base":
                continue
            pair_total += 1
            transformed_prediction = predictions.get(transformed.case_id)
            transformed_is_correct = (
                prediction_correct(transformed, transformed_prediction)
                if transformed_prediction
                else False
            )
            if base_is_correct:
                eligible_for_flip += 1
                if not transformed_is_correct:
                    conditional_flips += 1

            if base_prediction and transformed_prediction:
                is_consistent = (
                    base_prediction.status == "ok"
                    and transformed_prediction.status == "ok"
                    and values_equivalent(
                        base_prediction.value,
                        transformed_prediction.value,
                        base.answer,
                    )
                )
                if is_consistent:
                    pair_consistent += 1
                if (
                    base_prediction.status == "ok"
                    and transformed_prediction.status == "ok"
                ):
                    successful_pairs += 1
                    if not is_consistent:
                        silent_disagreements += 1

    latencies = [
        prediction.latency_ms
        for prediction in prediction_list
        if prediction.latency_ms is not None
    ]
    usage_keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    )
    token_usage = {
        key: sum(
            int(prediction.metadata.get("usage", {}).get(key, 0) or 0)
            for prediction in prediction_list
            if isinstance(prediction.metadata.get("usage"), dict)
        )
        for key in usage_keys
    }
    agent_names = sorted({prediction.agent for prediction in prediction_list})
    return {
        "benchmark_case_count": len(cases),
        "prediction_count": len(prediction_list),
        "agents": agent_names,
        "integrity": {
            "duplicate_predictions": duplicate_count,
            "unknown_prediction_ids": unknown_predictions,
            "missing_predictions": len(case_ids - set(predictions)),
        },
        "overall": _slice_metrics(cases, predictions),
        "base": _slice_metrics(base_cases, predictions),
        "transformed": _slice_metrics(transformed_cases, predictions),
        "paired": {
            "pair_comparisons": pair_total,
            "consistent": pair_consistent,
            "paired_consistency": _ratio(pair_consistent, pair_total),
            "base_correct_comparisons": eligible_for_flip,
            "conditional_flips": conditional_flips,
            "conditional_flip_rate": _optional_ratio(
                conditional_flips, eligible_for_flip
            ),
            "successful_pair_comparisons": successful_pairs,
            "silent_disagreements": silent_disagreements,
            "silent_disagreement_rate": _optional_ratio(
                silent_disagreements, successful_pairs
            ),
        },
        "by_variant": {
            name: _slice_metrics(items, predictions)
            for name, items in sorted(by_variant.items())
        },
        "by_family": {
            name: _slice_metrics(items, predictions)
            for name, items in sorted(by_family.items())
        },
        "confidence_intervals": _cluster_bootstrap(by_pair, predictions),
        "paired_changes_by_variant": _paired_changes_by_variant(by_pair, predictions),
        "latency_ms": {
            "count": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
        "token_usage": token_usage,
    }


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.2f%%" % (100.0 * value)


def report_markdown(metrics: Dict[str, Any]) -> str:
    overall = metrics["overall"]
    paired = metrics["paired"]
    lines = [
        "# DataInvariantBench evaluation",
        "",
        "Agents: `%s`" % ", ".join(metrics["agents"]),
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Overall accuracy | %s |" % _percent(overall["accuracy"]),
        "| Base accuracy | %s |" % _percent(metrics["base"]["accuracy"]),
        "| Transformed accuracy | %s |"
        % _percent(metrics["transformed"]["accuracy"]),
        "| Paired consistency | %s |"
        % _percent(paired["paired_consistency"]),
        "| Conditional flip rate | %s |"
        % _percent(paired["conditional_flip_rate"]),
        "| Silent disagreement rate | %s |"
        % _percent(paired["silent_disagreement_rate"]),
        "| Execution error rate | %s |"
        % _percent(overall["execution_error_rate"]),
        "",
        "## Accuracy by variant",
        "",
        "| Variant | Cases | Accuracy | Errors |",
        "|---|---:|---:|---:|",
    ]
    for variant, values in metrics["by_variant"].items():
        lines.append(
            "| %s | %d | %s | %d |"
            % (
                variant,
                values["count"],
                _percent(values["accuracy"]),
                values["execution_errors"],
            )
        )
    lines.extend(
        [
            "",
            "## Accuracy by family",
            "",
            "| Family | Cases | Accuracy | Errors |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, values in metrics["by_family"].items():
        lines.append(
            "| %s | %d | %s | %d |"
            % (
                family,
                values["count"],
                _percent(values["accuracy"]),
                values["execution_errors"],
            )
        )
    lines.extend(
        [
            "",
            "## Resource use",
            "",
            "| Field | Value |",
            "|---|---:|",
        ]
    )
    for name, value in metrics["token_usage"].items():
        if value:
            lines.append("| %s | %d |" % (name, value))
    if not any(metrics["token_usage"].values()):
        lines.append("| token usage | not reported by agent |")
    mean_latency = metrics["latency_ms"]["mean"]
    lines.append(
        "| mean latency (ms) | %s |"
        % ("n/a" if mean_latency is None else "%.2f" % mean_latency)
    )
    lines.extend(
        [
            "",
            "## Pair-cluster bootstrap intervals",
            "",
            "| Metric | Estimate | 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, values in metrics["confidence_intervals"]["metrics"].items():
        lines.append(
            "| %s | %s | %s--%s |"
            % (
                name,
                _percent(values["estimate"]),
                _percent(values["lower"]),
                _percent(values["upper"]),
            )
        )
    lines.extend(
        [
            "",
            "> This report is descriptive pilot output. Its bootstrap intervals are "
            "uncertainty summaries, not confirmatory results. Do not use it as a paper result "
            "until the task set, hypotheses, and protocol are frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(metrics: Dict[str, Any], output: Path) -> Tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "metrics.json"
    markdown_path = output / "report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(report_markdown(metrics))
    return json_path, markdown_path
