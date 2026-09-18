from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import List

from .agents import HTTPModelSQLAgent, NaiveSQLAgent, OracleAgent
from .evaluation import evaluate, write_report
from .storage import (
    read_benchmark,
    read_predictions,
    file_sha256,
    write_benchmark,
    write_json,
    write_predictions,
)
from .tasks import generate_cases
from .study import (
    DEFAULT_DBU_PRICE_USD,
    DEFAULT_GST_RATE,
    DEFAULT_USD_INR,
    build_study_plan,
    evaluate_databricks_study,
    model_specs_by_id,
    run_databricks_study,
)


def _parse_seeds(value: str) -> List[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _agent(name: str, prompt_condition: str = "contract"):
    if name == "oracle":
        return OracleAgent()
    if name == "naive":
        return NaiveSQLAgent()
    if name == "http-sql":
        return HTTPModelSQLAgent.from_environment(prompt_condition=prompt_condition)
    raise ValueError("unknown agent: %s" % name)


def command_generate(args: argparse.Namespace) -> int:
    cases = generate_cases(args.seeds)
    count = write_benchmark(cases, args.output)
    print("generated %d cases at %s" % (count, args.output))
    return 0


def command_run(args: argparse.Namespace) -> int:
    cases = read_benchmark(args.benchmark)
    if args.limit is not None:
        cases = cases[: args.limit]
    agent = _agent(args.agent, prompt_condition=args.prompt_condition)
    predictions = []
    for index, case in enumerate(cases, start=1):
        prediction = agent.predict(case)
        predictions.append(prediction)
        if args.progress:
            print(
                "[%d/%d] %s: %s"
                % (index, len(cases), case.case_id, prediction.status),
                file=sys.stderr,
            )
    count = write_predictions(predictions, args.output)
    run_metadata_path = Path(str(args.output) + ".run.json")
    write_json(
        {
            "agent": agent.name,
            "benchmark_manifest": str(args.benchmark / "manifest.jsonl"),
            "benchmark_manifest_sha256": file_sha256(
                args.benchmark / "manifest.jsonl"
            ),
            "case_count": count,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "prompt_condition": args.prompt_condition,
            "python": sys.version,
        },
        run_metadata_path,
    )
    print("wrote %d predictions to %s" % (count, args.output))
    print("wrote run metadata to %s" % run_metadata_path)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    cases = read_benchmark(args.benchmark)
    predictions = read_predictions(args.predictions)
    metrics = evaluate(cases, predictions)
    json_path, markdown_path = write_report(metrics, args.output)
    print("wrote metrics to %s" % json_path)
    print("wrote report to %s" % markdown_path)
    return 0


def _selected_cases(args: argparse.Namespace):
    cases = read_benchmark(args.benchmark)
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("case limit must be positive")
        cases = cases[: args.case_limit]
    return cases


def command_study_plan(args: argparse.Namespace) -> int:
    cases = _selected_cases(args)
    specs = model_specs_by_id(args.model)
    conditions = args.condition or ["schema", "contract"]
    plan = build_study_plan(
        cases,
        model_specs=specs,
        conditions=conditions,
        repetitions=args.repetitions,
        dbu_price_usd=args.dbu_price_usd,
        usd_inr=args.usd_inr,
        gst_rate=args.gst_rate,
        retry_allowance=args.retries,
    )
    if args.output:
        write_json(plan, args.output)
        print("wrote zero-cost study plan to %s" % args.output)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def command_databricks_run(args: argparse.Namespace) -> int:
    if not args.confirm_paid_run:
        raise ValueError(
            "paid inference is disabled; pass --confirm-paid-run after reviewing study-plan"
        )
    cases = _selected_cases(args)
    specs = model_specs_by_id(args.model)
    conditions = args.condition or ["schema", "contract"]
    result = run_databricks_study(
        cases=cases,
        benchmark_path=args.benchmark,
        output=args.output,
        model_specs=specs,
        conditions=conditions,
        repetitions=args.repetitions,
        budget_usd=args.budget_usd,
        workers=args.workers,
        retries=args.retries,
        dbu_price_usd=args.dbu_price_usd,
        progress=args.progress,
        requests_per_minute=args.requests_per_minute,
    )
    print(json.dumps(result["budget"], indent=2, sort_keys=True))
    print("wrote resumable study outputs to %s" % args.output)
    return 0


def command_study_evaluate(args: argparse.Namespace) -> int:
    cases = read_benchmark(args.benchmark)
    result = evaluate_databricks_study(cases, args.study_output, args.output)
    print(
        "evaluated %d runs; wrote aggregate report to %s"
        % (result["run_count"], args.output)
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datainvariant",
        description="Representation-invariance benchmark for data-analysis agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate benchmark cases")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--seeds", type=_parse_seeds, default=[0, 1, 2, 3, 4], help="default: 0,1,2,3,4"
    )
    generate.set_defaults(handler=command_generate)

    run = subparsers.add_parser("run", help="run an agent on generated cases")
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--agent", choices=["oracle", "naive", "http-sql"], required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--progress", action="store_true")
    run.add_argument(
        "--prompt-condition",
        choices=["schema", "contract"],
        default="contract",
        help="information exposed to an HTTP model; ignored by smoke agents",
    )
    run.set_defaults(handler=command_run)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate prediction JSONL"
    )
    evaluate_parser.add_argument("--benchmark", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(handler=command_evaluate)

    study_plan = subparsers.add_parser(
        "study-plan", help="estimate Databricks tokens and cost without making API calls"
    )
    study_plan.add_argument("--benchmark", type=Path, required=True)
    study_plan.add_argument("--output", type=Path)
    study_plan.add_argument("--case-limit", type=int)
    study_plan.add_argument("--model", action="append", help="repeat to select models")
    study_plan.add_argument(
        "--condition", choices=["schema", "contract"], action="append"
    )
    study_plan.add_argument("--repetitions", type=int, default=3)
    study_plan.add_argument("--retries", type=int, default=1)
    study_plan.add_argument("--dbu-price-usd", type=float, default=DEFAULT_DBU_PRICE_USD)
    study_plan.add_argument("--usd-inr", type=float, default=DEFAULT_USD_INR)
    study_plan.add_argument("--gst-rate", type=float, default=DEFAULT_GST_RATE)
    study_plan.set_defaults(handler=command_study_plan)

    databricks_run = subparsers.add_parser(
        "databricks-run", help="run the resumable, budget-controlled Databricks study"
    )
    databricks_run.add_argument("--benchmark", type=Path, required=True)
    databricks_run.add_argument("--output", type=Path, required=True)
    databricks_run.add_argument("--case-limit", type=int)
    databricks_run.add_argument("--model", action="append", help="repeat to select models")
    databricks_run.add_argument(
        "--condition", choices=["schema", "contract"], action="append"
    )
    databricks_run.add_argument("--repetitions", type=int, default=3)
    databricks_run.add_argument("--workers", type=int, default=4)
    databricks_run.add_argument("--retries", type=int, default=1)
    databricks_run.add_argument("--budget-usd", type=float, default=35.0)
    databricks_run.add_argument(
        "--dbu-price-usd", type=float, default=DEFAULT_DBU_PRICE_USD
    )
    databricks_run.add_argument("--progress", action="store_true")
    databricks_run.add_argument("--requests-per-minute", type=float, default=60.0)
    databricks_run.add_argument("--confirm-paid-run", action="store_true")
    databricks_run.set_defaults(handler=command_databricks_run)

    study_evaluate = subparsers.add_parser(
        "study-evaluate", help="evaluate and aggregate Databricks study repetitions"
    )
    study_evaluate.add_argument("--benchmark", type=Path, required=True)
    study_evaluate.add_argument("--study-output", type=Path, required=True)
    study_evaluate.add_argument("--output", type=Path, required=True)
    study_evaluate.set_defaults(handler=command_study_evaluate)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code = args.handler(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, "error: %s\n" % exc)
    raise SystemExit(exit_code)
