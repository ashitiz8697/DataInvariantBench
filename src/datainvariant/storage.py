from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

from .schema import BenchmarkCase, Prediction


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_benchmark(cases: Iterable[BenchmarkCase], output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    materialized = list(cases)

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for case in materialized:
            case_dir = cases_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            payload = case.to_dict()
            manifest.write(json.dumps(payload, sort_keys=True) + "\n")
            with (case_dir / "case.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            for table in case.tables:
                fieldnames = [column.name for column in table.columns]
                with (case_dir / (table.name + ".csv")).open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(table.rows)

    metadata = {
        "benchmark": "DataInvariantBench",
        "version": "0.2.0",
        "case_count": len(materialized),
        "pair_count": len({case.pair_id for case in materialized}),
        "families": sorted({case.family for case in materialized}),
        "variants": sorted({case.variant for case in materialized}),
        "manifest_sha256": file_sha256(output / "manifest.jsonl"),
    }
    with (output / "benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return len(materialized)


def read_benchmark(path: Path) -> List[BenchmarkCase]:
    manifest_path = path / "manifest.jsonl"
    cases = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(BenchmarkCase.from_dict(json.loads(line)))
    return cases


def write_predictions(predictions: Iterable[Prediction], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(predictions)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in materialized:
            handle.write(json.dumps(prediction.to_dict(), sort_keys=True) + "\n")
    return len(materialized)


def read_predictions(path: Path) -> List[Prediction]:
    predictions = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                predictions.append(Prediction.from_dict(json.loads(line)))
    return predictions


def write_json(value, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
