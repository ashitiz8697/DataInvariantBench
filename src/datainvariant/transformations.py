from __future__ import annotations

import calendar
import datetime as dt
import random
from typing import Dict, List, Sequence

from .schema import BenchmarkCase, ColumnSpec, TableSpec


def _table(case: BenchmarkCase, name: str) -> TableSpec:
    for table in case.tables:
        if table.name == name:
            return table
    raise KeyError("unknown table: %s" % name)


def permute_rows(base: BenchmarkCase, rng: random.Random) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--row-permutation", "row_permutation")
    for table in case.tables:
        rng.shuffle(table.rows)
    case.transformation_contract = {
        "name": "row_permutation",
        "semantic_effect": "none",
        "precondition": "row order is not part of the relational semantics",
    }
    case.tags.append("syntactic")
    return case


def permute_columns(base: BenchmarkCase, rng: random.Random) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--column-permutation", "column_permutation")
    for table in case.tables:
        rng.shuffle(table.columns)
        order = [column.name for column in table.columns]
        table.rows = [{name: row[name] for name in order} for row in table.rows]
    case.transformation_contract = {
        "name": "column_permutation",
        "semantic_effect": "none",
        "precondition": "columns are addressed by name rather than position",
    }
    case.tags.append("syntactic")
    return case


def rename_columns(
    base: BenchmarkCase, table_name: str, mapping: Dict[str, str]
) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--renamed-schema", "renamed_schema")
    table = _table(case, table_name)
    for column in table.columns:
        if column.name in mapping:
            old_name = column.name
            column.name = mapping[old_name]
            column.description = "%s Original semantic field: %s." % (
                column.description,
                old_name,
            )
    table.rows = [
        {mapping.get(name, name): value for name, value in row.items()}
        for row in table.rows
    ]
    if table.primary_key:
        table.primary_key = [mapping.get(name, name) for name in table.primary_key]
    case.transformation_contract = {
        "name": "renamed_schema",
        "semantic_effect": "column identifiers changed; descriptions preserve meaning",
        "mapping": mapping,
    }
    case.tags.extend(["semantic_schema", "renaming"])
    return case


def currency_usd_to_cents(base: BenchmarkCase) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--currency-cents", "currency_cents")
    table = _table(case, "orders")
    old_name = "unit_price_usd"
    new_name = "unit_price_cents"
    for column in table.columns:
        if column.name == old_name:
            column.name = new_name
            column.dtype = "INTEGER"
            column.unit = "USD cents"
            column.description = (
                "Price of one unit in integer US cents; divide by 100 to obtain USD."
            )
    for row in table.rows:
        row[new_name] = int(round(float(row.pop(old_name)) * 100.0))
    case.transformation_contract = {
        "name": "currency_cents",
        "semantic_effect": "values rescaled by 100 while monetary meaning is preserved",
        "source_unit": "USD",
        "target_unit": "USD cents",
        "inverse": "divide by 100",
    }
    case.tags.extend(["semantic_schema", "unit"])
    return case


def normalize_region_dimension(base: BenchmarkCase) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--normalized-region", "normalized_region")
    orders = _table(case, "orders")
    regions = sorted({str(row["region"]) for row in orders.rows})
    code_by_region = {region: "RG%02d" % (index + 1) for index, region in enumerate(regions)}

    for column in orders.columns:
        if column.name == "region":
            column.name = "region_id"
            column.description = "Foreign key into regions.region_id."
    for row in orders.rows:
        row["region_id"] = code_by_region[str(row.pop("region"))]
    orders.foreign_keys.append(
        {"column": "region_id", "references": "regions.region_id"}
    )

    dimension = TableSpec(
        name="regions",
        description="Region dimension mapping stable identifiers to display names.",
        columns=[
            ColumnSpec("region_id", "TEXT", "Stable region identifier."),
            ColumnSpec("region_name", "TEXT", "Human-readable region name."),
        ],
        rows=[
            {"region_id": code_by_region[region], "region_name": region}
            for region in regions
        ],
        primary_key=["region_id"],
    )
    case.tables.append(dimension)
    case.transformation_contract = {
        "name": "normalized_region",
        "semantic_effect": "region labels moved to a lossless dimension table",
        "join": "orders.region_id = regions.region_id",
    }
    case.tags.extend(["semantic_schema", "join", "normalization"])
    return case


def _parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def timestamps_to_ist(base: BenchmarkCase) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--timezone-ist", "timezone_ist")
    table = _table(case, "tickets")
    mapping = {
        "opened_at_utc": "opened_at_ist",
        "closed_at_utc": "closed_at_ist",
    }
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    for column in table.columns:
        if column.name in mapping:
            old_name = column.name
            column.name = mapping[old_name]
            column.unit = "ISO-8601 timestamp with +05:30 offset"
            column.description = (
                "The same instant as %s, represented in India Standard Time."
                % old_name
            )
    for row in table.rows:
        for old_name, new_name in mapping.items():
            value = row.pop(old_name)
            row[new_name] = (
                _parse_utc(value).astimezone(ist).isoformat(timespec="seconds")
                if value is not None
                else None
            )
    case.transformation_contract = {
        "name": "timezone_ist",
        "semantic_effect": "timestamps represent identical instants in UTC+05:30",
        "inverse": "convert to UTC",
    }
    case.tags.extend(["semantic_schema", "time", "timezone"])
    return case


def timestamps_to_unix_seconds(base: BenchmarkCase) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--unix-seconds", "unix_seconds")
    table = _table(case, "tickets")
    mapping = {
        "opened_at_utc": "opened_unix_seconds",
        "closed_at_utc": "closed_unix_seconds",
    }
    for column in table.columns:
        if column.name in mapping:
            old_name = column.name
            column.name = mapping[old_name]
            column.dtype = "INTEGER"
            column.unit = "seconds since Unix epoch (UTC)"
            column.description = "UTC instant encoded as Unix seconds."
    for row in table.rows:
        for old_name, new_name in mapping.items():
            value = row.pop(old_name)
            row[new_name] = (
                calendar.timegm(_parse_utc(value).utctimetuple())
                if value is not None
                else None
            )
    case.transformation_contract = {
        "name": "unix_seconds",
        "semantic_effect": "timestamps converted losslessly to Unix seconds",
        "duration_rule": "subtract timestamps and divide by 3600 for hours",
    }
    case.tags.extend(["semantic_schema", "time", "encoding"])
    return case


def explicit_missing_marker(base: BenchmarkCase) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--missing-marker", "missing_marker")
    table = _table(case, "sessions")
    for column in table.columns:
        if column.name == "converted":
            column.dtype = "TEXT"
            column.missing_value = "UNKNOWN"
            column.description = (
                "Conversion outcome encoded as '1' or '0'; 'UNKNOWN' means missing and "
                "must be excluded from the denominator."
            )
    for row in table.rows:
        value = row["converted"]
        row["converted"] = "UNKNOWN" if value is None else str(value)
    case.transformation_contract = {
        "name": "missing_marker",
        "semantic_effect": "SQL NULL replaced by a declared textual sentinel",
        "missing_marker": "UNKNOWN",
    }
    case.tags.extend(["semantic_schema", "missingness", "encoding"])
    return case


def booleans_to_text(base: BenchmarkCase) -> BenchmarkCase:
    case = base.clone(base.pair_id + "--text-booleans", "text_booleans")
    table = _table(case, "sessions")
    for column in table.columns:
        if column.name in ("eligible", "converted"):
            column.dtype = "TEXT"
            column.description = (
                "Boolean encoded as YES or NO. A missing conversion remains SQL NULL."
            )
    for row in table.rows:
        row["eligible"] = "YES" if int(row["eligible"]) == 1 else "NO"
        if row["converted"] is not None:
            row["converted"] = "YES" if int(row["converted"]) == 1 else "NO"
    case.transformation_contract = {
        "name": "text_booleans",
        "semantic_effect": "integer booleans converted losslessly to YES/NO labels",
    }
    case.tags.extend(["semantic_schema", "boolean", "encoding"])
    return case


def rescale_numeric_column(
    base: BenchmarkCase,
    table_name: str,
    old_name: str,
    new_name: str,
    scale_factor: float,
    dtype: str,
    unit: str,
    description: str,
    variant: str,
    inverse: str,
) -> BenchmarkCase:
    """Losslessly rescale a generated numeric column and declare its inverse."""
    if scale_factor == 0:
        raise ValueError("scale factor cannot be zero")
    case = base.clone(base.pair_id + "--" + variant.replace("_", "-"), variant)
    table = _table(case, table_name)
    found = False
    for column in table.columns:
        if column.name == old_name:
            found = True
            column.name = new_name
            column.dtype = dtype
            column.unit = unit
            column.description = description
    if not found:
        raise KeyError("unknown column: %s.%s" % (table_name, old_name))
    for row in table.rows:
        value = float(row.pop(old_name)) * scale_factor
        row[new_name] = int(round(value)) if dtype.upper() == "INTEGER" else value
    case.transformation_contract = {
        "name": variant,
        "semantic_effect": "numeric representation rescaled without changing meaning",
        "source_column": old_name,
        "target_column": new_name,
        "scale_factor": scale_factor,
        "inverse": inverse,
    }
    case.tags.extend(["semantic_schema", "unit"])
    return case


def integer_booleans_to_text(
    base: BenchmarkCase,
    table_name: str,
    columns: Sequence[str],
    variant: str,
    true_value: str = "YES",
    false_value: str = "NO",
) -> BenchmarkCase:
    """Encode selected 0/1 columns as declared text labels, preserving NULL."""
    case = base.clone(base.pair_id + "--" + variant.replace("_", "-"), variant)
    table = _table(case, table_name)
    selected = set(columns)
    found = set()
    for column in table.columns:
        if column.name in selected:
            found.add(column.name)
            column.dtype = "TEXT"
            column.description = (
                "Boolean encoded as %s for true and %s for false; SQL NULL remains "
                "missing." % (true_value, false_value)
            )
    if found != selected:
        raise KeyError("unknown boolean column(s): %s" % sorted(selected - found))
    for row in table.rows:
        for name in selected:
            if row[name] is not None:
                row[name] = true_value if int(row[name]) == 1 else false_value
    case.transformation_contract = {
        "name": variant,
        "semantic_effect": "integer booleans converted losslessly to text labels",
        "columns": sorted(selected),
        "true_value": true_value,
        "false_value": false_value,
    }
    case.tags.extend(["semantic_schema", "boolean", "encoding"])
    return case
