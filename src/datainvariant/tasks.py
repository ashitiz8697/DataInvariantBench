from __future__ import annotations

import datetime as dt
import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, Dict, List

from .schema import AnswerSpec, BenchmarkCase, ColumnSpec, TableSpec
from .transformations import (
    booleans_to_text,
    currency_usd_to_cents,
    explicit_missing_marker,
    integer_booleans_to_text,
    normalize_region_dimension,
    permute_columns,
    permute_rows,
    rename_columns,
    rescale_numeric_column,
    timestamps_to_ist,
    timestamps_to_unix_seconds,
)


def _round_half_up(value: Decimal, places: int = 2) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(value.quantize(quantum, rounding=ROUND_HALF_UP))


def _base_case(
    pair_id: str,
    family: str,
    question: str,
    tables: List[TableSpec],
    answer: AnswerSpec,
    tags: List[str],
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=pair_id + "--base",
        pair_id=pair_id,
        family=family,
        variant="base",
        question=question,
        tables=tables,
        answer=answer,
        transformation_contract={
            "name": "identity",
            "semantic_effect": "none",
        },
        tags=["base"] + tags,
    )


def make_revenue(seed: int) -> BenchmarkCase:
    rng = random.Random(1_000 + seed)
    rows = []
    price_cents_by_order: Dict[str, int] = {}
    statuses = ["completed", "completed", "completed", "pending", "cancelled"]
    for index in range(16):
        order_id = "O-%03d-%02d" % (seed, index + 1)
        price_cents = rng.randint(250, 25_000)
        price_cents_by_order[order_id] = price_cents
        rows.append(
            {
                "order_id": order_id,
                "quantity": rng.randint(1, 8),
                "unit_price_usd": price_cents / 100.0,
                "status": rng.choice(statuses),
            }
        )
    rows[0]["status"] = "completed"
    total_cents = sum(
        int(row["quantity"]) * price_cents_by_order[str(row["order_id"])]
        for row in rows
        if row["status"] == "completed"
    )
    value = total_cents / 100.0
    table = TableSpec(
        name="orders",
        description="One row per commerce order.",
        columns=[
            ColumnSpec("order_id", "TEXT", "Unique order identifier."),
            ColumnSpec("quantity", "INTEGER", "Number of units in the order.", "units"),
            ColumnSpec(
                "unit_price_usd", "REAL", "Price for one unit in US dollars.", "USD"
            ),
            ColumnSpec(
                "status",
                "TEXT",
                "Order lifecycle status; recognized orders have value 'completed'.",
            ),
        ],
        rows=rows,
        primary_key=["order_id"],
    )
    return _base_case(
        "revenue-s%03d" % seed,
        "recognized_revenue",
        "What is the total recognized revenue in USD from completed orders? "
        "Multiply quantity by unit price and round the final result to two decimal places "
        "using half-up rounding.",
        [table],
        AnswerSpec(
            value,
            "number",
            unit="USD",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["money", "filter", "aggregation", "multiplication"],
    )


def make_top_region(seed: int) -> BenchmarkCase:
    rng = random.Random(2_000 + seed)
    rows = []
    regions = ["East", "North", "South", "West"]
    statuses = ["completed", "completed", "completed", "pending", "cancelled"]
    for region in regions:
        for local_index in range(6):
            rows.append(
                {
                    "order_id": "R-%03d-%s-%02d" % (seed, region[0], local_index + 1),
                    "region": region,
                    "amount_usd": rng.randint(1_000, 80_000) / 100.0,
                    "status": rng.choice(statuses),
                }
            )
    for index, region in enumerate(regions):
        rows[index * 6]["status"] = "completed"
        rows[index * 6]["amount_usd"] += index * 0.13

    totals: Dict[str, float] = {region: 0.0 for region in regions}
    for row in rows:
        if row["status"] == "completed":
            totals[row["region"]] += row["amount_usd"]
    winner = sorted(totals, key=lambda region: (-totals[region], region))[0]

    table = TableSpec(
        name="orders",
        description="Order facts with a denormalized region label.",
        columns=[
            ColumnSpec("order_id", "TEXT", "Unique order identifier."),
            ColumnSpec("region", "TEXT", "Sales region display name."),
            ColumnSpec("amount_usd", "REAL", "Total order amount in US dollars.", "USD"),
            ColumnSpec(
                "status",
                "TEXT",
                "Order lifecycle status; recognized orders have value 'completed'.",
            ),
        ],
        rows=rows,
        primary_key=["order_id"],
    )
    return _base_case(
        "region-s%03d" % seed,
        "top_region",
        "Which region has the highest total recognized revenue from completed orders? "
        "Return the region name; break an exact tie alphabetically.",
        [table],
        AnswerSpec(winner, "text"),
        ["money", "filter", "grouping", "argmax"],
    )


def _utc_string(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_resolution(seed: int) -> BenchmarkCase:
    rng = random.Random(3_000 + seed)
    origin = dt.datetime(2026, 1, 3, 8, 0, 0)
    rows = []
    durations = []
    for index in range(18):
        opened = origin + dt.timedelta(days=index, hours=rng.randint(0, 8))
        priority = rng.choice(["high", "high", "normal", "low"])
        is_open = rng.random() < 0.18
        duration_hours = rng.randint(2, 72)
        closed = None if is_open else opened + dt.timedelta(hours=duration_hours)
        if priority == "high" and closed is not None:
            durations.append(duration_hours)
        rows.append(
            {
                "ticket_id": "T-%03d-%02d" % (seed, index + 1),
                "opened_at_utc": _utc_string(opened),
                "closed_at_utc": _utc_string(closed) if closed else None,
                "priority": priority,
            }
        )
    if not durations:
        rows[0]["priority"] = "high"
        rows[0]["closed_at_utc"] = _utc_string(origin + dt.timedelta(hours=12))
        durations = [12]
    value = _round_half_up(Decimal(sum(durations)) / Decimal(len(durations)), 2)

    table = TableSpec(
        name="tickets",
        description="Support tickets and their lifecycle timestamps.",
        columns=[
            ColumnSpec("ticket_id", "TEXT", "Unique ticket identifier."),
            ColumnSpec(
                "opened_at_utc",
                "TEXT",
                "Ticket creation instant as an ISO-8601 UTC timestamp.",
                "UTC timestamp",
            ),
            ColumnSpec(
                "closed_at_utc",
                "TEXT",
                "Ticket closure instant as an ISO-8601 UTC timestamp; NULL means open.",
                "UTC timestamp",
            ),
            ColumnSpec("priority", "TEXT", "Ticket priority label."),
        ],
        rows=rows,
        primary_key=["ticket_id"],
    )
    return _base_case(
        "resolution-s%03d" % seed,
        "average_resolution_hours",
        "What is the average resolution time in hours for closed high-priority tickets? "
        "Exclude open tickets and round the final result to two decimal places using "
        "half-up rounding.",
        [table],
        AnswerSpec(
            value,
            "number",
            unit="hours",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["time", "filter", "aggregation", "missingness"],
    )


def make_conversion(seed: int) -> BenchmarkCase:
    rng = random.Random(4_000 + seed)
    rows = []
    channels = ["direct", "email", "organic", "paid"]
    for index in range(30):
        eligible = 1 if rng.random() < 0.78 else 0
        roll = rng.random()
        converted = None if roll < 0.18 else (1 if roll < 0.53 else 0)
        rows.append(
            {
                "session_id": "S-%03d-%02d" % (seed, index + 1),
                "eligible": eligible,
                "converted": converted,
                "channel": rng.choice(channels),
            }
        )
    known_eligible = [
        row for row in rows if row["eligible"] == 1 and row["converted"] is not None
    ]
    if not known_eligible:
        rows[0]["eligible"] = 1
        rows[0]["converted"] = 1
        known_eligible = [rows[0]]
    value = _round_half_up(
        Decimal(100)
        * Decimal(sum(int(row["converted"]) for row in known_eligible))
        / Decimal(len(known_eligible)),
        2,
    )
    table = TableSpec(
        name="sessions",
        description="One row per product session.",
        columns=[
            ColumnSpec("session_id", "TEXT", "Unique session identifier."),
            ColumnSpec("eligible", "INTEGER", "Boolean: 1 is eligible and 0 is not."),
            ColumnSpec(
                "converted",
                "INTEGER",
                "Boolean outcome: 1 converted, 0 did not convert, NULL is unknown.",
                missing_value="SQL NULL",
            ),
            ColumnSpec("channel", "TEXT", "Acquisition channel; not used by this question."),
        ],
        rows=rows,
        primary_key=["session_id"],
    )
    return _base_case(
        "conversion-s%03d" % seed,
        "eligible_conversion_rate",
        "What percentage of eligible sessions converted among sessions with a known "
        "conversion outcome? Exclude unknown outcomes and round to two decimal places "
        "using half-up rounding.",
        [table],
        AnswerSpec(
            value,
            "number",
            unit="percent",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["boolean", "filter", "aggregation", "missingness"],
    )


def make_inventory(seed: int) -> BenchmarkCase:
    rng = random.Random(5_000 + seed)
    rows = []
    total_cents = 0
    for index in range(22):
        quantity = rng.randint(0, 180)
        cost_cents = rng.randint(75, 18_000)
        active = 1 if rng.random() < 0.8 else 0
        rows.append(
            {
                "sku": "SKU-%03d-%02d" % (seed, index + 1),
                "quantity_on_hand": quantity,
                "unit_cost_usd": cost_cents / 100.0,
                "active": active,
            }
        )
        if active:
            total_cents += quantity * cost_cents
    table = TableSpec(
        name="inventory",
        description="Current inventory balance by stock-keeping unit.",
        columns=[
            ColumnSpec("sku", "TEXT", "Unique stock-keeping unit."),
            ColumnSpec(
                "quantity_on_hand", "INTEGER", "Physical units currently held.", "units"
            ),
            ColumnSpec("unit_cost_usd", "REAL", "Cost per physical unit.", "USD"),
            ColumnSpec("active", "INTEGER", "Boolean: 1 is active and 0 is inactive."),
        ],
        rows=rows,
        primary_key=["sku"],
    )
    return _base_case(
        "inventory-s%03d" % seed,
        "active_inventory_value",
        "What is the total inventory value in USD for active SKUs? Multiply quantity "
        "on hand by unit cost and round the final result to two decimal places using "
        "half-up rounding.",
        [table],
        AnswerSpec(
            total_cents / 100.0,
            "number",
            unit="USD",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["inventory", "money", "filter", "multiplication"],
    )


def make_cashflow(seed: int) -> BenchmarkCase:
    rng = random.Random(6_000 + seed)
    rows = []
    net_cents = 0
    for index in range(28):
        amount_cents = rng.randint(1_000, 250_000)
        direction = rng.choice(["inflow", "inflow", "outflow"])
        settled = 1 if rng.random() < 0.82 else 0
        rows.append(
            {
                "transaction_id": "CF-%03d-%02d" % (seed, index + 1),
                "direction": direction,
                "amount_usd": amount_cents / 100.0,
                "settled": settled,
                "category": rng.choice(["customer", "supplier", "tax", "other"]),
            }
        )
        if settled:
            net_cents += amount_cents if direction == "inflow" else -amount_cents
    table = TableSpec(
        name="cash_transactions",
        description="Cash movements before and after settlement.",
        columns=[
            ColumnSpec("transaction_id", "TEXT", "Unique cash transaction identifier."),
            ColumnSpec(
                "direction", "TEXT", "Cash direction: inflow adds and outflow subtracts."
            ),
            ColumnSpec("amount_usd", "REAL", "Non-negative transaction magnitude.", "USD"),
            ColumnSpec("settled", "INTEGER", "Boolean: 1 is settled and 0 is pending."),
            ColumnSpec("category", "TEXT", "Accounting category; not used by this question."),
        ],
        rows=rows,
        primary_key=["transaction_id"],
    )
    return _base_case(
        "cashflow-s%03d" % seed,
        "settled_net_cash_flow",
        "What is settled net cash flow in USD? Add settled inflows, subtract settled "
        "outflows, ignore pending transactions, and round to two decimal places using "
        "half-up rounding.",
        [table],
        AnswerSpec(
            net_cents / 100.0,
            "number",
            unit="USD",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["finance", "money", "signed_aggregation", "filter"],
    )


def make_weighted_quality(seed: int) -> BenchmarkCase:
    rng = random.Random(7_000 + seed)
    rows = []
    weighted_numerator = Decimal(0)
    weight_total = Decimal(0)
    for index in range(18):
        score_hundredths = rng.randint(4_000, 10_000)
        weight_thousandths = rng.randint(10, 180)
        included = 1 if rng.random() < 0.82 else 0
        rows.append(
            {
                "metric_id": "QM-%03d-%02d" % (seed, index + 1),
                "score": score_hundredths / 100.0,
                "weight_fraction": weight_thousandths / 1000.0,
                "included": included,
            }
        )
        if included:
            weighted_numerator += Decimal(score_hundredths) * Decimal(weight_thousandths)
            weight_total += Decimal(weight_thousandths)
    value = _round_half_up(weighted_numerator / weight_total / Decimal(100), 2)
    table = TableSpec(
        name="quality_metrics",
        description="Component metrics used in a weighted quality assessment.",
        columns=[
            ColumnSpec("metric_id", "TEXT", "Unique metric identifier."),
            ColumnSpec("score", "REAL", "Metric score on a 0 to 100 scale.", "points"),
            ColumnSpec(
                "weight_fraction", "REAL", "Relative non-negative weighting fraction."
            ),
            ColumnSpec("included", "INTEGER", "Boolean: 1 is included and 0 excluded."),
        ],
        rows=rows,
        primary_key=["metric_id"],
    )
    return _base_case(
        "quality-s%03d" % seed,
        "weighted_quality_score",
        "What is the weighted average score among included quality metrics? Divide the "
        "sum of score times weight by the sum of weights and round to two decimal places "
        "using half-up rounding.",
        [table],
        AnswerSpec(
            value,
            "number",
            unit="points",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["quality", "weighted_average", "filter"],
    )


def make_sla(seed: int) -> BenchmarkCase:
    rng = random.Random(8_000 + seed)
    rows = []
    resolved_count = 0
    compliant_count = 0
    for index in range(32):
        response_minutes = rng.randint(2, 480)
        target_minutes = rng.choice([30, 60, 120, 240])
        resolved = 1 if rng.random() < 0.86 else 0
        rows.append(
            {
                "incident_id": "INC-%03d-%02d" % (seed, index + 1),
                "response_minutes": response_minutes,
                "target_minutes": target_minutes,
                "resolved": resolved,
                "severity": rng.choice(["sev1", "sev2", "sev3"]),
            }
        )
        if resolved:
            resolved_count += 1
            compliant_count += int(response_minutes <= target_minutes)
    value = _round_half_up(
        Decimal(100) * Decimal(compliant_count) / Decimal(resolved_count), 2
    )
    table = TableSpec(
        name="incidents",
        description="Operational incidents and first-response service targets.",
        columns=[
            ColumnSpec("incident_id", "TEXT", "Unique incident identifier."),
            ColumnSpec("response_minutes", "INTEGER", "Actual first-response duration.", "minutes"),
            ColumnSpec("target_minutes", "INTEGER", "Maximum compliant response duration.", "minutes"),
            ColumnSpec("resolved", "INTEGER", "Boolean: 1 is resolved and 0 is unresolved."),
            ColumnSpec("severity", "TEXT", "Incident severity; not used by this question."),
        ],
        rows=rows,
        primary_key=["incident_id"],
    )
    return _base_case(
        "sla-s%03d" % seed,
        "resolved_sla_compliance",
        "What percentage of resolved incidents met their response target? An incident "
        "meets the target when response duration is less than or equal to target duration. "
        "Round to two decimal places using half-up rounding.",
        [table],
        AnswerSpec(
            value,
            "number",
            unit="percent",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["operations", "duration", "comparison", "rate"],
    )


def make_energy(seed: int) -> BenchmarkCase:
    rng = random.Random(9_000 + seed)
    rows = []
    valid_wh = 0
    for index in range(30):
        energy_wh = rng.randint(100, 45_000)
        valid = 1 if rng.random() < 0.88 else 0
        rows.append(
            {
                "reading_id": "EN-%03d-%02d" % (seed, index + 1),
                "energy_kwh": energy_wh / 1000.0,
                "valid": valid,
                "source": rng.choice(["solar", "wind", "grid"]),
            }
        )
        if valid:
            valid_wh += energy_wh
    table = TableSpec(
        name="energy_readings",
        description="Interval energy readings from monitored sources.",
        columns=[
            ColumnSpec("reading_id", "TEXT", "Unique meter reading identifier."),
            ColumnSpec("energy_kwh", "REAL", "Energy measured in kilowatt-hours.", "kWh"),
            ColumnSpec("valid", "INTEGER", "Boolean: 1 is valid and 0 is invalid."),
            ColumnSpec("source", "TEXT", "Energy source; all sources are included."),
        ],
        rows=rows,
        primary_key=["reading_id"],
    )
    return _base_case(
        "energy-s%03d" % seed,
        "valid_energy_kwh",
        "What is the total energy in kWh across valid readings from all sources? Exclude "
        "invalid readings and round to two decimal places using half-up rounding.",
        [table],
        AnswerSpec(
            _round_half_up(Decimal(valid_wh) / Decimal(1000), 2),
            "number",
            unit="kWh",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["energy", "unit", "filter", "aggregation"],
    )


def make_population_density(seed: int) -> BenchmarkCase:
    rng = random.Random(10_000 + seed)
    rows = []
    total_population = 0
    total_area_hundredths = 0
    for index in range(14):
        population = rng.randint(20_000, 1_500_000)
        area_hundredths = rng.randint(1_000, 85_000)
        rows.append(
            {
                "municipality": "Municipality-%03d-%02d" % (seed, index + 1),
                "population": population,
                "area_sq_km": area_hundredths / 100.0,
                "region_type": rng.choice(["urban", "mixed", "rural"]),
            }
        )
        total_population += population
        total_area_hundredths += area_hundredths
    value = _round_half_up(
        Decimal(total_population) / (Decimal(total_area_hundredths) / Decimal(100)), 2
    )
    table = TableSpec(
        name="municipalities",
        description="Population and land area for municipalities in one study region.",
        columns=[
            ColumnSpec("municipality", "TEXT", "Unique municipality display name."),
            ColumnSpec("population", "INTEGER", "Resident population.", "people"),
            ColumnSpec("area_sq_km", "REAL", "Land area in square kilometres.", "km^2"),
            ColumnSpec("region_type", "TEXT", "Settlement type; all types are included."),
        ],
        rows=rows,
        primary_key=["municipality"],
    )
    return _base_case(
        "density-s%03d" % seed,
        "regional_population_density",
        "What is the aggregate population density of the study region in people per "
        "square kilometre? Divide total population by total land area (do not average "
        "municipality densities) and round to two decimal places using half-up rounding.",
        [table],
        AnswerSpec(
            value,
            "number",
            unit="people/km^2",
            tolerance=0.005,
            rounding=2,
            rounding_mode="half_up",
        ),
        ["public_data", "density", "ratio", "unit"],
    )

BUILDERS: List[Callable[[int], BenchmarkCase]] = [
    make_revenue,
    make_top_region,
    make_resolution,
    make_conversion,
    make_inventory,
    make_cashflow,
    make_weighted_quality,
    make_sla,
    make_energy,
    make_population_density,
]


def variants_for(base: BenchmarkCase, seed: int) -> List[BenchmarkCase]:
    rng = random.Random(10_000 + seed * 97 + len(base.family))
    common = [base, permute_rows(base, rng), permute_columns(base, rng)]

    if base.family == "recognized_revenue":
        return common + [
            currency_usd_to_cents(base),
            rename_columns(
                base,
                "orders",
                {
                    "quantity": "units_purchased",
                    "unit_price_usd": "price_each_usd",
                    "status": "lifecycle_state",
                },
            ),
        ]
    if base.family == "top_region":
        return common + [
            normalize_region_dimension(base),
            rename_columns(
                base,
                "orders",
                {
                    "region": "sales_territory",
                    "amount_usd": "recognized_amount_usd",
                    "status": "lifecycle_state",
                },
            ),
        ]
    if base.family == "average_resolution_hours":
        return common + [timestamps_to_ist(base), timestamps_to_unix_seconds(base)]
    if base.family == "eligible_conversion_rate":
        return common + [explicit_missing_marker(base), booleans_to_text(base)]
    if base.family == "active_inventory_value":
        return common + [
            rescale_numeric_column(
                base,
                "inventory",
                "unit_cost_usd",
                "unit_cost_cents",
                100.0,
                "INTEGER",
                "USD cents",
                "Cost per physical unit in integer US cents; divide by 100 for USD.",
                "cost_cents",
                "divide by 100",
            ),
            integer_booleans_to_text(base, "inventory", ["active"], "active_text"),
        ]
    if base.family == "settled_net_cash_flow":
        return common + [
            rescale_numeric_column(
                base,
                "cash_transactions",
                "amount_usd",
                "amount_cents",
                100.0,
                "INTEGER",
                "USD cents",
                "Non-negative transaction magnitude in integer US cents.",
                "amount_cents",
                "divide by 100",
            ),
            rename_columns(
                base,
                "cash_transactions",
                {
                    "direction": "cash_direction",
                    "amount_usd": "transaction_value_usd",
                    "settled": "is_finalized",
                },
            ),
        ]
    if base.family == "weighted_quality_score":
        return common + [
            rescale_numeric_column(
                base,
                "quality_metrics",
                "weight_fraction",
                "weight_percent",
                100.0,
                "REAL",
                "percent weight",
                "Relative weight expressed as a percentage; ratios must divide by the sum of weights.",
                "weight_percent",
                "divide by 100",
            ),
            integer_booleans_to_text(
                base, "quality_metrics", ["included"], "included_text"
            ),
        ]
    if base.family == "resolved_sla_compliance":
        seconds_case = rescale_numeric_column(
            base,
            "incidents",
            "response_minutes",
            "response_seconds",
            60.0,
            "INTEGER",
            "seconds",
            "Actual first-response duration in seconds.",
            "duration_seconds",
            "divide by 60",
        )
        seconds_case = rescale_numeric_column(
            seconds_case,
            "incidents",
            "target_minutes",
            "target_seconds",
            60.0,
            "INTEGER",
            "seconds",
            "Maximum compliant response duration in seconds.",
            "duration_seconds",
            "divide by 60",
        )
        seconds_case.case_id = base.pair_id + "--duration-seconds"
        seconds_case.transformation_contract = {
            "name": "duration_seconds",
            "semantic_effect": "both actual and target durations are rescaled from minutes to seconds",
            "mapping": {
                "response_minutes": "response_seconds",
                "target_minutes": "target_seconds",
            },
            "scale_factor": 60,
            "inverse": "divide both duration fields by 60",
        }
        return common + [
            seconds_case,
            rename_columns(
                base,
                "incidents",
                {
                    "response_minutes": "actual_response_minutes",
                    "target_minutes": "sla_limit_minutes",
                    "resolved": "is_resolved",
                },
            ),
        ]
    if base.family == "valid_energy_kwh":
        return common + [
            rescale_numeric_column(
                base,
                "energy_readings",
                "energy_kwh",
                "energy_wh",
                1000.0,
                "INTEGER",
                "Wh",
                "Energy in watt-hours; divide by 1000 to obtain kWh.",
                "energy_wh",
                "divide by 1000",
            ),
            integer_booleans_to_text(base, "energy_readings", ["valid"], "valid_text"),
        ]
    if base.family == "regional_population_density":
        return common + [
            rescale_numeric_column(
                base,
                "municipalities",
                "area_sq_km",
                "area_sq_miles",
                0.3861021585424458,
                "REAL",
                "mi^2",
                "Land area in square miles; divide by 0.3861021585424458 for square kilometres.",
                "area_sq_miles",
                "divide by 0.3861021585424458",
            ),
            rename_columns(
                base,
                "municipalities",
                {"population": "resident_count", "area_sq_km": "land_area_km2"},
            ),
        ]
    raise ValueError("unknown family: %s" % base.family)


def generate_cases(seeds: List[int]) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    for seed in seeds:
        for builder in BUILDERS:
            base = builder(seed)
            cases.extend(variants_for(base, seed))
    return cases
