from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from .databricks import (
    DatabricksOAuthTokenProvider,
    chat_completions_url,
)
from .schema import BenchmarkCase, Prediction, TableSpec


PROMPT_TEMPLATE_VERSION = "0.2.0-raw-scalar"


def _quote_identifier(value: str) -> str:
    return '"%s"' % value.replace('"', '""')


def _sqlite_type(dtype: str) -> str:
    normalized = dtype.upper()
    if normalized in ("INTEGER", "REAL", "TEXT", "BLOB"):
        return normalized
    return "TEXT"


def build_database(case: BenchmarkCase) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    for table in case.tables:
        definitions = ", ".join(
            "%s %s" % (_quote_identifier(column.name), _sqlite_type(column.dtype))
            for column in table.columns
        )
        connection.execute(
            "CREATE TABLE %s (%s)" % (_quote_identifier(table.name), definitions)
        )
        names = [column.name for column in table.columns]
        placeholders = ", ".join("?" for _ in names)
        insert_sql = "INSERT INTO %s (%s) VALUES (%s)" % (
            _quote_identifier(table.name),
            ", ".join(_quote_identifier(name) for name in names),
            placeholders,
        )
        connection.executemany(
            insert_sql,
            [[row.get(name) for name in names] for row in table.rows],
        )
    connection.commit()
    return connection


def execute_scalar_sql(case: BenchmarkCase, sql: str) -> Any:
    stripped = sql.strip()
    while stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("only SELECT or WITH queries are allowed")
    if ";" in stripped:
        raise ValueError("multiple SQL statements are not allowed")
    forbidden = re.compile(
        r"\b(attach|detach|pragma|insert|update|delete|drop|create|alter|replace|vacuum)\b",
        flags=re.IGNORECASE,
    )
    if forbidden.search(stripped):
        raise ValueError("query contains a forbidden SQL operation")

    connection = build_database(case)
    try:
        cursor = connection.execute(stripped)
        rows = cursor.fetchmany(2)
    finally:
        connection.close()
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError("query must return exactly one row and one column")
    return rows[0][0]


def _column_manifest(table: TableSpec) -> List[Dict[str, Any]]:
    return [
        {
            "name": column.name,
            "type": column.dtype,
            "description": column.description,
            "unit": column.unit,
            "missing_value": column.missing_value,
        }
        for column in table.columns
    ]


def render_sql_prompt(case: BenchmarkCase, condition: str = "contract") -> str:
    if condition not in ("schema", "contract"):
        raise ValueError("prompt condition must be 'schema' or 'contract'")
    manifest: Dict[str, Any] = {
        "question": case.question,
        "answer_type": case.answer.value_type,
        "answer_unit": case.answer.unit,
        "rounding": case.answer.rounding,
        "rounding_mode": case.answer.rounding_mode,
    }
    if condition == "contract":
        manifest["tables"] = [
            {
                "name": table.name,
                "description": table.description,
                "columns": _column_manifest(table),
                "primary_key": table.primary_key,
                "foreign_keys": table.foreign_keys,
            }
            for table in case.tables
        ]
        condition_instruction = (
            "Respect every semantic-contract field, including units, encodings, "
            "missing-value rules, keys, filters, rounding, and tie-breaking instructions."
        )
    else:
        manifest["tables"] = [
            {
                "name": table.name,
                "columns": [
                    {"name": column.name, "type": column.dtype}
                    for column in table.columns
                ],
            }
            for table in case.tables
        ]
        condition_instruction = (
            "Infer the relevant semantics from the question and the bare database schema."
        )
    return (
        "You are a careful data-analysis agent. The listed tables are available in a "
        "SQLite database. Write one read-only SQLite query that returns exactly one row "
        "and one column answering the question. The query's single cell must be the raw "
        "answer value (a number or plain text), not JSON, a labeled object, a sentence, "
        "or a value with units. Do not use json_object, json_array, or printf to wrap the "
        "answer. "
        + condition_instruction
        + " Return only a JSON response envelope of the form "
        "{\"sql\": \"SELECT ...\"}. The JSON requirement applies to your response, not "
        "to the SQL query's result. Do not include Markdown or an explanation.\n\n"
        "INPUT MANIFEST\n"
        + json.dumps(manifest, indent=2, sort_keys=True)
    )


class Agent(ABC):
    name: str

    @abstractmethod
    def predict(self, case: BenchmarkCase) -> Prediction:
        raise NotImplementedError


class ModelRequestError(RuntimeError):
    """An endpoint, authentication, rate-limit, or network failure."""


class OracleAgent(Agent):
    name = "oracle"

    def predict(self, case: BenchmarkCase) -> Prediction:
        base_queries = {
            "recognized_revenue": (
                "SELECT ROUND(SUM(quantity * unit_price_usd), 2) "
                "FROM orders WHERE status = 'completed'"
            ),
            "top_region": (
                "SELECT region FROM orders WHERE status = 'completed' "
                "GROUP BY region ORDER BY SUM(amount_usd) DESC, region ASC LIMIT 1"
            ),
            "average_resolution_hours": (
                "SELECT ROUND(AVG((CAST(strftime('%s', closed_at_utc) AS INTEGER) - "
                "CAST(strftime('%s', opened_at_utc) AS INTEGER)) / 3600.0), 2) "
                "FROM tickets "
                "WHERE priority = 'high' AND closed_at_utc IS NOT NULL"
            ),
            "eligible_conversion_rate": (
                "SELECT ROUND(100.0 * SUM(converted) / COUNT(*), 2) FROM sessions "
                "WHERE eligible = 1 AND converted IS NOT NULL"
            ),
            "active_inventory_value": (
                "SELECT ROUND(SUM(quantity_on_hand * unit_cost_usd), 2) "
                "FROM inventory WHERE active = 1"
            ),
            "settled_net_cash_flow": (
                "SELECT ROUND(SUM(CASE WHEN direction = 'inflow' THEN amount_usd "
                "ELSE -amount_usd END), 2) FROM cash_transactions WHERE settled = 1"
            ),
            "weighted_quality_score": (
                "SELECT ROUND(SUM(score * weight_fraction) / SUM(weight_fraction), 2) "
                "FROM quality_metrics WHERE included = 1"
            ),
            "resolved_sla_compliance": (
                "SELECT ROUND(100.0 * SUM(CASE WHEN response_minutes <= target_minutes "
                "THEN 1 ELSE 0 END) / COUNT(*), 2) FROM incidents WHERE resolved = 1"
            ),
            "valid_energy_kwh": (
                "SELECT CAST((SUM(CAST(ROUND(energy_kwh * 1000) AS INTEGER)) + 5) "
                "/ 10 AS INTEGER) / 100.0 FROM energy_readings WHERE valid = 1"
            ),
            "regional_population_density": (
                "SELECT ROUND(SUM(population) / SUM(area_sq_km), 2) FROM municipalities"
            ),
        }
        special_queries = {
            ("recognized_revenue", "currency_cents"): (
                "SELECT ROUND(SUM(quantity * unit_price_cents) / 100.0, 2) "
                "FROM orders WHERE status = 'completed'"
            ),
            ("recognized_revenue", "renamed_schema"): (
                "SELECT ROUND(SUM(units_purchased * price_each_usd), 2) "
                "FROM orders WHERE lifecycle_state = 'completed'"
            ),
            ("top_region", "normalized_region"): (
                "SELECT regions.region_name FROM orders JOIN regions "
                "ON orders.region_id = regions.region_id "
                "WHERE orders.status = 'completed' GROUP BY regions.region_name "
                "ORDER BY SUM(orders.amount_usd) DESC, regions.region_name ASC LIMIT 1"
            ),
            ("top_region", "renamed_schema"): (
                "SELECT sales_territory FROM orders "
                "WHERE lifecycle_state = 'completed' GROUP BY sales_territory "
                "ORDER BY SUM(recognized_amount_usd) DESC, sales_territory ASC LIMIT 1"
            ),
            ("average_resolution_hours", "timezone_ist"): (
                "SELECT ROUND(AVG((CAST(strftime('%s', closed_at_ist) AS INTEGER) - "
                "CAST(strftime('%s', opened_at_ist) AS INTEGER)) / 3600.0), 2) "
                "FROM tickets "
                "WHERE priority = 'high' AND closed_at_ist IS NOT NULL"
            ),
            ("average_resolution_hours", "unix_seconds"): (
                "SELECT ROUND(AVG((closed_unix_seconds - opened_unix_seconds) / "
                "3600.0), 2) FROM tickets WHERE priority = 'high' "
                "AND closed_unix_seconds IS NOT NULL"
            ),
            ("eligible_conversion_rate", "missing_marker"): (
                "SELECT ROUND(100.0 * SUM(CAST(converted AS INTEGER)) / COUNT(*), 2) "
                "FROM sessions WHERE eligible = 1 AND converted != 'UNKNOWN'"
            ),
            ("eligible_conversion_rate", "text_booleans"): (
                "SELECT ROUND(100.0 * SUM(CASE WHEN converted = 'YES' THEN 1 ELSE 0 "
                "END) / COUNT(*), 2) FROM sessions WHERE eligible = 'YES' "
                "AND converted IS NOT NULL"
            ),
            ("active_inventory_value", "cost_cents"): (
                "SELECT ROUND(SUM(quantity_on_hand * unit_cost_cents) / 100.0, 2) "
                "FROM inventory WHERE active = 1"
            ),
            ("active_inventory_value", "active_text"): (
                "SELECT ROUND(SUM(quantity_on_hand * unit_cost_usd), 2) "
                "FROM inventory WHERE active = 'YES'"
            ),
            ("settled_net_cash_flow", "amount_cents"): (
                "SELECT ROUND(SUM(CASE WHEN direction = 'inflow' THEN amount_cents "
                "ELSE -amount_cents END) / 100.0, 2) FROM cash_transactions "
                "WHERE settled = 1"
            ),
            ("settled_net_cash_flow", "renamed_schema"): (
                "SELECT ROUND(SUM(CASE WHEN cash_direction = 'inflow' THEN "
                "transaction_value_usd ELSE -transaction_value_usd END), 2) "
                "FROM cash_transactions WHERE is_finalized = 1"
            ),
            ("weighted_quality_score", "weight_percent"): (
                "SELECT ROUND(SUM(score * weight_percent) / SUM(weight_percent), 2) "
                "FROM quality_metrics WHERE included = 1"
            ),
            ("weighted_quality_score", "included_text"): (
                "SELECT ROUND(SUM(score * weight_fraction) / SUM(weight_fraction), 2) "
                "FROM quality_metrics WHERE included = 'YES'"
            ),
            ("resolved_sla_compliance", "duration_seconds"): (
                "SELECT ROUND(100.0 * SUM(CASE WHEN response_seconds <= target_seconds "
                "THEN 1 ELSE 0 END) / COUNT(*), 2) FROM incidents WHERE resolved = 1"
            ),
            ("resolved_sla_compliance", "renamed_schema"): (
                "SELECT ROUND(100.0 * SUM(CASE WHEN actual_response_minutes <= "
                "sla_limit_minutes THEN 1 ELSE 0 END) / COUNT(*), 2) FROM incidents "
                "WHERE is_resolved = 1"
            ),
            ("valid_energy_kwh", "energy_wh"): (
                "SELECT CAST((SUM(energy_wh) + 5) / 10 AS INTEGER) / 100.0 "
                "FROM energy_readings "
                "WHERE valid = 1"
            ),
            ("valid_energy_kwh", "valid_text"): (
                "SELECT CAST((SUM(CAST(ROUND(energy_kwh * 1000) AS INTEGER)) + 5) "
                "/ 10 AS INTEGER) / 100.0 FROM energy_readings WHERE valid = 'YES'"
            ),
            ("regional_population_density", "area_sq_miles"): (
                "SELECT ROUND(SUM(population) / "
                "SUM(area_sq_miles / 0.3861021585424458), 2) FROM municipalities"
            ),
            ("regional_population_density", "renamed_schema"): (
                "SELECT ROUND(SUM(resident_count) / SUM(land_area_km2), 2) "
                "FROM municipalities"
            ),
        }
        sql = special_queries.get((case.family, case.variant), base_queries[case.family])
        started = time.perf_counter()
        try:
            value = execute_scalar_sql(case, sql)
            status = "ok"
            error = None
        except Exception as exc:
            value = None
            status = "error"
            error = "%s: %s" % (type(exc).__name__, exc)
        return Prediction(
            case_id=case.case_id,
            agent=self.name,
            status=status,
            value=value,
            sql=sql,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "warning": "reference SQL validates benchmark integrity; not a model baseline"
            },
        )


class NaiveSQLAgent(Agent):
    """A deliberately representation-sensitive smoke baseline."""

    name = "naive-sql"

    SQL_BY_FAMILY = {
        "recognized_revenue": (
            "SELECT ROUND(SUM(quantity * unit_price_usd), 2) "
            "FROM orders WHERE status = 'completed'"
        ),
        "top_region": (
            "SELECT region FROM orders WHERE status = 'completed' "
            "GROUP BY region ORDER BY SUM(amount_usd) DESC, region ASC LIMIT 1"
        ),
        "average_resolution_hours": (
            "SELECT ROUND(AVG((julianday(closed_at_utc) - julianday(opened_at_utc)) "
            "* 24.0), 2) FROM tickets "
            "WHERE priority = 'high' AND closed_at_utc IS NOT NULL"
        ),
        "eligible_conversion_rate": (
            "SELECT ROUND(100.0 * SUM(converted) / COUNT(*), 2) FROM sessions "
            "WHERE eligible = 1 AND converted IS NOT NULL"
        ),
        "active_inventory_value": (
            "SELECT ROUND(SUM(quantity_on_hand * unit_cost_usd), 2) "
            "FROM inventory WHERE active = 1"
        ),
        "settled_net_cash_flow": (
            "SELECT ROUND(SUM(CASE WHEN direction = 'inflow' THEN amount_usd "
            "ELSE -amount_usd END), 2) FROM cash_transactions WHERE settled = 1"
        ),
        "weighted_quality_score": (
            "SELECT ROUND(SUM(score * weight_fraction) / SUM(weight_fraction), 2) "
            "FROM quality_metrics WHERE included = 1"
        ),
        "resolved_sla_compliance": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN response_minutes <= target_minutes "
            "THEN 1 ELSE 0 END) / COUNT(*), 2) FROM incidents WHERE resolved = 1"
        ),
        "valid_energy_kwh": (
            "SELECT ROUND(SUM(energy_kwh), 2) FROM energy_readings WHERE valid = 1"
        ),
        "regional_population_density": (
            "SELECT ROUND(SUM(population) / SUM(area_sq_km), 2) FROM municipalities"
        ),
    }

    def predict(self, case: BenchmarkCase) -> Prediction:
        started = time.perf_counter()
        sql = self.SQL_BY_FAMILY[case.family]
        try:
            value = execute_scalar_sql(case, sql)
            status = "ok"
            error = None
        except Exception as exc:  # expected in the brittle smoke baseline
            value = None
            status = "error"
            error = "%s: %s" % (type(exc).__name__, exc)
        return Prediction(
            case_id=case.case_id,
            agent=self.name,
            status=status,
            value=value,
            sql=sql,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _extract_json_object(content: str) -> Dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("model response did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value


def _content_text(content: Any) -> str:
    """Normalize OpenAI-compatible string or content-block responses."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                block_type = str(block.get("type", "text"))
                if block_type in ("text", "output_text"):
                    texts.append(block["text"])
        if texts:
            return "\n".join(texts)
    raise ValueError("model content did not contain a text response block")


class HTTPModelSQLAgent(Agent):
    """Text-to-SQL runner for an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: Optional[str] = None,
        token_provider: Optional[Callable[[], str]] = None,
        timeout_seconds: float = 90.0,
        prompt_condition: str = "contract",
        max_tokens: int = 256,
        temperature: Optional[float] = 0.0,
        reasoning_effort: Optional[str] = None,
        retries: int = 1,
        retry_base_seconds: float = 1.0,
        before_request: Optional[Callable[[], None]] = None,
    ) -> None:
        if prompt_condition not in ("schema", "contract"):
            raise ValueError("prompt condition must be 'schema' or 'contract'")
        self.url = url
        self.model = model
        self.api_key = api_key
        self.token_provider = token_provider
        self.timeout_seconds = timeout_seconds
        self.prompt_condition = prompt_condition
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self.retry_base_seconds = retry_base_seconds
        self.before_request = before_request
        self.name = "http-sql:%s:%s" % (model, prompt_condition)

    @classmethod
    def from_environment(
        cls, prompt_condition: str = "contract"
    ) -> "HTTPModelSQLAgent":
        url = os.environ.get("DATAINVARIANT_API_URL")
        model = os.environ.get("DATAINVARIANT_MODEL")
        token_provider: Optional[Callable[[], str]] = None
        api_key = os.environ.get("DATAINVARIANT_API_KEY")
        databricks_host = os.environ.get("DATABRICKS_HOST")
        if databricks_host:
            url = url or chat_completions_url(databricks_host)
            if os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get(
                "DATABRICKS_CLIENT_SECRET"
            ):
                token_provider = DatabricksOAuthTokenProvider.from_environment()
            elif os.environ.get("DATABRICKS_TOKEN"):
                api_key = os.environ["DATABRICKS_TOKEN"]
        if not url or not model:
            raise ValueError(
                "DATAINVARIANT_MODEL and either DATAINVARIANT_API_URL or "
                "DATABRICKS_HOST are required"
            )
        temperature_value = os.environ.get("DATAINVARIANT_TEMPERATURE", "0")
        temperature = (
            None
            if temperature_value.strip().lower() in ("none", "omit")
            else float(temperature_value)
        )
        return cls(
            url=url,
            model=model,
            api_key=api_key,
            token_provider=token_provider,
            timeout_seconds=float(os.environ.get("DATAINVARIANT_TIMEOUT", "90")),
            prompt_condition=prompt_condition,
            max_tokens=int(os.environ.get("DATAINVARIANT_MAX_TOKENS", "256")),
            temperature=temperature,
            reasoning_effort=os.environ.get("DATAINVARIANT_REASONING_EFFORT"),
            retries=int(os.environ.get("DATAINVARIANT_RETRIES", "1")),
        )

    def _call_model(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        body: Dict[str, Any]
        attempts = 0
        retryable_codes = {401, 408, 429, 500, 502, 503, 504}
        while True:
            attempts += 1
            if self.before_request is not None:
                self.before_request()
            headers = {"Content-Type": "application/json"}
            token = self.token_provider() if self.token_provider else self.api_key
            if token:
                headers["Authorization"] = "Bearer %s" % token
            request = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and hasattr(self.token_provider, "invalidate"):
                    self.token_provider.invalidate()  # type: ignore[attr-defined]
                if exc.code not in retryable_codes or attempts >= self.retries + 1:
                    raise ModelRequestError(
                        "model request failed with HTTP %d" % exc.code
                    ) from exc
                retry_after = 0.0
                try:
                    retry_after = float(exc.headers.get("Retry-After", 0) or 0)
                except (AttributeError, TypeError, ValueError):
                    retry_after = 0.0
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempts >= self.retries + 1:
                    raise ModelRequestError("model request failed after retries") from exc
                retry_after = 0.0
            time.sleep(
                max(retry_after, self.retry_base_seconds * (2 ** (attempts - 1)))
            )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("unexpected chat-completions response shape") from exc
        return _content_text(content), {
            "response_id": body.get("id"),
            "response_model": body.get("model"),
            "system_fingerprint": body.get("system_fingerprint"),
            "usage": body.get("usage"),
            "request_attempts": attempts,
        }

    def predict(self, case: BenchmarkCase) -> Prediction:
        started = time.perf_counter()
        raw_response = None
        sql = None
        response_metadata: Dict[str, Any] = {}
        prompt = render_sql_prompt(case, condition=self.prompt_condition)
        try:
            raw_response, response_metadata = self._call_model(prompt)
            payload = _extract_json_object(raw_response)
            sql = payload.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("response JSON must contain a non-empty sql string")
            value = execute_scalar_sql(case, sql)
            status = "ok"
            error = None
        except Exception as exc:
            value = None
            status = "error"
            error = "%s: %s" % (type(exc).__name__, exc)
            response_metadata["failure_class"] = (
                "infrastructure"
                if isinstance(exc, ModelRequestError)
                else "model_or_execution"
            )
        return Prediction(
            case_id=case.case_id,
            agent=self.name,
            status=status,
            value=value,
            sql=sql,
            raw_response=raw_response,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "model": self.model,
                "endpoint": urllib.parse.urlunsplit(
                    urllib.parse.urlsplit(self.url)._replace(query="", fragment="")
                ),
                "prompt_condition": self.prompt_condition,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
                "retry_limit": self.retries,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_characters": len(prompt),
                **response_metadata,
            },
        )
