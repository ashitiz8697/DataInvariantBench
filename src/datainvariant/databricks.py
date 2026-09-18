from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional


CHAT_COMPLETIONS_PATH = "/ai-gateway/mlflow/v1/chat/completions"
OAUTH_TOKEN_PATH = "/oidc/v1/token"


def normalize_workspace_host(value: str) -> str:
    """Validate a Databricks workspace origin without retaining URL decorations."""
    stripped = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(stripped)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(
            "DATABRICKS_HOST must be a plain https:// workspace URL without Markdown"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("DATABRICKS_HOST must not contain credentials or query parameters")
    if parsed.path not in ("", "/"):
        raise ValueError("DATABRICKS_HOST must contain only the workspace origin")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def chat_completions_url(host: str) -> str:
    return normalize_workspace_host(host) + CHAT_COMPLETIONS_PATH


class DatabricksOAuthTokenProvider:
    """Thread-safe OAuth client-credentials provider with early token refresh."""

    def __init__(
        self,
        host: str,
        client_id: str,
        client_secret: str,
        timeout_seconds: float = 30.0,
        refresh_margin_seconds: float = 120.0,
    ) -> None:
        if not client_id.strip() or not client_secret:
            raise ValueError("Databricks OAuth client ID and secret are required")
        self.host = normalize_workspace_host(host)
        self._client_id = client_id
        self._client_secret = client_secret
        self.timeout_seconds = timeout_seconds
        self.refresh_margin_seconds = refresh_margin_seconds
        self._access_token: Optional[str] = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "DatabricksOAuthTokenProvider":
        missing = [
            name
            for name in (
                "DATABRICKS_HOST",
                "DATABRICKS_CLIENT_ID",
                "DATABRICKS_CLIENT_SECRET",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError(
                "missing Databricks credential environment variables: %s"
                % ", ".join(missing)
            )
        return cls(
            host=os.environ["DATABRICKS_HOST"],
            client_id=os.environ["DATABRICKS_CLIENT_ID"],
            client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
            timeout_seconds=float(os.environ.get("DATABRICKS_OAUTH_TIMEOUT", "30")),
        )

    def invalidate(self) -> None:
        with self._lock:
            self._access_token = None
            self._expires_at = 0.0

    def __call__(self) -> str:
        now = time.time()
        if self._access_token and now < self._expires_at - self.refresh_margin_seconds:
            return self._access_token
        with self._lock:
            now = time.time()
            if self._access_token and now < self._expires_at - self.refresh_margin_seconds:
                return self._access_token
            self._refresh_locked(now)
            if not self._access_token:
                raise RuntimeError("Databricks OAuth response did not contain an access token")
            return self._access_token

    def _refresh_locked(self, now: float) -> None:
        credentials = (self._client_id + ":" + self._client_secret).encode("utf-8")
        body = urllib.parse.urlencode(
            {"grant_type": "client_credentials", "scope": "all-apis"}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.host + OAUTH_TOKEN_PATH,
            data=body,
            headers={
                "Authorization": "Basic "
                + base64.b64encode(credentials).decode("ascii"),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                "Databricks OAuth token request failed with HTTP %d" % exc.code
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Databricks OAuth token request failed") from exc
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Databricks OAuth response did not contain an access token")
        try:
            expires_in = max(60.0, float(payload.get("expires_in", 3600)))
        except (TypeError, ValueError):
            expires_in = 3600.0
        self._access_token = token
        self._expires_at = now + expires_in


@dataclass(frozen=True)
class DatabricksModelSpec:
    model_id: str
    display_name: str
    parameter_size: str
    input_dbu_per_million: float
    output_dbu_per_million: float
    max_output_tokens: int = 256
    expected_output_tokens: int = 80
    reasoning_effort: Optional[str] = None
    conservative_regional_multiplier: float = 1.0


PRIMARY_MODEL_SPECS: List[DatabricksModelSpec] = [
    DatabricksModelSpec(
        "system.ai.meta-llama-3-1-8b-instruct",
        "Meta Llama 3.1 8B Instruct",
        "8B",
        2.143,
        6.429,
    ),
    DatabricksModelSpec(
        "system.ai.gemma-3-12b",
        "Google Gemma 3 12B",
        "12B",
        2.143,
        7.143,
    ),
    DatabricksModelSpec(
        "system.ai.gpt-oss-20b",
        "OpenAI GPT-OSS 20B",
        "20.9B total / 3.6B active",
        1.000,
        4.286,
        max_output_tokens=512,
        expected_output_tokens=160,
        reasoning_effort="low",
    ),
    DatabricksModelSpec(
        "system.ai.meta-llama-3-3-70b-instruct",
        "Meta Llama 3.3 70B Instruct",
        "70B",
        7.143,
        21.429,
    ),
    DatabricksModelSpec(
        "system.ai.gpt-oss-120b",
        "OpenAI GPT-OSS 120B",
        "116.8B total / 5.1B active",
        2.143,
        8.571,
        max_output_tokens=512,
        expected_output_tokens=160,
        reasoning_effort="low",
    ),
    DatabricksModelSpec(
        "system.ai.claude-haiku-4-5",
        "Anthropic Claude Haiku 4.5",
        "not disclosed",
        14.286,
        71.429,
        conservative_regional_multiplier=1.10,
    ),
    DatabricksModelSpec(
        "system.ai.gpt-5-mini",
        "OpenAI GPT-5 Mini",
        "not disclosed",
        3.571,
        28.571,
        reasoning_effort="minimal",
        conservative_regional_multiplier=1.10,
    ),
]

PRIMARY_MODEL_BY_ID: Dict[str, DatabricksModelSpec] = {
    spec.model_id: spec for spec in PRIMARY_MODEL_SPECS
}
