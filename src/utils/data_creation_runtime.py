"""Small runtime pieces shared by the data-creation stages.

Why this exists: the stages run as separate processes under
`src/data_creation/run_data_creation.py`, and two things have to behave identically in
every one of them — the guard that keeps a development run off the KEYED services, and
the JSON-array encoding of an endpoint block, which has to match the endpoint pipeline's
byte for byte or `shared_showcase.endpoint_records` would read two encodings in one file.

THE KEYED-API GUARD. A development run (`run_data_creation.py --offline`) must not spend
a Dimensions, Altmetric or Anthropic call: those are made once, in the live run, after all
the code is tested. When `UKB_BLOCK_KEYED_APIS=1` is set, `install_keyed_api_guard()`
makes name resolution of those hosts raise, which stops `urllib`, `requests` and the
Anthropic SDK alike before a socket opens. Keyless public lookups — ClinicalTrials.gov and
NLM's MeSH SPARQL — are deliberately left open, so the MeSH stage is exercised for real.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

GUARD_ENV = "UKB_BLOCK_KEYED_APIS"

#: Hosts that need an API key. Matched as the name itself or any subdomain of it.
KEYED_HOSTS = (
    "dimensions.ai",        # app.dimensions.ai — DSL and auth
    "altmetric.com",        # www.altmetric.com/explorer/api
    "anthropic.com",        # api.anthropic.com — the collaborator classifier
)


class KeyedApiBlocked(RuntimeError):
    """A keyed API was called while the development guard was on."""


def _is_keyed(host: Any) -> bool:
    name = (host.decode() if isinstance(host, bytes) else str(host or "")).lower().rstrip(".")
    return any(name == h or name.endswith("." + h) for h in KEYED_HOSTS)


def install_keyed_api_guard(force: bool = False) -> bool:
    """Block keyed hosts in this process if the guard is requested. Returns whether it is on."""
    if not (force or os.environ.get(GUARD_ENV) == "1"):
        return False
    if getattr(socket.getaddrinfo, "_ukb_guard", False):
        return True
    original = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        if _is_keyed(host):
            raise KeyedApiBlocked(
                f"{host} needs an API key and {GUARD_ENV}=1 is set: keyed calls are made "
                "only in the live data-creation run"
            )
        return original(host, *args, **kwargs)

    guarded._ukb_guard = True           # type: ignore[attr-defined]
    socket.getaddrinfo = guarded
    return True


def json_array(values) -> str:
    """The endpoint pipeline's JSON-array encoding (`dimensions_endpoint_pipeline.json_array`)."""
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
