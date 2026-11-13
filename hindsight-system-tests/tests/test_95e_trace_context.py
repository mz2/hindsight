"""Gateway trace IDs reach request history in both the API and worker process."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.api.llm_traces_api import LLMTracesApi

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import extracted, fact, observes
from hindsight_system_tests.server import HindsightServer, start_hindsight_server, start_hindsight_worker

pytestmark = pytest.mark.asyncio
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACEPARENT = f"00-{TRACE_ID}-00f067aa0ba902b7-00"


@dataclass
class Deployment:
    api: HindsightServer
    worker: HindsightServer


@pytest.fixture
def traced_deployment(stub_server, tmp_path: Path) -> Iterator[Deployment]:
    env = {
        "HINDSIGHT_API_DATABASE_SCHEMA": "systest_trace_context",
        "HINDSIGHT_API_OTEL_TRACES_ENABLED": "true",
        "HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT": stub_server.url,
        # Correlation must work even when no spans are exported.
        "OTEL_TRACES_SAMPLER": "always_off",
    }
    api = start_hindsight_server(
        stub_url=stub_server.url, log_path=tmp_path / "api.log",
        extra_env={**env, "HINDSIGHT_API_WORKER_ENABLED": "false"},
    )
    try:
        worker = start_hindsight_worker(stub_url=stub_server.url, log_path=tmp_path / "worker.log", extra_env=env)
        try:
            yield Deployment(api, worker)
        finally:
            worker.stop()
    finally:
        api.stop()


@pytest.mark.parametrize("retain_async", [False, True])
async def test_request_history_continues_gateway_trace(traced_deployment: Deployment, llm, retain_async: bool) -> None:
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Theodora").returns(extracted(fact("Theodora lives in Reykjavík", who="Theodora")))
    llm.on_step("consolidate", contains="Theodora").answers_with(observes("Theodora is settled in Reykjavík"))
    answer = "Theodora lives in Reykjavík."
    query = "Where does Theodora live?"
    reflect_loop(llm, answer=answer, query=query)

    client = Hindsight(base_url=traced_deployment.api.url)
    client._api_client.set_default_header("traceparent", TRACEPARENT)
    try:
        await client.aretain(bank_id=bank, content=answer, retain_async=retain_async)
        await wait_until_settled(client, bank)
        reflected = await client.areflect(bank_id=bank, query=query)
        assert reflected.text.strip() == answer
        traces = LLMTracesApi(client._api_client)
        for _ in range(100):
            rows = (await traces.list_llm_requests(bank, limit=500)).items
            if {"retain", "reflect"} <= {row.operation for row in rows}:
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("retain and reflect request history did not arrive")
        for operation in ("retain", "reflect"):
            operation_rows = [r for r in rows if r.operation == operation]
            assert operation_rows and {r.trace_id for r in operation_rows} == {TRACE_ID}
        assert {r.parent_span_id for r in rows if r.operation == "retain"}.isdisjoint(
            {r.parent_span_id for r in rows if r.operation == "reflect"}
        )
    finally:
        await client.banks.delete_bank(bank)
        await client.aclose()
