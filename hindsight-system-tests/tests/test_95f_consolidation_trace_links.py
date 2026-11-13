"""Two requests contribute to one worker consolidation after the API restarts.

Retain history is disabled throughout ingestion. The worker's links must therefore
come from the saved memories, never from an in-process cache or request-history rows.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import Request, Response
from hindsight_client import Hindsight
from hindsight_client_api.api.llm_traces_api import LLMTracesApi
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import extracted, fact, observes
from hindsight_system_tests.server import start_hindsight_server, start_hindsight_worker

pytestmark = pytest.mark.asyncio
TRACE_A = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACE_B = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@dataclass
class Collector:
    endpoint: str
    spans: list[Span] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    async def receive(self, request: Request) -> Response:
        export = ExportTraceServiceRequest.FromString(await request.body())
        with self.lock:
            for resource in export.resource_spans:
                for scope in resource.scope_spans:
                    self.spans.extend(scope.spans)
        return Response(content=b"", media_type="application/x-protobuf")

    def snapshot(self) -> list[Span]:
        with self.lock:
            return list(self.spans)


async def test_consolidation_links_survive_restart_without_request_history(stub_server, llm, tmp_path: Path) -> None:
    collector_path = f"/trace-test-{uuid.uuid4().hex}"
    collector = Collector(endpoint=stub_server.url + collector_path)
    stub_server._server.config.app.add_api_route(collector_path + "/v1/traces", collector.receive, methods=["POST"])
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    env = {
        "HINDSIGHT_API_DATABASE_SCHEMA": "systest_consolidation_links",
        "HINDSIGHT_API_OTEL_TRACES_ENABLED": "true",
        "HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint,
        "HINDSIGHT_API_WORKER_ENABLED": "false",
        "OTEL_TRACES_SAMPLER": "always_on",
        "OTEL_BSP_SCHEDULE_DELAY": "100",
    }
    llm.on_step("extract_facts", contains="Alice").returns(extracted(fact("Alice lives in Lisbon", who="Alice")))
    llm.on_step("extract_facts", contains="Bob").returns(extracted(fact("Bob lives in Lisbon", who="Bob")))
    llm.on_step("consolidate").answers_with(observes("Alice and Bob live in Lisbon"))

    # No worker yet: the two retains durably enqueue their shared bank's work.
    api = start_hindsight_server(
        stub_url=stub_server.url, log_path=tmp_path / "ingest.log",
        extra_env={**env, "HINDSIGHT_API_LLM_TRACE_ENABLED": "false"},
    )
    client = Hindsight(base_url=api.url)
    try:
        for trace_id, content in [(TRACE_A, "Alice lives in Lisbon."), (TRACE_B, "Bob lives in Lisbon.")]:
            client._api_client.set_default_header("traceparent", f"00-{trace_id}-00f067aa0ba902b7-01")
            await client.aretain(bank_id=bank, content=content)
        history = await LLMTracesApi(client._api_client).list_llm_requests(bank, limit=500)
        assert history.total == 0
    finally:
        await client.aclose()
        api.stop()  # flushes retain spans, and discards all process-local context

    api = start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "api.log", extra_env=env)
    client = Hindsight(base_url=api.url)
    worker = None
    try:
        worker = start_hindsight_worker(
            stub_url=stub_server.url, log_path=tmp_path / "worker.log",
            extra_env={**env, "HINDSIGHT_API_WORKER_ENABLED": "true"},
        )
        await wait_until_settled(client, bank)
        for _ in range(100):
            spans = collector.snapshot()
            jobs = [s for s in spans if s.name == "hindsight.consolidation" and s.links]
            rows = (await LLMTracesApi(client._api_client).list_llm_requests(bank, limit=500)).items
            if jobs and any(r.operation == "consolidation" for r in rows):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("consolidation trace and model-call history did not arrive")

        origins = {(s.trace_id.hex(), s.span_id.hex()) for s in spans if s.name == "hindsight.retain" and s.kind == Span.SPAN_KIND_INTERNAL}
        assert {t for t, _ in origins} == {TRACE_A, TRACE_B}
        job = next(s for s in jobs if {link.trace_id.hex() for link in s.links} == {TRACE_A, TRACE_B})
        assert not job.parent_span_id
        assert job.trace_id.hex() not in {TRACE_A, TRACE_B}
        assert {(link.trace_id.hex(), link.span_id.hex()) for link in job.links} == origins
        assert len(job.links) == len(origins)  # one link per originating retain span
        assert not [r for r in rows if r.operation == "retain"]
        calls = [r for r in rows if r.operation == "consolidation"]
        assert calls and {r.trace_id for r in calls} == {job.trace_id.hex()}
        assert any(s.trace_id == job.trace_id and s.parent_span_id == job.span_id for s in spans)
        memories = await client.memory.list_memories(bank, limit=100)
        assert any(m.fact_type == "observation" for m in memories.items)
        (tmp_path / "trace-demo.json").write_text(json.dumps({
            "request_history_during_retain": 0,
            "api_restarted_before_consolidation": True,
            "originating_retains": [{"trace_id": t, "span_id": s} for t, s in sorted(origins)],
            "consolidation_trace_id": job.trace_id.hex(),
            "consolidation_span_id": job.span_id.hex(),
            "consolidation_parent_span_id": job.parent_span_id.hex(),
            "links": [{"trace_id": link.trace_id.hex(), "span_id": link.span_id.hex()} for link in job.links],
            "model_calls": [{"scope": r.scope, "trace_id": r.trace_id} for r in calls],
        }, indent=2) + "\n")
    finally:
        await client.banks.delete_bank(bank)
        await client.aclose()
        if worker:
            worker.stop()
        api.stop()
