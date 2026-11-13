"""Consolidation links to persistent memory origins, not request-history rows."""

import asyncio
from array import array
from collections.abc import Iterator
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from hindsight_api import tracing
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.postgresql import PostgresConnection
from hindsight_api.engine.memories.base import build_fact_records
from hindsight_api.engine.memories.pg import reads, writes
from hindsight_api.engine.retain.types import ProcessedFact
from hindsight_api.tracing import (
    create_consolidation_span,
    current_memory_trace_origin,
    link_consolidation_origin,
)

TRACE_A = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACE_B = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SPAN_A = "00f067aa0ba902b7"
SPAN_B = "bbbbbbbbbbbbbbbb"
PARENT_A = f"00-{TRACE_A}-{SPAN_A}-01"
PARENT_B = f"00-{TRACE_B}-{SPAN_B}-00"


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracing_enabled", True)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    yield exporter
    provider.shutdown()


def test_consolidation_is_a_root_with_unique_source_links(exporter: InMemorySpanExporter) -> None:
    incoming = TraceContextTextMapPropagator().extract({"traceparent": PARENT_A})
    token = context.attach(incoming)
    try:
        with create_consolidation_span("bank"):
            link_consolidation_origin(PARENT_A, "vendor=value")
            link_consolidation_origin(PARENT_A, "vendor=value")
            link_consolidation_origin(PARENT_B)
            link_consolidation_origin(None)
            link_consolidation_origin("invalid")
            with tracing.get_tracer().start_as_current_span("model-call"):
                pass
        assert trace.get_current_span().get_span_context().trace_id == int(TRACE_A, 16)
    finally:
        context.detach(token)
    spans = exporter.get_finished_spans()
    job = next(s for s in spans if s.name == "hindsight.consolidation")
    model = next(s for s in spans if s.name == "model-call")
    assert job.parent is None
    assert job.context.trace_id not in {int(TRACE_A, 16), int(TRACE_B, 16)}
    assert {(link.context.trace_id, link.context.span_id) for link in job.links} == {
        (int(TRACE_A, 16), int(SPAN_A, 16)),
        (int(TRACE_B, 16), int(SPAN_B, 16)),
    }
    assert job.links[0].context.trace_state.get("vendor") == "value"
    assert model.context.trace_id == job.context.trace_id and model.parent.span_id == job.context.span_id
    # A subsequent operation must not inherit the previous job's links.
    with create_consolidation_span("another-bank"):
        link_consolidation_origin(None)
    assert not exporter.get_finished_spans()[-1].links


def test_link_bookkeeping_is_bounded_and_truncation_is_visible(exporter: InMemorySpanExporter) -> None:
    with create_consolidation_span("bank"):
        for i in range(tracing.MAX_CONSOLIDATION_LINKS + 1):
            link_consolidation_origin(f"00-{i + 1:032x}-{SPAN_A}-01")
    (span,) = exporter.get_finished_spans()
    assert len(span.links) == tracing.MAX_CONSOLIDATION_LINKS
    assert span.attributes["hindsight.consolidation.links_truncated"] is True


@pytest.mark.asyncio
async def test_concurrent_banks_keep_separate_links(exporter: InMemorySpanExporter) -> None:
    async def run(bank: str, origin: str) -> None:
        with create_consolidation_span(bank):
            await asyncio.sleep(0)
            link_consolidation_origin(origin)

    await asyncio.gather(run("bank-a", PARENT_A), run("bank-b", PARENT_B))
    by_bank = {span.attributes["hindsight.bank_id"]: span for span in exporter.get_finished_spans()}
    assert [link.context.trace_id for link in by_bank["bank-a"].links] == [int(TRACE_A, 16)]
    assert [link.context.trace_id for link in by_bank["bank-b"].links] == [int(TRACE_B, 16)]


def _fact() -> ProcessedFact:
    return ProcessedFact(
        fact_text="Alice lives in Lisbon",
        fact_type="world",
        embedding=array("f", [0.1] * 384),
        occurred_start=None,
        occurred_end=None,
        mentioned_at=datetime.now(timezone.utc),
        context="",
        metadata={},
    )


def test_store_owned_records_preserve_origin_without_baggage() -> None:
    incoming = TraceContextTextMapPropagator().extract({"traceparent": PARENT_A, "tracestate": "vendor=value"})
    incoming = baggage.set_baggage("caller", "client-a", context=incoming)
    token = context.attach(incoming)
    try:
        (record,) = build_fact_records([str(uuid4())], [_fact()])
        assert record.origin_traceparent == PARENT_A
        assert record.origin_tracestate == "vendor=value"
        assert record.metadata_bag()["origin_traceparent"] == PARENT_A
        assert record.metadata_bag()["origin_tracestate"] == "vendor=value"
        assert "caller" not in record.metadata_bag() and "baggage" not in record.metadata_bag()
    finally:
        context.detach(token)
    token = context.attach(context.Context())
    try:
        assert current_memory_trace_origin().traceparent is None
    finally:
        context.detach(token)


@pytest.mark.asyncio
async def test_origins_survive_history_cleanup_and_memory_restore(pg0_db_url: str) -> None:
    bank = f"trace-origins-{uuid4().hex}"
    other = f"trace-origins-{uuid4().hex}"
    conn = await asyncpg.connect(pg0_db_url)
    # Direct SQL only creates fixture banks and deletes history, which has no
    # public purge API. Memory writes/reads and invalidation use the store.
    try:
        await conn.execute("INSERT INTO banks(bank_id) VALUES ($1), ($2)", bank, other)
        token = context.attach(TraceContextTextMapPropagator().extract({"traceparent": PARENT_A}))
        try:
            ids = await writes.insert_facts(
                conn=PostgresConnection(conn), ops=PostgreSQLOps(), bank_id=bank, facts=[_fact()]
            )
        finally:
            context.detach(token)
        await conn.execute(
            "INSERT INTO llm_requests(bank_id, trace_id, status) VALUES ($1, $2, 'success')", bank, TRACE_A
        )
        await conn.execute("DELETE FROM llm_requests WHERE bank_id = $1", bank)
        # A fresh connection represents the worker, with no in-process context.
        await conn.close()
        conn = await asyncpg.connect(pg0_db_url)
        rows = await reads.find_unconsolidated(
            conn=PostgresConnection(conn),
            fq_table=lambda name: name,
            bank_id=bank,
            fact_types=["world"],
            limit=10,
        )
        assert [r.origin_traceparent for r in rows] == [PARENT_A]
        assert (
            await reads.get_memories(
                conn=PostgresConnection(conn), fq_table=lambda name: name, bank_id=other, unit_ids=ids
            )
            == []
        )
        db = PostgresConnection(conn)
        async with db.transaction():
            assert await writes.invalidate_memory(
                conn=db, fq_table=lambda name: name, bank_id=bank, unit_id=ids[0], reason="test"
            )
            await writes.restore_memory(conn=db, fq_table=lambda name: name, bank_id=bank, unit_id=ids[0])
        restored = await reads.get_memories(
            conn=PostgresConnection(conn), fq_table=lambda name: name, bank_id=bank, unit_ids=ids
        )
        assert restored[0].origin_traceparent == PARENT_A
    finally:
        await conn.execute("DELETE FROM banks WHERE bank_id = ANY($1::text[])", [bank, other])
        await conn.close()
