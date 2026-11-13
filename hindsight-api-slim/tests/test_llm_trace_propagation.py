"""Distributed trace correlation without merging per-operation memory attribution."""

import json
import uuid

import asyncpg
import pytest
from opentelemetry import context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from hindsight_api.engine.llm_trace import (
    LLMTraceContext,
    LLMTraceRecorder,
    new_trace_context,
    reset_trace_context,
    set_trace_context,
)

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"


@pytest.mark.parametrize("flags", ["00", "01"])
def test_operation_uses_incoming_trace_even_when_unsampled(flags: str) -> None:
    parent = TraceContextTextMapPropagator().extract({"traceparent": f"00-{TRACE_ID}-{SPAN_ID}-{flags}"})
    token = context.attach(parent)
    try:
        first = new_trace_context("bank", "retain")
        second = new_trace_context("bank", "retain")
        assert first.trace_id == second.trace_id == TRACE_ID
        assert first.operation_span_id != second.operation_span_id
    finally:
        context.detach(token)


def test_operations_without_context_keep_independent_ids() -> None:
    token = context.attach(context.Context())
    try:
        first = new_trace_context("bank", "retain")
        second = new_trace_context("bank", "retain")
        assert uuid.UUID(first.trace_id) != uuid.UUID(second.trace_id)
        assert first.operation_span_id != second.operation_span_id
    finally:
        context.detach(token)


@pytest.mark.asyncio
async def test_memory_attribution_is_scoped_to_operation_and_bank(pg0_db_url: str) -> None:
    # Include the same operation ID in another bank to exercise bank isolation.
    pool = await asyncpg.create_pool(pg0_db_url, min_size=1, max_size=2)
    bank = f"trace-propagation-{uuid.uuid4().hex}"
    other_bank = f"{bank}-other"
    recorder = LLMTraceRecorder(
        pool_getter=lambda: pool, schema_getter=lambda: "public", enabled=True, allowed_scopes=[]
    )
    try:
        contexts = [
            LLMTraceContext(bank_id=bank_id, trace_id=TRACE_ID, operation_span_id=operation_id)
            for bank_id, operation_id in [(bank, "first"), (bank, "second"), (other_bank, "first")]
        ]
        for ctx in contexts:
            token = set_trace_context(ctx)
            try:
                recorder.record_llm_call(provider="mock", model="mock", scope="memory", messages=[])
            finally:
                reset_trace_context(token)
        recorder.attach_memory_ids(contexts[0], created=["memory-a"])
        recorder.attach_memory_ids(contexts[1], created=["memory-b"])
        await recorder._writes.drain("memory_ids")
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bank_id, parent_span_id, metadata FROM llm_requests WHERE bank_id = ANY($1::text[])",
                [bank, other_bank],
            )
        actual = {(r["bank_id"], r["parent_span_id"]): json.loads(r["metadata"]) for r in rows}
        assert actual == {
            (bank, "first"): {"memory_ids": ["memory-a"]},
            (bank, "second"): {"memory_ids": ["memory-b"]},
            (other_bank, "first"): {},
        }
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM llm_requests WHERE bank_id = ANY($1::text[])", [bank, other_bank])
        await pool.close()
