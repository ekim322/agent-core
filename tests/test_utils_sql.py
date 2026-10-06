"""Real read-only SQL behavior and deterministic worker cancellation."""

import asyncio
from threading import Event

import pytest

from agent_core.sql import SQLQueryError, VirtualTable


def test_schema_order_nulls_snapshot_and_empty_filter_notice():
    async def check():
        source = [{"a": 1, "b": 3}, {"a": 2}]
        table = VirtualTable("rows", source, columns=("b", "a"))
        source[0]["a"] = 99
        assert await table.run_sql_query("select * from rows order by a") == [
            {"b": 3, "a": 1},
            {"b": None, "a": 2},
        ]
        assert await table.run_sql_query(
            "with picked as (select a from rows) select sum(a) as total from picked"
        ) == [{"total": 3}]
        empty = await table.build_tool_result("select * from rows where a > 10")
        assert empty["sql_filter_notice"]["source_row_count"] == 2
        unfiltered = await table.build_tool_result(None)
        unfiltered["data"][0]["a"] = 88
        assert table.records[0]["a"] == 1
        assert (
            await VirtualTable("rows", [], columns=("a",)).run_sql_query(
                "select * from rows"
            )
            == []
        )

    asyncio.run(check())


@pytest.mark.parametrize(
    "query",
    [
        "delete from rows",
        "select * from rows; select * from rows",
        "select * from other",
        "select * from read_csv('/tmp/data.csv'), rows",
        "select * from information_schema.tables, rows",
        "with rows as (select 1) select * from rows",
        "select 1",
        "select * into new_table from rows",
        "with other as (with hidden as (select * from rows) select * from hidden) select * from hidden",
    ],
)
def test_untrusted_queries_cannot_escape_the_registered_relation(query):
    with pytest.raises(SQLQueryError):
        asyncio.run(VirtualTable("rows", [{"a": 1}]).run_sql_query(query))


def test_result_limits_and_ambiguous_column_names_fail_instead_of_losing_values():
    table = VirtualTable("rows", [{"a": 1}, {"a": 2}])
    with pytest.raises(SQLQueryError, match="more than 1"):
        asyncio.run(table.run_sql_query("select * from rows", max_result_rows=1))
    with pytest.raises(SQLQueryError, match="result column names"):
        asyncio.run(table.run_sql_query("select a, a as A from rows"))
    with pytest.raises(SQLQueryError, match="unique"):
        VirtualTable("rows", [{"a": 1, "A": 2}])
    with pytest.raises(ValueError, match="finite"):
        asyncio.run(
            table.run_sql_query("select * from rows", timeout_seconds=float("nan"))
        )


def test_real_recursive_query_is_interrupted_on_deadline():
    query = """with recursive n(x) as (
        select a from rows union all select x + 1 from n where x < 100000000
    ) select sum(x) from n"""
    with pytest.raises(SQLQueryError, match="execution limit"):
        asyncio.run(
            VirtualTable("rows", [{"a": 1}]).run_sql_query(query, timeout_seconds=0.05)
        )


def test_cancelled_caller_interrupts_and_joins_worker_owned_connection(monkeypatch):
    started, interrupted, closed = Event(), Event(), Event()

    class Connection:
        def register(self, *args):
            pass

        def execute(self, query):
            started.set()
            assert interrupted.wait(3), "worker must be interrupted"
            raise __import__("duckdb").InterruptException("interrupted")

        def interrupt(self):
            interrupted.set()

        def close(self):
            closed.set()

    monkeypatch.setattr(
        "agent_core.sql.table.duckdb.connect",
        lambda *args, **kwargs: Connection(),
    )

    async def check():
        task = asyncio.create_task(
            VirtualTable("rows", [{"a": 1}]).run_sql_query("select * from rows")
        )
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert interrupted.is_set() and closed.is_set()

    asyncio.run(check())
