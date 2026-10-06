"""An isolated DuckDB query over a snapshot of structured tool rows."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from threading import Timer
from typing import Any

import duckdb
import pyarrow as pa
from pydantic_core import to_jsonable_python
from sqlglot import exp, parse
from sqlglot.errors import ErrorLevel, SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from agent_core._validation import count, seconds
from agent_core.sql.executor import SQLQueryError, SQLQueryExecutor, _QueryControl


class VirtualTable:
    """Snapshot records; execute a single read-only query on a worker thread.

    The registered relation is the only physical table available. External
    access and extensions are disabled. SQL result limits do not truncate source
    rows when SQL is omitted. Inject SQLQueryExecutor to own dedicated bounded
    workers, or close the shared executor with SQLQueryExecutor.close_default.
    Cancellation interrupts DuckDB and joins cleanup before returning to the
    caller; Arrow conversion is not interruptible.
    """

    DEFAULT_MAX_RESULT_ROWS = 1000
    DEFAULT_TIMEOUT_SECONDS = 2.0
    MAX_QUERY_LENGTH = 20000
    _IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
    _CONFIG = dict(
        enable_external_access="false",
        allow_community_extensions="false",
        autoinstall_known_extensions="false",
        autoload_known_extensions="false",
        allow_unsigned_extensions="false",
        memory_limit="256MB",
        max_temp_directory_size="0B",
        threads="1",
        lock_configuration="true",
    )

    def __init__(
        self,
        name: str,
        records: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[str] = (),
        executor: SQLQueryExecutor | None = None,
    ) -> None:
        self._validate_identifier(name, "table")
        self.name = name
        self._executor = executor
        self.records = deepcopy([dict(row) for row in records])
        ordered: dict[str, str] = {}
        for column in columns:
            self._add_column(ordered, column, declared=True)
        for row in self.records:
            for column in row:
                self._add_column(ordered, column, declared=False)
        if not ordered:
            raise SQLQueryError("virtual table requires at least one column")
        self.columns = tuple(ordered.values())

    @classmethod
    def _validate_identifier(cls, name, kind) -> None:
        if not isinstance(name, str) or cls._IDENTIFIER.fullmatch(name) is None:
            raise SQLQueryError(
                f"SQL {kind} name must be a simple identifier: {name!r}"
            )

    @classmethod
    def _add_column(cls, ordered, name, *, declared) -> None:
        cls._validate_identifier(name, "column")
        key = name.casefold()
        previous = ordered.get(key)
        if previous is not None and (declared or previous != name):
            raise SQLQueryError(
                f"SQL column names must be unique ignoring case: {name!r}"
            )
        ordered[key] = name

    def _validate_read_query(self, text: str) -> str:
        """Parse one read query, check its operations and relations, then render SQL."""
        if not isinstance(text, str) or not text.strip():
            raise SQLQueryError("sql_query must not be blank")
        if len(text) > self.MAX_QUERY_LENGTH:
            raise SQLQueryError(
                f"sql_query cannot exceed {self.MAX_QUERY_LENGTH} characters"
            )
        try:
            statements = [
                node
                for node in parse(text, read="duckdb", error_level=ErrorLevel.RAISE)
                if node
            ]
        except SqlglotError as error:
            raise SQLQueryError(
                f"DuckDB query parsing failed: {error}"
            ) from error
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            raise SQLQueryError("sql_query requires exactly one read-only SELECT")
        query = statements[0]
        forbidden = (exp.DDL, exp.DML, exp.Command, exp.Into, exp.Lock)
        if any(isinstance(node, forbidden) for node in query.walk()):
            raise SQLQueryError("sql_query must not modify data or configuration")
        self._validate_relation_access(query)
        return query.sql(dialect="duckdb")

    def _validate_relation_access(self, query: exp.Query) -> None:
        """Allow the registered table and lexically visible CTEs as query sources."""
        ctes = {node.alias_or_name.casefold() for node in query.find_all(exp.CTE)}
        if self.name.casefold() in ctes:
            raise SQLQueryError("a CTE cannot shadow the registered table")
        for relation in query.find_all(exp.Table):
            if (
                not isinstance(relation.this, exp.Identifier)
                or relation.db
                or relation.catalog
            ):
                raise SQLQueryError(
                    "only the registered table and query CTEs may be read"
                )
        # Resolve CTEs lexically. A CTE declared inside a subquery cannot grant
        # access to a same-named physical relation elsewhere in the query.
        normalized = query.copy()
        for identifier in normalized.find_all(exp.Identifier):
            identifier.set("this", identifier.this.casefold())
        source_found = False
        try:
            for scope in traverse_scope(normalized):
                for _, source in scope.selected_sources.values():
                    if isinstance(source, exp.Table):
                        if source.name != self.name.casefold():
                            raise SQLQueryError(
                                f"sql_query can reference only {self.name!r}; found {source.name!r}"
                            )
                        source_found = True
        except SqlglotError as error:
            raise SQLQueryError(
                f"sql_query has invalid relation scopes: {error}"
            ) from error
        if not source_found:
            raise SQLQueryError(f"sql_query must read from {self.name!r}")

    def _records_to_arrow(self) -> pa.Table:
        try:
            if not self.records:
                return pa.table(
                    {name: pa.array([], type=pa.null()) for name in self.columns}
                )
            return pa.Table.from_pylist(
                [{name: row.get(name) for name in self.columns} for row in self.records]
            )
        except (pa.ArrowException, ValueError, TypeError, OverflowError) as error:
            raise SQLQueryError(
                f"virtual table records cannot be converted: {error}"
            ) from error

    async def run_sql_query(
        self,
        sql_query: str,
        *,
        max_result_rows: int = DEFAULT_MAX_RESULT_ROWS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Validate SQL before dispatch, then await its bounded worker execution.

        Oversized results raise SQLQueryError. The submission deadline includes
        parsing, executor queue wait, Arrow conversion, setup and fetching.
        Excess work is rejected. Cancellation interrupts the connection and waits
        for worker cleanup; noninterruptible conversion can outlast the budget.
        """
        count(max_result_rows, "max_result_rows", minimum=1)
        seconds(timeout_seconds, "timeout_seconds", positive=True)
        deadline = time.monotonic() + timeout_seconds
        sql = self._validate_read_query(sql_query)
        executor = self._executor
        if executor is None:
            executor = SQLQueryExecutor.default()
        return await executor.run(
            lambda control: self._execute_query(
                sql, max_result_rows, deadline, control
            ),
            deadline=deadline,
        )

    def _execute_query(
        self,
        sql: str,
        max_result_rows: int,
        deadline: float,
        control: _QueryControl,
    ) -> list[dict[str, Any]]:
        """Own the query deadline and connection through conversion and cleanup."""
        remaining = deadline - time.monotonic()
        if remaining <= 0 or control.stopped:
            raise SQLQueryError("sql_query exceeded its execution limit before execution")
        timer = Timer(remaining, control.interrupt)
        timer.daemon = True
        timer.start()
        connection = None
        try:
            arrow = self._records_to_arrow()
            connection = duckdb.connect(":memory:", config=self._CONFIG)
            control.attach(connection)
            connection.register(self.name, arrow)
            cursor = connection.execute(sql)
            names = [field[0] for field in cursor.description]
            if len({name.casefold() for name in names}) != len(names):
                raise SQLQueryError("SQL result column names must be unique")
            rows = cursor.fetchmany(max_result_rows + 1)
            if control.stopped:
                raise SQLQueryError(
                    "sql_query exceeded its execution limit"
                )
            if len(rows) > max_result_rows:
                raise SQLQueryError(
                    f"Result contains more than {max_result_rows} rows; narrow the query with LIMIT or aggregation"
                )
            return [
                dict(
                    zip(
                        names,
                        to_jsonable_python(
                            row, inf_nan_mode="null", serialize_unknown=True
                        ),
                        strict=True,
                    )
                )
                for row in rows
            ]
        except duckdb.Error as error:
            if control.stopped:
                raise SQLQueryError(
                    "sql_query exceeded its execution limit"
                ) from error
            raise SQLQueryError(f"DuckDB could not complete the query: {error}") from error
        finally:
            timer.cancel()
            timer.join()
            control.detach()
            if connection is not None:
                connection.close()

    async def build_tool_result(
        self,
        sql_query: str | None,
        *,
        max_result_rows: int = DEFAULT_MAX_RESULT_ROWS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, object]:
        """Return source or filtered rows, explaining when a filter removes all rows.

        Omitting SQL returns a copy of every source row; query limits apply only
        when SQL is supplied.
        """
        rows = (
            deepcopy(self.records)
            if sql_query is None
            else await self.run_sql_query(
                sql_query,
                max_result_rows=max_result_rows,
                timeout_seconds=timeout_seconds,
            )
        )
        payload: dict[str, object] = {"data": rows}
        if sql_query is not None and self.records and not rows:
            payload["sql_filter_notice"] = {
                "status": "empty_after_filter",
                "table_name": self.name,
                "source_row_count": len(self.records),
                "result_row_count": 0,
                "message": "Source rows were available, but this query selected none.",
            }
        return payload


__all__ = ["VirtualTable", "SQLQueryError"]
