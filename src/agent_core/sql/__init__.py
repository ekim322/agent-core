"""Optional read-only SQL over structured tool results.

Install the sql extra. VirtualTable owns query validation; SQLQueryExecutor owns bounded workers;
applications own fetching and interpreting the source rows.
"""

from agent_core.sql.executor import (
    SQLQueryCapacityExceeded,
    SQLQueryError,
    SQLQueryExecutor,
)
from agent_core.sql.table import VirtualTable

__all__ = ["SQLQueryCapacityExceeded", "SQLQueryError", "SQLQueryExecutor", "VirtualTable"]
