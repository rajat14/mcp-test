# app/api/tools.py
"""
Tools router: exposes the contract-first lineage tool endpoints backed by SQLiteService.

Endpoints:
 - GET  /health
 - GET  /tools/registry
 - GET  /tools/databases
 - GET  /tools/tables?database=<optional>
 - GET  /tools/schema/{table}
 - GET  /tools/table-lineage/{table}
 - GET  /tools/column-lineage/{table}/{column}
 - GET  /tools/search?keyword=...&limit=...
 - GET  /tools/row-count/{table}
 - GET  /tools/distinct-values/{table}/{column}?limit=...
 - POST /tools/cache/invalidate
"""

import os
import time
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from pydantic import BaseModel

# from app import services
from services.schemas import (
    DatabasesList,
    TablesList,
    TableSchema,
    RowCountResp,
    DistinctValuesResp,
)
from services.sqlite_service import (
    SQLiteService,
    NotFoundError,
    ValidationError as ServiceValidationError,
    SQLiteServiceError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])

# Config
DB_PATH = os.environ.get("MOCK_LINEAGE_DB", "mock_lineage.db")
CONCEPTUAL_DBS = {
    "sales_db": ["customers", "orders", "payments"],
    "fraud_db": ["transactions", "alerts"],
}
DISTINCT_LIMIT_CAP = 1000
SEARCH_LIMIT_CAP = 2000

# ---- Dependency: service instance (singleton per process) ----
_service_instance: Optional[SQLiteService] = None


def get_sqlite_service() -> SQLiteService:
    global _service_instance
    if _service_instance is None:
        _service_instance = SQLiteService(db_path=DB_PATH, conceptual_dbs=CONCEPTUAL_DBS)
    return _service_instance


# ---- Helper: trace / timing decorator-like util ----
def _start_trace(request: Request) -> str:
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    request.state.trace_id = trace_id
    request.state.start_time = time.time()
    return trace_id


def _end_trace(request: Request, operation: str, extra: Optional[Dict[str, Any]] = None):
    duration = (time.time() - getattr(request.state, "start_time", time.time()))
    trace_id = getattr(request.state, "trace_id", None)
    logger.info("[%s] %s completed in %.3fs params=%s", trace_id or "-", operation, duration, extra or {})


# ---- Admin / Health endpoints ----
@router.get("/health")
def health_check(request: Request, svc: SQLiteService = Depends(get_sqlite_service)):
    trace_id = _start_trace(request)
    try:
        # simple sanity check: list tables
        tables = svc.list_tables()
        _end_trace(request, "health_check", {"table_count": len(tables)})
        return {"status": "ok", "tables": len(tables), "trace_id": trace_id}
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DB health check failed")


@router.get("/registry")
def get_tool_registry(request: Request, svc: SQLiteService = Depends(get_sqlite_service)):
    """
    Return a machine-readable registry describing the available tools (input/output contract hints).
    This uses Pydantic schema metadata where available so orchestrators can inspect contracts.
    """
    trace_id = _start_trace(request)
    # Compose registry entries
    registry = {
        "tools": [
            {
                "name": "list_databases",
                "path": "/tools/databases",
                "method": "GET",
                "read_only": True,
                "description": "List conceptual database names",
                "response_schema": DatabasesResponse.schema(),
            },
            {
                "name": "list_tables",
                "path": "/tools/tables",
                "method": "GET",
                "read_only": True,
                "description": "List tables (optionally for a conceptual database)",
                "response_schema": TablesResponse.schema(),
            },
            {
                "name": "get_table_schema",
                "path": "/tools/schema/{table}",
                "method": "GET",
                "read_only": True,
                "description": "Return TableSchema for a given table",
                "response_schema": TableSchema.schema(),
            },
            {
                "name": "get_table_lineage",
                "path": "/tools/table-lineage/{table}",
                "method": "GET",
                "read_only": True,
                "description": "Lineage mappings where the table is source or target",
                "response_schema": {"type": "array", "items": LineageMapping.schema()},
            },
            {
                "name": "get_column_lineage",
                "path": "/tools/column-lineage/{table}/{column}",
                "method": "GET",
                "read_only": True,
                "description": "Lineage mappings for a specific column",
                "response_schema": {"type": "array", "items": LineageMapping.schema()},
            },
            {
                "name": "search_entities",
                "path": "/tools/search",
                "method": "GET",
                "read_only": True,
                "description": "Fuzzy search for tables/columns by keyword",
                "response_schema": {"type": "array", "items": {"type": "object"}},
            },
            {
                "name": "row_count",
                "path": "/tools/row-count/{table}",
                "method": "GET",
                "read_only": True,
                "description": "Row count for a table",
                "response_schema": RowCountResponse.schema(),
            },
            {
                "name": "distinct_values",
                "path": "/tools/distinct-values/{table}/{column}",
                "method": "GET",
                "read_only": True,
                "description": "Distinct values for a column",
                "response_schema": DistinctValuesResponse.schema(),
            },
        ]
    }
    _end_trace(request, "get_tool_registry")
    return {"trace_id": trace_id, "registry": registry}


# ---- Core tool endpoints ----
@router.get("/databases", response_model=DatabasesList)
def api_list_databases(request: Request, svc: SQLiteService = Depends(get_sqlite_service)):
    trace_id = _start_trace(request)
    try:
        dbs = svc.list_databases()
        _end_trace(request, "list_databases", {"count": len(dbs)})
        return DatabasesResponse(databases=dbs)
    except SQLiteServiceError as e:
        logger.exception("list_databases failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables", response_model=TablesList)
def api_list_tables(
    request: Request,
    database: Optional[str] = Query(None, description="Optional conceptual database name"),
    svc: SQLiteService = Depends(get_sqlite_service),
):
    trace_id = _start_trace(request)
    try:
        tables = svc.list_tables(database)
        _end_trace(request, "list_tables", {"database": database, "count": len(tables)})
        return TablesResponse(database=database or "all", tables=tables)
    except NotFoundError as e:
        logger.warning("list_tables: unknown database %s", database)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("list_tables failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema/{table}", response_model=TableSchema)
def api_get_schema(request: Request, table: str, svc: SQLiteService = Depends(get_sqlite_service)):
    trace_id = _start_trace(request)
    try:
        schema = svc.get_table_schema(table)
        _end_trace(request, "get_table_schema", {"table": table, "cols": len(schema.columns)})
        return schema
    except ServiceValidationError as e:
        logger.warning("get_table_schema validation failed for %s: %s", table, e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except NotFoundError as e:
        logger.warning("get_table_schema not found: %s", table)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("get_table_schema failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/table-lineage/{table}", response_model=List[LineageMapping])
def api_table_lineage(request: Request, table: str, svc: SQLiteService = Depends(get_sqlite_service)):
    trace_id = _start_trace(request)
    try:
        mappings = svc.get_table_lineage(table)
        _end_trace(request, "get_table_lineage", {"table": table, "mappings": len(mappings)})
        return mappings
    except ServiceValidationError as e:
        logger.warning("get_table_lineage validation failed for %s: %s", table, e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except NotFoundError as e:
        logger.warning("get_table_lineage not found: %s", table)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("get_table_lineage failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/column-lineage/{table}/{column}", response_model=List[LineageMapping])
def api_column_lineage(
    request: Request,
    table: str,
    column: str,
    svc: SQLiteService = Depends(get_sqlite_service),
):
    trace_id = _start_trace(request)
    try:
        mappings = svc.get_column_lineage(table, column)
        _end_trace(request, "get_column_lineage", {"table": table, "column": column, "mappings": len(mappings)})
        return mappings
    except ServiceValidationError as e:
        logger.warning("get_column_lineage validation failed for %s.%s: %s", table, column, e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except NotFoundError as e:
        logger.warning("get_column_lineage not found: %s.%s", table, column)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("get_column_lineage failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def api_search(
    request: Request,
    keyword: str = Query(..., min_length=1, description="Keyword to search in table and column names"),
    limit: int = Query(200, ge=1, le=SEARCH_LIMIT_CAP),
    svc: SQLiteService = Depends(get_sqlite_service),
):
    trace_id = _start_trace(request)
    try:
        results = svc.search_entities(keyword, limit=limit)
        _end_trace(request, "search_entities", {"keyword": keyword, "returned": len(results)})
        return {"trace_id": trace_id, "keyword": keyword, "results": results}
    except ServiceValidationError as e:
        logger.warning("search_entities validation failed for %s: %s", keyword, e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("search_entities failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/row-count/{table}", response_model=RowCountResp)
def api_row_count(request: Request, table: str, svc: SQLiteService = Depends(get_sqlite_service)):
    trace_id = _start_trace(request)
    try:
        cnt = svc.row_count(table)
        _end_trace(request, "row_count", {"table": table, "count": cnt})
        return RowCountResponse(table=table, count=cnt)
    except ServiceValidationError as e:
        logger.warning("row_count validation failed for %s: %s", table, e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except NotFoundError as e:
        logger.warning("row_count not found: %s", table)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("row_count failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distinct-values/{table}/{column}", response_model=DistinctValuesResp)
def api_distinct_values(
    request: Request,
    table: str,
    column: str,
    limit: int = Query(50, ge=1, le=DISTINCT_LIMIT_CAP),
    svc: SQLiteService = Depends(get_sqlite_service),
):
    trace_id = _start_trace(request)
    try:
        vals = svc.distinct_values(table, column, limit=limit)
        _end_trace(request, "distinct_values", {"table": table, "column": column, "returned": len(vals)})
        return DistinctValuesResponse(table=table, column=column, values=vals)
    except ServiceValidationError as e:
        logger.warning("distinct_values validation failed for %s.%s: %s", table, column, e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except NotFoundError as e:
        logger.warning("distinct_values not found: %s.%s", table, column)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SQLiteServiceError as e:
        logger.exception("distinct_values failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Cache management / admin ----
class CacheInvalidateRequest(BaseModel):
    tables: Optional[List[str]] = None  # if None -> clear all cached schemas


@router.post("/cache/invalidate")
def api_cache_invalidate(request: Request, body: CacheInvalidateRequest, svc: SQLiteService = Depends(get_sqlite_service)):
    trace_id = _start_trace(request)
    try:
        if body.tables:
            for t in body.tables:
                svc.invalidate_table_schema(t)
            msg = f"invalidated schemas for {len(body.tables)} tables"
        else:
            svc.invalidate_table_schema()
            svc.invalidate_table_cache()
            msg = "invalidated all table schema and table list cache"
        _end_trace(request, "cache_invalidate", {"detail": msg})
        return {"trace_id": trace_id, "message": msg}
    except ServiceValidationError as e:
        logger.warning("cache_invalidate validation error: %s", e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception:
        logger.exception("cache_invalidate failed")
        raise HTTPException(status_code=500, detail="Cache invalidation failed")
