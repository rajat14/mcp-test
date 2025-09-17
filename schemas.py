# schemas.py
from typing import List, Optional, Any
from pydantic import BaseModel

class ColumnInfo(BaseModel):
    cid: int
    name: str
    type: str
    notnull: bool
    default_value: Optional[Any]
    pk: bool

class TableSchema(BaseModel):
    table: str
    columns: List[ColumnInfo]

class LineageMapping(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    transformation: Optional[str]

class DatabasesList(BaseModel):
    databases: List[str]

class TablesList(BaseModel):
    database: str
    tables: List[str]

class RowCountResp(BaseModel):
    table: str
    row_count: int

class DistinctValuesResp(BaseModel):
    table: str
    column: str
    distinct_values: List[Any]

# Query endpoint
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class PlanStep(BaseModel):
    tool: str
    params: dict

class QueryResponse(BaseModel):
    session_id: Optional[str]
    plan: List[PlanStep]
    results: Any
    message: Optional[str] = None
