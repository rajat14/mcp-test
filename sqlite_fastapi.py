from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
from sqlite_api_service import SQLiteAPIService
from enhanced_langgraph_service import EnhancedLanggraphSQLiteService

app = FastAPI(
    title="SQLite Database API",
    description="FastAPI server for SQLite database operations with LangGraph orchestration",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
sqlite_service = SQLiteAPIService("fastapi_database.db")
langgraph_service = EnhancedLanggraphSQLiteService("fastapi_database.db")

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class TableRequest(BaseModel):
    table_name: str

class ExecutionStep(BaseModel):
    step_id: str
    api_name: str
    params: Dict[str, Any]
    status: str
    result: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None

class OrchestrationResponse(BaseModel):
    query: str
    execution_plan: List[ExecutionStep]
    final_response: str
    total_duration_ms: float

# Basic API Routes
@app.get("/")
async def root():
    return {"message": "SQLite Database API with LangGraph Orchestration"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

# SQLite API Endpoints
@app.get("/api/tables")
async def list_tables():
    """List all tables in the database"""
    try:
        result = sqlite_service.list_tables()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table/describe")
async def describe_table(request: TableRequest):
    """Describe a specific table"""
    try:
        result = sqlite_service.describe_table({"table_name": request.table_name})
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def execute_query(request: QueryRequest):
    """Execute a SQL query"""
    try:
        result = sqlite_service.execute_query({"query": request.query})
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# LangGraph Orchestration Endpoints
@app.post("/api/orchestrate")
async def orchestrate_query(request: QueryRequest):
    """Process query with LangGraph orchestration and return detailed execution steps"""
    try:
        import time
        start_time = time.time()
        
        # Get the workflow execution details
        execution_details = await langgraph_service.process_query_with_orchestration_details(request.query)
        
        end_time = time.time()
        total_duration = (end_time - start_time) * 1000
        
        return OrchestrationResponse(
            query=request.query,
            execution_plan=execution_details["steps"],
            final_response=execution_details["final_response"],
            total_duration_ms=total_duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simple-query")
async def simple_query(request: QueryRequest):
    """Process query with simple response (no orchestration details)"""
    try:
        result = await langgraph_service.process_query_with_chaining(request.query)
        return {"query": request.query, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/api/available-tools")
async def get_available_tools():
    """Get list of available database tools"""
    return sqlite_service.get_available_tools()

@app.get("/api/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing"""
    return {
        "simple_queries": [
            "Show me all tables",
            "Describe the employees table",
            "Get all employees"
        ],
        "sql_queries": [
            "SELECT * FROM employees WHERE salary > 70000",
            "SELECT department, COUNT(*) FROM employees GROUP BY department",
            "SELECT * FROM projects WHERE status = 'In Progress'"
        ],
        "orchestration_queries": [
            "List all tables and describe the first one",
            "Get employee count and department count",
            "Show me projects and their department information"
        ]
    }

# WebSocket endpoint for real-time orchestration updates (optional)
@app.websocket("/ws/orchestration")
async def websocket_orchestration(websocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            if query:
                # Stream execution steps in real-time
                async for step in langgraph_service.process_query_with_streaming(query):
                    await websocket.send_json(step)
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Database API: http://localhost:8000/api/tables")
    print("🧠 Orchestration API: http://localhost:8000/api/orchestrate")
    
    uvicorn.run(
        "fastapi_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
)
