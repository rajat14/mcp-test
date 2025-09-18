from fastapi import UploadFile, File
from fastapi import APIRouter
from src.core.instrumentation.logger.logger import Logger
from src.core.instrumentation.decorators.transaction_data import initialize_transaction_data
from src.core.instrumentation.decorators.function_timer import timer_start, timer_end
from src.core.instrumentation.decorators.log_entry_exit import log_entry_exit
from src.domain.dto.lineage_dto import LineageResponse, LineageStatusResponse, OpenLineageResponse
from src.services.lineage.lineage_service import LineageService

from src.web.util.request_util import RequestUtil

logger = Logger()

router = APIRouter(
    tags=["lineage"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/lineage",
    openapi_extra={
        "mcp_description": "Extracts lineage information from uploaded files and stores it in a vectorDB. Supports various file types such as SAS, Spark, and SQL. "
                           "This endpoint does not return the lineage data directly; instead, it processes the file and stores the extracted lineage information for future queries.",
        "mcp_scenarios": [
            "when the input is formdata containing a file upload",
            "when you need to extract and process lineage information from different file types",
            "upload a SAS, Spark, or SQL file to analyze its lineage"
        ],
        "mcp_file_upload": True,
        "mcp_file_upload_description": "Upload a file containing lineage information. Supported types include 'sas', 'spark', 'log', 'xml', 'sql' etc. Default is 'sas'."
    }
)
@initialize_transaction_data()
@timer_end()
@timer_start()
@log_entry_exit(api_router=True)
def post_lineage(
        file: UploadFile = File(...),
        type: str = "sas"
) -> LineageResponse:
    file_name, file_contents = RequestUtil.extract_uploaded_file_contents(file)
    workflow_details = LineageService().post_lineage_process(
        file_name=file_name,
        file_contents=file_contents,
        type=type
    )
    return LineageResponse(**workflow_details)

@router.get(
    "/lineage/status/{workflow_name}",
    openapi_extra={
        "mcp_description": "Retrieves the status of a previously submitted lineage extraction workflow. "
                           "This endpoint allows users to check whether the lineage extraction process has completed successfully, is still in progress, or has encountered any errors.",
        "mcp_scenarios": [
            "when you need to check the status of a lineage extraction workflow",
            "check if the lineage extraction process is complete or still running"
        ],
        "examples": [
            "what is the of a job named 'lineage_extraction_12345'?",
            "get the status of the workflow 'lineage_job_67890'",
            "check the progress of 'data_lineage_workflow_abcde'"
        ]
    }
)
@initialize_transaction_data()
@timer_end()
@timer_start()
@log_entry_exit(api_router=True)
def lineage_status(
        workflow_name: str
) -> LineageStatusResponse:
    status = LineageService().get_lineage_status(
        workflow_name=workflow_name
    )
    return LineageStatusResponse(workflow_name=workflow_name, status=status)

@router.get(
    "/lineage",
    response_model=OpenLineageResponse,
    openapi_extra={
        "mcp_description": "Fetches lineage information for a specified dataset or column within a given namespace. "
                           "Users can specify whether they want to retrieve upstream and/or downstream lineage, as well as the depth of the lineage graph. "
                           "This endpoint is useful for understanding data dependencies and the flow of data within an organization."
    }
)
@initialize_transaction_data()
@timer_end()
@timer_start()
@log_entry_exit(api_router=True)
def get_lineage(
        lineage_type: str,
        namespace: str,
        table_name: str,
        column_name: str = None,
        upstream: bool = True,
        downstream: bool = True,
        depth: int = 1,
        include_transformations: bool = False
) -> OpenLineageResponse:
    lineage = LineageService().get_lineage(
        lineage_type=lineage_type,
        namespace=namespace,
        table_name=table_name,
        column_name=column_name,
        upstream=upstream,
        downstream=downstream,
        depth=depth,
        include_transformations=include_transformations
    )
    return OpenLineageResponse(lineage=lineage)


