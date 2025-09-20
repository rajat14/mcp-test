{
  "tools": [
    {
      "name": "Health",
      "title": null,
      "description": "API endpoint: GET /health",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Insight",
      "title": null,
      "description": "Generates Insights for a given namespace.table_name.column (we call this a nodeId)combination. It queries lineage for a given nodeId and generates upstream and downstream insights based on the lineage graph. ",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "namespace": {
                "type": "string",
                "description": "Field namespace for InsightRequest"
              },
              "table_name": {
                "type": "string",
                "description": "Field table_name for InsightRequest"
              },
              "column_name": {
                "type": "string",
                "description": "Field column_name for InsightRequest"
              },
              "insight": {
                "type": "string",
                "description": "Field insight for InsightResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Summarize And Format Responses",
      "title": null,
      "description": "responds and formats multiple api responses into a single coherent response. It takes a user query and additional information as input and generates a summarized response.The additional information can be a sql query and/or data and/or json and/or other relevant information.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "user_query": {
                "type": "string",
                "description": "Field user_query for SummarizeAndFormatRequest"
              },
              "additional_information": {
                "type": "string",
                "description": "Field additional_information for SummarizeAndFormatRequest"
              },
              "response": {
                "type": "string",
                "description": "Field response for SummarizeAndFormatResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Execute With Langgraph",
      "title": null,
      "description": "API endpoint: POST /langgraph/execute",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "user_query": {
                "type": "string",
                "description": "Field user_query for McpExecuteRequest"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Execute Custom Workflow",
      "title": null,
      "description": "API endpoint: POST /langgraph/workflow/custom",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Post Lineage",
      "title": null,
      "description": "Extracts lineage information from uploaded files and stores it in a vectorDB. Supports various file types such as SAS, Spark, and SQL. This endpoint does not return the lineage data directly; instead, it processes the file and stores the extracted lineage information for future queries.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "workflow_name": {
                "type": "string",
                "description": "Field workflow_name for LineageResponse"
              },
              "workflow_id": {
                "type": "string",
                "description": "Field workflow_id for LineageResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Lineage Status",
      "title": null,
      "description": "Retrieves the status of a previously submitted lineage extraction workflow. This endpoint allows users to check whether the lineage extraction process has completed successfully, is still in progress, or has encountered any errors.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "workflow_name": {
                "type": "string",
                "description": "Field workflow_name for LineageStatusResponse"
              },
              "status": {
                "type": "string",
                "description": "Field status for LineageStatusResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Get Lineage",
      "title": null,
      "description": "Fetches lineage information for a specified dataset or column within a given namespace. Users can specify whether they want to retrieve upstream and/or downstream lineage, as well as the depth of the lineage graph. This endpoint is useful for understanding data dependencies and the flow of data within an organization.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "lineage": {
                "type": "string",
                "description": "Field lineage for OpenLineageResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Smart Data Retrieval",
      "title": null,
      "description": "retrieves similar database.table_name.column_name combinations based on a user query.  Use this api when you have a database/table/column in natural language and you want to find the actual column names. This api does not provide any additional information about the columns like data type, description etc. It only provides the database, table and column names.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "user_query": {
                "type": "string",
                "description": "Field user_query for SmartDataRetrievalRequest"
              },
              "results": {
                "type": "string",
                "description": "Field results for SmartDataRetrievalResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Text To Sql",
      "title": null,
      "description": "This api generates SQL query over openlineage data model given a natural language input. This generated SQL query can then be executed using the /text_to_sql/execute api.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "user_query": {
                "type": "string",
                "description": "Field user_query for TextToSqlRequest"
              },
              "sql": {
                "type": "string",
                "description": "Field sql for TextToSqlResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    },
    {
      "name": "Execute Sql",
      "title": null,
      "description": "Execute a SQL query and return the actual data results. Use this AFTER generating SQL with text_to_sql.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_params": {
            "type": "object",
            "properties": {},
            "additionalProperties": true
          },
          "request_body": {
            "type": "object",
            "properties": {
              "sql_query": {
                "type": "string",
                "description": "Field sql_query for TextToSqlExecuteRequest"
              },
              "result": {
                "type": "string",
                "description": "Field result for TextToSqlExecuteResponse"
              }
            },
            "additionalProperties": true
          }
        },
        "required": []
      },
      "outputSchema": null,
      "annotations": null,
      "_meta": null
    }
  ]
}
R
