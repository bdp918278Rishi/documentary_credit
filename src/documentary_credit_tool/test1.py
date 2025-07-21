import sys
import json
import argparse
from mcp.server.fastmcp import FastMCP

# Initialize MCP
mcp = FastMCP(name="DocumentaryCreditValidator")

# Register tool with MCP
@mcp.tool(name="Documentary_Credit_Discrepancy_Tool", description="Runs the LC document validation tool")
def documentary_credit_check_tool(
    lc_pdf_path: str,
    swift_pdf_path: str,
) -> dict:
    
    """
    Documentary credit check tool which will Validates LC documents against SWIFT clauses.
    """
    from .tool import UserParameters, ToolParameters, run_tool

    config = UserParameters()
    params = ToolParameters(lc_pdf_path=lc_pdf_path, swift_pdf_path=swift_pdf_path)
    result = run_tool(config, params)
    extracted_path = result.get("extracted_text_folder_path")
    discrepancy_path = result.get("discrepancy_individual_folder_path")
    missing_docs = result.get("missing_document")
    return extracted_path, discrepancy_path, missing_docs

# Main entrypoint
def main():
        mcp.run(transport="stdio")
    

if __name__ == "__main__":
    main()
