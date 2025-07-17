import sys
import json
import argparse
from mcp.server.fastmcp import FastMCP
from .tool import UserParameters, ToolParameters, run_tool

# Initialize MCP
mcp = FastMCP(name="DocumentaryCreditValidator")

# Register tool with MCP
@mcp.tool(name="run_document_check", description="Runs the LC document validation tool")
def run_document_check(
    lc_pdf_path: str,
    swift_pdf_path: str,
) -> dict:
    """
    Entry point for MCP tool. Validates LC documents against SWIFT clauses.
    """
    config = UserParameters()
    params = ToolParameters(lc_pdf_path=lc_pdf_path, swift_pdf_path=swift_pdf_path)
    result = run_tool(config, params)
    return result

# Main entrypoint
def main():
        mcp.run(transport="stdio")
    

if __name__ == "__main__":
    main()
