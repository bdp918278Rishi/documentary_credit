# src/documentary_credit_tool/test.py

from mcp.server.fastmcp import FastMCP
from .tool import UserParameters, ToolParameters, run_tool
import json
import argparse
import sys

mcp = FastMCP(name="DocumentaryCreditValidator")

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

def main():
    if "--mcp-stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--user-params", required=True)
        parser.add_argument("--tool-params", required=True)
        args = parser.parse_args()

        user_dict = json.loads(args.user_params)
        tool_dict = json.loads(args.tool_params)

        config = UserParameters(**user_dict)
        params = ToolParameters(**tool_dict)

        output = run_tool(config, params)
        print(json.dumps(output, indent=2))
