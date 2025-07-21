import os, json
from typing import Annotated, Literal
from pydantic import Field

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Sample-MCP")


@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"


@mcp.prompt()
def echo_prompt(message: str) -> str:
    """Create an echo prompt"""
    return f"Please process this message: {message}"


@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"


@mcp.tool()
def calculator_tool(
    a: Annotated[float, Field(description="first number")],
    b: Annotated[float, Field(description="second number")],
    operator: Annotated[Literal["+", "-", "*", "/"], Field(description="operator")],
) -> str:
    """
    Calculator tool which can do basic addition, subtraction, multiplication, and division.
    Division by 0 is not allowed.
    """
    res = None
    if operator == "+":
        res = a + b
    elif operator == "-":
        res = a - b
    elif operator == "*":
        res = a * b
    elif operator == "/":
        res = float(a / b)
    else:
        raise ValueError("Invalid operator")
    return str(res)

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


# Get configuration from environment variables
def get_config():
    return {
        "host": os.environ.get("CLOUDERA_ML_HOST", ""),
        "api_key": os.environ.get("CLOUDERA_ML_API_KEY", "")
    }



def main():
    print("Starting MCP server...")
    mcp.run(transport="stdio")


def testrepo():
    print("Testing.... testing.... testing....")


if __name__ == "__main__":
    main()