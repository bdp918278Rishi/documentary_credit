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
    if "--mcp-stdio" in sys.argv:
        # MCP stdio transport (used in production)
        mcp.run(transport="stdio")
    elif "--user-params" in sys.argv and "--tool-params" in sys.argv:
        # CLI testing mode
        parser = argparse.ArgumentParser(description="Run the Documentary Credit Validation Tool")
        parser.add_argument("--user-params", required=True, help="JSON string of UserParameters")
        parser.add_argument("--tool-params", required=True, help="JSON string of ToolParameters")

        args = parser.parse_args()

        try:
            user_dict = json.loads(args.user_params)
            tool_dict = json.loads(args.tool_params)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON input: {e}")
            sys.exit(1)

        try:
            config = UserParameters(**user_dict)
            params = ToolParameters(**tool_dict)
        except Exception as e:
            print(f"❌ Error initializing parameters: {e}")
            sys.exit(1)

        output = run_tool(config, params)
        print("✅ Output:")
        print(json.dumps(output, indent=2))
    else:
        # Show usage help
        print("\n❌ Missing required arguments.\n")
        print("📘 Usage:")
        print("  - MCP Mode:")
        print("      documentary_credit_tool --mcp-stdio")
        print("  - CLI Mode:")
        print("      documentary_credit_tool \\")
        print("         --user-params '{\"param1\": \"value\"}' \\")
        print("         --tool-params '{\"lc_pdf_path\": \"path/to/lc.pdf\", \"swift_pdf_path\": \"path/to/swift.pdf\"}'\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
