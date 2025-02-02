import math
import re

import numexpr
from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


async def mcp(server: str = None, tool: str = None, arguments: dict = None):
    """
    Dynamic MCP (Model Context Protocol) client for discovering and interacting with MCP servers.

    Key Features:
    - Server discovery and capability exploration
    - Dynamic tool execution
    - Configuration management across multiple locations

    Configuration:
        The client looks for mcp_config.json in the following locations (in order):
        1. Current directory
        2. ~/.config/autogen/mcp_config.json
        3. Path specified in MCP_CONFIG_PATH environment variable

    Usage:
        1. List Available Servers:
        result = await mcp(tool='list_available_servers')

        2. Discover Server Tools:
        result = await mcp(
            server='server_name',
            tool='tool_details'
        )

        3. Execute Tool:
        result = await mcp(
            server='server_name',
            tool='tool_name',
            arguments={'param1': 'value1'}
        )

    Args:
        server (str, optional): Name of the MCP server to connect to
        tool (str, optional): Either a tool name to execute or one of these special commands:
            - 'list_available_servers': Lists all enabled servers
            - 'tool_details': Returns detailed information about available tools
        arguments (dict, optional): Key-value pairs of arguments for the tool

    Returns:
        str: Tool execution results, server listings, or error messages

    Example:
        # List available servers
        servers = await mcp(tool='list_available_servers')
        
        # Get tool details for a specific server
        tools = await mcp(server='brave-search', tool='tool_details')
        
        # Execute a search
        results = await mcp(
            server='brave-search',
            tool='brave_web_search',
            arguments={'query': 'python programming'}
        )

    Notes:
        - Requires proper configuration in mcp_config.json
        - Supports NPX-based tools with automatic path detection
        - Handles environment variables from configuration
    """
    try:
        import json
        import asyncio
        import os
        import platform
        from pathlib import Path
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Check multiple config locations
        possible_paths = [
            Path('mcp_config.json'),  # Current directory
            Path.home() / '.config' / 'autogen' / 'mcp_config.json',  # User config dir
            Path(os.getenv('MCP_CONFIG_PATH', 'mcp_config.json')),  # Environment variable
        ]

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if not config_path:
            paths_checked = '\n'.join(str(p) for p in possible_paths)
            return f"Error: No configuration file found. Checked:\n{paths_checked}"

        # Get system-specific npx path
        system = platform.system()
        if system == "Darwin":  # macOS
            default_npx = Path("/opt/homebrew/bin/npx")
        elif system == "Windows":
            default_npx = Path(os.getenv("APPDATA")) / "npm/npx.cmd"
        else:  # Linux and others
            default_npx = Path("/usr/local/bin/npx")

        # Find npx in PATH if default doesn't exist
        npx_path = str(default_npx if default_npx.exists() else "npx")

        # Load config
        with open(config_path) as f:
            config_data = json.load(f)
            servers = config_data.get('mcpServers', {})

        # Handle list_available_servers
        if tool == 'list_available_servers':
            enabled_servers = [name for name, cfg in servers.items() if cfg.get('enabled', True)]
            return json.dumps(enabled_servers, indent=2)

        # Validate server
        if not server:
            return "Error: Server parameter required for tool operations"
        if server not in servers:
            return f"Error: Server {server} not found"
        if not servers[server].get('enabled', True):
            return f"Error: Server {server} is disabled in configuration"

        # Build server connection
        config = servers[server]
        command = npx_path if config['command'] == 'npx' else config['command']
        env = os.environ.copy()
        env.update(config.get('env', {}))

        arguments = arguments or {}

        # Connect to server and execute tool
        async with stdio_client(StdioServerParameters(
            command=command, 
            args=config.get('args', []), 
            env=env
        )) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Handle tool_details
                if tool == 'tool_details':
                    result = await session.list_tools()
                    return json.dumps([{
                        'name': t.name,
                        'description': t.description,
                        'input_schema': t.inputSchema
                    } for t in result.tools], indent=2)

                # Execute requested tool
                if not tool:
                    return "Error: Tool name required"

                result = await session.call_tool(tool, arguments=arguments)
                return str(result)

    except Exception as e:
        return f"Error: {str(e)}"


mcp_tool: BaseTool = tool(mcp)
mcp_tool.name = "MCP"
mcp_tool.description = "Dynamic MCP client that adapts to available server capabilities"

# Export both tools
__all__ = ["calculator", "mcp_tool"]