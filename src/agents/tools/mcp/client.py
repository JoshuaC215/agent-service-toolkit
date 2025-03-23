from abc import ABC
from typing import Any, Dict, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from core.settings import settings

class MCPClient(ABC):
    """Base client for all MCP servers"""
    
    def __init__(self):
        if not settings.MCP_ENABLED:
            raise ValueError("MCP is not enabled")
        
        # Set by child classes
        self.command: str
        self.args: list[str]
    
    async def execute(self, tool: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Execute an MCP tool"""
        async with stdio_client(StdioServerParameters(
            command=self.command,
            args=self.args
        )) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, arguments or {})