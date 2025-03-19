from pathlib import Path
from typing import Any, List
from ..client import MCPClient
import os

class FileServer(MCPClient):
    """Simple file operations server with basic list and read capabilities"""

    def __init__(self):
        super().__init__()
        # Get the agent-service-toolkit directory
        self.agent_home = os.getenv('AGENT_HOME')
        if not self.agent_home:
            raise ValueError("AGENT_HOME environment variable must be set")

        self.command = "npx"
        # Only allow access to agent_home directory
        self.args = [
            "@modelcontextprotocol/server-filesystem",
            self.agent_home
        ]

    async def list_directory(self, path: str = "./") -> Any:
        """List contents of a directory relative to agent-service-toolkit root"""
        try:
            result = await self.execute("list_directory", {"path": path})
            return result.content[0].text if result.content else "Empty directory"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    async def read_file(self, path: str) -> Any:
        """Read contents of a file relative to agent-service-toolkit root"""
        try:
            result = await self.execute("read_file", {"path": path})
            return result.content[0].text if result.content else ""
        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def write_file(self, path: str, content: str) -> Any:
        """Write contents to a file, creating it if it doesn't exist"""
        try:
            result = await self.execute("write_file", {"path": path, "content": content})
            return result.content[0].text if result.content else "File written successfully"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    async def create_directory(self, path: str) -> Any:
        """Create a new directory or ensure it exists"""
        try:
            result = await self.execute("create_directory", {"path": path})
            return result.content[0].text if result.content else "Directory created successfully"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    async def move_file(self, source: str, destination: str) -> Any:
        """Move or rename files and directories"""
        try:
            result = await self.execute("move_file", {"source": source, "destination": destination})
            return result.content[0].text if result.content else "File moved successfully"
        except Exception as e:
            return f"Error moving file: {str(e)}"

    async def search_files(self, path: str, pattern: str, exclude_patterns: List[str] = None) -> Any:
        """Recursively search for files/directories"""
        try:
            result = await self.execute("search_files", {"path": path, "pattern": pattern, "excludePatterns": exclude_patterns or []})
            return result.content[0].text if result.content else "No files found"
        except Exception as e:
            return f"Error searching files: {str(e)}"

    async def get_file_info(self, path: str) -> Any:
        """Get detailed file/directory metadata"""
        try:
            result = await self.execute("get_file_info", {"path": path})
            return result.content[0].text if result.content else "No file info available"
        except Exception as e:
            return f"Error getting file info: {str(e)}"

    async def list_allowed_directories(self) -> Any:
        """List all directories the server is allowed to access"""
        try:
            return self.args[1:]  # Assuming args[1] holds the allowed directories
        except Exception as e:
            return f"Error listing allowed directories: {str(e)}"