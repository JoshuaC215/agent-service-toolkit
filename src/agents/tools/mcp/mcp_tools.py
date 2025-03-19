import math
import re
import asyncio
from typing import Any, List

import numexpr
from langchain_core.tools import BaseTool, tool

from agents.tools.mcp.adapters.file_adapters import FileServer

def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.
    
    Useful for when you need to answer questions about math using numexpr.
    Only input math expressions.
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},
                local_dict=local_dict,
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )

def file_list_func(path: str = "./") -> str:
    """Lists contents of a directory."""
    file_server = FileServer()
    return asyncio.run(file_server.list_directory(path))

def file_read_func(path: str) -> str:
    """Reads and returns the contents of a file."""
    file_server = FileServer()
    return asyncio.run(file_server.read_file(path))

def file_write_func(path: str, content: str) -> str:
    """Writes content to a file, creating it if necessary."""
    file_server = FileServer()
    return asyncio.run(file_server.write_file(path, content))

def file_create_directory_func(path: str) -> str:
    """Creates or ensures that a directory exists."""
    file_server = FileServer()
    return asyncio.run(file_server.create_directory(path))

def file_move_func(source: str, destination: str) -> str:
    """Moves or renames files and directories."""
    file_server = FileServer()
    return asyncio.run(file_server.move_file(source, destination))

def file_search_files_func(path: str, pattern: str, exclude_patterns: List[str] = None) -> str:
    """Searches recursively for files/directories matching a pattern."""
    file_server = FileServer()
    return asyncio.run(file_server.search_files(path, pattern, exclude_patterns))

def file_get_file_info_func(path: str) -> str:
    """Retrieves metadata for a file."""
    file_server = FileServer()
    return asyncio.run(file_server.get_file_info(path))

def file_list_allowed_directories_func() -> Any:
    """Lists directories the server is allowed to access."""
    file_server = FileServer()
    return asyncio.run(file_server.list_allowed_directories())

# Create tools using the langchain_core tool decorator

calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"
calculator.description = "Calculates mathematical expressions."

file_list: BaseTool = tool(file_list_func)
file_list.name = "ListDirectory"
file_list.description = "Lists the contents of a directory."

file_read: BaseTool = tool(file_read_func)
file_read.name = "ReadFile"
file_read.description = "Reads a file's contents."

file_write: BaseTool = tool(file_write_func)
file_write.name = "WriteFile"
file_write.description = "Writes content to a file."

file_create_directory: BaseTool = tool(file_create_directory_func)
file_create_directory.name = "CreateDirectory"
file_create_directory.description = "Creates or ensures a directory exists."

file_move: BaseTool = tool(file_move_func)
file_move.name = "MoveFile"
file_move.description = "Moves or renames a file or directory."

file_search_files: BaseTool = tool(file_search_files_func)
file_search_files.name = "SearchFiles"
file_search_files.description = "Searches recursively for files matching a pattern."

file_get_file_info: BaseTool = tool(file_get_file_info_func)
file_get_file_info.name = "GetFileInfo"
file_get_file_info.description = "Provides metadata for a file."

file_list_allowed_directories: BaseTool = tool(file_list_allowed_directories_func)
file_list_allowed_directories.name = "ListAllowedDirectories"
file_list_allowed_directories.description = "Lists allowed directories for file operations."

__all__ = [
    "calculator", "file_list", "file_read", "file_write",
    "file_create_directory", "file_move", "file_search_files",
    "file_get_file_info", "file_list_allowed_directories"
]