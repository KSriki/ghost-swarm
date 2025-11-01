"""
Filesystem MCP Server for Ghost Swarm.

Provides tools and resources for file operations with security controls.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import structlog

from mcp_server.base import BaseMCPServer

logger = structlog.get_logger(__name__)


class FilesystemMCPServer(BaseMCPServer):
    """
    MCP Server for filesystem operations.
    
    Provides:
    - Tools: read_file, write_file, list_directory, search_files
    - Resources: File contents at specific paths
    - Prompts: File analysis templates
    """
    
    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        readonly: bool = False,
    ) -> None:
        """
        Initialize filesystem MCP server.
        
        Args:
            allowed_directories: List of allowed directory paths
            readonly: If True, only allow read operations
        """
        super().__init__(name="filesystem", version="1.0.0")
        
        self.allowed_directories = [
            Path(d).resolve() for d in (allowed_directories or ["."])
        ]
        self.readonly = readonly
        
        logger.info(
            "filesystem_server_initialized",
            allowed_dirs=len(self.allowed_directories),
            readonly=readonly,
        )
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories."""
        resolved_path = Path(path).resolve()
        
        return any(
            resolved_path.is_relative_to(allowed_dir)
            for allowed_dir in self.allowed_directories
        )
    
    async def setup(self) -> None:
        """Setup filesystem tools and resources."""
        
        # Tool: Read file
        self.register_tool(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file to read",
                    },
                },
                "required": ["path"],
            },
            handler=self._read_file,
        )
        
        # Tool: Write file (if not readonly)
        if not self.readonly:
            self.register_tool(
                name="write_file",
                description="Write content to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._write_file,
            )
        
        # Tool: List directory
        self.register_tool(
            name="list_directory",
            description="List contents of a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to directory",
                    },
                },
                "required": ["path"],
            },
            handler=self._list_directory,
        )
        
        # Tool: Search files
        self.register_tool(
            name="search_files",
            description="Search for files matching a pattern",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File pattern (e.g., '*.py')",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
            handler=self._search_files,
        )
        
        # Prompt: File analysis
        self.register_prompt(
            name="analyze_file",
            description="Analyze a file's contents",
            arguments=[
                {
                    "name": "path",
                    "description": "Path to file",
                    "required": True,
                }
            ],
            template="""Analyze the following file: {path}

Please provide:
1. A summary of the file's purpose
2. Key components or sections
3. Any issues or improvements
4. Dependencies or relationships""",
        )
    
    async def _read_file(self, path: str) -> dict[str, Any]:
        """Read file contents."""
        if not self._is_path_allowed(path):
            raise PermissionError(f"Access denied: {path}")
        
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not file_path.is_file():
            raise IsADirectoryError(f"Not a file: {path}")
        
        content = file_path.read_text()
        
        return {
            "path": str(path),
            "content": content,
            "size": file_path.stat().st_size,
        }
    
    async def _write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write content to file."""
        if self.readonly:
            raise PermissionError("Server is in readonly mode")
        
        if not self._is_path_allowed(path):
            raise PermissionError(f"Access denied: {path}")
        
        file_path = Path(path)
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path.write_text(content)
        
        return {
            "path": str(path),
            "bytes_written": len(content),
            "success": True,
        }
    
    async def _list_directory(self, path: str) -> dict[str, Any]:
        """List directory contents."""
        if not self._is_path_allowed(path):
            raise PermissionError(f"Access denied: {path}")
        
        dir_path = Path(path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        entries = []
        for entry in dir_path.iterdir():
            entries.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else 0,
            })
        
        return {
            "path": str(path),
            "entries": entries,
            "count": len(entries),
        }
    
    async def _search_files(
        self,
        pattern: str,
        directory: str = ".",
    ) -> dict[str, Any]:
        """Search for files matching pattern."""
        if not self._is_path_allowed(directory):
            raise PermissionError(f"Access denied: {directory}")
        
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Invalid directory: {directory}")
        
        matches = list(dir_path.glob(pattern))
        
        files = [
            {
                "path": str(match.relative_to(dir_path)),
                "size": match.stat().st_size if match.is_file() else 0,
            }
            for match in matches
        ]
        
        return {
            "pattern": pattern,
            "directory": str(directory),
            "matches": files,
            "count": len(files),
        }


async def main() -> None:
    """Run the filesystem MCP server."""
    from common import configure_logging
    
    configure_logging()
    
    server = FilesystemMCPServer(
        allowed_directories=["./data", "./docs"],
        readonly=False,
    )
    
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())