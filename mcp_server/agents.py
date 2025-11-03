"""
Agent Management MCP Server for Ghost Swarm.

Provides tools for managing and interacting with Ghost agents via MCP.
Integrates with our Agent2Agent framework.
"""

import asyncio
from typing import Any
from uuid import UUID

import structlog

from common import (
    AgentInfo,
    TaskRequest,
    TaskResult,
    Message,
    MessageType,
    get_logger,
)
from mcp_server.base import BaseMCPServer

logger = get_logger(__name__)


class AgentManagementMCPServer(BaseMCPServer):
    """
    MCP Server for Ghost agent management.
    
    Provides tools for:
    - Listing available agents
    - Sending tasks to agents
    - Querying agent status
    - Getting agent capabilities
    """
    
    def __init__(self) -> None:
        """Initialize agent management MCP server."""
        super().__init__(name="ghost-agents", version="1.0.0")
        
        # Registry of known agents
        self.agents: dict[str, AgentInfo] = {}
        
        logger.info("agent_management_server_initialized")
    
    async def setup(self) -> None:
        """Setup agent management tools."""
        
        # Tool: List agents
        self.register_tool(
            name="list_agents",
            description="List all available Ghost agents",
            parameters={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "Filter by agent role (optional)",
                        "enum": ["orchestrator", "worker", "router", "evaluator", "optimizer"],
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by agent status (optional)",
                        "enum": ["idle", "busy", "error", "offline"],
                    },
                },
            },
            handler=self._list_agents,
        )
        
        # Tool: Get agent info
        self.register_tool(
            name="get_agent_info",
            description="Get detailed information about a specific agent",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "ID of the agent",
                    },
                },
                "required": ["agent_id"],
            },
            handler=self._get_agent_info,
        )
        
        # Tool: Send task to agent
        self.register_tool(
            name="send_task",
            description="Send a task to a Ghost agent",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "ID of target agent",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Type of task",
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Task parameters",
                    },
                },
                "required": ["agent_id", "task_type", "description"],
            },
            handler=self._send_task,
        )
        
        # Tool: Get agent capabilities
        self.register_tool(
            name="get_agent_capabilities",
            description="Get the capabilities of an agent",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "ID of the agent",
                    },
                },
                "required": ["agent_id"],
            },
            handler=self._get_agent_capabilities,
        )
        
        # Tool: Find best agent for task
        self.register_tool(
            name="find_best_agent",
            description="Find the best agent for a specific task type",
            parameters={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of task",
                    },
                    "required_capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required capabilities (optional)",
                    },
                },
                "required": ["task_type"],
            },
            handler=self._find_best_agent,
        )
        
        # Prompt: Agent coordination
        self.register_prompt(
            name="coordinate_agents",
            description="Coordinate multiple agents for a complex task",
            arguments=[
                {
                    "name": "task",
                    "description": "Description of the overall task",
                    "required": True,
                },
                {
                    "name": "agent_ids",
                    "description": "Comma-separated agent IDs",
                    "required": True,
                },
            ],
            template="""Coordinate the following agents to complete a complex task:

Task: {task}
Agents: {agent_ids}

Please:
1. Break down the task into subtasks
2. Assign subtasks to appropriate agents based on their capabilities
3. Define the execution order and dependencies
4. Specify how results should be aggregated""",
        )
    
    def register_agent(self, agent_info: AgentInfo) -> None:
        """
        Register an agent with the MCP server.
        
        Args:
            agent_info: Agent information
        """
        self.agents[agent_info.agent_id] = agent_info
        logger.info("agent_registered", agent_id=agent_info.agent_id)
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("agent_unregistered", agent_id=agent_id)
    
    async def _list_agents(
        self,
        role: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List available agents with optional filtering."""
        agents = list(self.agents.values())
        
        # Apply filters
        if role:
            agents = [a for a in agents if a.role.value == role]
        
        if status:
            agents = [a for a in agents if a.status.value == status]
        
        return {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "role": a.role.value,
                    "status": a.status.value,
                    "current_load": a.current_load,
                    "max_load": a.max_load,
                    "capabilities": a.capabilities,
                }
                for a in agents
            ],
            "total": len(agents),
        }
    
    async def _get_agent_info(self, agent_id: str) -> dict[str, Any]:
        """Get detailed agent information."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent.agent_id,
            "role": agent.role.value,
            "status": agent.status.value,
            "capabilities": agent.capabilities,
            "current_load": agent.current_load,
            "max_load": agent.max_load,
            "is_available": agent.is_available,
            "metadata": agent.metadata,
        }
    
    async def _send_task(
        self,
        agent_id: str,
        task_type: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a task to an agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        # Create task request
        task = TaskRequest(
            task_type=task_type,
            description=description,
            parameters=parameters or {},
        )
        
        # In a real implementation, this would use A2A to send the task
        # For now, return task details
        return {
            "task_id": str(task.task_id),
            "agent_id": agent_id,
            "task_type": task_type,
            "status": "submitted",
            "message": f"Task submitted to agent {agent_id}",
        }
    
    async def _get_agent_capabilities(self, agent_id: str) -> dict[str, Any]:
        """Get agent capabilities."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent.agent_id,
            "capabilities": agent.capabilities,
            "role": agent.role.value,
        }
    
    async def _find_best_agent(
        self,
        task_type: str,
        required_capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find the best agent for a task."""
        candidates = []
        
        for agent in self.agents.values():
            # Check if agent is available
            if not agent.is_available:
                continue
            
            # Check required capabilities
            if required_capabilities:
                has_capabilities = all(
                    cap in agent.capabilities
                    for cap in required_capabilities
                )
                if not has_capabilities:
                    continue
            
            # Calculate score based on load
            score = 1.0 - (agent.current_load / agent.max_load)
            
            candidates.append({
                "agent": agent,
                "score": score,
            })
        
        if not candidates:
            return {
                "found": False,
                "message": "No suitable agent found",
            }
        
        # Select best candidate
        best = max(candidates, key=lambda x: x["score"])
        agent = best["agent"]
        
        return {
            "found": True,
            "agent_id": agent.agent_id,
            "role": agent.role.value,
            "capabilities": agent.capabilities,
            "current_load": agent.current_load,
            "score": best["score"],
        }


async def main() -> None:
    """Run the agent management MCP server."""
    import os
    from common import configure_logging
    
    configure_logging()
    
    # Get configuration from environment
    transport = os.getenv("MCP_TRANSPORT", "http")  # Default to HTTP in Docker
    port = int(os.getenv("MCP_PORT", "8081"))
    
    server = AgentManagementMCPServer()
    await server.run(transport=transport, port=port)


if __name__ == "__main__":
    asyncio.run(main())