"""
Examples of using uvloop and executors in Ghost Swarm MCP servers and agents.

Demonstrates:
1. CPU-bound MCP tools (embeddings, data processing)
2. I/O-bound MCP tools (API calls)
3. Parallel agent execution
4. Best practices for concurrency
"""

import asyncio
from typing import Any, Dict, List
import time

import structlog
from anthropic import Anthropic

# Import concurrency utilities
from common.concurrency import (
    ExecutorPool,
    get_executor_pool,
    run_cpu_bound,
    run_io_bound,
)
from mcp_server.base import BaseMCPServer

logger = structlog.get_logger(__name__)


# ============================================================================
# Example 1: CPU-Intensive MCP Server (RAG/Embeddings)
# ============================================================================

class RAGMCPServer(BaseMCPServer):
    """
    MCP server for RAG operations with CPU-intensive embedding generation.
    """
    
    async def setup(self) -> None:
        """Register RAG tools."""
        
        # Register CPU-intensive embedding tool
        self.register_tool(
            name="generate_embeddings",
            description="Generate embeddings for text (CPU-intensive)",
            parameters={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to embed",
                    },
                },
                "required": ["texts"],
            },
            handler=self._generate_embeddings,
            is_cpu_intensive=True,  # Automatically uses process pool!
        )
        
        # Register regular async tool
        self.register_tool(
            name="search_vectors",
            description="Search vector database (I/O-bound)",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            handler=self._search_vectors,
            is_cpu_intensive=False,  # Regular async execution
        )
    
    async def _generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings (CPU-intensive).
        
        This will automatically run in process pool due to is_cpu_intensive=True.
        """
        # Simulate CPU-intensive embedding generation
        embeddings = []
        for text in texts:
            # In real implementation: use sentence-transformers or similar
            embedding = [hash(text) % 1000 / 1000.0 for _ in range(384)]
            embeddings.append(embedding)
            time.sleep(0.1)  # Simulate computation
        
        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
        }
    
    async def _search_vectors(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search vector database (I/O-bound).
        
        Runs as regular async operation.
        """
        # In real implementation: query ChromaDB, Pinecone, etc.
        await asyncio.sleep(0.05)  # Simulate network I/O
        
        return {
            "query": query,
            "results": [
                {"id": i, "score": 0.9 - (i * 0.1), "text": f"Result {i}"}
                for i in range(top_k)
            ],
        }


# ============================================================================
# Example 2: Agent with Parallel Claude API Calls
# ============================================================================

class ParallelReasoningAgent:
    """
    Agent that makes multiple Claude API calls in parallel using thread pool.
    """
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.executor_pool = get_executor_pool()
    
    def _sync_claude_call(self, prompt: str) -> str:
        """
        Synchronous Claude API call.
        
        This will be executed in thread pool by run_in_thread().
        """
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    async def parallel_reasoning(self, prompts: List[str]) -> List[str]:
        """
        Execute multiple Claude calls in parallel using thread pool.
        
        Perfect for I/O-bound API calls - thread pool prevents blocking.
        """
        logger.info("parallel_reasoning_start", prompt_count=len(prompts))
        
        # Execute all Claude calls concurrently in thread pool
        tasks = [
            self.executor_pool.run_in_thread(self._sync_claude_call, prompt)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks)
        
        logger.info("parallel_reasoning_complete", results_count=len(results))
        return results
    
    async def multi_perspective_analysis(self, topic: str) -> Dict[str, str]:
        """
        Analyze a topic from multiple perspectives in parallel.
        """
        prompts = [
            f"Analyze {topic} from a technical perspective.",
            f"Analyze {topic} from a business perspective.",
            f"Analyze {topic} from a user perspective.",
            f"Analyze {topic} from a security perspective.",
        ]
        
        perspectives = ["technical", "business", "user", "security"]
        results = await self.parallel_reasoning(prompts)
        
        return dict(zip(perspectives, results))


# ============================================================================
# Example 3: Hybrid MCP Server (CPU + I/O)
# ============================================================================

class DataProcessingMCPServer(BaseMCPServer):
    """
    MCP server with both CPU and I/O intensive operations.
    """
    
    async def setup(self) -> None:
        """Register mixed workload tools."""
        
        # CPU-intensive: data transformation
        self.register_tool(
            name="transform_data",
            description="Transform large dataset (CPU-intensive)",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "operation": {"type": "string"},
                },
                "required": ["data", "operation"],
            },
            handler=self._transform_data,
            is_cpu_intensive=True,
        )
        
        # I/O-intensive: fetch and process
        self.register_tool(
            name="fetch_and_analyze",
            description="Fetch data from API and analyze",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
            handler=self._fetch_and_analyze,
            is_cpu_intensive=False,
        )
    
    async def _transform_data(self, data: List[Any], operation: str) -> Dict[str, Any]:
        """CPU-intensive transformation."""
        # This runs in process pool automatically
        transformed = []
        for item in data:
            # Simulate heavy computation
            result = sum(hash(str(item)) % 1000 for _ in range(1000))
            transformed.append(result)
        
        return {
            "original_count": len(data),
            "transformed_count": len(transformed),
            "operation": operation,
        }
    
    async def _fetch_and_analyze(self, url: str) -> Dict[str, Any]:
        """
        I/O-intensive fetch followed by CPU work.
        
        Demonstrates mixing I/O and CPU operations.
        """
        # Fetch data (I/O-bound) - runs in thread pool
        data = await self.run_in_thread(self._fetch_data, url)
        
        # Analyze data (CPU-bound) - runs in process pool
        analysis = await self.run_in_process(self._analyze_data, data)
        
        return {
            "url": url,
            "data_size": len(data),
            "analysis": analysis,
        }
    
    def _fetch_data(self, url: str) -> str:
        """Sync I/O operation."""
        time.sleep(0.1)  # Simulate network delay
        return f"Data from {url}"
    
    def _analyze_data(self, data: str) -> Dict[str, int]:
        """CPU-intensive analysis."""
        time.sleep(0.2)  # Simulate heavy computation
        return {
            "word_count": len(data.split()),
            "char_count": len(data),
        }


# ============================================================================
# Example 4: Orchestrator with Parallel Agent Dispatch
# ============================================================================

class ConcurrentOrchestrator:
    """
    Orchestrator that dispatches tasks to multiple agents in parallel.
    """
    
    def __init__(self):
        self.executor_pool = get_executor_pool()
        self.agents = {}  # agent_id -> agent instance
    
    async def dispatch_to_agents_parallel(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Dispatch multiple tasks to agents in parallel.
        
        Each agent processes its task concurrently.
        """
        logger.info("dispatching_tasks", count=len(tasks))
        
        # Create tasks for parallel execution
        agent_tasks = [
            self._dispatch_single_task(task)
            for task in tasks
        ]
        
        # Execute all in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [
            r for r in results if not isinstance(r, Exception)
        ]
        
        logger.info(
            "dispatch_complete",
            total=len(tasks),
            successful=len(successful_results),
        )
        
        return successful_results
    
    async def _dispatch_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task."""
        # Simulate agent processing
        await asyncio.sleep(0.1)
        return {"task_id": task.get("id"), "status": "completed"}


# ============================================================================
# Example 5: Running Everything Together
# ============================================================================

async def main():
    """Demonstrate all concurrency patterns."""
    
    logger.info("=== Ghost Swarm Concurrency Examples ===")
    
    # 1. RAG Server with CPU-intensive operations
    logger.info("\n1. RAG MCP Server with CPU-intensive embeddings")
    rag_server = RAGMCPServer("rag-server")
    await rag_server.setup()
    
    # This will run in process pool automatically
    embeddings_result = await rag_server.tools["generate_embeddings"](
        texts=["Hello world", "Goodbye world", "Testing embeddings"]
    )
    logger.info("embeddings_generated", result=embeddings_result)
    
    # 2. Parallel Claude API calls (requires API key)
    # Uncomment if you want to test with real API
    # logger.info("\n2. Parallel Claude API calls")
    # agent = ParallelReasoningAgent(api_key="your-api-key")
    # analysis = await agent.multi_perspective_analysis("AI Safety")
    # logger.info("multi_perspective_complete", perspectives=list(analysis.keys()))
    
    # 3. Data processing server (mixed workload)
    logger.info("\n3. Data Processing MCP Server")
    data_server = DataProcessingMCPServer("data-server")
    await data_server.setup()
    
    # CPU-intensive transform
    transform_result = await data_server.tools["transform_data"](
        data=[1, 2, 3, 4, 5],
        operation="hash_sum",
    )
    logger.info("transform_complete", result=transform_result)
    
    # 4. Parallel orchestration
    logger.info("\n4. Parallel Task Orchestration")
    orchestrator = ConcurrentOrchestrator()
    
    tasks = [{"id": i, "type": "analysis"} for i in range(5)]
    results = await orchestrator.dispatch_to_agents_parallel(tasks)
    logger.info("orchestration_complete", results_count=len(results))
    
    logger.info("\n=== All Examples Complete ===")


if __name__ == "__main__":
    # uvloop is already installed by importing common.concurrency
    asyncio.run(main())