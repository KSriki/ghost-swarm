"""
Concurrency utilities for Ghost Swarm.

Provides optimized async runtime with uvloop and executor pools for
CPU-bound and I/O-bound operations in MCP servers and agents.
"""

import asyncio
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def install_uvloop() -> None:
    """
    Install uvloop as the default asyncio event loop.
    
    uvloop is 2-4x faster than the default asyncio event loop.
    Safe to call multiple times - only installs once.
    """
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("uvloop_installed", version=uvloop.__version__)
    except ImportError:
        logger.warning("uvloop_not_available", message="Falling back to standard asyncio")


class ExecutorPool:
    """
    Manages process and thread pools for concurrent execution.
    
    - ProcessPoolExecutor: For CPU-bound tasks (embeddings, data processing)
    - ThreadPoolExecutor: For I/O-bound tasks (API calls, database queries)
    """
    
    def __init__(
        self,
        max_process_workers: Optional[int] = None,
        max_thread_workers: Optional[int] = None,
    ):
        """
        Initialize executor pools.
        
        Args:
            max_process_workers: Number of processes (default: CPU count)
            max_thread_workers: Number of threads (default: CPU count * 5)
        """
        # Default sizing
        cpu_count = os.cpu_count() or 4
        self.max_process_workers = max_process_workers or cpu_count
        self.max_thread_workers = max_thread_workers or (cpu_count * 5)
        
        # Initialize pools lazily
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        
        logger.info(
            "executor_pool_configured",
            process_workers=self.max_process_workers,
            thread_workers=self.max_thread_workers,
        )
    
    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.max_process_workers
            )
            logger.debug("process_pool_created", workers=self.max_process_workers)
        return self._process_pool
    
    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.max_thread_workers,
                thread_name_prefix="ghost-io-",
            )
            logger.debug("thread_pool_created", workers=self.max_thread_workers)
        return self._thread_pool
    
    async def run_in_process(
        self, 
        func: Callable[..., T], 
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute CPU-bound function in process pool.
        
        Use for:
        - Embedding generation
        - Data processing/transformation
        - Complex computations
        - Anything that would block the GIL
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from function execution
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_pool,
            lambda: func(*args, **kwargs),
        )
    
    async def run_in_thread(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute I/O-bound function in thread pool.
        
        Use for:
        - Synchronous API calls (Claude, OpenAI)
        - Database queries (if using sync driver)
        - File I/O operations
        - Any blocking I/O
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from function execution
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs),
        )
    
    async def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown all executor pools.
        
        Args:
            wait: Wait for pending tasks to complete
        """
        if self._process_pool:
            self._process_pool.shutdown(wait=wait)
            logger.info("process_pool_shutdown")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=wait)
            logger.info("thread_pool_shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Sync shutdown
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)


# Global executor pool instance
_global_executor_pool: Optional[ExecutorPool] = None


def get_executor_pool() -> ExecutorPool:
    """
    Get global executor pool singleton.
    
    Creates pool on first access with sensible defaults.
    """
    global _global_executor_pool
    
    if _global_executor_pool is None:
        # Get config from environment
        max_processes = os.getenv("MAX_PROCESS_WORKERS")
        max_threads = os.getenv("MAX_THREAD_WORKERS")
        
        _global_executor_pool = ExecutorPool(
            max_process_workers=int(max_processes) if max_processes else None,
            max_thread_workers=int(max_threads) if max_threads else None,
        )
    
    return _global_executor_pool


async def run_cpu_bound(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Convenience function for CPU-bound operations.
    
    Example:
        result = await run_cpu_bound(compute_embeddings, text_batch)
    """
    pool = get_executor_pool()
    return await pool.run_in_process(func, *args, **kwargs)


async def run_io_bound(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Convenience function for I/O-bound operations.
    
    Example:
        response = await run_io_bound(requests.post, url, json=data)
    """
    pool = get_executor_pool()
    return await pool.run_in_thread(func, *args, **kwargs)


async def cleanup_executors() -> None:
    """Cleanup global executor pools."""
    global _global_executor_pool
    
    if _global_executor_pool:
        await _global_executor_pool.shutdown()
        _global_executor_pool = None


# Install uvloop when module is imported
if sys.platform != "win32":  # uvloop doesn't support Windows
    install_uvloop()