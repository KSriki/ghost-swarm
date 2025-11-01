"""Ghost Swarm - AI Agent System."""

import asyncio
import sys

from common import configure_logging, get_logger

logger = get_logger(__name__)


async def main() -> None:
    """Main entry point for Ghost Swarm."""
    configure_logging()

    logger.info("ghost_swarm_starting", version="0.1.0")

    print("""
    👻🐝 Ghost Swarm - AI Agent System
    ═══════════════════════════════════
    
    Usage:
        Orchestrator:  python -m ghosts.orchestrator.orchestrator
        Worker:        python -m ghosts.worker.worker
        
    For more information, see README.md
    """)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown_requested")
        sys.exit(0)