"""
Agent2Agent (A2A) Communication Framework.

Based on Google's Agent2Agent protocol for inter-agent communication.
Implements a WebSocket-based RPC-like protocol with idempotency guarantees.
"""

import asyncio
import json
from typing import Any, Callable, Coroutine
from uuid import UUID

import structlog
import websockets
from websockets.server import WebSocketServerProtocol

from common.config.settings import get_settings
from common.models.messages import Message, MessageType

logger = structlog.get_logger(__name__)


class A2AProtocol:
    """
    Agent2Agent protocol handler.

    Implements idempotent message handling and routing between agents.
    """

    def __init__(self) -> None:
        """Initialize the A2A protocol handler."""
        self.handlers: dict[MessageType, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self.processed_messages: set[UUID] = set()
        self.max_processed_history: int = 10000

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.handlers[message_type] = handler
        logger.info("registered_handler", message_type=message_type)

    async def handle_message(self, raw_message: str) -> dict[str, Any]:
        """
        Handle an incoming message with idempotency guarantee.

        Args:
            raw_message: Raw JSON message string

        Returns:
            Response dictionary
        """
        try:
            message_data = json.loads(raw_message)
            message = Message[dict[str, Any]](**message_data)

            # Idempotency check
            if message.id in self.processed_messages:
                logger.warning(
                    "duplicate_message_ignored",
                    message_id=str(message.id),
                    sender=message.sender_id,
                )
                return {
                    "status": "duplicate",
                    "message_id": str(message.id),
                    "error": "Message already processed",
                }

            # Mark as processed
            self.processed_messages.add(message.id)
            if len(self.processed_messages) > self.max_processed_history:
                # Remove oldest entries to prevent memory growth
                oldest = list(self.processed_messages)[: len(self.processed_messages) // 2]
                self.processed_messages -= set(oldest)

            # Route to appropriate handler
            handler = self.handlers.get(message.type)
            if handler is None:
                logger.error("no_handler_found", message_type=message.type)
                return {
                    "status": "error",
                    "message_id": str(message.id),
                    "error": f"No handler for message type: {message.type}",
                }

            # Execute handler
            result = await handler(message)
            logger.info(
                "message_processed",
                message_id=str(message.id),
                message_type=message.type,
                sender=message.sender_id,
            )

            return {
                "status": "success",
                "message_id": str(message.id),
                "result": result,
            }

        except json.JSONDecodeError as e:
            logger.error("invalid_json", error=str(e))
            return {"status": "error", "error": f"Invalid JSON: {e}"}
        except Exception as e:
            logger.error("message_handling_error", error=str(e), exc_info=True)
            return {"status": "error", "error": str(e)}


class A2AServer:
    """WebSocket server for Agent2Agent communication."""

    def __init__(self) -> None:
        """Initialize the A2A server."""
        self.settings = get_settings()
        self.protocol = A2AProtocol()
        self.clients: set[WebSocketServerProtocol] = set()
        self.running: bool = False

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """
        Handle a connected client.

        Args:
            websocket: Connected WebSocket client
        """
        self.clients.add(websocket)
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info("client_connected", client_id=client_id, total_clients=len(self.clients))

        try:
            async for message in websocket:
                if isinstance(message, str):
                    response = await self.protocol.handle_message(message)
                    await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            logger.info("client_disconnected", client_id=client_id)
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, message: Message[Any]) -> None:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
        """
        if not self.clients:
            logger.warning("no_clients_for_broadcast")
            return

        message_json = message.model_dump_json()
        await asyncio.gather(
            *[client.send(message_json) for client in self.clients],
            return_exceptions=True,
        )

    async def start(self) -> None:
        """Start the A2A server."""
        self.running = True
        logger.info(
            "starting_a2a_server",
            host=self.settings.a2a_host,
            port=self.settings.a2a_port,
        )

        async with websockets.serve(
            self.handle_client,
            self.settings.a2a_host,
            self.settings.a2a_port,
        ):
            await asyncio.Future()  # Run forever

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register a message handler.

        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.protocol.register_handler(message_type, handler)


class A2AClient:
    """WebSocket client for Agent2Agent communication."""

    def __init__(self, agent_id: str) -> None:
        """
        Initialize the A2A client.

        Args:
            agent_id: Unique identifier for this agent
        """
        self.agent_id = agent_id
        self.settings = get_settings()
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.connected: bool = False

    async def connect(self) -> None:
        """Connect to the A2A server."""
        try:
            self.websocket = await websockets.connect(self.settings.a2a_url)
            self.connected = True
            logger.info("connected_to_a2a_server", agent_id=self.agent_id)
        except Exception as e:
            logger.error("connection_failed", agent_id=self.agent_id, error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect from the A2A server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("disconnected_from_a2a_server", agent_id=self.agent_id)

    async def send_message(self, message: Message[Any]) -> dict[str, Any]:
        """
        Send a message to another agent.

        Args:
            message: Message to send

        Returns:
            Response from the server
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to A2A server")

        message_json = message.model_dump_json()
        await self.websocket.send(message_json)

        response_raw = await self.websocket.recv()
        if isinstance(response_raw, str):
            return json.loads(response_raw)
        else:
            raise ValueError("Expected string response from server")

    async def receive_messages(
        self,
        handler: Callable[[Message[Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Continuously receive and handle messages.

        Args:
            handler: Async function to handle received messages
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to A2A server")

        try:
            async for message_raw in self.websocket:
                if isinstance(message_raw, str):
                    message_data = json.loads(message_raw)
                    message = Message[dict[str, Any]](**message_data)
                    await handler(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("connection_closed", agent_id=self.agent_id)
            self.connected = False