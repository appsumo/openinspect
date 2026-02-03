"""
Agent bridge - bidirectional communication between sandbox and control plane.

This module handles:
- WebSocket connection to control plane Durable Object
- Heartbeat loop for connection health
- ACP (Agent Client Protocol) communication with Claude Code
- Event forwarding from ACP agent to control plane
- Command handling from control plane (prompt, stop, snapshot)
- Git identity configuration per prompt author
"""

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import websockets

# ACP SDK - handles JSON-RPC over stdio
from acp import (
    Client as ACPClientBase,
)
from acp import (
    RequestError,
    spawn_agent_process,
    text_block,
)
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCall,
    ToolCallStart,
    ToolCallUpdate,
)
from websockets import ClientConnection, State
from websockets.exceptions import InvalidStatus

from .log_config import configure_logging, get_logger
from .types import GitUser

if TYPE_CHECKING:
    from acp.core import ClientSideConnection

configure_logging()


class TokenResolution(NamedTuple):
    """Result of GitHub token resolution."""

    token: str
    source: str


class SessionTerminatedError(Exception):
    """Raised when the control plane has terminated the session (HTTP 410).

    This is a non-recoverable error - the bridge should exit gracefully
    rather than retry. The session can be restored via user action (sending
    a new prompt), which will trigger snapshot restoration on the control plane.
    """

    pass


class ACPClient(ACPClientBase):
    """
    ACP Client implementation for sandbox environment.

    Handles permission requests, file operations, and terminal management
    for the Claude Code agent running in the sandbox.
    """

    def __init__(self, bridge: "AgentBridge"):
        self.bridge = bridge
        self.log = bridge.log

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Auto-approve all permissions in sandbox environment.

        The sandbox runs in an isolated container with no access to user
        systems, so all permissions can be safely granted.
        """
        # Find an "allow" option to select
        allow_option = None
        for opt in options:
            if opt.kind in ("allow_once", "allow_always"):
                allow_option = opt
                break

        if allow_option:
            self.log.debug(
                "acp.permission_auto_approve",
                option_id=allow_option.option_id,
                option_kind=allow_option.kind,
                tool=tool_call.title if tool_call else None,
            )
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=allow_option.option_id)
            )
        else:
            # Fallback: if no allow option found, use first option
            first_option = options[0] if options else None
            if first_option:
                self.log.debug(
                    "acp.permission_fallback",
                    option_id=first_option.option_id,
                    option_kind=first_option.kind,
                )
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=first_option.option_id)
                )
            # No options at all - this shouldn't happen
            self.log.warn("acp.permission_no_options")
            raise ValueError("No permission options provided")

    async def write_text_file(
        self,
        session_id: str,
        path: str,
        text: str,
        **kwargs: Any,
    ) -> None:
        """Write a text file to the filesystem."""
        self.log.debug("acp.write_file", path=path)
        Path(path).write_text(text)

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        **kwargs: Any,
    ) -> str:
        """Read a text file from the filesystem."""
        self.log.debug("acp.read_file", path=path)
        return Path(path).read_text()

    async def create_terminal(
        self,
        session_id: str,
        command: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a terminal - not implemented in sandbox context."""
        raise RequestError.method_not_found("create_terminal not implemented")

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Send terminal output - not implemented in sandbox context."""
        raise RequestError.method_not_found("terminal_output not implemented")

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> int:
        """Wait for terminal exit - not implemented in sandbox context."""
        raise RequestError.method_not_found("wait_for_terminal_exit not implemented")

    async def kill_terminal_command(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> None:
        """Kill terminal command - not implemented in sandbox context."""
        raise RequestError.method_not_found("kill_terminal_command not implemented")

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> None:
        """Release terminal - not implemented in sandbox context."""
        raise RequestError.method_not_found("release_terminal not implemented")

    async def session_update(
        self,
        session_id: str,
        update: Any,
        source: str | None = None,
    ) -> None:
        """Handle session updates from the agent.

        This is the main callback for receiving streaming events from Claude Code.
        We forward these to the control plane as appropriate event types.
        """
        await self.bridge._handle_acp_session_update(session_id, update, source)


class AgentBridge:
    """
    Bridge between sandbox ACP agent and control plane.

    Handles:
    - WebSocket connection management with reconnection
    - Heartbeat for connection health
    - ACP subprocess management and communication
    - Event streaming from ACP to control plane
    - Command handling (prompt, stop, snapshot, shutdown)
    - Git identity management per prompt author
    """

    HEARTBEAT_INTERVAL = 30.0
    RECONNECT_BACKOFF_BASE = 2.0
    RECONNECT_MAX_DELAY = 60.0
    PROMPT_MAX_DURATION = 5400.0

    def __init__(
        self,
        sandbox_id: str,
        session_id: str,
        control_plane_url: str,
        auth_token: str,
    ):
        self.sandbox_id = sandbox_id
        self.session_id = session_id
        self.control_plane_url = control_plane_url
        self.auth_token = auth_token

        # Logger
        self.log = get_logger(
            "bridge",
            service="sandbox",
            sandbox_id=sandbox_id,
            session_id=session_id,
        )

        self.ws: ClientConnection | None = None
        self.shutdown_event = asyncio.Event()
        self.git_sync_complete = asyncio.Event()

        # ACP state
        self.acp_conn: ClientSideConnection | None = None
        self.acp_process: asyncio.subprocess.Process | None = None
        self.acp_session_id: str | None = None
        self._acp_context: Any | None = None  # Async context manager for spawn_agent_process
        self.acp_session_id_file = Path(tempfile.gettempdir()) / "acp-session-id"
        self.repo_path = Path("/workspace")

        # Current prompt tracking
        self._current_message_id: str | None = None
        self._prompt_task: asyncio.Task[None] | None = None
        # Cache tool titles by tool_call_id (ToolCallUpdate loses the title)
        self._tool_titles: dict[str, str] = {}

    @property
    def ws_url(self) -> str:
        """WebSocket URL for control plane connection."""
        url = self.control_plane_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{url}/sessions/{self.session_id}/ws?type=sandbox"

    async def run(self) -> None:
        """Main bridge loop with reconnection handling.

        Handles reconnection for transient errors (network issues, etc.) but
        exits gracefully for terminal errors like HTTP 410 (session terminated).
        """
        self.log.info("bridge.run_start")

        await self._load_session_id()
        reconnect_attempts = 0

        try:
            while not self.shutdown_event.is_set():
                try:
                    await self._connect_and_run()
                    reconnect_attempts = 0
                except SessionTerminatedError as e:
                    # Non-recoverable: session has been terminated by control plane
                    self.log.info(
                        "bridge.disconnect",
                        reason="session_terminated",
                        detail=str(e),
                    )
                    self.shutdown_event.set()
                    break
                except websockets.ConnectionClosed as e:
                    self.log.warn(
                        "bridge.disconnect",
                        reason="connection_closed",
                        ws_close_code=e.code,
                    )
                except Exception as e:
                    error_str = str(e)
                    # Check for fatal HTTP errors that shouldn't trigger retry
                    if self._is_fatal_connection_error(error_str):
                        self.log.error(
                            "bridge.disconnect",
                            reason="fatal_error",
                            exc=e,
                        )
                        self.shutdown_event.set()
                        break
                    self.log.warn(
                        "bridge.disconnect",
                        reason="connection_error",
                        detail=error_str,
                    )

                if self.shutdown_event.is_set():
                    break

                reconnect_attempts += 1
                delay = min(
                    self.RECONNECT_BACKOFF_BASE**reconnect_attempts,
                    self.RECONNECT_MAX_DELAY,
                )
                self.log.info(
                    "bridge.reconnect",
                    attempt=reconnect_attempts,
                    delay_s=round(delay, 1),
                )
                await asyncio.sleep(delay)

        finally:
            await self._cleanup_acp()

    def _is_fatal_connection_error(self, error_str: str) -> bool:
        """Check if a connection error is fatal and shouldn't trigger retry.

        Fatal errors indicate the session is invalid or terminated, not a
        transient network issue. These include:
        - HTTP 401 (Unauthorized): Auth token invalid or expired
        - HTTP 403 (Forbidden): Access denied
        - HTTP 404 (Not Found): Session doesn't exist
        - HTTP 410 (Gone): Session terminated, sandbox stopped/stale

        For these errors, retrying is futile - the bridge should exit and
        allow the control plane to spawn a new sandbox if needed.
        """
        fatal_patterns = [
            "HTTP 401",  # Unauthorized
            "HTTP 403",  # Forbidden
            "HTTP 404",  # Session not found
            "HTTP 410",  # Session terminated (stopped/stale)
        ]
        return any(pattern in error_str for pattern in fatal_patterns)

    async def _connect_and_run(self) -> None:
        """Connect to control plane and handle messages.

        Raises:
            SessionTerminatedError: If the control plane rejects the connection
                with HTTP 410 (session stopped/stale).
        """
        additional_headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "X-Sandbox-ID": self.sandbox_id,
        }

        try:
            async with websockets.connect(
                self.ws_url,
                additional_headers=additional_headers,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                self.ws = ws
                self.log.info("bridge.connect", outcome="success")

                await self._send_event(
                    {
                        "type": "ready",
                        "sandboxId": self.sandbox_id,
                        "acpSessionId": self.acp_session_id,
                    }
                )

                heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                background_tasks: set[asyncio.Task[None]] = set()

                try:
                    async for message in ws:
                        if self.shutdown_event.is_set():
                            break

                        try:
                            cmd = json.loads(message)
                            task = await self._handle_command(cmd)
                            if task:
                                background_tasks.add(task)
                                task.add_done_callback(background_tasks.discard)
                        except json.JSONDecodeError as e:
                            self.log.warn("bridge.invalid_message", exc=e)
                        except Exception as e:
                            self.log.error("bridge.command_error", exc=e)

                finally:
                    heartbeat_task.cancel()
                    for task in background_tasks:
                        task.cancel()
                    self.ws = None

        except InvalidStatus as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403, 404, 410):
                raise SessionTerminatedError(
                    f"Session rejected by control plane (HTTP {status})."
                ) from e
            raise

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat events."""
        while not self.shutdown_event.is_set():
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

            if self.ws and self.ws.state == State.OPEN:
                await self._send_event(
                    {
                        "type": "heartbeat",
                        "sandboxId": self.sandbox_id,
                        "status": "ready",
                        "timestamp": time.time(),
                    }
                )

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Send event to control plane."""
        event_type = event.get("type", "unknown")

        if not self.ws:
            self.log.debug("bridge.send_failed", event_type=event_type, reason="ws_none")
            return
        if self.ws.state != State.OPEN:
            self.log.debug(
                "bridge.send_failed",
                event_type=event_type,
                reason=f"ws_state_{self.ws.state}",
            )
            return

        event["sandboxId"] = self.sandbox_id
        event["timestamp"] = event.get("timestamp", time.time())

        try:
            await self.ws.send(json.dumps(event))
        except Exception as e:
            self.log.error("bridge.send_error", event_type=event_type, exc=e)

    async def _handle_command(self, cmd: dict[str, Any]) -> asyncio.Task[None] | None:
        """Handle command from control plane.

        Long-running commands (like prompt) are run as background tasks to keep
        the WebSocket listener responsive to other commands (like push).

        Returns a Task for long-running commands, None for immediate commands.
        """
        cmd_type = cmd.get("type")
        self.log.debug("bridge.command_received", cmd_type=cmd_type)

        if cmd_type == "prompt":
            message_id = cmd.get("messageId") or cmd.get("message_id", "unknown")
            task = asyncio.create_task(self._handle_prompt(cmd))
            self._prompt_task = task

            def handle_task_exception(t: asyncio.Task[None], mid: str = message_id) -> None:
                if t.cancelled():
                    asyncio.create_task(
                        self._send_event(
                            {
                                "type": "execution_complete",
                                "messageId": mid,
                                "success": False,
                                "error": "Task was cancelled",
                            }
                        )
                    )
                elif exc := t.exception():
                    asyncio.create_task(
                        self._send_event(
                            {
                                "type": "execution_complete",
                                "messageId": mid,
                                "success": False,
                                "error": str(exc),
                            }
                        )
                    )

            task.add_done_callback(handle_task_exception)
            return task
        elif cmd_type == "stop":
            await self._handle_stop()
        elif cmd_type == "snapshot":
            await self._handle_snapshot()
        elif cmd_type == "shutdown":
            await self._handle_shutdown()
        elif cmd_type == "git_sync_complete":
            self.git_sync_complete.set()
        elif cmd_type == "push":
            await self._handle_push(cmd)
        else:
            self.log.debug("bridge.unknown_command", cmd_type=cmd_type)
        return None

    async def _ensure_acp_agent(self) -> None:
        """Ensure ACP agent subprocess is running and connected.

        Uses spawn_agent_process from the ACP SDK which handles proper
        stdio stream setup for JSON-RPC communication.
        """
        if self.acp_conn is not None and self.acp_process is not None:
            # Check if process is still alive
            if self.acp_process.returncode is None:
                return
            # Process died, clean up
            self.log.warn("acp.process_died", returncode=self.acp_process.returncode)
            await self._cleanup_acp()

        self.log.info("acp.spawn_start")

        # Determine working directory
        workdir = self.repo_path
        repo_dirs = list(self.repo_path.glob("*/.git"))
        if repo_dirs:
            workdir = repo_dirs[0].parent

        # Build environment for Claude Code ACP
        # OAuth token takes precedence over API key (uses user's subscription, much cheaper)
        env = {**os.environ}
        oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if oauth_token:
            self.log.info("acp.auth_method", method="oauth_token")
            env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
            env.pop("ANTHROPIC_API_KEY", None)  # Don't use API key when OAuth available
        else:
            self.log.info("acp.auth_method", method="api_key")

        # Create ACP client for handling callbacks
        client = ACPClient(self)

        # ACP command is configurable - defaults to "claude-code-acp" but can be
        # set to "opencode acp" or other ACP-compatible agents via ACP_COMMAND env var
        # Supports space-separated command + args (e.g., "opencode acp")
        acp_command_str = os.environ.get("ACP_COMMAND", "claude-code-acp")
        acp_parts = acp_command_str.split()
        acp_command = acp_parts[0]
        acp_args = acp_parts[1:]
        self.log.info("acp.command", command=acp_command, args=acp_args)

        # Use spawn_agent_process from ACP SDK - it's an async context manager
        # We manually manage the context to keep the connection alive
        self._acp_context = spawn_agent_process(
            client,
            acp_command,
            *acp_args,
            env=env,
            cwd=workdir,
        )

        # Enter the context manually (it's wrapped in @asynccontextmanager)
        self.acp_conn, self.acp_process = await self._acp_context.__aenter__()

        self.log.info("acp.spawn_complete", pid=self.acp_process.pid)

        # Initialize connection
        init_response = await self.acp_conn.initialize(
            protocol_version=1,
            capabilities={},
        )
        self.log.info(
            "acp.initialized",
            protocol_version=init_response.protocol_version,
        )

    async def _ensure_acp_session(self) -> str:
        """Ensure ACP session exists and return session ID."""
        await self._ensure_acp_agent()

        if self.acp_session_id:
            return self.acp_session_id

        if self.acp_conn is None:
            raise RuntimeError("ACP connection not initialized")

        # Determine working directory
        workdir = self.repo_path
        repo_dirs = list(self.repo_path.glob("*/.git"))
        if repo_dirs:
            workdir = repo_dirs[0].parent

        # Create new session
        response: NewSessionResponse = await self.acp_conn.new_session(
            cwd=str(workdir),
            mcp_servers=[],
        )

        session_id = response.session_id
        if not session_id:
            raise RuntimeError("ACP session creation failed - no session ID returned")

        self.acp_session_id = session_id
        self.log.info("acp.session_created", acp_session_id=self.acp_session_id)

        await self._save_session_id()
        return session_id

    async def _cleanup_acp(self) -> None:
        """Clean up ACP connection and process.

        Properly exits the spawn_agent_process context manager to ensure
        clean shutdown of the subprocess and connection.
        """
        # Exit the async context manager properly
        if self._acp_context is not None:
            try:
                # Exit the context manager (it's wrapped in @asynccontextmanager)
                await self._acp_context.__aexit__(None, None, None)
            except Exception as e:
                self.log.debug("acp.context_close_error", exc=e)
            self._acp_context = None

        # The context manager should have cleaned up, but ensure conn/process are None
        self.acp_conn = None
        self.acp_process = None

    async def _handle_prompt(self, cmd: dict[str, Any]) -> None:
        """Handle prompt command - send to ACP agent and stream response."""
        message_id = cmd.get("messageId") or cmd.get("message_id", "unknown")
        content = cmd.get("content", "")
        model = cmd.get("model")
        author_data = cmd.get("author", {})
        start_time = time.time()
        outcome = "success"

        self.log.info(
            "prompt.start",
            message_id=message_id,
            model=model,
        )

        # Set current message ID for event forwarding
        self._current_message_id = message_id
        self._tool_titles = {}  # Reset tool title cache for new prompt

        # Configure git identity if provided
        github_name = author_data.get("githubName")
        github_email = author_data.get("githubEmail")
        if github_name and github_email:
            await self._configure_git_identity(
                GitUser(
                    name=github_name,
                    email=github_email,
                )
            )

        try:
            # Ensure ACP agent is running and session exists
            session_id = await self._ensure_acp_session()

            if self.acp_conn is None:
                raise RuntimeError("ACP connection not initialized")

            # Set model if specified (optional - not all ACP agents support this)
            if model:
                try:
                    self.log.debug("acp.set_model", model=model)
                    await self.acp_conn.set_session_model(
                        session_id=session_id,
                        model_id=model,
                    )
                except Exception as e:
                    # Model setting not supported by this agent - continue without it
                    self.log.debug("acp.set_model_unsupported", model=model, error=str(e))

            # Send prompt via ACP
            response: PromptResponse = await self.acp_conn.prompt(
                session_id=session_id,
                prompt=[text_block(content)],
            )

            self.log.info(
                "prompt.complete",
                stop_reason=response.stop_reason,
            )

            await self._send_event(
                {
                    "type": "execution_complete",
                    "messageId": message_id,
                    "success": True,
                }
            )

        except Exception as e:
            outcome = "error"
            self.log.error("prompt.error", exc=e, message_id=message_id)
            await self._send_event(
                {
                    "type": "execution_complete",
                    "messageId": message_id,
                    "success": False,
                    "error": str(e),
                }
            )
        finally:
            self._current_message_id = None
            duration_ms = int((time.time() - start_time) * 1000)
            self.log.info(
                "prompt.run",
                message_id=message_id,
                model=model,
                outcome=outcome,
                duration_ms=duration_ms,
            )

    async def _handle_acp_session_update(
        self,
        session_id: str,
        update: Any,
        source: str | None,
    ) -> None:
        """Handle session updates from ACP agent.

        Maps ACP session updates to control plane event types.
        """
        message_id = self._current_message_id
        if not message_id:
            self.log.warn(
                "acp.session_update_dropped",
                reason="no_message_id",
                update_type=type(update).__name__,
            )
            return

        # Handle different update types based on session_update field or isinstance
        update_type = getattr(update, "session_update", None)
        self.log.debug(
            "acp.session_update",
            update_class=type(update).__name__,
            update_type=update_type,
            message_id=message_id,
        )

        if isinstance(update, AgentMessageChunk) or update_type == "agent_message_chunk":
            # Agent is sending a message chunk - content is a single block, not a list
            content = update.content
            if content:
                await self._forward_content_block(message_id, content)

        elif isinstance(update, AgentThoughtChunk) or update_type == "agent_thought_chunk":
            # Agent is thinking (internal reasoning) - content is a single block
            # Send incremental chunks - frontend concatenates them
            content = update.content
            if content and hasattr(content, "text") and content.text:
                await self._send_event(
                    {
                        "type": "token",
                        "content": content.text,
                        "messageId": message_id,
                        "isThought": True,
                    }
                )

        elif isinstance(update, ToolCallStart):
            # Tool call is starting - cache the title for later updates
            tool_name = update.title or update.kind or "Tool"
            self._tool_titles[update.tool_call_id] = tool_name
            self.log.debug(
                "acp.tool_call_start",
                tool_call_id=update.tool_call_id,
                title=update.title,
                kind=update.kind,
                status=update.status,
            )
            await self._send_event(
                {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": update.raw_input or {},
                    "callId": update.tool_call_id,
                    "status": "pending",
                    "output": "",
                    "messageId": message_id,
                }
            )

        elif isinstance(update, ToolCallUpdate | ToolCall):
            # Tool call update (progress, completion, etc.)
            # Use cached title from ToolCallStart since updates often have title=None
            tool_name = update.title or self._tool_titles.get(update.tool_call_id, "Tool")
            self.log.debug(
                "acp.tool_call_update",
                tool_call_id=update.tool_call_id,
                title=update.title,
                cached_title=self._tool_titles.get(update.tool_call_id),
                status=update.status,
            )
            status = "running"
            if update.status == "completed":
                status = "completed"
            elif update.status == "failed":
                status = "error"

            event: dict[str, Any] = {
                "type": "tool_call",
                "tool": tool_name,
                "args": update.raw_input or {},
                "callId": update.tool_call_id,
                "status": status,
                "messageId": message_id,
            }

            # Add output if available
            if update.raw_output:
                event["output"] = update.raw_output
            elif update.content:
                # Content might be single block or list depending on SDK version
                content = update.content
                if isinstance(content, list):
                    output_parts = []
                    for c in content:
                        if hasattr(c, "text") and c.text:
                            output_parts.append(c.text)
                    event["output"] = "\n".join(output_parts)
                elif hasattr(content, "text") and content.text:
                    event["output"] = content.text

            await self._send_event(event)

        else:
            # Log unknown update types for debugging
            self.log.debug(
                "acp.unknown_update_type",
                update_type=type(update).__name__,
            )

    async def _forward_content_block(self, message_id: str, content: Any) -> None:
        """Forward a content block as an appropriate event.

        For text content, we send incremental chunks. The frontend concatenates
        these chunks to build the full response for live streaming display.
        """
        # Handle text content - check both isinstance and duck typing
        # The ACP SDK may return different concrete types
        is_text = isinstance(content, TextContentBlock)
        has_text_attr = hasattr(content, "text") and hasattr(content, "type")

        if is_text or (has_text_attr and getattr(content, "type", None) == "text"):
            text = getattr(content, "text", "")
            # Skip empty text chunks
            if not text:
                return
            # Send incremental chunk - frontend concatenates them
            await self._send_event(
                {
                    "type": "token",
                    "content": text,
                    "messageId": message_id,
                }
            )
            return

        if isinstance(content, ToolCall):
            # Tool call in progress or completed
            # Use cached title from ToolCallStart, or cache new title if available
            if content.title:
                self._tool_titles[content.tool_call_id] = content.title
            tool_name = (
                content.title
                or self._tool_titles.get(content.tool_call_id)
                or getattr(content, "kind", None)
                or "Tool"
            )

            self.log.debug(
                "acp.tool_call_content",
                tool_call_id=content.tool_call_id,
                title=content.title,
                cached_title=self._tool_titles.get(content.tool_call_id),
                kind=getattr(content, "kind", None),
                status=content.status,
            )
            status = "running"
            if content.status == "completed":
                status = "completed"
            elif content.status == "failed":
                status = "error"
            elif content.status == "pending":
                status = "pending"

            await self._send_event(
                {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": content.raw_input or {},
                    "callId": content.tool_call_id,
                    "status": status,
                    "output": content.raw_output or "",
                    "messageId": message_id,
                }
            )

    async def _handle_stop(self) -> None:
        """Handle stop command - cancel current execution."""
        self.log.info("bridge.stop")

        # Cancel current prompt task if running
        if self._prompt_task and not self._prompt_task.done():
            self._prompt_task.cancel()

        # Send cancel notification via ACP if session exists
        if self.acp_conn and self.acp_session_id:
            try:
                await self.acp_conn.cancel(session_id=self.acp_session_id)
                self.log.info("acp.cancel_sent")
            except Exception as e:
                self.log.warn("acp.cancel_error", exc=e)

    async def _handle_snapshot(self) -> None:
        """Handle snapshot command - prepare for snapshot."""
        self.log.info("bridge.snapshot_prepare")
        await self._send_event(
            {
                "type": "snapshot_ready",
                "acpSessionId": self.acp_session_id,
            }
        )

    async def _handle_shutdown(self) -> None:
        """Handle shutdown command - graceful shutdown."""
        self.log.info("bridge.shutdown_requested")
        self.shutdown_event.set()

    def _resolve_github_token(self, cmd: dict[str, Any]) -> TokenResolution:
        """Resolve GitHub token with priority ordering.

        Token priority:
        1. Fresh app token from command (just-in-time from control plane)
        2. Startup app token from env (may be expired for long sessions)
        3. No auth (will fail for private repos)

        Returns:
            TokenResolution with token and source description for logging.
        """
        if cmd.get("githubToken"):
            return TokenResolution(cmd["githubToken"], "fresh from command")
        elif os.environ.get("GITHUB_APP_TOKEN"):
            return TokenResolution(os.environ["GITHUB_APP_TOKEN"], "from env")
        else:
            return TokenResolution("", "none")

    async def _handle_push(self, cmd: dict[str, Any]) -> None:
        """Handle push command - push current branch to GitHub."""
        branch_name = cmd.get("branchName", "")
        repo_owner = cmd.get("repoOwner") or os.environ.get("REPO_OWNER", "")
        repo_name = cmd.get("repoName") or os.environ.get("REPO_NAME", "")

        github_token, token_source = self._resolve_github_token(cmd)
        self.log.info(
            "git.push_start",
            branch_name=branch_name,
            repo_owner=repo_owner,
            repo_name=repo_name,
            token_source=token_source,
        )

        repo_dirs = list(self.repo_path.glob("*/.git"))
        if not repo_dirs:
            self.log.warn("git.push_error", reason="no_repository")
            await self._send_event(
                {
                    "type": "push_error",
                    "error": "No repository found",
                }
            )
            return

        repo_dir = repo_dirs[0].parent

        try:
            refspec = f"HEAD:refs/heads/{branch_name}"

            if not github_token or not repo_owner or not repo_name:
                self.log.warn("git.push_error", reason="missing_credentials")
                await self._send_event(
                    {
                        "type": "push_error",
                        "error": "Push failed - GitHub authentication token is required",
                        "branchName": branch_name,
                    }
                )
                return

            push_url = (
                f"https://x-access-token:{github_token}@github.com/{repo_owner}/{repo_name}.git"
            )

            result = await asyncio.create_subprocess_exec(
                "git",
                "push",
                push_url,
                refspec,
                "-f",
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _stdout, _stderr = await result.communicate()

            if result.returncode != 0:
                self.log.warn("git.push_failed", branch_name=branch_name)
                await self._send_event(
                    {
                        "type": "push_error",
                        "error": "Push failed - authentication may be required",
                        "branchName": branch_name,
                    }
                )
            else:
                self.log.info("git.push_complete", branch_name=branch_name)
                await self._send_event(
                    {
                        "type": "push_complete",
                        "branchName": branch_name,
                    }
                )

        except Exception as e:
            self.log.error("git.push_error", exc=e, branch_name=branch_name)
            await self._send_event(
                {
                    "type": "push_error",
                    "error": str(e),
                    "branchName": branch_name,
                }
            )

    async def _configure_git_identity(self, user: GitUser) -> None:
        """Configure git identity for commit attribution."""
        self.log.debug("git.identity_configure", git_name=user.name, git_email=user.email)

        repo_dirs = list(self.repo_path.glob("*/.git"))
        if not repo_dirs:
            self.log.debug("git.identity_skip", reason="no_repository")
            return

        repo_dir = repo_dirs[0].parent

        try:
            subprocess.run(
                ["git", "config", "--local", "user.name", user.name],
                cwd=repo_dir,
                check=True,
            )
            subprocess.run(
                ["git", "config", "--local", "user.email", user.email],
                cwd=repo_dir,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            self.log.error("git.identity_error", exc=e)

    async def _load_session_id(self) -> None:
        """Load ACP session ID from file if it exists."""
        if self.acp_session_id_file.exists():
            try:
                self.acp_session_id = self.acp_session_id_file.read_text().strip()
                self.log.info(
                    "acp.session.loaded",
                    acp_session_id=self.acp_session_id,
                )
            except Exception as e:
                self.log.error("acp.session.load_error", exc=e)

    async def _save_session_id(self) -> None:
        """Save ACP session ID to file for persistence."""
        if self.acp_session_id:
            try:
                self.acp_session_id_file.write_text(self.acp_session_id)
            except Exception as e:
                self.log.error("acp.session.save_error", exc=e)


async def main():
    """Entry point for bridge process."""
    parser = argparse.ArgumentParser(description="Open-Inspect Agent Bridge")
    parser.add_argument("--sandbox-id", required=True, help="Sandbox ID")
    parser.add_argument("--session-id", required=True, help="Session ID for WebSocket connection")
    parser.add_argument("--control-plane", required=True, help="Control plane URL")
    parser.add_argument("--token", required=True, help="Auth token")

    args = parser.parse_args()

    bridge = AgentBridge(
        sandbox_id=args.sandbox_id,
        session_id=args.session_id,
        control_plane_url=args.control_plane,
        auth_token=args.token,
    )

    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())
