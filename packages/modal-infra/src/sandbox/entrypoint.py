#!/usr/bin/env python3
"""
Sandbox entrypoint - manages bridge lifecycle and git sync.

Runs as PID 1 inside the sandbox. Responsibilities:
1. Perform git sync with latest code
2. Start bridge process for control plane communication
3. Monitor bridge process and restart on crash with exponential backoff
4. Handle graceful shutdown on SIGTERM/SIGINT

Note: The ACP agent (Claude Code) is spawned by the bridge on-demand,
not by this entrypoint. This simplifies the startup sequence.
"""

import asyncio
import json
import os
import signal
import time
from pathlib import Path

import httpx

from .log_config import configure_logging, get_logger

configure_logging()


class SandboxSupervisor:
    """
    Supervisor process for sandbox lifecycle management.

    Manages:
    - Git synchronization with base branch
    - Bridge process for control plane communication
    - Process monitoring with crash recovery
    """

    # Configuration
    MAX_RESTARTS = 5
    BACKOFF_BASE = 2.0
    BACKOFF_MAX = 60.0

    def __init__(self):
        self.bridge_process: asyncio.subprocess.Process | None = None
        self.shutdown_event = asyncio.Event()
        self.git_sync_complete = asyncio.Event()

        # Configuration from environment (set by Modal/SandboxManager)
        self.sandbox_id = os.environ.get("SANDBOX_ID", "unknown")
        self.control_plane_url = os.environ.get("CONTROL_PLANE_URL", "")
        self.sandbox_token = os.environ.get("SANDBOX_AUTH_TOKEN", "")
        self.repo_owner = os.environ.get("REPO_OWNER", "")
        self.repo_name = os.environ.get("REPO_NAME", "")
        self.github_app_token = os.environ.get("GITHUB_APP_TOKEN", "")

        # Set GH_TOKEN for GitHub CLI (gh) - uses the same GitHub App token
        if self.github_app_token:
            os.environ["GH_TOKEN"] = self.github_app_token

        # Parse session config if provided
        session_config_json = os.environ.get("SESSION_CONFIG", "{}")
        self.session_config = json.loads(session_config_json)

        # Paths
        self.workspace_path = Path("/workspace")
        self.repo_path = self.workspace_path / self.repo_name

        # Logger
        session_id = self.session_config.get("session_id", "")
        self.log = get_logger(
            "supervisor",
            service="sandbox",
            sandbox_id=self.sandbox_id,
            session_id=session_id,
        )

    async def perform_git_sync(self) -> bool:
        """
        Clone repository if needed, then synchronize with latest changes.

        Returns:
            True if sync completed successfully, False otherwise
        """
        self.log.debug(
            "git.sync_start",
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            repo_path=str(self.repo_path),
            has_github_token=bool(self.github_app_token),
        )

        # Clone the repository if it doesn't exist
        if not self.repo_path.exists():
            if not self.repo_owner or not self.repo_name:
                self.log.info("git.skip_clone", reason="no_repo_configured")
                self.git_sync_complete.set()
                return True

            self.log.info(
                "git.clone_start",
                repo_owner=self.repo_owner,
                repo_name=self.repo_name,
                authenticated=bool(self.github_app_token),
            )

            # Use authenticated URL if GitHub App token is available
            if self.github_app_token:
                clone_url = f"https://x-access-token:{self.github_app_token}@github.com/{self.repo_owner}/{self.repo_name}.git"
            else:
                clone_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"

            result = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth",
                "1",
                clone_url,
                str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                self.log.error(
                    "git.clone_error",
                    stderr=stderr.decode(),
                    exit_code=result.returncode,
                )
                self.git_sync_complete.set()
                return False

            self.log.info("git.clone_complete", repo_path=str(self.repo_path))

        try:
            # Configure remote URL with auth token if available
            if self.github_app_token:
                auth_url = f"https://x-access-token:{self.github_app_token}@github.com/{self.repo_owner}/{self.repo_name}.git"
                await asyncio.create_subprocess_exec(
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    auth_url,
                    cwd=self.repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            # Fetch latest changes
            result = await asyncio.create_subprocess_exec(
                "git",
                "fetch",
                "origin",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()

            if result.returncode != 0:
                stderr = await result.stderr.read() if result.stderr else b""
                self.log.error(
                    "git.fetch_error",
                    stderr=stderr.decode(),
                    exit_code=result.returncode,
                )
                return False

            # Get the base branch (default to main)
            base_branch = self.session_config.get("branch", "main")

            # Rebase onto latest
            result = await asyncio.create_subprocess_exec(
                "git",
                "rebase",
                f"origin/{base_branch}",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()

            if result.returncode != 0:
                # Check if there's actually a rebase in progress before trying to abort
                rebase_merge = self.repo_path / ".git" / "rebase-merge"
                rebase_apply = self.repo_path / ".git" / "rebase-apply"
                if rebase_merge.exists() or rebase_apply.exists():
                    await asyncio.create_subprocess_exec(
                        "git",
                        "rebase",
                        "--abort",
                        cwd=self.repo_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                self.log.warn("git.rebase_error", base_branch=base_branch)

            # Get current SHA
            result = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            current_sha = stdout.decode().strip()
            self.log.info("git.sync_complete", head_sha=current_sha)

            self.git_sync_complete.set()
            return True

        except Exception as e:
            self.log.error("git.sync_error", exc=e)
            self.git_sync_complete.set()  # Allow agent to proceed anyway
            return False

    async def start_bridge(self) -> None:
        """Start the agent bridge process."""
        self.log.info("bridge.start")

        if not self.control_plane_url:
            self.log.info("bridge.skip", reason="no_control_plane_url")
            return

        # Get session_id from config (required for WebSocket connection)
        session_id = self.session_config.get("session_id", "")
        if not session_id:
            self.log.info("bridge.skip", reason="no_session_id")
            return

        # Run bridge as a module (works with relative imports)
        self.bridge_process = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "sandbox.bridge",
            "--sandbox-id",
            self.sandbox_id,
            "--session-id",
            session_id,
            "--control-plane",
            self.control_plane_url,
            "--token",
            self.sandbox_token,
            env=os.environ,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Start log forwarder for bridge
        asyncio.create_task(self._forward_bridge_logs())
        self.log.info("bridge.started")

        # Check if bridge exited immediately during startup
        await asyncio.sleep(0.5)
        if self.bridge_process.returncode is not None:
            exit_code = self.bridge_process.returncode
            # Bridge exited immediately - read any error output
            stdout, _ = await self.bridge_process.communicate()
            if exit_code == 0:
                self.log.warn("bridge.early_exit", exit_code=exit_code)
            else:
                self.log.error(
                    "bridge.startup_crash",
                    exit_code=exit_code,
                    output=stdout.decode() if stdout else "",
                )

    async def _forward_bridge_logs(self) -> None:
        """Forward bridge stdout to supervisor stdout."""
        if not self.bridge_process or not self.bridge_process.stdout:
            return

        try:
            async for line in self.bridge_process.stdout:
                # Bridge already prefixes its output with [bridge], don't double it
                print(line.decode().rstrip())
        except Exception as e:
            print(f"[supervisor] Bridge log forwarding error: {e}")

    async def monitor_processes(self) -> None:
        """Monitor child processes and restart on crash."""
        bridge_restart_count = 0

        while not self.shutdown_event.is_set():
            # Check bridge process
            if self.bridge_process and self.bridge_process.returncode is not None:
                exit_code = self.bridge_process.returncode

                if exit_code == 0:
                    # Graceful exit: shutdown command, session terminated, or fatal
                    # connection error. Propagate shutdown rather than restarting.
                    self.log.info(
                        "bridge.graceful_exit",
                        exit_code=exit_code,
                    )
                    self.shutdown_event.set()
                    break
                else:
                    # Crash: restart with backoff and retry limit
                    bridge_restart_count += 1
                    self.log.error(
                        "bridge.crash",
                        exit_code=exit_code,
                        restart_count=bridge_restart_count,
                    )

                    if bridge_restart_count > self.MAX_RESTARTS:
                        self.log.error(
                            "bridge.max_restarts",
                            restart_count=bridge_restart_count,
                        )
                        await self._report_fatal_error(
                            f"Bridge crashed {bridge_restart_count} times, giving up"
                        )
                        self.shutdown_event.set()
                        break

                    delay = min(self.BACKOFF_BASE**bridge_restart_count, self.BACKOFF_MAX)
                    self.log.info(
                        "bridge.restart",
                        delay_s=round(delay, 1),
                        restart_count=bridge_restart_count,
                    )
                    await asyncio.sleep(delay)
                    await self.start_bridge()

            await asyncio.sleep(1.0)

    async def _report_fatal_error(self, message: str) -> None:
        """Report a fatal error to the control plane."""
        self.log.error("supervisor.fatal", message=message)

        if not self.control_plane_url:
            return

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.control_plane_url}/sandbox/{self.sandbox_id}/error",
                    json={"error": message, "fatal": True},
                    headers={"Authorization": f"Bearer {self.sandbox_token}"},
                    timeout=5.0,
                )
        except Exception as e:
            self.log.error("supervisor.report_error_failed", exc=e)

    async def configure_git_identity(self) -> None:
        """Configure git identity from session owner."""
        git_user = self.session_config.get("git_user")
        if not git_user or not self.repo_path.exists():
            return

        try:
            await asyncio.create_subprocess_exec(
                "git",
                "config",
                "--local",
                "user.name",
                git_user["name"],
                cwd=self.repo_path,
            )
            await asyncio.create_subprocess_exec(
                "git",
                "config",
                "--local",
                "user.email",
                git_user["email"],
                cwd=self.repo_path,
            )
            self.log.info(
                "git.identity_configured",
                git_name=git_user["name"],
                git_email=git_user["email"],
            )
        except Exception as e:
            self.log.error("git.identity_error", exc=e)

    async def _quick_git_fetch(self) -> None:
        """
        Quick fetch to check if we're behind after snapshot restore.

        When restored from a snapshot, the workspace already has all changes.
        This just checks if the remote has new commits since the snapshot.
        """
        if not self.repo_path.exists():
            self.log.info("git.quick_fetch_skip", reason="no_repo_path")
            return

        try:
            # Configure remote URL with auth token if available
            if self.github_app_token:
                auth_url = f"https://x-access-token:{self.github_app_token}@github.com/{self.repo_owner}/{self.repo_name}.git"
                await asyncio.create_subprocess_exec(
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    auth_url,
                    cwd=self.repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            # Fetch from origin
            result = await asyncio.create_subprocess_exec(
                "git",
                "fetch",
                "--quiet",
                "origin",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                self.log.warn(
                    "git.quick_fetch_error",
                    stderr=stderr.decode(),
                    exit_code=result.returncode,
                )
                return

            # Check if we're behind the remote
            # Get the current branch
            result = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            current_branch = stdout.decode().strip()

            # Check if we have an upstream set
            result = await asyncio.create_subprocess_exec(
                "git",
                "rev-list",
                "--count",
                f"HEAD..origin/{current_branch}",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                commits_behind = int(stdout.decode().strip() or "0")
                self.log.info(
                    "git.snapshot_status",
                    commits_behind=commits_behind,
                    current_branch=current_branch,
                )
            else:
                self.log.debug("git.snapshot_status_unknown", reason="no_upstream")

        except Exception as e:
            self.log.error("git.quick_fetch_error", exc=e)

    async def run(self) -> None:
        """Main supervisor loop."""
        startup_start = time.time()

        self.log.info(
            "supervisor.start",
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
        )

        # Check if restored from snapshot
        restored_from_snapshot = os.environ.get("RESTORED_FROM_SNAPSHOT") == "true"
        if restored_from_snapshot:
            self.log.info("supervisor.restored_from_snapshot")

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._handle_signal(s)))

        git_sync_success = False
        try:
            # Phase 1: Git sync
            if restored_from_snapshot:
                # Restored from snapshot - just do a quick fetch to check for updates
                await self._quick_git_fetch()
                self.git_sync_complete.set()
                git_sync_success = True
            else:
                # Fresh sandbox - full git clone and sync
                git_sync_success = await self.perform_git_sync()

            # Phase 2: Configure git identity (if repo was cloned)
            await self.configure_git_identity()

            # Phase 3: Start bridge (which will spawn ACP agent on-demand)
            await self.start_bridge()

            # Emit sandbox.startup wide event
            duration_ms = int((time.time() - startup_start) * 1000)
            self.log.info(
                "sandbox.startup",
                repo_owner=self.repo_owner,
                repo_name=self.repo_name,
                restored_from_snapshot=restored_from_snapshot,
                git_sync_success=git_sync_success,
                duration_ms=duration_ms,
                outcome="success",
            )

            # Phase 4: Monitor processes
            await self.monitor_processes()

        except Exception as e:
            self.log.error("supervisor.error", exc=e)
            await self._report_fatal_error(str(e))

        finally:
            await self.shutdown()

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        self.log.info("supervisor.signal", signal_name=sig.name)
        self.shutdown_event.set()

    async def shutdown(self) -> None:
        """Graceful shutdown of all processes."""
        self.log.info("supervisor.shutdown_start")

        # Terminate bridge (which will clean up its ACP subprocess)
        if self.bridge_process and self.bridge_process.returncode is None:
            self.bridge_process.terminate()
            try:
                await asyncio.wait_for(self.bridge_process.wait(), timeout=10.0)
            except TimeoutError:
                self.bridge_process.kill()

        self.log.info("supervisor.shutdown_complete")


async def main():
    """Entry point for the sandbox supervisor."""
    supervisor = SandboxSupervisor()
    await supervisor.run()


if __name__ == "__main__":
    asyncio.run(main())
