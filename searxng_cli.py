#!/usr/bin/env python3
"""SearXNG CLI - Command line interface for SearXNG search engine."""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
import subprocess
import platform

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

# SearXNG is now installed as external dependency via git+https://github.com/searxng/searxng.git


# Configure logging to suppress noisy errors from SearXNG engines
def configure_logging():
    """Configure logging to suppress engine errors that don't affect overall functionality."""
    # Set SearXNG loggers to WARNING level to suppress INFO/ERROR for individual engine failures
    logging.getLogger("searx").setLevel(logging.CRITICAL)
    logging.getLogger("searx.engines").setLevel(logging.CRITICAL)
    logging.getLogger("searx.search").setLevel(logging.CRITICAL)
    logging.getLogger("searx.network").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)


# Configure logging early
configure_logging()

# Import vendored SearXNG modules
import searx
import searx.engines
import searx.preferences
import searx.search
import searx.webadapter
from searx.search.models import EngineRef, SearchQuery
from searx.search.processors import PROCESSORS
from searx.results import ResultContainer
from searx.compat import tomllib
import threading
from timeit import default_timer
from uuid import uuid4

app = typer.Typer(
    name="searxng-cli",
    help="Privacy-respecting metasearch engine CLI with MCP server support",
    no_args_is_help=True,
)
console = Console()

# Shared AI system prompt for consistent tool usage across all modes
BASE_AI_SYSTEM_PROMPT = """You have access to powerful web search and URL fetching tools. When researching topics, you should:

**PRIORITIZE PARALLEL OPERATIONS FOR SPEED:**
- ALWAYS prefer multi_web_search over single web_search when you need multiple searches
- ALWAYS prefer fetch_urls over single fetch_url when you need multiple URLs  
- Use parallel tools aggressively - users want comprehensive data delivered quickly
- Run multiple related searches simultaneously rather than sequentially
- Fetch all relevant URLs at once rather than one-by-one

**TOOL USAGE GUIDELINES:**
- multi_web_search: Use for related queries like ["topic overview", "recent developments", "expert opinions"]
- fetch_urls: Use when you find multiple relevant URLs in search results
- Be thorough but efficient - parallel execution lets you gather more data faster
- Don't hesitate to use 3-5 searches or 5-10 URL fetches if it improves your response"""

# Default engines for different categories
DEFAULT_ENGINES = {
    "general": ["duckduckgo", "startpage", "brave"],
    "images": ["duckduckgo_images", "bing_images"],
    "videos": ["youtube_noapi", "vimeo"],
    "news": ["duckduckgo", "reuters"],
    "science": ["arxiv", "google_scholar"],
    "it": ["github", "stackoverflow"],
}

# Popular engines that users might want to enable/disable
COMMON_ENGINES = [
    "google",
    "duckduckgo",
    "bing",
    "startpage",
    "brave",
    "qwant",
    "google_images",
    "bing_images",
    "duckduckgo_images",
    "youtube_noapi",
    "vimeo",
    "dailymotion",
    "google_news",
    "bing_news",
    "reuters",
    "github",
    "gitlab",
    "stackoverflow",
    "arxiv",
    "google_scholar",
    "pubmed",
]


# Sessions management sub-app
sessions_app = typer.Typer(help="Manage chat sessions")


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of sessions to show"),
    all: bool = typer.Option(False, "--all", "-a", help="Show all sessions"),
    sort: str = typer.Option(
        "updated_at",
        "--sort",
        help="Sort by one of: updated_at, created_at, title, model",
    ),
    reverse: bool = typer.Option(True, "--desc/--asc", help="Sort order"),
):
    """List chat sessions with metadata."""
    items = session_store.list_sessions(
        limit=None if all else limit, sort_by=sort, reverse=reverse
    )
    if not items:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title="Chat Sessions")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Model", style="green")
    table.add_column("Created", style="blue")
    table.add_column("Updated", style="blue")
    table.add_column("Msgs", justify="right", style="magenta")

    for s in items:
        sid = s.get("session_id", "")
        table.add_row(
            (sid[:8] + "...") if sid else "",
            (s.get("title") or "(untitled)")[:60],
            (s.get("model") or "").split("/")[-1],
            (s.get("created_at") or ""),
            (s.get("updated_at") or ""),
            str(s.get("messages_count", len(s.get("messages", []) or []))),
        )

    console.print(table)


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session ID or unique prefix"),
    preview: int = typer.Option(0, "--preview", "-p", help="Show last N messages"),
):
    """Show details for a session and optionally preview messages."""
    try:
        resolved = session_store.find(session_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    try:
        s = session_store.load(resolved)
    except FileNotFoundError:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)
    s = session_store._normalize_session(s)

    console.print("[bold]Session Details[/bold]")
    console.print(f"ID: [cyan]{s.get('session_id')}[/cyan]")
    console.print(f"Title: {s.get('title') or '(untitled)'}")
    console.print(f"Model: [green]{s.get('model')}[/green]")
    console.print(f"Created: {s.get('created_at')}")
    console.print(f"Updated: {s.get('updated_at')}")
    console.print(f"Messages: {len(s.get('messages', []) or [])}")
    console.print(f"Transcript: {s.get('markdown_path') or 'N/A'}")

    if preview > 0:
        msgs = s.get("messages", []) or []
        msgs = msgs[-preview:]
        if msgs:
            console.print("\n[bold]Recent Messages[/bold]")
            for m in msgs:
                role = m.get("role") or "assistant"
                style = "green" if role == "user" else "blue"
                content = str(m.get("content", ""))
                head = content.splitlines()[0]
                shown = content if len(content) <= 120 else head[:117] + "..."
                console.print(f"[{style}]{role.title()}:[/{style}] {shown}")


@sessions_app.command("rm")
def sessions_rm(
    session_id: str = typer.Argument(..., help="Session ID or unique prefix"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Confirm deletion without prompt"
    ),
    keep_transcript: bool = typer.Option(
        False, "--keep-transcript", help="Do not delete transcript file"
    ),
):
    """Remove a session JSON (and transcript unless kept)."""
    try:
        resolved = session_store.find(session_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not yes:
        if not typer.confirm(f"Delete session {resolved[:8]}... ?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return
    session_store.delete(resolved, keep_transcript=keep_transcript)
    console.print(f"[green]Session deleted:[/green] {resolved}")


app.add_typer(sessions_app, name="sessions")


# Profile management for API keys and configurations
class ProfileManager:
    """Manages user profiles for API keys and configurations."""

    def __init__(self):
        self.config_dir = self._get_config_dir()
        self.profiles_file = self.config_dir / "profiles.toml"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _get_config_dir(self) -> Path:
        """Get XDG-compliant config directory."""
        if platform.system() == "Windows":
            config_home = os.environ.get(
                "APPDATA", os.path.expanduser("~\\AppData\\Roaming")
            )
        else:
            config_home = os.environ.get(
                "XDG_CONFIG_HOME", os.path.expanduser("~/.config")
            )
        return Path(config_home) / "searxng"

    def _get_default_model(self, provider_type: str) -> str:
        """Get default model for each provider type."""
        defaults = {
            "openrouter": "openrouter/openai/gpt-5",
            "anthropic": "anthropic/claude-3-5-sonnet-20241022",
            "openai": "openai/gpt-5",
            "google": "gemini/gemini-2.0-flash-exp",
        }
        return defaults.get(provider_type, "openai/gpt-5")

    def _get_default_base_url(self, provider_type: str) -> Optional[str]:
        """Get default base URL for each provider type."""
        defaults = {
            "openrouter": "https://openrouter.ai/api/v1",
            "anthropic": None,  # Use LiteLLM default
            "openai": None,  # Use LiteLLM default
            "google": None,  # Use LiteLLM default
        }
        return defaults.get(provider_type)

    def _load_profiles(self) -> Dict[str, Any]:
        """Load profiles from TOML file."""
        if not self.profiles_file.exists():
            return {"profiles": {}, "settings": {}}

        try:
            with open(self.profiles_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            console.print(f"[red]Error loading profiles: {e}[/red]")
            return {"profiles": {}, "settings": {}}

    def _save_profiles(self, data: Dict[str, Any]):
        """Save profiles to TOML file."""
        import toml

        try:
            with open(self.profiles_file, "w") as f:
                toml.dump(data, f)
            # Set restrictive permissions
            os.chmod(self.profiles_file, 0o600)
        except Exception as e:
            console.print(f"[red]Error saving profiles: {e}[/red]")
            raise typer.Exit(1)

    def add_profile(
        self,
        name: str,
        provider_type: str,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        force: bool = False,
    ):
        """Add a new profile."""
        data = self._load_profiles()

        # Check if profile exists
        if name in data["profiles"] and not force:
            if not typer.confirm(f"Profile '{name}' already exists. Overwrite?"):
                console.print("[yellow]Profile not added.[/yellow]")
                return

        # Set defaults
        if not base_url:
            base_url = self._get_default_base_url(provider_type)
        if not model:
            model = self._get_default_model(provider_type)

        # Create profile
        profile = {
            "type": provider_type,
            "api_key": api_key,
            "default_model": model,
        }
        if base_url:
            profile["base_url"] = base_url

        data["profiles"][name] = profile

        # Set as default if it's the first profile
        if not data["settings"].get("default_profile"):
            data["settings"]["default_profile"] = name

        self._save_profiles(data)
        console.print(f"[green]Profile '{name}' added successfully.[/green]")

        if (
            not data["settings"].get("default_profile")
            or data["settings"]["default_profile"] == name
        ):
            console.print(f"[blue]Profile '{name}' set as default.[/blue]")

    def list_profiles(self):
        """List all profiles."""
        data = self._load_profiles()
        if not data["profiles"]:
            console.print("[yellow]No profiles found.[/yellow]")
            console.print(f"[dim]Profiles are stored in: {self.profiles_file}[/dim]")
            return

        table = Table(title="SearXNG Profiles")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Model", style="green")
        table.add_column("Base URL", style="blue")
        table.add_column("Default", style="yellow")

        default_profile = data["settings"].get("default_profile")
        for name, profile in data["profiles"].items():
            is_default = "✓" if name == default_profile else ""
            base_url = profile.get("base_url", "Default")
            table.add_row(
                name, profile["type"], profile["default_model"], base_url, is_default
            )

        console.print(table)
        console.print(f"[dim]Profiles stored in: {self.profiles_file}[/dim]")

    def show_profile(self, name: str):
        """Show details of a specific profile."""
        data = self._load_profiles()
        if name not in data["profiles"]:
            console.print(f"[red]Profile '{name}' not found.[/red]")
            raise typer.Exit(1)

        profile = data["profiles"][name]
        is_default = name == data["settings"].get("default_profile")

        console.print(f"\n[bold]Profile: {name}[/bold]")
        console.print(f"Type: {profile['type']}")
        console.print(f"Model: {profile['default_model']}")
        console.print(f"Base URL: {profile.get('base_url', 'Default')}")
        console.print(f"API Key: {profile['api_key'][:8]}...")
        console.print(f"Default: {'Yes' if is_default else 'No'}")

    def remove_profile(self, name: str):
        """Remove a profile."""
        data = self._load_profiles()
        if name not in data["profiles"]:
            console.print(f"[red]Profile '{name}' not found.[/red]")
            raise typer.Exit(1)

        if not typer.confirm(f"Remove profile '{name}'?"):
            console.print("[yellow]Profile not removed.[/yellow]")
            return

        del data["profiles"][name]

        # Clear default if removing default profile
        if data["settings"].get("default_profile") == name:
            data["settings"]["default_profile"] = None
            # Set new default if profiles remain
            if data["profiles"]:
                new_default = next(iter(data["profiles"]))
                data["settings"]["default_profile"] = new_default
                console.print(
                    f"[blue]Profile '{new_default}' set as new default.[/blue]"
                )

        self._save_profiles(data)
        console.print(f"[green]Profile '{name}' removed.[/green]")

    def set_default(self, name: str):
        """Set default profile."""
        data = self._load_profiles()
        if name not in data["profiles"]:
            console.print(f"[red]Profile '{name}' not found.[/red]")
            raise typer.Exit(1)

        data["settings"]["default_profile"] = name
        self._save_profiles(data)
        console.print(f"[green]Profile '{name}' set as default.[/green]")

    def edit_profile(self, name: str):
        """Edit a profile using system editor."""
        data = self._load_profiles()
        if name not in data["profiles"]:
            console.print(f"[red]Profile '{name}' not found.[/red]")
            raise typer.Exit(1)

        editor = os.environ.get(
            "EDITOR", "nano" if platform.system() != "Windows" else "notepad"
        )
        try:
            subprocess.run([editor, str(self.profiles_file)])
            console.print(f"[green]Profile file opened in {editor}.[/green]")
        except Exception as e:
            console.print(f"[red]Error opening editor: {e}[/red]")
            raise typer.Exit(1)

    def test_profile(self, name: str):
        """Test a profile by making a simple API call."""
        data = self._load_profiles()
        if name not in data["profiles"]:
            console.print(f"[red]Profile '{name}' not found.[/red]")
            raise typer.Exit(1)

        profile = data["profiles"][name]

        # Test with a simple query
        console.print(f"[blue]Testing profile '{name}'...[/blue]")

        # Set environment variables temporarily
        old_env = {}
        try:
            # Map provider types to environment variables
            env_map = {
                "openrouter": "OPENROUTER_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
            }

            env_key = env_map.get(profile["type"])
            if env_key:
                old_env[env_key] = os.environ.get(env_key)
                os.environ[env_key] = profile["api_key"]

            if profile.get("base_url"):
                old_env["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL")
                os.environ["OPENAI_BASE_URL"] = profile["base_url"]

            # Make test call
            import litellm

            response = litellm.completion(
                model=profile["default_model"],
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )

            console.print(f"[green]✓ Profile '{name}' is working![/green]")
            console.print(f"Model: {profile['default_model']}")
            console.print(f"Response: {response.choices[0].message.content}")

        except Exception as e:
            console.print(f"[red]✗ Profile '{name}' test failed: {e}[/red]")
        finally:
            # Restore environment
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def get_profile(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get profile by name or default profile."""
        data = self._load_profiles()

        if name:
            return data["profiles"].get(name)
        else:
            default_name = data["settings"].get("default_profile")
            if default_name:
                return data["profiles"].get(default_name)

        return None


# Global profile manager instance
profile_manager = ProfileManager()


class GlobalConfig:
    """Global configuration for SearXNG CLI."""

    def __init__(self):
        self.config_dir = profile_manager.config_dir  # Reuse same config dir
        self.config_file = self.config_dir / "config.toml"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config_data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    def _save_config(self, data: Dict[str, Any]):
        """Save configuration to TOML file."""
        try:
            import tomli_w

            with open(self.config_file, "wb") as f:
                tomli_w.dump(data, f)
            self._config_data = data
        except Exception:
            pass

    def get_default_model(self) -> Optional[str]:
        """Get the global default model."""
        return self._config_data.get("default_model")

    def set_default_model(self, model: Optional[str]):
        """Set the global default model."""
        data = self._config_data.copy()
        if model:
            data["default_model"] = model
        else:
            data.pop("default_model", None)
        self._save_config(data)


# Global config instance
global_config = GlobalConfig()


# --------------------
# CLI Proxy API Config
# --------------------
class CLIProxyAPIConfig:
    """Manages CLI Proxy API configuration for searxng integration.

    Configuration is stored in ~/.config/searxng/config.toml under [cli-proxy-api] section.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or profile_manager.config_dir
        self.config_file = self.config_dir / "config.toml"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load config from TOML file."""
        if not self.config_file.exists():
            return {}
        try:
            with open(self.config_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return {}

    def _save_config(self, data: Dict[str, Any]):
        """Save config to TOML file."""
        import toml

        try:
            import toml

            with open(self.config_file, "w", encoding="utf-8") as f:
                toml.dump(data, f)
            self._config_data = data
        except Exception:
            pass
            self._lock_file = None

    def _find_free_port(self) -> int:
        """Find an available port using OS ephemeral port allocation."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        return port

    def _resolve_path(self, path: str, base_dir: Path) -> str:
        """Resolve relative paths against base directory."""
        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return str(p)
        # Handle ~ expansion
        if path.startswith("~"):
            return str(Path(path).expanduser())
        # Resolve relative to base_dir
        return str((base_dir / p).resolve())

    def _create_temp_config(self, source_config: Path, port: int) -> Path:
        """Create temporary config with modified port and security settings."""
        import yaml

        with open(source_config, "r") as f:
            config = yaml.safe_load(f)

        base_dir = source_config.parent

        # Modify port and FORCE localhost binding for security
        config["port"] = port
        config["host"] = "127.0.0.1"  # Always force localhost

        # Resolve relative paths to absolute
        if "auth-dir" in config:
            config["auth-dir"] = self._resolve_path(config["auth-dir"], base_dir)

        # Create temp file with secure permissions
        temp_dir = (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "searxng"
            / "cli-proxy"
        )
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_config = temp_dir / f"config-{port}.yaml"

        # Write with restrictive permissions (0600)
        fd = os.open(str(temp_config), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)

        return temp_config

    def start(self, source_config: Optional[Path] = None) -> bool:
        """Start cli-proxy-api subprocess.

        Args:
            source_config: Path to cli-proxy-api config. If None, auto-detects.

        Returns:
            True if started successfully, False otherwise.
        """
        import time

        if self._started and self.is_running():
            return True

        # Auto-detect config if not provided
        if source_config is None:
            source_config = cli_proxy_config.find_cli_proxy_config()
            if source_config is None:
                return False

        # Acquire lock to prevent concurrent starts
        if not self._acquire_lock():
            # Another instance may be starting, wait and check
            time.sleep(1)
            return False

        try:
            # Find free port using ephemeral allocation
            self.port = self._find_free_port()
            cli_proxy_log.debug(f"Starting cli-proxy-api on port {self.port}")

            # Create temp config with our port
            self.temp_config_path = self._create_temp_config(source_config, self.port)
            cli_proxy_log.debug(f"Created temp config at {self.temp_config_path}")

            # Set up signal handlers before starting process
            self._setup_signal_handlers()

            # Start subprocess
            # Set NO_PROXY to ensure localhost calls don't go through HTTP proxy
            env = os.environ.copy()
            no_proxy = env.get("NO_PROXY", "")
            if no_proxy:
                env["NO_PROXY"] = f"{no_proxy},127.0.0.1,localhost"
            else:
                env["NO_PROXY"] = "127.0.0.1,localhost"

            self.process = subprocess.Popen(
                ["cli-proxy-api", "-config", str(self.temp_config_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except FileNotFoundError:
            cli_proxy_log.error("cli-proxy-api binary not found on PATH")
            self._release_lock()
            return False
        except Exception as e:
            cli_proxy_log.error(f"Failed to start cli-proxy-api: {e}")
            self._release_lock()
            raise

        # Wait for server to be ready
        if not self._wait_for_ready(timeout=15):
            cli_proxy_log.error(
                f"cli-proxy-api failed to start (timeout waiting for server on port {self.port})"
            )
            self.stop()
            return False

        self._started = True
        self._restart_count = 0
        self._start_time = time.time()  # Track when proxy started for retry logic
        cli_proxy_log.info(f"cli-proxy-api started successfully on port {self.port}")
        return True

    def _wait_for_ready(self, timeout: int = 15) -> bool:
        """Wait for cli-proxy-api to be ready with exponential backoff."""
        import time

        start = time.time()
        url = f"http://127.0.0.1:{self.port}"
        delay = 0.1

        while time.time() - start < timeout:
            # Check if process died
            if self.process and self.process.poll() is not None:
                return False  # Process exited

            try:
                # Try the root endpoint first (doesn't require auth)
                resp = httpx.get(url, timeout=2)
                if resp.status_code == 200:
                    return True
            except httpx.ConnectError:
                pass  # Not ready yet
            except Exception:
                pass

            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)  # Exponential backoff, max 2s

        return False

    def is_running(self) -> bool:
        """Check if subprocess is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_base_url(self) -> Optional[str]:
        """Get base URL for API requests (WITHOUT /v1 - litellm adds it)."""
        if not self._started or self.port is None:
            return None
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        """Stop subprocess and cleanup."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass
            self.process = None

        # Cleanup temp config
        if self.temp_config_path and self.temp_config_path.exists():
            try:
                self.temp_config_path.unlink()
            except Exception:
                pass
            self.temp_config_path = None

        self._release_lock()
        self._started = False
        self.port = None

    def restart(self, source_config: Optional[Path] = None) -> bool:
        """Restart the subprocess with backoff protection.

        Args:
            source_config: Path to cli-proxy-api config. If None, auto-detects.

        Returns:
            True if restarted successfully, False otherwise.
        """
        import time

        if self._restart_count >= self._max_restarts:
            cli_proxy_log.error(
                f"cli-proxy-api max restarts ({self._max_restarts}) exceeded"
            )
            return False

        self._restart_count += 1
        cli_proxy_log.warning(
            f"cli-proxy-api process died, attempting restart {self._restart_count}/{self._max_restarts}"
        )
        # Exponential backoff between restarts
        backoff = min(2**self._restart_count, 30)
        cli_proxy_log.debug(f"Waiting {backoff}s before restart")
        time.sleep(backoff)

        self.stop()
        return self.start(source_config)

    def ensure_running(self, source_config: Optional[Path] = None) -> bool:
        """Ensure proxy is running, starting if necessary.

        Args:
            source_config: Path to cli-proxy-api config. If None, auto-detects.

        Returns:
            True if running, False if failed to start.
        """
        if self.is_running():
            return True
        return self.start(source_config)

    def get_models(self, force_refresh: bool = False) -> List[str]:
        """Fetch available models from cli-proxy-api.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            List of model IDs available from the proxy.
        """
        import time

        if not self.is_running():
            return []

        # Check cache (5 minute TTL)
        if not force_refresh and hasattr(self, "_models_cache"):
            cache_age = time.time() - getattr(self, "_models_cache_time", 0)
            if cache_age < 300:  # 5 minutes
                return self._models_cache

        try:
            url = f"http://127.0.0.1:{self.port}/v1/models"

            # Retry logic only if proxy was just started (within last 5 seconds)
            just_started = (
                hasattr(self, "_start_time") and (time.time() - self._start_time) < 5
            )
            max_attempts = 3 if just_started else 1

            for attempt in range(max_attempts):
                resp = httpx.get(url, timeout=5)
                if resp.status_code != 200:
                    return []

                data = resp.json()
                # OpenAI format: {"data": [{"id": "model-name", ...}, ...]}
                models = []
                for model in data.get("data", []):
                    model_id = model.get("id")
                    if model_id:
                        models.append(model_id)

                if models:
                    # Cache results
                    self._models_cache = models
                    self._models_cache_time = time.time()
                    return models

                # Empty result on fresh start, wait and retry
                if attempt < max_attempts - 1:
                    time.sleep(0.5)

            # Return empty after retries
            return []
        except Exception:
            return []

    def get_prefixed_models(self, force_refresh: bool = False) -> List[str]:
        """Get models with cli-proxy-api/ prefix for litellm routing.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            List of model IDs prefixed with 'cli-proxy-api/'.
        """
        return [f"cli-proxy-api/{m}" for m in self.get_models(force_refresh)]


# CLI Proxy API config instance
cli_proxy_config = CLIProxyAPIConfig()


# --------------------
# Session persistence
# --------------------
class SessionStore:
    """Lightweight JSON session storage under the user config dir.

    Files live at: ~/.config/searxng/sessions/<session_id>.json (XDG on *nix)
    """

    def __init__(self, base_config_dir: Optional[Path] = None):
        self.base_config_dir = base_config_dir or profile_manager.config_dir
        self.sessions_dir = self.base_config_dir / "sessions"

    def _ensure_dir(self) -> None:
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Defer permission errors until actual use sites handle/report
            pass

    def _path(self, session_id: str) -> Path:
        self._ensure_dir()
        return self.sessions_dir / f"{session_id}.json"

    def create(
        self,
        model: str,
        base_url: Optional[str],
        profile: Optional[str],
        markdown_path: Optional[str] = None,
        initial_messages: Optional[List[Dict[str, str]]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        session = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "model": model,
            "base_url": base_url,
            "profile": profile,
            "title": title or "",
            "markdown_path": markdown_path,
            "messages": initial_messages or [],
        }
        self.save(session)
        return session

    def load(self, session_id: str) -> Dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, session: Dict[str, Any]) -> None:
        session["updated_at"] = datetime.utcnow().isoformat()
        path = self._path(session["session_id"]) if isinstance(session, dict) else None
        if not path:
            raise ValueError("Invalid session object")
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            # Best effort; permissions may not be supported on all platforms
            pass

    def append_message(
        self, session: Dict[str, Any], role: str, content: str, **extra: Any
    ) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat(),
        }
        msg.update({k: v for k, v in extra.items() if v is not None})
        session.setdefault("messages", []).append(msg)
        # Auto-title based on the first user message if not set
        try:
            if not session.get("title") and role == "user":
                preview = (content or "").strip().splitlines()[0]
                if preview:
                    session["title"] = preview[:80]
        except Exception:
            # Non-fatal: continue without setting a title
            pass
        self.save(session)
        return session

    # ---- Extended helpers for session discovery/management ----
    def _iter_session_files(self):
        """Yield Path objects for session JSON files (best-effort).

        Ensures the sessions directory exists and yields any '*.json' within it.
        """
        self._ensure_dir()
        try:
            for p in sorted(self.sessions_dir.glob("*.json")):
                if p.is_file():
                    yield p
        except Exception:
            return

    def _normalize_session(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy with normalized metadata fields.

        - Ensure 'title' exists (may derive from first user message)
        - Ensure 'updated_at' is present (fallback to last message ts)
        - Attach 'messages_count' for quick display
        """
        s = dict(d)
        msgs = s.get("messages", []) or []
        # Title
        if not s.get("title"):
            try:
                first_user = next(
                    (m for m in msgs if m.get("role") == "user" and m.get("content")),
                    None,
                )
                if first_user:
                    pv = str(first_user.get("content", "")).strip().splitlines()[0]
                    s["title"] = pv[:80] if pv else ""
            except Exception:
                pass
        # updated_at fallback
        if not s.get("updated_at"):
            try:
                last_ts = next(
                    (m.get("ts") for m in reversed(msgs) if m.get("ts")), None
                )
                if last_ts:
                    s["updated_at"] = last_ts
            except Exception:
                pass
        # messages_count
        try:
            s["messages_count"] = len(msgs)
        except Exception:
            s["messages_count"] = 0
        return s

    def list_sessions(
        self,
        limit: Optional[int] = None,
        sort_by: str = "updated_at",
        reverse: bool = True,
    ) -> List[Dict[str, Any]]:
        """List sessions with normalized metadata.

        Args:
            limit: Max number of sessions to return (None = all)
            sort_by: Field to sort by: 'updated_at' | 'created_at' | 'title' | 'model'
            reverse: Sort descending when True
        """
        items: List[Dict[str, Any]] = []
        for p in self._iter_session_files():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items.append(self._normalize_session(data))
            except Exception:
                # Skip unreadable/corrupted files silently (listing is best-effort)
                continue

        key = (
            (lambda s: s.get("created_at", ""))
            if sort_by == "created_at"
            else (lambda s: s.get("title", ""))
            if sort_by == "title"
            else (lambda s: s.get("model", ""))
            if sort_by == "model"
            else (lambda s: s.get("updated_at", ""))
        )
        try:
            items.sort(key=key, reverse=reverse)
        except Exception:
            pass

        return items if limit in (None, 0) else items[: int(limit)]

    def find(self, partial_id: str) -> str:
        """Resolve a partial ID (substring or prefix) to a full session_id.

        Raises ValueError if none or multiple matches are found.
        """
        partial = (partial_id or "").strip()
        if not partial:
            raise ValueError("Empty session id/prefix")
        matches: List[str] = []
        for p in self._iter_session_files():
            sid = p.stem
            if partial in sid:
                matches.append(sid)
        if not matches:
            raise ValueError(f"Session not found: {partial_id}")
        if len(matches) > 1:
            # Prefer unique prefix of at least 6 chars
            exact_prefix = [m for m in matches if m.startswith(partial)]
            if len(exact_prefix) == 1:
                return exact_prefix[0]
            raise ValueError(
                "Ambiguous session prefix; matches: "
                + ", ".join(m[:8] for m in matches[:5])
            )
        return matches[0]

    def delete(self, session_id: str, keep_transcript: bool = False) -> None:
        """Delete the JSON session file and (optionally) its transcript file."""
        # Load to fetch transcript path; ignore if load fails
        transcript_path = None
        try:
            data = self.load(session_id)
            transcript_path = data.get("markdown_path")
        except Exception:
            pass
        # Remove JSON
        try:
            self._path(session_id).unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Remove transcript
        if transcript_path and not keep_transcript:
            try:
                Path(transcript_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass


session_store = SessionStore()


def _initialize_cli_proxy_config(model: str) -> Tuple[str, str, str]:
    """Initialize config for cli-proxy-api models.

    Args:
        model: Model string starting with 'cli-proxy-api/'

    Returns:
        Tuple of (litellm_model, base_url, api_key)

    Raises:
        ValueError: If cli-proxy-api is unavailable or fails to start
    """
    # Ensure cli-proxy-api is available
    if not cli_proxy_config.is_cli_proxy_available():
        raise ValueError(
            "cli-proxy-api binary not found on PATH. "
            "Install from: https://github.com/router-for-me/CLIProxyAPI"
        )

    # Find config
    config_path = cli_proxy_config.find_cli_proxy_config()
    if not config_path:
        raise ValueError(
            "No cli-proxy-api config found. "
            "Set one with: searxng cli-proxy-api set-config /path/to/config.yaml"
        )

    # Start/get manager
    manager = CLIProxyAPIManager.get_instance()
    if not manager.ensure_running(config_path):
        # Check if process is still running (might be waiting for OAuth)
        if manager.is_running():
            raise ValueError(
                "cli-proxy-api started but not responding. "
                "It may be waiting for OAuth authentication. "
                f"Run 'cli-proxy-api -config {config_path}' manually to complete OAuth setup."
            )
        raise ValueError(
            "Failed to start cli-proxy-api. "
            "Check the config file and try: searxng cli-proxy-api start"
        )

    # Extract actual model name (remove prefix)
    # Handle models that may contain / like "moonshotai/kimi-k2:free"
    actual_model = model[len("cli-proxy-api/") :]

    # For litellm, use openai/ prefix with custom base_url
    # This tells litellm to use OpenAI-compatible API format
    litellm_model = f"openai/{actual_model}"

    # CRITICAL: base_url WITH /v1 - litellm needs the full path
    base_url = f"{manager.get_base_url()}/v1"

    # Return dummy api_key - proxy handles actual auth
    return litellm_model, base_url, "cli-proxy-api-managed"


def initialize_ai_config(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    profile: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Initialize AI configuration from profiles, CLI Proxy API, and environment.

    This is the single source of truth for AI configuration across:
    - CLI commands (ask, chat)
    - MCP server (stdio and HTTP)
    - Library interface

    Args:
        model: Explicit model override (use 'cli-proxy-api/model-name' for proxy models)
        base_url: Explicit base URL override
        profile: Specific profile name to use (None = use default)

    Returns:
        Tuple of (model, base_url, api_key) configured and ready to use.
        api_key is None for standard providers (uses env vars), or a dummy
        key for cli-proxy-api models.
    """
    import os

    # Check if model is a cli-proxy-api model
    if model and model.startswith("cli-proxy-api/"):
        return _initialize_cli_proxy_config(model)

    # If no model specified, check for global default model first
    if not model:
        global_default = global_config.get_default_model()
        if global_default:
            model = global_default

    # Re-check cli-proxy-api model after global default
    if model and model.startswith("cli-proxy-api/"):
        return _initialize_cli_proxy_config(model)

    # If still no model specified, check for cli-proxy-api default model
    if not model and cli_proxy_config.is_enabled():
        default_model = cli_proxy_config.get_default_model()
        if default_model:
            return _initialize_cli_proxy_config(f"cli-proxy-api/{default_model}")

    # Load profile configuration
    profile_config = None
    if profile:
        profile_config = profile_manager.get_profile(profile)
        if not profile_config:
            raise ValueError(f"Profile '{profile}' not found")
    elif not model and not base_url:
        # No explicit overrides, try default profile
        profile_config = profile_manager.get_profile()  # Gets default profile

    # Apply profile configuration if available
    if profile_config:
        # Set model from profile if not explicitly provided
        if not model:
            model = profile_config.get("default_model")

        # Set base_url from profile if not explicitly provided
        if not base_url and profile_config.get("base_url"):
            base_url = profile_config["base_url"]

        # Set API key environment variable based on provider type
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        env_key = env_map.get(profile_config.get("type"))
        if env_key and not os.environ.get(env_key):
            os.environ[env_key] = profile_config["api_key"]

    # Set default model if still not set
    if not model:
        model = "openai/gpt-5"

    # Return None for api_key - litellm will use env vars
    return model, base_url, None


class CLISearch:
    """Custom search class that works without Flask context."""

    def __init__(self, search_query: SearchQuery):
        self.search_query = search_query
        self.result_container = ResultContainer()
        self.start_time = None
        self.actual_timeout = None

    def _get_requests(self):
        """Get search requests for all selected engines."""
        requests = []
        default_timeout = 0

        for engineref in self.search_query.engineref_list:
            if engineref.name not in PROCESSORS:
                continue

            processor = PROCESSORS[engineref.name]

            # Skip suspended engines
            if processor.extend_container_if_suspended(self.result_container):
                continue

            # Get request parameters
            request_params = processor.get_params(self.search_query, engineref.category)
            if request_params is None:
                continue

            requests.append((engineref.name, self.search_query.query, request_params))
            default_timeout = max(default_timeout, processor.engine.timeout)

        # Set timeout
        max_request_timeout = searx.settings["outgoing"]["max_request_timeout"]
        actual_timeout = default_timeout
        query_timeout = self.search_query.timeout_limit

        if max_request_timeout is None and query_timeout is None:
            pass
        elif max_request_timeout is None and query_timeout is not None:
            actual_timeout = min(default_timeout, query_timeout)
        elif max_request_timeout is not None and query_timeout is None:
            actual_timeout = min(default_timeout, max_request_timeout)
        elif max_request_timeout is not None and query_timeout is not None:
            actual_timeout = min(query_timeout, max_request_timeout)

        return requests, actual_timeout

    def search_multiple_requests(self, requests):
        """Execute multiple search requests in parallel."""
        search_id = str(uuid4())

        for engine_name, query, request_params in requests:
            processor = PROCESSORS[engine_name]

            def _search_wrapper():
                try:
                    processor.search(
                        query,
                        request_params,
                        self.result_container,
                        self.start_time,
                        self.actual_timeout,
                    )
                except Exception as e:
                    self.result_container.add_unresponsive_engine(engine_name, str(e))

            th = threading.Thread(target=_search_wrapper, name=search_id)
            th._timeout = False
            th._engine_name = engine_name
            th.start()

        # Wait for all threads to complete
        for th in threading.enumerate():
            if th.name == search_id:
                remaining_time = max(
                    0.0, self.actual_timeout - (default_timer() - self.start_time)
                )
                th.join(remaining_time)
                if th.is_alive():
                    th._timeout = True
                    self.result_container.add_unresponsive_engine(
                        th._engine_name, "timeout"
                    )

    def search(self) -> ResultContainer:
        """Perform the search."""
        self.start_time = default_timer()

        requests, self.actual_timeout = self._get_requests()

        if requests:
            self.search_multiple_requests(requests)

        return self.result_container


def json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf8")
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type ({type(obj)}) not serializable")


def initialize_searx():
    """Initialize SearXNG search system."""
    try:
        # Use the unified initialize() function (SearXNG API changed upstream)
        searx.search.initialize()
        return True
    except Exception as e:
        console.print(f"[red]Error initializing SearXNG: {e}[/red]")
        return False


def get_available_engines() -> Dict[str, List[str]]:
    """Get available engines organized by category."""
    engines_by_category = {}
    for name, engine in searx.engines.engines.items():
        for category in engine.categories:
            if category not in engines_by_category:
                engines_by_category[category] = []
            engines_by_category[category].append(name)
    return engines_by_category


def create_search_query(
    query: str,
    category: str = "general",
    engines: Optional[List[str]] = None,
    lang: str = "all",
    safe_search: int = 0,
    page: int = 1,
    time_range: Optional[str] = None,
) -> SearchQuery:
    """Create a SearchQuery object."""

    # Get available engines for the category
    available_engines = get_available_engines()
    category_engines = available_engines.get(category, [])

    # Use specified engines or default engines for the category
    if engines:
        # Filter to only include engines that exist and are in the category
        target_engines = [e for e in engines if e in category_engines]
        if not target_engines:
            # Fallback to default engines if none of the specified engines are available
            target_engines = [
                e
                for e in DEFAULT_ENGINES.get(category, category_engines[:3])
                if e in category_engines
            ]
    else:
        # Use default engines for the category
        target_engines = [
            e
            for e in DEFAULT_ENGINES.get(category, category_engines[:3])
            if e in category_engines
        ]

    # If still no engines, use first 3 available engines in category
    if not target_engines and category_engines:
        target_engines = category_engines[:3]

    # Create EngineRef objects
    engine_refs = [EngineRef(name, category) for name in target_engines]

    return SearchQuery(
        query=query,
        engineref_list=engine_refs,
        lang=lang,
        safesearch=safe_search,
        pageno=page,
        time_range=time_range,
    )


def format_results_human(results_dict: Dict[str, Any]) -> None:
    """Format and display search results in human-readable format."""
    search_info = results_dict["search"]
    results = results_dict["results"]

    # Display search info
    console.print(f"\n[bold blue]Search:[/bold blue] {search_info['q']}")

    # Show which engines were used
    engines_used = set()
    for result in results:
        if result.get("engine"):
            engines_used.add(result["engine"])

    if engines_used:
        engines_str = ", ".join(sorted(engines_used))
        console.print(f"[green]Engines:[/green] {engines_str}")

    console.print(
        f"[dim]Language: {search_info['lang']}, Page: {search_info['pageno']}, Results: {len(results)}[/dim]\n"
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Display results
    for i, result in enumerate(results[:10], 1):  # Show top 10 results
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [bold]{result.get('title', 'No title')}[/bold]"
        )
        if result.get("content"):
            # Truncate content to reasonable length
            content = (
                result["content"][:200] + "..."
                if len(result["content"]) > 200
                else result["content"]
            )
            console.print(f"   {content}")
        console.print(f"   [link]{result.get('url', 'No URL')}[/link]")
        if result.get("engine"):
            console.print(f"   [dim]Source: {result['engine']}[/dim]")
        console.print()

    # Display suggestions if any
    if results_dict.get("suggestions"):
        console.print("[bold yellow]Suggestions:[/bold yellow]")
        for suggestion in results_dict["suggestions"][:5]:
            console.print(f"  • {suggestion}")
        console.print()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    category: str = typer.Option("general", "--category", "-c", help="Search category"),
    output_format: str = typer.Option(
        "human", "--format", "-f", help="Output format: human or json"
    ),
    engines: Optional[str] = typer.Option(
        None, "--engines", "-e", help="Comma-separated list of engines to use"
    ),
    disable_engines: Optional[str] = typer.Option(
        None, "--disable", "-d", help="Comma-separated list of engines to disable"
    ),
    language: str = typer.Option("all", "--lang", "-l", help="Search language"),
    safe_search: int = typer.Option(
        0, "--safe", "-s", help="Safe search level (0=off, 1=moderate, 2=strict)"
    ),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    time_range: Optional[str] = typer.Option(
        None, "--time", "-t", help="Time range: day, week, month, year"
    ),
):
    """Search using SearXNG engines."""

    if not initialize_searx():
        raise typer.Exit(1)

    # Parse engines
    target_engines = None
    if engines:
        target_engines = [e.strip() for e in engines.split(",")]
    elif disable_engines:
        # Get all engines for category and remove disabled ones
        available_engines = get_available_engines()
        category_engines = available_engines.get(category, [])
        disabled = [e.strip() for e in disable_engines.split(",")]
        target_engines = [e for e in category_engines if e not in disabled]

    try:
        # Create search query
        search_query = create_search_query(
            query=query,
            category=category,
            engines=target_engines,
            lang=language,
            safe_search=safe_search,
            page=page,
            time_range=time_range,
        )

        # Show which engines will be used (for human-readable output)
        if output_format.lower() != "json":
            engine_names = [ref.name for ref in search_query.engineref_list]
            if engine_names:
                console.print(
                    f"[dim]Using engines: {', '.join(sorted(engine_names))}[/dim]"
                )

        # Perform search
        result_container = CLISearch(search_query).search()

        # Prepare results
        results_dict = {
            "search": {
                "q": search_query.query,
                "pageno": search_query.pageno,
                "lang": search_query.lang,
                "safesearch": search_query.safesearch,
                "timerange": search_query.time_range,
                "category": category,
            },
            "results": result_container.get_ordered_results(),
            "infoboxes": result_container.infoboxes,
            "suggestions": list(result_container.suggestions),
            "answers": list(result_container.answers),
            "number_of_results": result_container.number_of_results,
        }

        # Remove parsed_url from results for cleaner output
        for result in results_dict["results"]:
            if "parsed_url" in result:
                del result["parsed_url"]

        # Output results
        if output_format.lower() == "json":
            print(
                json.dumps(
                    results_dict, indent=2, ensure_ascii=False, default=json_serial
                )
            )
        else:
            format_results_human(results_dict)

    except Exception as e:
        console.print(f"[red]Error performing search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def engines(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    show_common: bool = typer.Option(
        False, "--common", help="Show only common engines"
    ),
):
    """List available search engines."""

    if not initialize_searx():
        raise typer.Exit(1)

    engines_by_category = get_available_engines()

    if category:
        if category not in engines_by_category:
            console.print(f"[red]Category '{category}' not found.[/red]")
            console.print(
                f"Available categories: {', '.join(sorted(engines_by_category.keys()))}"
            )
            raise typer.Exit(1)

        console.print(f"\n[bold blue]Engines in '{category}' category:[/bold blue]")
        engines_list = engines_by_category[category]
        if show_common:
            engines_list = [e for e in engines_list if e in COMMON_ENGINES]

        # Display engines in columns for better readability
        engines_list = sorted(engines_list)
        for i in range(0, len(engines_list), 3):
            row_engines = engines_list[i : i + 3]
            formatted_engines = []
            for engine in row_engines:
                marker = "⭐" if engine in COMMON_ENGINES else " "
                formatted_engines.append(f"{marker} {engine}")
            console.print("  ".join(f"{engine:<25}" for engine in formatted_engines))
        console.print()
    else:
        # Show all engines in a compact format
        if show_common:
            console.print("\n[bold blue]Common Search Engines:[/bold blue]")
            all_engines = []
            for engines_list in engines_by_category.values():
                all_engines.extend([e for e in engines_list if e in COMMON_ENGINES])
            all_engines = sorted(set(all_engines))

            # Display in 4 columns
            for i in range(0, len(all_engines), 4):
                row_engines = all_engines[i : i + 4]
                console.print("  ".join(f"⭐ {engine:<20}" for engine in row_engines))
        else:
            console.print("\n[bold blue]Search Engines by Category:[/bold blue]")
            for cat in sorted(engines_by_category.keys()):
                engines_list = sorted(engines_by_category[cat])
                common_count = len([e for e in engines_list if e in COMMON_ENGINES])
                total_count = len(engines_list)

                if common_count > 0:
                    console.print(
                        f"[cyan]{cat:<18}[/cyan] {total_count} engines ({common_count} common)"
                    )
                else:
                    console.print(f"[dim]{cat:<18}[/dim] {total_count} engines")

        console.print(
            f"\n[dim]Use --category <name> to see engines in a specific category[/dim]"
        )
        console.print(f"[dim]Use --common to show only popular engines[/dim]")


@app.command()
def categories():
    """List available search categories."""

    if not initialize_searx():
        raise typer.Exit(1)

    engines_by_category = get_available_engines()

    console.print("\n[bold blue]Available Categories:[/bold blue]")
    for category in sorted(engines_by_category.keys()):
        engine_count = len(engines_by_category[category])
        console.print(f"  • [cyan]{category}[/cyan] ({engine_count} engines)")
    console.print()


# MCP Server Implementation
async def fetch_url_content(url: str) -> Dict[str, Any]:
    """Fetch URL content using Jina.ai's reader service."""
    # Properly encode the URL to handle special characters and parameters
    encoded_url = quote(url, safe="")
    jina_url = f"https://r.jina.ai/{encoded_url}"

    # Use JINA_API_KEY environment variable if available
    headers = {}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()

            # Try to parse as JSON first
            try:
                content_data = response.json()
                return {
                    "success": True,
                    "title": content_data.get("title", ""),
                    "content": content_data.get("content", ""),
                    "url": content_data.get("url", url),
                    "timestamp": content_data.get("timestamp", ""),
                }
            except json.JSONDecodeError:
                # If not JSON, return as plain text
                return {
                    "success": True,
                    "title": "",
                    "content": response.text,
                    "url": url,
                    "timestamp": "",
                }
    except httpx.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching URL: {str(e)}",
            "url": url,
        }


async def fetch_multiple_urls_async(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple URLs in parallel using existing network infrastructure, preserving order."""
    if not urls:
        return []

    # Import here to avoid circular imports
    from searx.network import Request, multi_requests

    # Use JINA_API_KEY environment variable if available
    headers = {}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"

    # Create requests for all URLs
    requests = []
    for url in urls:
        # Properly encode the URL to handle special characters and parameters
        encoded_url = quote(url, safe="")
        jina_url = f"https://r.jina.ai/{encoded_url}"
        requests.append(Request.get(jina_url, headers=headers, timeout=30.0))

    # Execute requests in parallel
    responses = multi_requests(requests)

    # Process responses while preserving order
    results = []
    for i, (url, response) in enumerate(zip(urls, responses)):
        if isinstance(response, Exception):
            # Handle errors
            results.append(
                {
                    "success": False,
                    "error": f"Error fetching URL: {str(response)}",
                    "url": url,
                    "index": i,
                }
            )
        else:
            try:
                # Check if request was successful
                response.raise_for_status()

                # Try to parse as JSON first
                try:
                    content_data = response.json()
                    results.append(
                        {
                            "success": True,
                            "title": content_data.get("title", ""),
                            "content": content_data.get("content", ""),
                            "url": content_data.get("url", url),
                            "timestamp": content_data.get("timestamp", ""),
                            "index": i,
                        }
                    )
                except json.JSONDecodeError:
                    # If not JSON, return as plain text
                    results.append(
                        {
                            "success": True,
                            "title": "",
                            "content": response.text,
                            "url": url,
                            "timestamp": "",
                            "index": i,
                        }
                    )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "error": f"HTTP error: {str(e)}",
                        "url": url,
                        "index": i,
                    }
                )

    return results


async def perform_search_async(
    query: str,
    category: str = "general",
    engines: Optional[List[str]] = None,
    language: str = "all",
    safe_search: int = 0,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Async wrapper for search functionality."""
    # Initialize SearXNG if not already done
    if not initialize_searx():
        return {"error": "Failed to initialize SearXNG"}

    try:
        # Create search query
        search_query = create_search_query(
            query=query,
            category=category,
            engines=engines,
            lang=language,
            safe_search=safe_search,
            page=1,
        )

        # Perform search
        result_container = CLISearch(search_query).search()

        # Get results
        results = result_container.get_ordered_results()[:max_results]

        # Clean up results for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "engine": result.get("engine", ""),
            }
            clean_results.append(clean_result)

        return {
            "success": True,
            "query": query,
            "category": category,
            "engines_used": [ref.name for ref in search_query.engineref_list],
            "results": clean_results,
            "suggestions": list(result_container.suggestions),
            "number_of_results": result_container.number_of_results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Search error: {str(e)}",
            "query": query,
        }


async def perform_multi_search_async(
    queries: List[str],
    category: str = "general",
    engines: Optional[List[str]] = None,
    language: str = "all",
    safe_search: int = 0,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Perform multiple search queries in parallel."""
    if not queries:
        return {"error": "No queries provided"}

    # Create tasks for parallel execution
    tasks = []
    for query in queries:
        task = perform_search_async(
            query=query,
            category=category,
            engines=engines,
            language=language,
            safe_search=safe_search,
            max_results=max_results,
        )
        tasks.append(task)

    try:
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        search_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                search_results.append(
                    {
                        "success": False,
                        "query": queries[i],
                        "error": f"Search failed: {str(result)}",
                    }
                )
            else:
                search_results.append(result)

        # Create a more structured response that clearly maps queries to results
        structured_results = []
        for i, (query, result) in enumerate(zip(queries, search_results)):
            structured_result = {
                "query_index": i + 1,
                "query": query,
                "search_result": result,
            }
            structured_results.append(structured_result)

        return {
            "success": True,
            "summary": {
                "total_queries": len(queries),
                "successful_queries": sum(
                    1 for r in search_results if r.get("success", False)
                ),
                "queries_executed": queries,
            },
            "results": structured_results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Multi-search error: {str(e)}",
            "queries": queries,
        }


@app.command("fetch-urls")
def fetch_urls_command(
    urls: List[str] = typer.Argument(..., help="URLs to fetch content from"),
    output_format: str = typer.Option(
        "human", "--format", "-f", help="Output format: human or json"
    ),
    max_concurrent: int = typer.Option(
        5, "--concurrent", "-c", help="Maximum number of concurrent requests"
    ),
):
    """Fetch content from multiple URLs in parallel using Jina.ai's reader service.

    This command fetches content from one or more URLs concurrently, maintaining the
    order of results to match the input order. Each URL is processed through Jina.ai's
    reader service to extract clean, readable content.

    Examples:

    # Fetch single URL
    searxng-cli fetch-urls "https://example.com"

    # Fetch multiple URLs in parallel
    searxng-cli fetch-urls "https://site1.com" "https://site2.com" "https://site3.com"

    # Output in human-readable format
    searxng-cli fetch-urls "https://example.com" --format human

    # Control concurrency
    searxng-cli fetch-urls url1 url2 url3 --concurrent 2

    Environment variables:
    - JINA_API_KEY: Optional API key for enhanced Jina.ai features
    """
    if not urls:
        console.print("[red]Error: At least one URL is required[/red]")
        raise typer.Exit(1)

    # Limit concurrent requests to reasonable number
    if max_concurrent > 20:
        console.print(
            "[yellow]Warning: Limiting concurrent requests to 20 for stability[/yellow]"
        )
        max_concurrent = 20

    try:
        if len(urls) == 1:
            # Single URL - use existing async function but run it
            import asyncio

            result = asyncio.run(fetch_url_content(urls[0]))
            results = [result]
        else:
            # Multiple URLs - use parallel fetching
            import asyncio

            results = asyncio.run(fetch_multiple_urls_async(urls))

        if output_format.lower() == "json":
            # JSON output
            output = json.dumps(
                results, indent=2, ensure_ascii=False, default=json_serial
            )
            console.print(output)
        else:
            # Human-readable output
            for i, result in enumerate(results):
                if result.get("success"):
                    console.print(
                        f"\n[bold green]✓ URL {i + 1}:[/bold green] {result['url']}"
                    )
                    if result.get("title"):
                        console.print(f"[bold]Title:[/bold] {result['title']}")
                    if result.get("content"):
                        console.print(f"[bold]Content:[/bold] {result['content']}")
                    if result.get("timestamp"):
                        console.print(f"[dim]Timestamp: {result['timestamp']}[/dim]")
                else:
                    console.print(
                        f"\n[bold red]✗ URL {i + 1}:[/bold red] {result['url']}"
                    )
                    console.print(
                        f"[red]Error: {result.get('error', 'Unknown error')}[/red]"
                    )

                if i < len(results) - 1:  # Add separator between results
                    console.print("[dim]" + "─" * 50 + "[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error fetching URLs: {e}[/red]")
        raise typer.Exit(1)


def get_mcp_tools():
    """Get the list of MCP tools available."""
    from mcp.types import Tool

    return [
        Tool(
            name="web_search",
            description="Search the web using SearXNG's search engines",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "category": {
                        "type": "string",
                        "description": "Search category (general, images, videos, news, science, it)",
                        "default": "general",
                    },
                    "engines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search engines to use",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language",
                        "default": "all",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="chat",
            description="Send a chat message with optional sessionId to continue a conversation thread. IMPORTANT: This tool returns a sessionId in the response - you should store and reuse this sessionId for subsequent messages in the same conversation to maintain context and conversation history. When starting a new conversation, omit sessionId. When continuing an existing conversation, always pass the sessionId from the previous response.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "User message to send",
                    },
                    "sessionId": {
                        "type": "string",
                        "description": "Existing session ID to continue the same conversation (obtained from previous chat responses). Omit for new conversations, include for continuing existing conversations.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: from profile or openai/gpt-5)",
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Custom API base URL (optional, overrides profile setting)",
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile name to use for API keys and config (default: uses default profile)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum number of tool calling iterations (default: 200)",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="multi_web_search",
            description="Search the web with multiple queries in parallel using SearXNG",
            inputSchema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of search queries to execute in parallel",
                    },
                    "category": {
                        "type": "string",
                        "description": "Search category (general, images, videos, news, science, it)",
                        "default": "general",
                    },
                    "engines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search engines to use",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return per query",
                        "default": 10,
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language",
                        "default": "all",
                    },
                },
                "required": ["queries"],
            },
        ),
        Tool(
            name="fetch_url",
            description="Fetch and extract content from a single URL using Jina.ai's reader service.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch content from",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="fetch_urls",
            description="Fetch and extract content from multiple URLs in parallel using Jina.ai's reader service.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of URLs to fetch content from in parallel",
                    },
                },
                "required": ["urls"],
            },
        ),
        Tool(
            name="ask",
            description="Ask an AI assistant with access to web search and URL fetching tools. The assistant can run parallel searches and fetch content from multiple URLs to provide comprehensive research and answers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Question or research request to ask the AI assistant",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: from profile or openai/gpt-5)",
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Custom API base URL (optional, overrides profile setting)",
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile name to use for API keys and config (default: uses default profile)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum number of tool calling iterations (default: 200)",
                    },
                },
                "required": ["prompt"],
            },
        ),
    ]


async def ask_ai_async(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    profile: Optional[str] = None,
    max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Core async function for asking AI with web search tools.
    This is used by both the CLI command, MCP server, and library interface.

    Args:
        prompt: The question or research request
        model: Optional model override (defaults to profile or "openai/gpt-5")
        base_url: Optional base URL override
        profile: Optional profile name to use (defaults to default profile)
        max_iterations: Maximum number of tool calling iterations (default: 200)
    """
    import litellm
    import os
    import sys

    # Initialize configuration from profiles or CLI Proxy API
    try:
        model, base_url, api_key = initialize_ai_config(model, base_url, profile)
    except ValueError as e:
        # CLI Proxy API or profile errors
        return {
            "success": False,
            "error": str(e),
            "prompt": prompt,
            "model": model or "unknown",
        }

    # Check for API keys after profile initialization (skip if using cli-proxy-api)
    if api_key is None:  # Standard provider, check env vars
        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        }

        # Check if we have at least one API key
        if not any(api_keys.values()):
            return {
                "success": False,
                "error": "No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY",
                "prompt": prompt,
                "model": model,
            }

    # Define tools for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using SearXNG's search engines",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {
                            "type": "string",
                            "description": "Search category",
                            "default": "general",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multi_web_search",
                "description": "Search the web with multiple queries in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of search queries",
                        },
                        "category": {
                            "type": "string",
                            "description": "Search category",
                            "default": "general",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results per query",
                            "default": 10,
                        },
                    },
                    "required": ["queries"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch and extract content from a single URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_urls",
                "description": "Fetch and extract content from multiple URLs in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs to fetch",
                        }
                    },
                    "required": ["urls"],
                },
            },
        },
    ]

    try:
        # Use base system prompt with ask-specific additions
        system_content = (
            BASE_AI_SYSTEM_PROMPT
            + """\n\n**ASK MODE - COMPREHENSIVE RESEARCH:**
- Provide thorough, well-researched responses with comprehensive coverage
- Use extensive parallel searches to gather complete information
- Include relevant data, statistics, examples, and expert perspectives
- Cite sources and provide context for your findings
- Aim for depth and completeness in your analysis"""
        )

        user_prompt = f"User request: {prompt}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        # Prepare completion arguments
        completion_args = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        # Add api_base if provided (overrides environment variable)
        # Note: litellm uses api_base, not base_url, for OpenAI-compatible endpoints
        if base_url:
            completion_args["api_base"] = base_url

        # Add api_key if provided (for cli-proxy-api, pass per-call not via env)
        if api_key:
            completion_args["api_key"] = api_key

        # Make initial request to the LLM
        response = litellm.completion(**completion_args)

        # Handle tool calls iteratively with limit
        iteration_count = 0
        while (
            response.choices[0].message.tool_calls and iteration_count < max_iterations
        ):
            iteration_count += 1
            # Add the assistant's message with tool calls
            messages.append(response.choices[0].message.model_dump())

            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Log tool usage to stderr for shell piping friendliness
                stderr_console = Console(file=sys.stderr, force_terminal=True)
                if function_name == "web_search":
                    stderr_console.print(
                        f"🔍 [cyan]Searching:[/cyan] {function_args.get('query', 'N/A')}"
                    )
                elif function_name == "multi_web_search":
                    queries = function_args.get("queries", [])
                    stderr_console.print(
                        f"🔍 [cyan]Multi-search:[/cyan] {', '.join(queries[:3])}{'...' if len(queries) > 3 else ''}"
                    )
                elif function_name == "fetch_url":
                    stderr_console.print(
                        f"📄 [blue]Fetching:[/blue] {function_args.get('url', 'N/A')}"
                    )
                elif function_name == "fetch_urls":
                    urls = function_args.get("urls", [])
                    stderr_console.print(
                        f"📄 [blue]Fetching {len(urls)} URLs:[/blue] {urls[0] if urls else 'N/A'}{'...' if len(urls) > 1 else ''}"
                    )
                else:
                    stderr_console.print(f"🔧 [magenta]Tool:[/magenta] {function_name}")

                # Execute the tool using our existing handler
                tool_result = await handle_tool_call(function_name, function_args)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                    }
                )

            # Get next response from LLM
            response = litellm.completion(**completion_args)

        # Return the final response
        final_response = response.choices[0].message.content
        return {
            "success": True,
            "model": model,
            "prompt": prompt,
            "response": final_response,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error calling {model}: {str(e)}",
            "prompt": prompt,
            "model": model,
        }


async def handle_tool_call(name: str, arguments: dict):
    """Shared tool call handler for both stdio and HTTP versions."""
    if name == "web_search":
        query = arguments.get("query")
        category = arguments.get("category", "general")
        engines = arguments.get("engines")
        max_results = arguments.get("max_results", 10)
        language = arguments.get("language", "all")

        result = await perform_search_async(
            query=query,
            category=category,
            engines=engines,
            language=language,
            max_results=max_results,
        )

        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)

    elif name == "chat":
        # Conversational, resumable chat via session store
        message = arguments.get("message")
        if not message:
            return json.dumps({"error": "message is required"})

        provided_session_id = arguments.get("sessionId")
        model = arguments.get("model")
        base_url = arguments.get("base_url")
        profile = arguments.get("profile")
        max_iterations = arguments.get("max_iterations", 200)

        try:
            # Initialize config early to get defaults and env set
            resolved_model, resolved_base, resolved_api_key = initialize_ai_config(
                model, base_url, profile
            )

            if provided_session_id:
                session = session_store.load(provided_session_id)
                session_id = session["session_id"]
            else:
                session = session_store.create(
                    model=resolved_model,
                    base_url=resolved_base,
                    profile=profile,
                    title="",
                )
                session_id = session["session_id"]

            # Prepare messages: only keep role/content for LLM
            hist = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in session.get("messages", [])
                if m.get("role") and m.get("content") is not None
            ]

            # Append the new user message for this turn
            hist.append({"role": "user", "content": message})

            # Call conversational async
            result = await ask_ai_conversational_async(
                messages=hist,
                model=resolved_model,
                base_url=resolved_base,
                profile=profile,
                max_iterations=max_iterations,
            )

            if result.get("success"):
                # Persist updated messages back to the session
                session["model"] = result.get("model", resolved_model)
                session["messages"] = result.get("messages", hist)
                session_store.save(session)

                return json.dumps(
                    {
                        "success": True,
                        "sessionId": session_id,
                        "model": session["model"],
                        "response": result.get("response", ""),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            else:
                return json.dumps(
                    {
                        "success": False,
                        "sessionId": session_id,
                        "error": result.get("error", "Unknown error"),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
        except FileNotFoundError:
            return json.dumps({"error": f"Session not found: {provided_session_id}"})
        except Exception as e:
            return json.dumps({"error": f"Chat error: {str(e)}"})

    elif name == "multi_web_search":
        queries = arguments.get("queries")
        if not queries:
            return json.dumps({"error": "Queries array is required"})

        if not isinstance(queries, list):
            return json.dumps({"error": "Queries must be an array"})

        if not queries:
            return json.dumps({"error": "Queries array cannot be empty"})

        category = arguments.get("category", "general")
        engines = arguments.get("engines")
        max_results = arguments.get("max_results", 10)
        language = arguments.get("language", "all")

        result = await perform_multi_search_async(
            queries=queries,
            category=category,
            engines=engines,
            language=language,
            max_results=max_results,
        )

        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)

    elif name == "fetch_url":
        url = arguments.get("url")
        if not url:
            return json.dumps({"error": "URL is required"})

        # Single URL
        result = await fetch_url_content(url)
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)

    elif name == "fetch_urls":
        urls = arguments.get("urls")
        if not urls:
            return json.dumps({"error": "URLs array is required"})

        if not isinstance(urls, list):
            return json.dumps({"error": "URLs must be an array"})

        if not urls:
            return json.dumps({"error": "URLs array cannot be empty"})

        # Multiple URLs - use parallel fetching
        result = await fetch_multiple_urls_async(urls)
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)

    elif name == "ask":
        prompt = arguments.get("prompt")
        if not prompt:
            return json.dumps({"error": "Prompt is required"})

        model = arguments.get("model")  # Will use profile default if not specified
        base_url = arguments.get("base_url")  # Optional custom base URL
        profile = arguments.get("profile")  # Optional profile name

        # Use the shared ask_ai_async function with profile support
        max_iterations = arguments.get("max_iterations", 200)
        result = await ask_ai_async(prompt, model, base_url, profile, max_iterations)
        return json.dumps(result, indent=2, ensure_ascii=False, default=json_serial)

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


@app.command()
def models(
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Refresh model cache"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter by source: cli-proxy-api, profile"
    ),
):
    """List available AI models from all sources.

    Shows models available through:
    - CLI Proxy API (if enabled and configured)
    - Configured profiles
    """
    all_models = []
    model_sources = {}  # Map model -> source

    # Get CLI Proxy API models if enabled and available
    if cli_proxy_config.is_enabled() and cli_proxy_config.is_cli_proxy_available():
        config_path = cli_proxy_config.find_cli_proxy_config()
        if config_path:
            manager = CLIProxyAPIManager.get_instance()
            # Start proxy if not running
            if manager.ensure_running(config_path):
                proxy_models = manager.get_models(force_refresh=refresh)
                for m in proxy_models:
                    prefixed = f"cli-proxy-api/{m}"
                    all_models.append(prefixed)
                    model_sources[prefixed] = "CLI Proxy API"

    # Get profile-based models (show default model from each profile)
    data = profile_manager._load_profiles()
    for name, profile in data.get("profiles", {}).items():
        default_model = profile.get("default_model")
        if default_model:
            all_models.append(default_model)
            model_sources[default_model] = f"Profile: {name}"

    # Filter by source if requested
    if source:
        source_lower = source.lower()
        if source_lower == "cli-proxy-api":
            all_models = [m for m in all_models if m.startswith("cli-proxy-api/")]
        elif source_lower == "profile":
            all_models = [m for m in all_models if not m.startswith("cli-proxy-api/")]

    # Output
    if json_output:
        output = {
            "models": [
                {"id": m, "source": model_sources.get(m, "Unknown")}
                for m in sorted(all_models)
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        if not all_models:
            console.print("[yellow]No models available.[/yellow]")
            console.print()
            console.print("[dim]Tips:[/dim]")
            console.print(
                "[dim]  - Enable CLI Proxy API: searxng cli-proxy-api enable[/dim]"
            )
            console.print(
                "[dim]  - Add a profile: searxng profile add <name> <api-key>[/dim]"
            )
            return

        # Get default model for indicator
        default_model = None
        if cli_proxy_config.is_enabled():
            dm = cli_proxy_config.get_default_model()
            if dm:
                default_model = f"cli-proxy-api/{dm}"

        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("", style="yellow")  # Default indicator

        for model in sorted(all_models):
            is_default = model == default_model
            table.add_row(
                model,
                model_sources.get(model, "Unknown"),
                "(default)" if is_default else "",
            )

        console.print(table)
        console.print()
        console.print(f"[dim]Total: {len(all_models)} models[/dim]")
        if default_model:
            console.print(f"[dim]Default: {default_model}[/dim]")


@app.command()
def mcp_server(
    remote: bool = typer.Option(
        False, "--remote", help="Start as remote HTTP server instead of stdio"
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", help="Host to bind to (default: 0.0.0.0)"
    ),
    port: int = typer.Option(8000, "--port", help="Port to bind to (default: 8000)"),
    claude_cfg: bool = typer.Option(
        False,
        "--claude-cfg",
        help="Output Claude Code compatible JSON config instead of starting server",
    ),
):
    """Start the MCP server for Model Context Protocol integration.

    This starts an MCP server that communicates over stdio by default, or over HTTP
    when using the --remote flag. Provides web search and URL content retrieval
    tools for AI applications like Claude.

    Available tools:
    - web_search: Search using SearXNG's 180+ search engines
    - fetch_url: Retrieve and extract content from URL(s) using Jina.ai (supports arrays)

    Usage with Claude Desktop (stdio):
    Add this to your claude_desktop_config.json:
    {
      "mcpServers": {
        "searxng": {
          "command": "searxng",
          "args": ["mcp-server"]
        }
      }
    }

    Usage as remote HTTP server:
    searxng mcp-server --remote --host 0.0.0.0 --port 8000

    Then connect MCP clients to: http://your-server:8000

    Environment variables:
    - JINA_API_KEY: Optional API key for enhanced Jina.ai features

    Usage with Claude Code:
    claude --mcp-config "$(searxng mcp-server --claude-cfg)"
    """
    # Handle --claude-cfg flag to output JSON config
    if claude_cfg:
        launch_config = {
            "mcpServers": {
                "searxng": {
                    "type": "stdio",
                    "command": "searxng",
                    "args": ["mcp-server"],
                }
            }
        }
        # Output just the JSON config for Claude Code
        print(json.dumps(launch_config))
        return

    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        import mcp.server.stdio
        import mcp.types
    except ImportError:
        console.print(
            "[red]MCP library not found. Please install with: pip install mcp[/red]"
        )
        raise typer.Exit(1)

    # Create MCP server
    server = Server("searxng-cli")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available MCP tools."""
        return get_mcp_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls for stdio version."""
        content_text = await handle_tool_call(name, arguments)
        return [TextContent(type="text", text=content_text)]

    async def run_server():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    # Choose server type based on remote flag
    if remote:
        console.print(
            f"[blue]Starting MCP remote HTTP server on {host}:{port}...[/blue]"
        )
        asyncio.run(run_http_server(server, host, port))
    else:
        console.print("[blue]Starting MCP server on stdio...[/blue]")
        print("Server is running on stdio. Use Ctrl+C to stop.", file=sys.stderr)

        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            print("\nMCP server stopped.", file=sys.stderr)
        except Exception as e:
            print(f"Error running MCP server: {e}", file=sys.stderr)
            raise typer.Exit(1)


async def run_http_server(mcp_server, host: str, port: int):
    """Run the MCP server over HTTP with JSON-RPC 2.0 and SSE support."""
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import Response, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        from sse_starlette.sse import EventSourceResponse
        import uvicorn
    except ImportError:
        console.print(
            "[red]FastAPI and dependencies not found. Please install with: pip install fastapi uvicorn sse-starlette[/red]"
        )
        raise typer.Exit(1)

    # Store active sessions
    sessions = {}

    app = FastAPI(
        title="SearXNG MCP Server",
        description="Model Context Protocol server for SearXNG search engine",
        version="0.1.0",
    )

    # Add CORS middleware for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def validate_origin(request: Request):
        """Validate Origin header to prevent DNS rebinding attacks."""
        origin = request.headers.get("origin")
        if origin:
            # For localhost/0.0.0.0, allow common variations
            if host in ["127.0.0.1", "localhost", "0.0.0.0"]:
                allowed_origins = [
                    "http://localhost",
                    "http://127.0.0.1",
                    f"http://localhost:{port}",
                    f"http://127.0.0.1:{port}",
                ]
                if not any(origin.startswith(allowed) for allowed in allowed_origins):
                    raise HTTPException(status_code=403, detail="Forbidden origin")
            elif not origin.startswith(f"http://{host}"):
                raise HTTPException(status_code=403, detail="Forbidden origin")

    def create_jsonrpc_response(id: Any, result: Any = None, error: Any = None):
        """Create a JSON-RPC 2.0 response."""
        response = {"jsonrpc": "2.0", "id": id}
        if error:
            response["error"] = error
        else:
            response["result"] = result
        return response

    def create_jsonrpc_error(code: int, message: str, data: Any = None):
        """Create a JSON-RPC 2.0 error object."""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        return error

    @app.get("/")
    async def get_endpoint(request: Request):
        """Handle GET requests for SSE stream or capabilities."""
        validate_origin(request)

        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept:
            # Start SSE stream
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"created": datetime.now()}

            async def event_stream():
                # Send initial endpoint event
                yield {
                    "event": "endpoint",
                    "data": json.dumps(
                        {
                            "method": "notifications/initialized",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {"tools": {"listChanged": True}},
                                "serverInfo": {
                                    "name": "searxng-cli",
                                    "version": "0.1.0",
                                },
                            },
                        }
                    ),
                }

                # Keep connection alive
                try:
                    while True:
                        await asyncio.sleep(30)
                        yield {"event": "ping", "data": ""}
                except asyncio.CancelledError:
                    sessions.pop(session_id, None)
                    return

            return EventSourceResponse(
                event_stream(), headers={"Mcp-Session-Id": session_id}
            )
        else:
            # Return server capabilities
            return {
                "name": "searxng-cli",
                "version": "0.1.0",
                "capabilities": {"tools": {}, "resources": {}},
                "protocolVersion": "2024-11-05",
            }

    @app.post("/register")
    async def register_client(request: Request):
        """Handle OAuth 2.0 dynamic client registration."""
        validate_origin(request)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Simple client registration response
        # In production, you'd validate and store client info
        client_id = str(uuid.uuid4())

        return {
            "client_id": client_id,
            "client_id_issued_at": int(datetime.now().timestamp()),
            "registration_access_token": str(uuid.uuid4()),
            "registration_client_uri": f"http://{host}:{port}/register/{client_id}",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",  # No auth required for this demo
        }

    @app.post("/")
    async def post_endpoint(request: Request):
        """Handle POST requests with JSON-RPC 2.0."""
        validate_origin(request)

        try:
            body = await request.json()
        except Exception:
            return Response(
                content=json.dumps(
                    create_jsonrpc_response(
                        None, error=create_jsonrpc_error(-32700, "Parse error")
                    )
                ),
                media_type="application/json",
                status_code=400,
            )

        # Validate JSON-RPC 2.0 format
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            return Response(
                content=json.dumps(
                    create_jsonrpc_response(
                        body.get("id"),
                        error=create_jsonrpc_error(-32600, "Invalid Request"),
                    )
                ),
                media_type="application/json",
                status_code=400,
            )

        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        # Generate session ID if not present
        session_id = request.headers.get("mcp-session-id")
        if not session_id:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"created": datetime.now()}

        response_headers = {"Mcp-Session-Id": session_id}

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "searxng-cli", "version": "0.1.0"},
                }

            elif method == "tools/list":
                # Use shared tool definitions
                tools = get_mcp_tools()
                result = {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema,
                        }
                        for tool in tools
                    ]
                }

            elif method == "notifications/initialized":
                # Client is notifying that it has initialized
                # For notifications (id is null), we should return 204 No Content
                if request_id is None:
                    return Response(status_code=204, headers=response_headers)
                else:
                    result = {"status": "initialized"}

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name:
                    return Response(
                        content=json.dumps(
                            create_jsonrpc_response(
                                request_id,
                                error=create_jsonrpc_error(
                                    -32602, "Invalid params: missing tool name"
                                ),
                            )
                        ),
                        media_type="application/json",
                        status_code=400,
                        headers=response_headers,
                    )

                # Use shared tool handler
                content_text = await handle_tool_call(tool_name, arguments)

                # Check for error responses
                try:
                    result_data = json.loads(content_text)
                    if (
                        isinstance(result_data, dict)
                        and "error" in result_data
                        and result_data.get("error", "").startswith("Unknown tool:")
                    ):
                        return Response(
                            content=json.dumps(
                                create_jsonrpc_response(
                                    request_id,
                                    error=create_jsonrpc_error(
                                        -32601, result_data["error"]
                                    ),
                                )
                            ),
                            media_type="application/json",
                            status_code=404,
                            headers=response_headers,
                        )
                except json.JSONDecodeError:
                    pass  # Content is not JSON, treat as regular response

                result = {"content": [{"type": "text", "text": content_text}]}

            else:
                return Response(
                    content=json.dumps(
                        create_jsonrpc_response(
                            request_id,
                            error=create_jsonrpc_error(
                                -32601, f"Method not found: {method}"
                            ),
                        )
                    ),
                    media_type="application/json",
                    status_code=404,
                    headers=response_headers,
                )

            return Response(
                content=json.dumps(
                    create_jsonrpc_response(request_id, result),
                    ensure_ascii=False,
                    default=json_serial,
                ),
                media_type="application/json",
                headers=response_headers,
            )

        except Exception as e:
            console.print(f"[red]Error handling request: {e}[/red]")
            return Response(
                content=json.dumps(
                    create_jsonrpc_response(
                        request_id,
                        error=create_jsonrpc_error(-32603, f"Internal error: {str(e)}"),
                    )
                ),
                media_type="application/json",
                status_code=500,
                headers=response_headers,
            )

    # Start server
    print(f"MCP HTTP server starting on {host}:{port}", file=sys.stderr)
    try:
        config = uvicorn.Config(
            app=app, host=host, port=port, log_level="info", access_log=False
        )
        server = uvicorn.Server(config)
        await server.serve()
    except KeyboardInterrupt:
        print("\nMCP HTTP server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Error running MCP HTTP server: {e}", file=sys.stderr)
        raise typer.Exit(1)


@app.command("multi-search")
def multi_search_command(
    queries: List[str] = typer.Argument(
        ..., help="Search queries to execute in parallel"
    ),
    output_format: str = typer.Option(
        "human", "--format", "-f", help="Output format: human or json"
    ),
    category: str = typer.Option("general", "--category", "-c", help="Search category"),
    engines: Optional[List[str]] = typer.Option(
        None, "--engines", "-e", help="Search engines to use"
    ),
    max_results: int = typer.Option(
        10, "--max-results", "-n", help="Maximum results per query"
    ),
    language: str = typer.Option("all", "--language", "-l", help="Search language"),
):
    """Execute multiple search queries in parallel."""
    if not queries:
        console.print("[red]Error: At least one query is required[/red]")
        raise typer.Exit(1)

    try:
        import asyncio

        result = asyncio.run(
            perform_multi_search_async(
                queries=queries,
                category=category,
                engines=engines,
                language=language,
                max_results=max_results,
            )
        )

        if output_format.lower() == "json":
            # JSON output
            output = json.dumps(
                result, indent=2, ensure_ascii=False, default=json_serial
            )
            console.print(output)
        else:
            # Human-readable output
            if result.get("success"):
                summary = result.get("summary", {})
                console.print(f"[bold green]✓ Multi-Search Results[/bold green]")
                console.print(
                    f"Executed {summary.get('total_queries', 0)} queries, {summary.get('successful_queries', 0)} successful"
                )
                console.print()

                for i, query_result in enumerate(result.get("results", [])):
                    query_info = query_result.get("search_result", {})
                    query_text = query_result.get("query", "")
                    query_index = query_result.get("query_index", i + 1)

                    if query_info.get("success"):
                        console.print(
                            f"[bold blue]Query {query_index}:[/bold blue] {query_text}"
                        )
                        console.print(
                            f"[dim]Found {len(query_info.get('results', []))} results[/dim]"
                        )

                        for j, search_result in enumerate(
                            query_info.get("results", [])[:3]
                        ):  # Show top 3
                            console.print(
                                f"  {j + 1}. [bold]{search_result.get('title', 'No title')}[/bold]"
                            )
                            console.print(
                                f"     [link]{search_result.get('url', '')}[/link]"
                            )
                            if search_result.get("content"):
                                content = (
                                    search_result["content"][:100] + "..."
                                    if len(search_result["content"]) > 100
                                    else search_result["content"]
                                )
                                console.print(f"     {content}")

                        if len(query_info.get("results", [])) > 3:
                            console.print(
                                f"     [dim]... and {len(query_info.get('results', [])) - 3} more results[/dim]"
                            )
                    else:
                        console.print(
                            f"[bold red]Query {query_index} (FAILED):[/bold red] {query_text}"
                        )
                        console.print(
                            f"[red]  Error: {query_info.get('error', 'Unknown error')}[/red]"
                        )

                    if (
                        i < len(result.get("results", [])) - 1
                    ):  # Add separator between queries
                        console.print("[dim]" + "─" * 60 + "[/dim]")
            else:
                console.print(
                    f"[red]Multi-search failed: {result.get('error', 'Unknown error')}[/red]"
                )

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error executing multi-search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(
    prompt: Optional[str] = typer.Argument(
        None, help="Question or research request (use '-' or omit to read from stdin)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (format: provider/model)"
    ),
    format_output: str = typer.Option(
        "human", "--format", "-f", help="Output format: human or json"
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Custom API base URL (overrides OPENAI_BASE_URL env var)",
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Profile to use for API key and configuration"
    ),
    max_iterations: int = typer.Option(
        200,
        "--max-iterations",
        help="Maximum number of tool calling iterations (default: 200)",
    ),
):
    """Ask an AI assistant with web search tools (one-shot Q&A). For interactive conversations, use 'searxng chat'."""
    import litellm
    import os
    import sys

    # Handle stdin input for prompt
    if prompt is None or prompt == "-":
        if sys.stdin.isatty():
            # Interactive mode - show prompt on stderr so it doesn't interfere with piping
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print(
                "[yellow]Reading prompt from stdin (press Ctrl+D when done):[/yellow]"
            )

        try:
            # Read all available input from stdin
            prompt_lines = []
            for line in sys.stdin:
                prompt_lines.append(line.rstrip("\n\r"))
            prompt = "\n".join(prompt_lines).strip()
        except KeyboardInterrupt:
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print("\n[red]Cancelled.[/red]")
            raise typer.Exit(1)

        if not prompt:
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print("[red]Error: No prompt provided.[/red]")
            raise typer.Exit(1)

    async def run_chat():
        # Use the shared ask function with profile support
        result = await ask_ai_async(
            prompt=prompt,
            model=model,
            base_url=base_url,
            profile=profile,
            max_iterations=max_iterations,
        )

        # Log model info to stderr if successful
        if result.get("success"):
            stderr_console = Console(file=sys.stderr, force_terminal=True)
            stderr_console.print(
                f"[dim]Using model: [blue]{result.get('model', 'unknown')}[/blue][/dim]"
            )

        if format_output.lower() == "json":
            # JSON output goes to stdout for piping
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if result["success"]:
                # Just the response content goes to stdout for piping
                print(result["response"])
            else:
                # Errors go to stderr
                stderr_console.print(f"[red]Error: {result['error']}[/red]")
                raise typer.Exit(1)

    # Run the async chat function
    asyncio.run(run_chat())


async def ask_ai_conversational_async(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    profile: Optional[str] = None,
    max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Core async function for conversational AI chat with web search tools.
    Takes a conversation history as input and returns the response with updated history.

    Args:
        messages: Conversation history as list of role/content dicts
        model: Optional model override (defaults to profile or "openai/gpt-5")
        base_url: Optional base URL override
        profile: Optional profile name to use (defaults to default profile)
        max_iterations: Maximum number of tool calling iterations (default: 200)
    """
    import litellm
    import os

    # Initialize configuration from profiles or CLI Proxy API
    try:
        model, base_url, api_key = initialize_ai_config(model, base_url, profile)
    except ValueError as e:
        # CLI Proxy API or profile errors
        return {
            "success": False,
            "error": str(e),
            "model": model or "unknown",
            "messages": messages,
        }

    # Check for API keys after profile initialization (skip if using cli-proxy-api)
    if api_key is None:  # Standard provider, check env vars
        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        }

        # Check if we have at least one API key
        if not any(api_keys.values()):
            return {
                "success": False,
                "error": "No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY",
                "model": model,
                "messages": messages,
            }

    # Define tools for the LLM (same as ask function)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using SearXNG's search engines",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {
                            "type": "string",
                            "description": "Search category",
                            "default": "general",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multi_web_search",
                "description": "Search the web with multiple queries in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of search queries",
                        },
                        "category": {
                            "type": "string",
                            "description": "Search category",
                            "default": "general",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results per query",
                            "default": 10,
                        },
                    },
                    "required": ["queries"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch and extract content from a single URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_urls",
                "description": "Fetch and extract content from multiple URLs in parallel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs to fetch",
                        }
                    },
                    "required": ["urls"],
                },
            },
        },
    ]

    try:
        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            system_content = (
                BASE_AI_SYSTEM_PROMPT
                + """\n\n**CHAT MODE - CONVERSATIONAL RESEARCH:**
- Engage naturally in conversation while maintaining research capabilities
- Reference previous messages and build on earlier searches when relevant
- Balance thoroughness with conversational flow - be comprehensive but not overwhelming
- Use context from conversation history to make more targeted searches
- Provide concise but informative responses that invite further questions"""
            )

            system_message = {"role": "system", "content": system_content}
            messages = [system_message] + messages

        # Prepare completion arguments
        completion_args = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        # Add api_base if provided (overrides environment variable)
        # Note: litellm uses api_base, not base_url, for OpenAI-compatible endpoints
        if base_url:
            completion_args["api_base"] = base_url

        # Add api_key if provided (for cli-proxy-api managed models)
        if api_key:
            completion_args["api_key"] = api_key

        # Make initial request to the LLM
        response = litellm.completion(**completion_args)

        # Handle tool calls iteratively with limit
        iteration_count = 0
        while (
            response.choices[0].message.tool_calls and iteration_count < max_iterations
        ):
            iteration_count += 1
            # Add the assistant's message with tool calls
            messages.append(response.choices[0].message.model_dump())

            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Log tool usage to stderr for shell piping friendliness
                stderr_console = Console(file=sys.stderr, force_terminal=True)
                if function_name == "web_search":
                    stderr_console.print(
                        f"🔍 [cyan]Searching:[/cyan] {function_args.get('query', 'N/A')}"
                    )
                elif function_name == "multi_web_search":
                    queries = function_args.get("queries", [])
                    stderr_console.print(
                        f"🔍 [cyan]Multi-search:[/cyan] {', '.join(queries[:3])}{'...' if len(queries) > 3 else ''}"
                    )
                elif function_name == "fetch_url":
                    stderr_console.print(
                        f"📄 [blue]Fetching:[/blue] {function_args.get('url', 'N/A')}"
                    )
                elif function_name == "fetch_urls":
                    urls = function_args.get("urls", [])
                    stderr_console.print(
                        f"📄 [blue]Fetching {len(urls)} URLs:[/blue] {urls[0] if urls else 'N/A'}{'...' if len(urls) > 1 else ''}"
                    )
                else:
                    stderr_console.print(f"🔧 [magenta]Tool:[/magenta] {function_name}")

                # Execute the tool using our existing handler
                tool_result = await handle_tool_call(function_name, function_args)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                    }
                )

            # Update completion args with new messages
            completion_args["messages"] = messages

            # Get next response from LLM
            response = litellm.completion(**completion_args)

        # Add the final assistant response to messages
        final_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": final_response})

        # Return the response and updated conversation history
        return {
            "success": True,
            "model": model,
            "response": final_response,
            "messages": messages,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error calling {model}: {str(e)}",
            "model": model,
            "messages": messages,
        }


@app.command()
def chat(
    initial_message: Optional[str] = typer.Argument(
        None, help="Initial message to send (use '-' to read from stdin)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (format: provider/model)"
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Custom API base URL (overrides OPENAI_BASE_URL env var)",
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Profile to use for API key and configuration"
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Existing session ID to resume; creates a new one if omitted",
    ),
    new: bool = typer.Option(
        False,
        "--new",
        help="Force a new session even if --session is provided",
    ),
    no_reprint: bool = typer.Option(
        False,
        "--no-reprint",
        help="Do not print previous messages when resuming a session",
    ),
    reprint_last: Optional[int] = typer.Option(
        None,
        "--reprint-last",
        help="When reprinting, only show the last N messages",
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        "-t",
        help="Set or update a human-friendly session title",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume the most recently updated session",
    ),
    max_iterations: int = typer.Option(
        200,
        "--max-iterations",
        help="Maximum number of tool calling iterations (default: 200)",
    ),
):
    """Start or resume an interactive chat session with history persistence.

    - New session: no --session (or use --new)
    - Resume: pass --session <id>, history is reloaded
    """
    import litellm
    import os
    import sys

    # Resolve session, model/base_url/profile before starting interactive loop
    active_session = None
    selected_profile = profile

    # Convenience: --resume picks most recent session when --session not provided
    if resume and not session and not new:
        most_recent = session_store.list_sessions(
            limit=1, sort_by="updated_at", reverse=True
        )
        if most_recent:
            session = most_recent[0].get("session_id")
            console.print(f"[dim]Resuming most recent session: {session[:8]}...[/dim]")

    if session and not new:
        try:
            try:
                active_session = session_store.load(session)
            except FileNotFoundError:
                # Allow partial/prefix matching
                resolved = session_store.find(session)
                active_session = session_store.load(resolved)
            # Fill missing CLI args from stored session metadata
            if model is None:
                model = active_session.get("model")
            if base_url is None:
                base_url = active_session.get("base_url")
            if selected_profile is None:
                selected_profile = active_session.get("profile")
        except FileNotFoundError:
            console.print(f"[red]Session not found: {session}[/red]")
            raise typer.Exit(1)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    # Initialize configuration from profiles (sets env vars, defaults)
    model, base_url, api_key = initialize_ai_config(model, base_url, selected_profile)

    async def run_interactive_chat():
        import signal
        from datetime import datetime
        from pathlib import Path

        # Initialize conversation history (load if resuming)
        messages: List[Dict[str, str]] = []
        session_obj: Optional[Dict[str, Any]] = None
        session_id: Optional[str] = None

        # Handle stdin input if requested
        first_message = None
        if initial_message == "-":
            if sys.stdin.isatty():
                stderr_console = Console(file=sys.stderr, force_terminal=True)
                stderr_console.print(
                    "[red]Error: stdin is not available (no pipe detected)[/red]"
                )
                raise typer.Exit(1)
            first_message = sys.stdin.read().strip()
        elif initial_message:
            first_message = initial_message

        # Setup chat history directory (XDG Base Directory spec)
        data_home = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        chat_dir = Path(data_home) / "searxng-ai-kit" / "chats"
        chat_dir.mkdir(parents=True, exist_ok=True)

        # Compute model display name
        model_name = model.split("/")[-1] if "/" in model else model

        # Establish or resume session JSON + markdown transcript
        if active_session and not new:
            session_obj = active_session
            session_id = session_obj["session_id"]
            messages = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in session_obj.get("messages", [])
                if m.get("role") and m.get("content") is not None
            ]
            # Use existing markdown file if present, else create new
            mp = session_obj.get("markdown_path")
            if mp:
                chat_file = Path(mp)
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                chat_file = chat_dir / f"chat-{timestamp}-{model_name}.md"
                with open(chat_file, "w", encoding="utf-8") as f:
                    f.write(f"# SearXNG AI Kit Chat Session\n\n")
                    f.write(
                        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \\n"
                    )
                    f.write(f"**Model:** {model}  \\n")
                    f.write(f"**Session:** {session_id}  \\n\\n")
                    f.write("---\n\n")
                session_obj["markdown_path"] = str(chat_file)
                session_store.save(session_obj)
            # Update title on resume if provided
            if title:
                session_obj["title"] = title
                session_store.save(session_obj)
        else:
            # New session: create JSON session and markdown
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            chat_file = chat_dir / f"chat-{timestamp}-{model_name}.md"
            with open(chat_file, "w", encoding="utf-8") as f:
                f.write(f"# SearXNG AI Kit Chat Session\n\n")
                f.write(
                    f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \\n"
                )
                f.write(f"**Model:** {model}  \\n")
                f.write(f"**Session:** (pending)  \\n\\n")
                f.write("---\n\n")
            session_obj = session_store.create(
                model=model,
                base_url=base_url,
                profile=selected_profile,
                markdown_path=str(chat_file),
                title=title,
            )
            session_id = session_obj["session_id"]

        # Setup console for input/output
        stderr_console = Console(file=sys.stderr, force_terminal=True)
        stderr_console.print(f"[dim]SearXNG AI Kit - Interactive Chat[/dim]")
        stderr_console.print(f"[dim]Using model: [blue]{model}[/blue][/dim]")
        stderr_console.print(f"[dim]Chat history: {chat_file}[/dim]")
        stderr_console.print(f"[dim]Session ID: [cyan]{session_id}[/cyan][/dim]")
        stderr_console.print(
            f"[dim]Type 'exit', 'quit', or press Ctrl+C to end the conversation[/dim]"
        )
        stderr_console.print()

        # Helper to reprint conversation history when resuming
        def _reprint_conversation_history(
            prior_messages: List[Dict[str, str]],
            model_label: str,
            last_n: Optional[int] = None,
        ) -> None:
            if not prior_messages:
                return
            to_print = prior_messages
            if last_n is not None and last_n > 0:
                to_print = prior_messages[-last_n:]
            stderr_console.print("[dim]─── Previous Conversation ───[/dim]")
            for m in to_print:
                role = m.get("role")
                content = m.get("content")
                if role == "user":
                    stderr_console.print(f"[bold green]You:[/bold green] {content}")
                elif role == "assistant":
                    stderr_console.print(
                        f"[bold blue]{model_label}:[/bold blue] {content}"
                    )
                else:
                    continue
                stderr_console.print()
            stderr_console.print("[dim]─── Resuming Session ───[/dim]\n")

        # Reprint history if resuming (before first prompt)
        if active_session and not new and not no_reprint and messages:
            _reprint_conversation_history(messages, model_name, last_n=reprint_last)

        # Setup signal handling for graceful shutdown
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            # For SIGINT, let it raise KeyboardInterrupt as usual
            if signum == signal.SIGINT:
                raise KeyboardInterrupt()

        # Install signal handlers
        # Only handle SIGTERM with custom handler, let SIGINT work normally
        signal.signal(signal.SIGTERM, signal_handler)

        # Enable bracketed paste mode for proper multi-line input handling
        import termios
        import tty

        def enable_bracketed_paste():
            if sys.stdin.isatty():
                sys.stdout.write("\x1b[?2004h")  # Enable bracketed paste
                sys.stdout.flush()

        def disable_bracketed_paste():
            if sys.stdin.isatty():
                sys.stdout.write("\x1b[?2004l")  # Disable bracketed paste
                sys.stdout.flush()

        def get_multiline_input(prompt=""):
            """Get input that properly handles bracketed paste and multi-line content."""
            stderr_console.print(prompt, end="")

            lines = []
            in_paste = False

            try:
                while True:
                    line = input()

                    # Check for bracketed paste start/end sequences
                    if line.startswith("\x1b[200~"):
                        in_paste = True
                        line = line[6:]  # Remove paste start sequence

                    if line.endswith("\x1b[201~"):
                        in_paste = False
                        line = line[:-6]  # Remove paste end sequence
                        lines.append(line)
                        break

                    lines.append(line)

                    # If not in paste mode and we have content, break
                    if not in_paste and lines and lines[-1].strip():
                        break

                    # Show continuation prompt for multi-line input
                    if in_paste:
                        continue
                    else:
                        stderr_console.print("... ", end="")

            except (EOFError, KeyboardInterrupt):
                return None

            return "\n".join(lines).strip()

        enable_bracketed_paste()

        # Process initial message if provided (works for new or resumed)
        if first_message:
            messages.append({"role": "user", "content": first_message})
            with open(chat_file, "a", encoding="utf-8") as f:
                f.write(f"## You\n\n{first_message}\n\n")
            stderr_console.print(f"[bold green]You:[/bold green] {first_message}")
            stderr_console.print()

            # Show spinner while getting AI response
            from rich.spinner import Spinner
            from rich.live import Live

            spinner = Spinner("dots", text="[dim]Thinking... [/dim]")

            try:
                with Live(spinner, console=stderr_console, refresh_per_second=10):
                    # Check for shutdown request during AI processing
                    if shutdown_requested:
                        stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                        return

                    # Get AI response
                    result = await ask_ai_conversational_async(
                        messages=messages,
                        model=model,
                        base_url=base_url,
                        profile=selected_profile,
                        max_iterations=max_iterations,
                    )
            except KeyboardInterrupt:
                stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                return

            if result["success"]:
                # Update conversation history with the response and persist session JSON
                messages = result["messages"]
                session_obj["messages"] = messages
                session_store.save(session_obj)

                # Display the response with model name instead of "Assistant"
                stderr_console.print(f"[bold blue]{model_name}:[/bold blue] ", end="")
                print(result["response"])
                stderr_console.print()

                # Save assistant response to markdown file
                with open(chat_file, "a", encoding="utf-8") as f:
                    f.write(f"## {model_name}\n\n{result['response']}\n\n")
            else:
                # Display error
                stderr_console.print(f"[red]Error: {result['error']}[/red]")
                stderr_console.print()

                # Save error to markdown file
                with open(chat_file, "a", encoding="utf-8") as f:
                    f.write(f"## Error\n\n{result['error']}\n\n")

        try:
            while True:
                # Get user input with multi-line support
                try:
                    user_input = get_multiline_input("[bold green]You:[/bold green] ")
                    if user_input is None:  # Handle Ctrl-C/Ctrl-D
                        stderr_console.print("\n[yellow]Goodbye![/yellow]")
                        break
                    user_input = user_input.strip()
                except (EOFError, KeyboardInterrupt):
                    stderr_console.print("\n[yellow]Goodbye![/yellow]")
                    break

                # Handle exit commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    stderr_console.print("[yellow]Goodbye![/yellow]")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Add user message to history and append to transcript
                messages.append({"role": "user", "content": user_input})
                with open(chat_file, "a", encoding="utf-8") as f:
                    f.write(f"## You\n\n{user_input}\n\n")

                # Show spinner while getting AI response
                from rich.spinner import Spinner
                from rich.live import Live

                spinner = Spinner("dots", text="[dim]Thinking... [/dim]")

                try:
                    with Live(spinner, console=stderr_console, refresh_per_second=10):
                        # Check for shutdown request during AI processing
                        if shutdown_requested:
                            stderr_console.print(
                                "\n[yellow]Interrupted. Goodbye![/yellow]"
                            )
                            break

                        # Get AI response
                        result = await ask_ai_conversational_async(
                            messages=messages,
                            model=model,
                            base_url=base_url,
                            profile=selected_profile,
                            max_iterations=max_iterations,
                        )
                except KeyboardInterrupt:
                    stderr_console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                    break

                if result["success"]:
                    # Update conversation history with the response and persist session JSON
                    messages = result["messages"]
                    session_obj["messages"] = messages
                    session_store.save(session_obj)

                    # Display the response with model name instead of "Assistant"
                    stderr_console.print(
                        f"[bold blue]{model_name}:[/bold blue] ", end=""
                    )
                    print(result["response"])
                    stderr_console.print()

                    # Save assistant response to markdown file
                    with open(chat_file, "a", encoding="utf-8") as f:
                        f.write(f"## {model_name}\n\n{result['response']}\n\n")
                else:
                    # Display error
                    stderr_console.print(f"[red]Error: {result['error']}[/red]")
                    stderr_console.print()

                    # Save error to markdown file
                    with open(chat_file, "a", encoding="utf-8") as f:
                        f.write(f"## Error\n\n{result['error']}\n\n")

        except Exception as e:
            stderr_console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        finally:
            # Clean up bracketed paste mode
            disable_bracketed_paste()

    # Run the interactive chat
    asyncio.run(run_interactive_chat())


# Profile management commands
profile_app = typer.Typer(help="Manage user profiles for API keys and configurations")


@profile_app.command()
def add(
    name: str = typer.Argument(..., help="Profile name"),
    api_key: str = typer.Argument(..., help="API key"),
    type: str = typer.Option(
        "openai",
        "--type",
        "-t",
        help="Provider type: openrouter, anthropic, openai, google",
    ),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom base URL"),
    model: Optional[str] = typer.Option(None, "--model", help="Default model"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing profile without confirmation"
    ),
):
    """Add a new profile."""
    if type not in ["openrouter", "anthropic", "openai", "google"]:
        console.print(
            f"[red]Error: Invalid provider type '{type}'. Must be one of: openrouter, anthropic, openai, google[/red]"
        )
        raise typer.Exit(1)

    profile_manager.add_profile(name, type, api_key, base_url, model, force)


@profile_app.command(name="list")
def list_cmd():
    """List all profiles."""
    profile_manager.list_profiles()


@profile_app.command()
def show(name: str = typer.Argument(..., help="Profile name")):
    """Show profile details."""
    profile_manager.show_profile(name)


@profile_app.command()
def remove(name: str = typer.Argument(..., help="Profile name")):
    """Remove a profile."""
    profile_manager.remove_profile(name)


@profile_app.command()
def set_default(name: str = typer.Argument(..., help="Profile name")):
    """Set default profile."""
    profile_manager.set_default(name)


@profile_app.command()
def edit(name: str = typer.Argument(..., help="Profile name")):
    """Edit a profile using system editor."""
    profile_manager.edit_profile(name)


@profile_app.command()
def test(name: str = typer.Argument(..., help="Profile name")):
    """Test a profile by making an API call."""
    profile_manager.test_profile(name)


# Global default model commands
@app.command()
def set_default_model(
    model: str = typer.Argument(
        ..., help="Model to set as default (format: provider/model)"
    ),
):
    """Set the global default model for all AI operations.

    Examples:
        searxng set-default-model openai/gpt-4
        searxng set-default-model anthropic/claude-3-sonnet
        searxng set-default-model cli-proxy-api/model-name
        searxng set-default-model openrouter/openai/gpt-4
    """
    global_config.set_default_model(model)
    console.print(f"[green]✓[/green] Global default model set to: {model}")
    console.print(
        f"[dim]This will be used for 'searxng ask' and 'searxng chat' when no --model is specified.[/dim]"
    )


@app.command()
def clear_default_model():
    """Clear the global default model setting."""
    global_config.set_default_model(None)
    console.print("[green]✓[/green] Global default model cleared")
    console.print("[dim]Will use profile defaults or hardcoded fallbacks.[/dim]")


@app.command()
def default_model():
    """Show the current global default model."""
    model = global_config.get_default_model()
    if model:
        console.print(f"Global default model: {model}")
    else:
        console.print("No global default model set")
        console.print("[dim]Use 'searxng set-default-model <model>' to set one[/dim]")


# Add profile command group to main app
app.add_typer(profile_app, name="profile")


# CLI Proxy API management commands
cli_proxy_app = typer.Typer(help="Manage CLI Proxy API integration")


@cli_proxy_app.command()
def status():
    """Show CLI Proxy API integration status."""
    status_info = cli_proxy_config.get_status()

    console.print("\n[bold]CLI Proxy API Status[/bold]\n")

    # Binary availability
    if status_info["binary_available"]:
        console.print("[green]✓[/green] cli-proxy-api binary found on PATH")
    else:
        console.print("[red]✗[/red] cli-proxy-api binary not found on PATH")
        console.print(
            "  [dim]Install from: https://github.com/router-for-me/CLIProxyAPI[/dim]"
        )

    # Integration enabled
    if status_info["enabled"]:
        console.print("[green]✓[/green] Integration enabled")
    else:
        console.print("[yellow]○[/yellow] Integration disabled")

    # Config file
    if status_info["config_found"]:
        console.print(f"[green]✓[/green] Config found: {status_info['config_path']}")
    else:
        console.print("[yellow]○[/yellow] No config file found")
        console.print(
            "  [dim]Checked: ~/.cli-proxy-api/config.yaml, ~/.config/cli-proxy-api/config.yaml[/dim]"
        )

    # Explicit config path
    if status_info["explicit_config_path"]:
        console.print(
            f"  [dim]Explicit path set: {status_info['explicit_config_path']}[/dim]"
        )

    # Default model
    if status_info["default_model"]:
        console.print(
            f"[green]✓[/green] Default model: cli-proxy-api/{status_info['default_model']}"
        )
    else:
        console.print("[dim]○[/dim] No default model set")
        console.print(
            "  [dim]Set with: searxng cli-proxy-api set-default-model <model>[/dim]"
        )

    # Process status (only if binary and config available)
    if status_info["binary_available"] and status_info["config_found"]:
        manager = CLIProxyAPIManager.get_instance()
        if manager.is_running():
            console.print(f"[green]✓[/green] Process running on port {manager.port}")
            # Try to get model count
            models = manager.get_models()
            if models:
                console.print(f"[green]✓[/green] {len(models)} models available")
            else:
                console.print(
                    "[yellow]○[/yellow] No models returned (check auth/OAuth)"
                )
        else:
            console.print("[dim]○[/dim] Process not started (starts on first use)")

    # Overall readiness
    console.print()
    if status_info["binary_available"] and status_info["config_found"]:
        if status_info["enabled"]:
            console.print("[green]Ready to use CLI Proxy API for AI requests[/green]")
        else:
            console.print(
                "[yellow]CLI Proxy API available but disabled. Run 'searxng cli-proxy-api enable' to enable.[/yellow]"
            )
    else:
        missing = []
        if not status_info["binary_available"]:
            missing.append("binary")
        if not status_info["config_found"]:
            missing.append("config")
        console.print(f"[red]Not ready: missing {', '.join(missing)}[/red]")


@cli_proxy_app.command()
def enable():
    """Enable CLI Proxy API integration."""
    cli_proxy_config.set_enabled(True)
    console.print("[green]CLI Proxy API integration enabled.[/green]")

    # Check if actually usable
    status_info = cli_proxy_config.get_status()
    if not status_info["binary_available"]:
        console.print(
            "[yellow]Warning: cli-proxy-api binary not found on PATH[/yellow]"
        )
    if not status_info["config_found"]:
        console.print("[yellow]Warning: No cli-proxy-api config file found[/yellow]")


@cli_proxy_app.command()
def disable():
    """Disable CLI Proxy API integration."""
    cli_proxy_config.set_enabled(False)
    console.print("[green]CLI Proxy API integration disabled.[/green]")


@cli_proxy_app.command(name="set-config")
def set_config(
    path: str = typer.Argument(..., help="Path to cli-proxy-api config.yaml file"),
):
    """Set explicit path to cli-proxy-api config file."""
    # Expand and validate path
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        console.print(f"[red]Error: File not found: {config_path}[/red]")
        raise typer.Exit(1)

    if not config_path.is_file():
        console.print(f"[red]Error: Not a file: {config_path}[/red]")
        raise typer.Exit(1)

    cli_proxy_config.set_config_path(str(config_path))
    console.print(f"[green]Config path set to: {config_path}[/green]")


@cli_proxy_app.command(name="clear-config")
def clear_config():
    """Clear explicit config path (use auto-detection)."""
    cli_proxy_config.set_config_path(None)
    console.print("[green]Config path cleared. Will use auto-detection.[/green]")

    # Show what will be auto-detected
    found = cli_proxy_config.find_cli_proxy_config()
    if found:
        console.print(f"[dim]Auto-detected config: {found}[/dim]")
    else:
        console.print("[dim]No config file found in standard locations.[/dim]")


@cli_proxy_app.command(name="set-default-model")
def set_default_model(
    model: str = typer.Argument(
        ..., help="Model name (e.g., claude-sonnet-4-5-20250929)"
    ),
):
    """Set the default model for cli-proxy-api.

    When set, 'searxng ask' and 'searxng chat' will use this model
    automatically without needing --model flag.

    Example:
        searxng cli-proxy-api set-default-model claude-sonnet-4-5-20250929
        searxng ask "question"  # Uses cli-proxy-api/claude-sonnet-4-5-20250929
    """
    # Strip cli-proxy-api/ prefix if provided
    if model.startswith("cli-proxy-api/"):
        model = model[len("cli-proxy-api/") :]

    cli_proxy_config.set_default_model(model)
    console.print(f"[green]Default model set to: cli-proxy-api/{model}[/green]")
    console.print(
        "[dim]Use 'searxng ask' or 'searxng chat' without --model to use this default[/dim]"
    )


@cli_proxy_app.command(name="clear-default-model")
def clear_default_model():
    """Clear the default model setting."""
    cli_proxy_config.set_default_model(None)
    console.print("[green]Default model cleared.[/green]")


@cli_proxy_app.command(name="start")
def start_proxy():
    """Start cli-proxy-api subprocess (for testing)."""
    # Check prerequisites
    if not cli_proxy_config.is_cli_proxy_available():
        console.print("[red]Error: cli-proxy-api binary not found on PATH[/red]")
        raise typer.Exit(1)

    config_path = cli_proxy_config.find_cli_proxy_config()
    if not config_path:
        console.print("[red]Error: No cli-proxy-api config file found[/red]")
        console.print(
            "[dim]Set one with: searxng cli-proxy-api set-config /path/to/config.yaml[/dim]"
        )
        raise typer.Exit(1)

    console.print(f"[blue]Starting cli-proxy-api with config: {config_path}[/blue]")

    manager = CLIProxyAPIManager.get_instance()
    if manager.start(config_path):
        console.print(f"[green]Started successfully on port {manager.port}[/green]")
        console.print(f"[dim]Base URL: {manager.get_base_url()}[/dim]")
        console.print("[dim]Press Ctrl-C to stop[/dim]")

        # Keep running until interrupted
        import time

        try:
            while manager.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop()
            console.print("[yellow]Stopped.[/yellow]")
    else:
        console.print("[red]Failed to start cli-proxy-api[/red]")
        raise typer.Exit(1)


@cli_proxy_app.command(name="stop")
def stop_proxy():
    """Stop cli-proxy-api subprocess if running."""
    manager = CLIProxyAPIManager.get_instance()
    if manager.is_running():
        manager.stop()
        console.print("[green]Stopped cli-proxy-api[/green]")
    else:
        console.print("[yellow]cli-proxy-api is not running[/yellow]")


# Add cli-proxy-api command group to main app
app.add_typer(cli_proxy_app, name="cli-proxy-api")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
