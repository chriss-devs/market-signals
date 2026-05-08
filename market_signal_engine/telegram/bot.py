"""Telegram Bot — 16 commands + push alerts with importance tags."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TelegramBot:
    """Async Telegram bot using Telegram HTTP API directly (no framework deps).

    Provides:
      - 16 slash commands per plan spec
      - Push alerts with CRITICAL/IMPORTANT/WATCH tags
      - Subscription management for push alerts
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        alerts_enabled: bool = False,
    ) -> None:
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._alerts_enabled = alerts_enabled
        self._base_url = f"https://api.telegram.org/bot{self._token}"
        self._subscribers: set[str] = set()
        self._last_update_id: int = 0

    @property
    def configured(self) -> bool:
        return bool(self._token)

    @property
    def alerts_enabled(self) -> bool:
        return self._alerts_enabled and self.configured

    # ── HTTP helpers ────────────────────────────────────────────────────

    def _api_call(self, method: str, params: dict | None = None) -> dict:
        """Make a synchronous Telegram API call."""
        if not self.configured:
            return {"ok": False, "description": "Telegram not configured"}
        url = f"{self._base_url}/{method}"
        body = json.dumps(params or {}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.error(f"Telegram API error ({method}): {e}")
            return {"ok": False, "description": str(e)}

    def _send(self, chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to a chat. Returns True on success."""
        result = self._api_call("sendMessage", {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        return result.get("ok", False)

    # ── Core push ───────────────────────────────────────────────────────

    def push_alert(self, message: str, to_subscribers: bool = True) -> bool:
        """Push an alert to the configured chat + subscribers."""
        if not self.alerts_enabled:
            return False
        ok = self._send(self._chat_id, message)
        if to_subscribers:
            for sub in list(self._subscribers):
                self._send(sub, message)
        return ok

    def push_critical(self, message: str) -> bool:
        """Push a critical alert — always sends to all subscribers."""
        if not self.configured:
            return False
        ok = self._send(self._chat_id, message)
        for sub in list(self._subscribers):
            self._send(sub, message)
        return ok

    # ── Command handlers (called by polling or webhook) ─────────────────

    def handle_command(self, command: str, args: str, chat_id: str) -> str | None:
        """Route a command to its handler. Returns response text or None."""
        cmd = command.lstrip("/").lower().split("@")[0]  # strip bot mention

        handlers = {
            "start": self._cmd_start,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "signals": self._cmd_signals,
            "signal": self._cmd_signal,
            "markets": self._cmd_markets,
            "market": self._cmd_market,
            "crypto": self._cmd_crypto,
            "macro": self._cmd_macro,
            "news": self._cmd_news,
            "forex": self._cmd_forex,
            "system": self._cmd_system,
            "agents": self._cmd_agents,
            "reasoning": self._cmd_reasoning,
            "accuracy": self._cmd_accuracy,
            "subscribe": self._cmd_subscribe,
            "unsubscribe": self._cmd_unsubscribe,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                return handler(args, chat_id)
            except Exception as e:
                logger.error(f"Command /{cmd} error: {e}")
                return f"Error: {e}"
        return None

    # ── Command implementations ─────────────────────────────────────────

    def _cmd_start(self, args: str, chat_id: str) -> str:
        from market_signal_engine.telegram.formatter import format_welcome
        return format_welcome()

    def _cmd_help(self, args: str, chat_id: str) -> str:
        from market_signal_engine.telegram.formatter import format_help
        return format_help()

    def _cmd_status(self, args: str, chat_id: str) -> str:
        return (
            "\U0001F4CA *System Status*\n\n"
            "  Collectors: YFinance, Binance, DeFi Llama active\n"
            "  Agents: 11/26 Tier 1 built (Batch A + B complete)\n"
            "  DB: SQLite (dev) / PostgreSQL (prod)\n"
            "  Alerts: " + ("✅ Enabled" if self.alerts_enabled else "❌ Disabled") + "\n"
            "  Uptime: check */system* for details"
        )

    def _cmd_signals(self, args: str, chat_id: str) -> str:
        return (
            "\U0001F4E1 *Recent Signals*\n\n"
            "  No signals stored yet. Run the pipeline to generate signals.\n"
            "  Use */signal <id>* once signals are available."
        )

    def _cmd_signal(self, args: str, chat_id: str) -> str:
        if not args.strip():
            return "Usage: `/signal <id>`\nExample: `/signal 5`"
        return f"\U0001F50D *Signal #{args.strip()}*\n\n  Signal detail not yet available — pipeline pending."

    def _cmd_markets(self, args: str, chat_id: str) -> str:
        return "\U0001F4CA *Market Overview*\n\n  Market data not yet available — run collectors first."

    def _cmd_market(self, args: str, chat_id: str) -> str:
        if not args.strip():
            return "Usage: `/market <symbol>`\nExample: `/market AAPL`"
        return f"\U0001F4CA *{args.strip().upper()}*\n\n  Asset data not yet available — run collectors first."

    def _cmd_crypto(self, args: str, chat_id: str) -> str:
        return "\U0001F4CA *Crypto Overview*\n\n  Crypto data not yet available — run collectors first."

    def _cmd_macro(self, args: str, chat_id: str) -> str:
        return "\U0001F30D *Macro Indicators*\n\n  Macro data not yet available — run collectors first."

    def _cmd_news(self, args: str, chat_id: str) -> str:
        return "\U0001F4F0 *News*\n\n  News data not yet available."

    def _cmd_forex(self, args: str, chat_id: str) -> str:
        return "\U0001F4B1 *Forex*\n\n  Forex data not yet available."

    def _cmd_system(self, args: str, chat_id: str) -> str:
        import platform
        return (
            "\U0001F4BB *System Info*\n\n"
            f"  Platform: {platform.system()} {platform.release()}\n"
            f"  Python: {platform.python_version()}\n"
            f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"  Alerts: {'ON' if self.alerts_enabled else 'OFF'}"
        )

    def _cmd_agents(self, args: str, chat_id: str) -> str:
        return (
            "\U0001F916 *Agents (11/26 built)*\n\n"
            "*Tier 1 Batch A:* Tech(1), Sentiment(2), Macro(4), Fundamental(5), Momentum(9)\n"
            "*Tier 1 Batch B:* Pattern(6), VolProfile(7), Structure(8), Vol(10), Fear(11), Whale(12)\n"
            "*Meta:* Meta-Agent(13)\n\n"
            "Use */accuracy* for performance leaderboard.\n"
            "Use */agent <name>* for details."
        )

    def _cmd_reasoning(self, args: str, chat_id: str) -> str:
        if not args.strip():
            return "Usage: `/reasoning <signal_id>`\nExample: `/reasoning 3`"
        return f"\U0001F9E0 *Reasoning for Signal #{args.strip()}*\n\n  Reasoning not yet available — pipeline pending."

    def _cmd_accuracy(self, args: str, chat_id: str) -> str:
        return (
            "\U0001F3AF *Agent Accuracy*\n\n"
            "  No performance data yet — agents need predictions + outcomes.\n"
            "  Accuracy is tracked via EMA (0.9 decay) per agent."
        )

    def _cmd_subscribe(self, args: str, chat_id: str) -> str:
        if chat_id in self._subscribers:
            return "✅ You're already subscribed to push alerts."
        self._subscribers.add(chat_id)
        return (
            "✅ *Subscribed!*\n\n"
            "You'll receive push alerts for:\n"
            "  \U0001F534 CRITICAL signals (confidence ≥ 85%)\n"
            "  \U0001F7E0 IMPORTANT signals (confidence ≥ 75%)\n"
            "  \U0001F7E1 WATCH signals (confidence ≥ 65%)\n\n"
            "Use /unsubscribe to stop."
        )

    def _cmd_unsubscribe(self, args: str, chat_id: str) -> str:
        self._subscribers.discard(chat_id)
        return "\U0001F44B Unsubscribed from push alerts. Use /subscribe to re-enable."

    # ── Polling ─────────────────────────────────────────────────────────

    async def start_polling(self, interval: float = 2.0) -> None:
        """Poll for updates and handle commands. Runs forever."""
        if not self.configured:
            logger.warning("Telegram bot not configured — polling disabled")
            return

        logger.info("Telegram bot polling started")
        while True:
            try:
                updates = self._api_call("getUpdates", {
                    "offset": self._last_update_id + 1,
                    "timeout": 30,
                })
                if updates.get("ok") and updates.get("result"):
                    for update in updates["result"]:
                        self._last_update_id = max(self._last_update_id, update["update_id"])
                        self._process_update(update)
            except Exception as e:
                logger.error(f"Telegram polling error: {e}")
            await asyncio.sleep(interval)

    def _process_update(self, update: dict) -> None:
        """Process a single Telegram update."""
        msg = update.get("message", {})
        text = msg.get("text", "")
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            command = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            response = self.handle_command(command, args, chat_id)
            if response:
                self._send(chat_id, response)


# Global singleton
_bot: TelegramBot | None = None


def get_bot() -> TelegramBot:
    global _bot
    if _bot is None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        alerts = os.environ.get("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
        _bot = TelegramBot(token=token, chat_id=chat_id, alerts_enabled=alerts)
    return _bot
