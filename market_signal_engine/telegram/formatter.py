"""Telegram message formatting — clear, scannable alerts with importance tags.

Importance levels:
  CRITICAL  — consensus confidence >= 0.85, strong multi-agent agreement, or crash signals
  IMPORTANT — confidence >= 0.75, Meta-Agent consensus, or regime change detected
  WATCH     — confidence >= 0.65, interesting divergence, or buildup forming
  INFO      — routine status updates, signal summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class SignalAlert:
    """Structured alert data for formatting."""
    symbol: str
    direction: str          # bullish / bearish / neutral
    confidence: float       # 0.0 - 1.0
    consensus_count: int    # agents agreeing
    total_agents: int       # agents voting
    reasons: list[str]
    dispersion: float = 0.0 # consensus dispersion (0 = perfect agreement)
    recommendation: str = ""
    signal_id: int | None = None


def importance_level(alert: SignalAlert) -> str:
    """Determine importance tag for a signal alert."""
    c = alert.confidence
    agree_ratio = alert.consensus_count / max(alert.total_agents, 1)
    d = alert.dispersion

    if c >= 0.85 and agree_ratio >= 0.6:
        return "CRITICAL"
    if c >= 0.85 or (c >= 0.75 and agree_ratio >= 0.7):
        return "CRITICAL"
    if c >= 0.75:
        return "IMPORTANT"
    if c >= 0.65 or (agree_ratio >= 0.5 and c >= 0.55):
        return "IMPORTANT"
    if d > 0.4 and c >= 0.5:
        return "WATCH"
    if c >= 0.55:
        return "WATCH"
    return "INFO"


TAG_EMOJI = {
    "CRITICAL": "\U0001F534",   # red circle
    "IMPORTANT": "\U0001F7E0",  # orange circle
    "WATCH": "\U0001F7E1",     # yellow circle
    "INFO": "\U0001F535",      # blue circle
}

TAG_LABEL = {
    "CRITICAL": "**CRITICAL**",
    "IMPORTANT": "**IMPORTANT**",
    "WATCH": "WATCH",
    "INFO": "INFO",
}

DIRECTION_EMOJI = {
    "bullish": "\U0001F4C8",   # chart up
    "bearish": "\U0001F4C9",   # chart down
    "neutral": "\U0001F4CA",   # chart flat
}


def format_signal_alert(alert: SignalAlert) -> str:
    """Format a single signal as a clear, scannable Telegram message."""
    tag = importance_level(alert)
    emoji = TAG_EMOJI[tag]
    tag_text = TAG_LABEL[tag]
    dir_emoji = DIRECTION_EMOJI.get(alert.direction, "")

    # Confidence bar: 10 chars
    bar_filled = int(alert.confidence * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)

    lines = [
        f"{emoji} {tag_text}  {dir_emoji} {alert.symbol} — {alert.direction.upper()}",
        "",
        f"  Confidence: {alert.confidence*100:.0f}%  [{bar}]",
        f"  Consensus:  {alert.consensus_count}/{alert.total_agents} agents agree",
    ]

    if alert.dispersion > 0.3:
        lines.append(f"  Dispersion:  {alert.dispersion:.0%} — mixed opinions")
    elif alert.dispersion < 0.15 and alert.confidence > 0.6:
        lines.append(f"  Agreement:   strong consensus")

    lines.append("")

    if alert.reasons:
        lines.append("Why:")
        for r in alert.reasons[:4]:
            lines.append(f"  • {r}")

    lines.append("")

    if alert.recommendation:
        lines.append(f"\U0001F4A1 {alert.recommendation}")
    elif tag == "CRITICAL":
        lines.append(f"\U0001F4A1 Review immediately — strong signal with high conviction")
    elif tag == "IMPORTANT":
        lines.append(f"\U0001F4A1 Review this signal — check reasoning above")

    if alert.signal_id:
        lines.append(f"\n#signal-{alert.signal_id}")

    return "\n".join(lines)


def format_signal_summary(alerts: list[SignalAlert]) -> str:
    """Format a batch summary of multiple signals."""
    if not alerts:
        return "\U0001F4AD No active signals right now."

    criticals = [a for a in alerts if importance_level(a) == "CRITICAL"]
    importants = [a for a in alerts if importance_level(a) == "IMPORTANT"]
    watches = [a for a in alerts if importance_level(a) == "WATCH"]

    lines = [
        "\U0001F4E1 *Signal Summary*",
        "",
    ]

    if criticals:
        lines.append(f"\U0001F534 *CRITICAL ({len(criticals)})*")
        for a in criticals:
            dir_emoji = DIRECTION_EMOJI.get(a.direction, "")
            lines.append(f"  {dir_emoji} {a.symbol} {a.direction} ({a.confidence*100:.0f}%)")
        lines.append("")

    if importants:
        lines.append(f"\U0001F7E0 *IMPORTANT ({len(importants)})*")
        for a in importants:
            dir_emoji = DIRECTION_EMOJI.get(a.direction, "")
            lines.append(f"  {dir_emoji} {a.symbol} {a.direction} ({a.confidence*100:.0f}%)")
        lines.append("")

    if watches:
        lines.append(f"\U0001F7E1 *WATCH ({len(watches)})*")
        for a in watches:
            dir_emoji = DIRECTION_EMOJI.get(a.direction, "")
            lines.append(f"  {dir_emoji} {a.symbol} {a.direction} ({a.confidence*100:.0f}%)")
        lines.append("")

    if not any([criticals, importants, watches]):
        lines.append("\U0001F535 No significant signals — all clear")

    return "\n".join(lines)


def format_agent_status(
    agent_name: str, accuracy: float, weight: float, predictions: int,
    last_direction: str = "", last_confidence: float = 0.0,
) -> str:
    """Format a single agent's status line."""
    bar = "█" * int(accuracy * 10) + "░" * (10 - int(accuracy * 10))
    return (
        f"*{agent_name}*\n"
        f"  Accuracy: {accuracy*100:.1f}% [{bar}]\n"
        f"  Weight: {weight:.3f}  |  Predictions: {predictions}\n"
        f"  Last: {last_direction} ({last_confidence*100:.0f}%)"
    )


def format_help() -> str:
    """Format the help/command list message."""
    return """
\U0001F916 *Market Signal Engine — Commands*

*/start* — Welcome + status overview
*/help* — This command list
*/status* — System status (collectors, agents, DB)
*/signals* — Recent signals (last 10)
*/signal <id>* — Full signal detail with agent breakdown
*/markets* — Market overview (top assets)
*/market <sym>* — Single asset detail
*/macro* — Macro indicators snapshot
*/agents* — All agent statuses, weights, accuracy
*/agent <name>* — Single agent details
*/reasoning <id>* — Per-agent reasoning for a signal
*/accuracy* — Agent accuracy leaderboard
*/subscribe* — Enable push alerts
*/unsubscribe* — Disable push alerts
*/system* — Resource usage + uptime
""".strip()


def format_welcome() -> str:
    return """
\U0001F916 *Market Signal Engine* v0.3

26 AI agents analyzing crypto + global stocks.
Self-improving consensus engine with real-time alerts.

Use */help* for all commands.
Use */subscribe* for push alerts on high-confidence signals.
""".strip()
