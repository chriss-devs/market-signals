"""APScheduler job definitions — periodic data collection, analysis, and signal emission."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Default assets to track
DEFAULT_ASSETS = [
    # Crypto
    "BTC", "ETH", "SOL", "DOGE",
    # Stocks
    "AAPL", "NVDA", "TSLA", "MSFT", "SPY",
    # Chains
    "Ethereum", "Solana",
]

# Collector symbols for DeFi Llama (chain-level data)
DEFILLAMA_SYMBOLS = ["Ethereum", "Solana"]


class PipelineScheduler:
    """Manages periodic pipeline execution using APScheduler."""

    def __init__(self) -> None:
        self._scheduler = None
        self._running = False
        self._assets = DEFAULT_ASSETS.copy()
        self._collector_interval = int(os.environ.get("COLLECTOR_INTERVAL", "300"))
        self._analysis_interval = int(os.environ.get("ANALYSIS_INTERVAL", "600"))
        self._signal_interval = int(os.environ.get("SIGNAL_CHECK_INTERVAL", "60"))

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start all scheduled jobs."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
        except ImportError:
            logger.error("APScheduler not installed. Run: pip install apscheduler")
            return

        self._scheduler = AsyncIOScheduler()
        self._running = True

        # Job 1: Data collection — runs every COLLECTOR_INTERVAL seconds
        self._scheduler.add_job(
            self._collect_data,
            "interval",
            seconds=self._collector_interval,
            id="collect_data",
            next_run_time=datetime.now(timezone.utc),
        )

        # Job 2: Agent analysis — runs every ANALYSIS_INTERVAL seconds
        self._scheduler.add_job(
            self._run_analysis,
            "interval",
            seconds=self._analysis_interval,
            id="run_analysis",
            next_run_time=datetime.now(timezone.utc),
        )

        # Job 3: Signal check + alerts — runs every SIGNAL_CHECK_INTERVAL seconds
        self._scheduler.add_job(
            self._check_signals,
            "interval",
            seconds=self._signal_interval,
            id="check_signals",
            next_run_time=datetime.now(timezone.utc),
        )

        # Job 4: Self-tuning — runs daily
        self._scheduler.add_job(
            self._self_tune_agents,
            "cron",
            hour=1,
            minute=17,
            id="self_tune",
        )

        self._scheduler.start()
        logger.info(
            f"Scheduler started: collect={self._collector_interval}s, "
            f"analyze={self._analysis_interval}s, signal={self._signal_interval}s"
        )

    def stop(self) -> None:
        """Stop all scheduled jobs."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("Scheduler stopped")

    # ── Job implementations ──────────────────────────────────────────────

    async def _collect_data(self) -> None:
        """Collect raw data for all assets from all available collectors."""
        logger.debug("Collecting data...")
        from market_signal_engine.collectors.base import BaseCollector
        from market_signal_engine.collectors.yfinance_collector import YFinanceCollector
        from market_signal_engine.collectors.binance_collector import BinanceCollector
        from market_signal_engine.collectors.dexscreener_collector import DEXScreenerCollector
        from market_signal_engine.collectors.defillama_collector import DefiLlamaCollector
        from market_signal_engine.collectors.blockchain_collector import BlockchainCollector

        collectors: list[BaseCollector] = [
            YFinanceCollector(),
            BinanceCollector(),
            DEXScreenerCollector(),
            DefiLlamaCollector(),
            BlockchainCollector(),
        ]

        # Collect for each asset (collectors handle rate limiting internally)
        for sym in self._assets[:6]:  # limit batch size per cycle
            for collector in collectors:
                try:
                    result = collector.collect(sym)
                    if not result.is_ok and result.error:
                        logger.debug(f"{collector.source}/{sym}: {result.error}")
                except Exception as e:
                    logger.debug(f"Collector {collector.source}/{sym} failed: {e}")

        # DeFi Llama chain-level data
        for chain_sym in DEFILLAMA_SYMBOLS:
            try:
                collector = DefiLlamaCollector()
                collector.collect(chain_sym)
            except Exception:
                pass

        # Stablecoin data
        try:
            collector = DefiLlamaCollector()
            collector.collect("stablecoins")
        except Exception:
            pass

    async def _run_analysis(self) -> None:
        """Run agent analysis cycle for all tracked assets and persist results."""
        logger.debug("Running analysis cycle...")
        from market_signal_engine.collectors.yfinance_collector import YFinanceCollector
        from market_signal_engine.jobs.orchestrator import AgentOrchestrator
        from market_signal_engine.database.repository import (
            create_prediction,
            create_signal,
            get_or_create_asset,
            seed_agent_performance,
            upsert_agent_performance,
        )

        # Ensure all 26 agents have performance rows
        seed_agent_performance()

        orchestrator = AgentOrchestrator()
        collector = YFinanceCollector()

        for sym in self._assets[:6]:
            try:
                result = collector.collect(sym)
                summary = orchestrator.run_cycle(sym, [result])

                if summary.get("error"):
                    logger.warning(f"Analysis skipped for {sym}: {summary['error']}")
                    continue

                # Determine asset type
                asset_type = "stock" if sym in ("AAPL", "NVDA", "TSLA", "MSFT", "SPY") else "crypto"
                asset = get_or_create_asset(sym, asset_type)

                # Extract price from collector result
                price = None
                if result.is_ok and result.data:
                    price = result.data.get("price")

                # Compute price levels
                direction = summary["consensus"]["direction"]
                entry_price = price
                stop_loss = None
                take_profit = None

                if price and direction == "bullish":
                    stop_loss = round(price * 0.95, 2)
                    take_profit = round(price * 1.10, 2)
                elif price and direction == "bearish":
                    stop_loss = round(price * 1.05, 2)
                    take_profit = round(price * 0.90, 2)

                # Persist signal
                signal = create_signal(
                    asset_id=asset.id,
                    direction=direction,
                    confidence=summary["consensus"]["confidence"],
                    agent_weights=summary["consensus"]["agent_weights"],
                    consensus_dispersion=summary["consensus"]["dispersion"],
                    price=price,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                # Persist each agent prediction
                for r in summary.get("agent_results", []):
                    create_prediction(
                        signal_id=signal.id,
                        agent_name=r["agent_name"],
                        agent_id=r["agent_id"],
                        tier=r.get("tier", 1),
                        category=r.get("category", "Unknown"),
                        vote=r["direction"],
                        confidence=r["confidence"],
                        reasoning=r.get("reasoning", ""),
                    )

                # Update agent performance records
                for r in summary.get("agent_results", []):
                    try:
                        perf = orchestrator._registry.get(r["agent_name"])
                        if perf:
                            p = perf.get_performance()
                            upsert_agent_performance(
                                agent_name=p.agent_name,
                                agent_id=p.agent_id,
                                tier=r.get("tier", 1),
                                category=r.get("category", "Unknown"),
                                accuracy_ema=p.accuracy_ema,
                                total_predictions=p.total_predictions,
                                correct_predictions=p.correct_predictions,
                                weight=p.weight,
                            )
                    except Exception:
                        pass

                logger.info(
                    f"Signal #{signal.id}: {direction.upper()} {sym} "
                    f"({summary['consensus']['confidence']*100:.0f}%, "
                    f"{summary['predictions']} agents, "
                    f"disp={summary['consensus']['dispersion']:.2f})"
                )

                if summary.get("should_alert"):
                    self._push_alert(summary)

            except Exception as e:
                logger.error(f"Analysis failed for {sym}: {e}")

    async def _check_signals(self) -> None:
        """Check for new signals and push alerts."""
        # This runs frequently — checks if last analysis produced actionable signals
        logger.debug("Checking signals...")

    async def _self_tune_agents(self) -> None:
        """Daily self-tuning of all agents from historical outcomes."""
        logger.info("Self-tuning agents...")
        # In a full implementation, this loads historical outcomes from DB
        # and calls agent.self_tune() for each agent

    # ── Alert dispatch ───────────────────────────────────────────────────

    def _push_alert(self, summary: dict) -> None:
        """Push a Telegram alert if the signal is important enough."""
        try:
            from market_signal_engine.telegram.bot import get_bot
            from market_signal_engine.telegram.formatter import SignalAlert, format_signal_alert

            bot = get_bot()
            if not bot.alerts_enabled:
                return

            alert = SignalAlert(
                symbol=summary["symbol"],
                direction=summary["consensus"]["direction"],
                confidence=summary["consensus"]["confidence"],
                consensus_count=summary["consensus"]["vote_tally"].get(
                    summary["consensus"]["direction"], 0
                ),
                total_agents=summary["predictions"],
                reasons=[r.get("reasoning", "") for r in summary.get("agent_results", [])[:4]],
                dispersion=summary["consensus"]["dispersion"],
            )
            message = format_signal_alert(alert)
            bot.push_alert(message)
            logger.info(f"Alert pushed: {summary['alert_summary']}")
        except Exception as e:
            logger.error(f"Failed to push alert: {e}")


# Global instance
_scheduler: PipelineScheduler | None = None


def get_scheduler() -> PipelineScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = PipelineScheduler()
    return _scheduler
