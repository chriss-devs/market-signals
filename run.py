from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from market_signal_engine.dashboard.routes import router as dashboard_router
from market_signal_engine.chatbot.routes import router as chatbot_router

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "market_signal_engine" / "dashboard" / "templates"
STATIC_DIR = BASE_DIR / "market_signal_engine" / "dashboard" / "static"


def _register_all_agents() -> None:
    """Register all Tier 1 agents with the global registry."""
    from market_signal_engine.agents.registry import get_registry

    reg = get_registry()

    def _reg(cls, tier: int, category: str) -> None:
        reg.register(lambda c=cls: c(), {"tier": tier, "category": category})

    from market_signal_engine.agents.technical_analysis import TechnicalAnalysisAgent
    from market_signal_engine.agents.sentiment_analysis import SentimentAnalysisAgent
    from market_signal_engine.agents.onchain_analysis import OnChainAnalysisAgent
    from market_signal_engine.agents.macro_analysis import MacroAnalysisAgent
    from market_signal_engine.agents.fundamental_analysis import FundamentalAnalysisAgent
    from market_signal_engine.agents.pattern_recognition import PatternRecognitionAgent
    from market_signal_engine.agents.volume_profile import VolumeProfileAgent
    from market_signal_engine.agents.market_structure import MarketStructureAgent
    from market_signal_engine.agents.momentum import MomentumAgent
    from market_signal_engine.agents.volatility import VolatilityAgent
    from market_signal_engine.agents.fear_greed import FearGreedAgent
    from market_signal_engine.agents.whale_tracking import WhaleTrackingAgent
    from market_signal_engine.agents.meta_agent import MetaAgent
    from market_signal_engine.agents.backtesting import BacktestingAgent
    from market_signal_engine.agents.calibration import CalibrationAgent
    from market_signal_engine.agents.correlation import CorrelationAgent
    from market_signal_engine.agents.intermarket import IntermarketAgent
    from market_signal_engine.agents.defi import DeFiAgent
    from market_signal_engine.agents.l1_l2_analysis import L1L2AnalysisAgent
    from market_signal_engine.agents.economic_calendar import EconomicCalendarAgent
    from market_signal_engine.agents.social_media import SocialMediaAgent
    from market_signal_engine.agents.alert_recommendation import AlertRecommendationAgent
    from market_signal_engine.agents.sector_rotation import SectorRotationAgent
    from market_signal_engine.agents.statistical_arbitrage import StatisticalArbitrageAgent
    from market_signal_engine.agents.geopolitical import GeopoliticalAgent
    from market_signal_engine.agents.nft_gaming import NFTGamingAgent

    _reg(TechnicalAnalysisAgent, tier=1, category="Technical")
    _reg(SentimentAnalysisAgent, tier=1, category="Sentiment")
    _reg(OnChainAnalysisAgent, tier=1, category="On-Chain")
    _reg(MacroAnalysisAgent, tier=1, category="Macro")
    _reg(FundamentalAnalysisAgent, tier=1, category="Fundamental")
    _reg(PatternRecognitionAgent, tier=1, category="Technical")
    _reg(VolumeProfileAgent, tier=1, category="Technical")
    _reg(MarketStructureAgent, tier=1, category="Technical")
    _reg(MomentumAgent, tier=1, category="Momentum")
    _reg(VolatilityAgent, tier=1, category="Volatility")
    _reg(FearGreedAgent, tier=1, category="Sentiment")
    _reg(WhaleTrackingAgent, tier=1, category="On-Chain")
    _reg(MetaAgent, tier=1, category="Meta")
    _reg(BacktestingAgent, tier=1, category="Validation")
    _reg(CalibrationAgent, tier=1, category="Validation")
    _reg(CorrelationAgent, tier=2, category="Cross-Market")
    _reg(IntermarketAgent, tier=2, category="Cross-Market")
    _reg(DeFiAgent, tier=2, category="DeFi")
    _reg(L1L2AnalysisAgent, tier=2, category="Chain")
    _reg(EconomicCalendarAgent, tier=2, category="Macro")
    _reg(SocialMediaAgent, tier=2, category="Sentiment")
    _reg(AlertRecommendationAgent, tier=2, category="Signal")
    _reg(SectorRotationAgent, tier=3, category="Cross-Market")
    _reg(StatisticalArbitrageAgent, tier=3, category="Quant")
    _reg(GeopoliticalAgent, tier=3, category="Macro")
    _reg(NFTGamingAgent, tier=3, category="Alternative")

    logger.info(f"Registered {reg.agent_count} agents")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: register agents, start scheduler. Shutdown: stop scheduler."""
    _register_all_agents()

    from market_signal_engine.database.connection import init_db
    init_db()
    logger.info("Database initialized")

    from market_signal_engine.jobs.scheduler import get_scheduler
    scheduler = get_scheduler()
    scheduler.start()
    logger.info("Pipeline scheduler started")

    yield

    scheduler.stop()
    logger.info("Pipeline scheduler stopped")


app = FastAPI(title="Market Signal Engine", version="0.3.0", lifespan=_lifespan)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def ts_fmt(ts: str) -> str:
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S UTC")
    except Exception:
        return ts


def ago(ts: str) -> str:
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        mins = int(delta.total_seconds() / 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins}m ago"
        hrs = mins // 60
        if hrs < 24:
            return f"{hrs}h ago"
        return f"{hrs // 24}d ago"
    except Exception:
        return ts


templates.env.filters["pct"] = pct
templates.env.filters["ts_fmt"] = ts_fmt
templates.env.filters["ago"] = ago

app.state.templates = templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(dashboard_router)
app.include_router(chatbot_router)

def main() -> None:
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
