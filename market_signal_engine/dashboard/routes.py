import random
import time
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

# ── Seed for reproducible randomness ────────────────────────────────────────
RNG = random.Random(42)


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ago(minutes: int) -> str:
    return (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"


# ── Agent definitions (all 26 from the plan) ────────────────────────────────
AGENT_DEFS: list[dict[str, Any]] = [
    # Tier 1 — Core Analysis (15)
    {"id": 1,  "name": "Technical Analysis",   "tier": 1, "category": "Technical"},
    {"id": 2,  "name": "Sentiment Analysis",   "tier": 1, "category": "Sentiment"},
    {"id": 3,  "name": "On-Chain Analysis",    "tier": 1, "category": "On-Chain"},
    {"id": 4,  "name": "Macro Analysis",       "tier": 1, "category": "Macro"},
    {"id": 5,  "name": "Fundamental Analysis",  "tier": 1, "category": "Fundamental"},
    {"id": 6,  "name": "Pattern Recognition",  "tier": 1, "category": "Technical"},
    {"id": 7,  "name": "Volume Profile",       "tier": 1, "category": "Technical"},
    {"id": 8,  "name": "Market Structure",     "tier": 1, "category": "Technical"},
    {"id": 9,  "name": "Momentum",             "tier": 1, "category": "Momentum"},
    {"id": 10, "name": "Volatility",           "tier": 1, "category": "Volatility"},
    {"id": 11, "name": "Fear & Greed",         "tier": 1, "category": "Sentiment"},
    {"id": 12, "name": "Whale Tracking",       "tier": 1, "category": "On-Chain"},
    {"id": 13, "name": "Meta-Agent",           "tier": 1, "category": "Meta"},
    {"id": 14, "name": "Backtesting",          "tier": 1, "category": "Validation"},
    {"id": 15, "name": "Calibration",          "tier": 1, "category": "Validation"},
    # Tier 2 — High Value (7)
    {"id": 16, "name": "Correlation",          "tier": 2, "category": "Cross-Market"},
    {"id": 17, "name": "Intermarket",          "tier": 2, "category": "Cross-Market"},
    {"id": 18, "name": "DeFi",                 "tier": 2, "category": "DeFi"},
    {"id": 19, "name": "L1/L2 Analysis",       "tier": 2, "category": "Chain"},
    {"id": 20, "name": "Economic Calendar",    "tier": 2, "category": "Macro"},
    {"id": 21, "name": "Social Media",         "tier": 2, "category": "Sentiment"},
    {"id": 22, "name": "Alert/Recommendation", "tier": 2, "category": "Signal"},
    # Tier 3 — Specialized (4)
    {"id": 23, "name": "Sector Rotation",      "tier": 3, "category": "Cross-Market"},
    {"id": 24, "name": "Statistical Arbitrage","tier": 3, "category": "Quant"},
    {"id": 25, "name": "Geopolitical",         "tier": 3, "category": "Macro"},
    {"id": 26, "name": "NFT/Gaming",           "tier": 3, "category": "Alternative"},
]

SIGNAL_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AAPL", "NVDA", "SPX", "TSLA", "DOGE/USDT"]
SIGNAL_DIRECTIONS = ["bullish", "bearish", "neutral"]

# ── Dynamic agent state ─────────────────────────────────────────────────────
def _build_agent_state(agent: dict) -> dict:
    """Generate realistic-looking agent state from seed agent definition."""
    RNG.seed(agent["id"] * 137 + 42)
    accuracy = round(RNG.uniform(0.52, 0.84), 3)
    weight = round(accuracy**2 / (RNG.uniform(0.6, 0.9) * 50), 4)
    status = RNG.choice(["active"] * 8 + ["idle"] * 2)
    direction = RNG.choice(SIGNAL_DIRECTIONS)
    return {
        **agent,
        "status": status,
        "weight": weight,
        "accuracy": accuracy,
        "predictions_count": RNG.randint(120, 3400),
        "last_prediction": _ago(RNG.randint(1, 180)),
        "current_signal": direction,
    }


def _get_all_agents() -> list[dict]:
    return [_build_agent_state(a) for a in AGENT_DEFS]


# ── Mock signals ────────────────────────────────────────────────────────────
def _build_signals(count: int = 12) -> list[dict]:
    RNG.seed(7)
    signals: list[dict] = []
    for i in range(count):
        sig_id = 1042 - i
        asset = RNG.choice(SIGNAL_ASSETS)
        direction = RNG.choice(SIGNAL_DIRECTIONS)
        confidence = round(RNG.uniform(0.55, 0.94), 3)
        dispersion = round(RNG.uniform(0.05, 0.45), 3)
        signals.append({
            "id": sig_id,
            "asset": asset,
            "direction": direction,
            "confidence": confidence,
            "consensus_dispersion": dispersion,
            "timestamp": _ago(i * 25 + RNG.randint(0, 15)),
            "meta_agent_version": "0.2.1",
        })
    return signals


MOCK_SIGNALS = _build_signals()


def _agent_votes_for_signal(signal_id: int) -> list[dict]:
    RNG.seed(signal_id * 31)
    votes: list[dict] = []
    reasons_pool = [
        "RSI(14) oversold at 28.3 with bullish divergence on 4h",
        "MACD crossover confirmed on daily; histogram turning positive",
        "Price rejected below VWAP; accumulation at support zone $62.4k",
        "Rising DEX volume + declining exchange reserves = supply squeeze",
        "DXY weakening, T10Y2Y steepening — risk-on macro backdrop",
        "Social sentiment z-score +2.1, Twitter volume spiking on keyword",
        "Whale accumulation detected: 3 wallets added $14M in past 6h",
        "Double-bottom pattern confirmed with neckline break on volume",
        "Funding rate flipped negative while price holds — squeeze setup",
        "Volatility regime shifting from high to normal — expansion likely",
        "Fear & Greed at 22 (extreme fear) — contrarian buy signal",
        "SMA 50 crossing above SMA 200 with volume confirmation",
        "Order book shows thick bid wall at support, thin asks above",
        "TVL growth accelerating +22% WoW while token price flat",
        "Correlation with SPX breaking down — idiosyncratic move starting",
        "Economic surprise index negative, dovish pivot probability rising",
        "Insider accumulation detected via on-chain clustering analysis",
        "Volatility term structure in backwardation — near-term premium",
    ]
    for agent in AGENT_DEFS:
        vote = RNG.choice(SIGNAL_DIRECTIONS)
        confidence = round(RNG.uniform(0.45, 0.92), 3)
        votes.append({
            "agent_name": agent["name"],
            "agent_id": agent["id"],
            "tier": agent["tier"],
            "category": agent["category"],
            "vote": vote,
            "confidence": confidence,
            "reasoning": RNG.choice(reasons_pool),
        })
    return votes


# ── Routes ──────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    agents = _get_all_agents()
    active_count = sum(1 for a in agents if a["status"] == "active")
    avg_accuracy = round(sum(a["accuracy"] for a in agents) / len(agents), 3)
    top_agent = max(agents, key=lambda a: a["accuracy"])
    return request.app.state.templates.TemplateResponse("dashboard.html", {
        "request": request,
        "agents": agents,
        "signals": MOCK_SIGNALS[:6],
        "stats": {
            "active_agents": active_count,
            "total_agents": 26,
            "avg_accuracy": avg_accuracy,
            "top_agent": top_agent["name"],
            "top_accuracy": top_agent["accuracy"],
            "signals_today": len(MOCK_SIGNALS),
        },
    })


@router.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    agents = _get_all_agents()
    tier1 = [a for a in agents if a["tier"] == 1]
    tier2 = [a for a in agents if a["tier"] == 2]
    tier3 = [a for a in agents if a["tier"] == 3]
    return request.app.state.templates.TemplateResponse("agents.html", {
        "request": request,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "stats": {
            "active_agents": sum(1 for a in agents if a["status"] == "active"),
            "total_agents": 26,
            "avg_accuracy": round(sum(a["accuracy"] for a in agents) / len(agents), 3),
        },
    })


@router.get("/signal/{signal_id}", response_class=HTMLResponse)
async def signal_detail(request: Request, signal_id: int):
    signal = next((s for s in MOCK_SIGNALS if s["id"] == signal_id), None)
    if not signal:
        return HTMLResponse("<h1>Signal not found</h1>", status_code=404)
    votes = _agent_votes_for_signal(signal_id)
    bullish = [v for v in votes if v["vote"] == "bullish"]
    bearish = [v for v in votes if v["vote"] == "bearish"]
    neutral = [v for v in votes if v["vote"] == "neutral"]
    agents = _get_all_agents()
    return request.app.state.templates.TemplateResponse("signal_detail.html", {
        "request": request,
        "signal": signal,
        "votes": votes,
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "bullish_pct": round(len(bullish) / 26 * 100, 1),
        "bearish_pct": round(len(bearish) / 26 * 100, 1),
        "neutral_pct": round(len(neutral) / 26 * 100, 1),
        "stats": {
            "active_agents": sum(1 for a in agents if a["status"] == "active"),
            "total_agents": 26,
            "avg_accuracy": round(sum(a["accuracy"] for a in agents) / len(agents), 3),
        },
    })


# ── API Endpoints ───────────────────────────────────────────────────────────

@router.get("/api/agents/summary")
async def api_agents_summary():
    agents = _get_all_agents()
    return JSONResponse({
        "total": 26,
        "active": sum(1 for a in agents if a["status"] == "active"),
        "avg_accuracy": round(sum(a["accuracy"] for a in agents) / len(agents), 3),
        "agents": agents,
    })


@router.get("/api/agents/{name}/history")
async def api_agent_history(name: str):
    RNG.seed(hash(name) % 10000)
    history: list[dict] = []
    base_acc = RNG.uniform(0.48, 0.72)
    for i in range(30):
        day = (datetime.utcnow() - timedelta(days=29 - i)).strftime("%Y-%m-%d")
        acc = round(base_acc + RNG.uniform(-0.04, 0.04) + i * 0.003, 3)
        history.append({"date": day, "accuracy": min(acc, 0.92), "predictions": RNG.randint(3, 25)})
    return JSONResponse({"agent": name, "history": history})


@router.get("/api/signals/{signal_id}/agents")
async def api_signal_agents(signal_id: int):
    signal = next((s for s in MOCK_SIGNALS if s["id"] == signal_id), None)
    if not signal:
        return JSONResponse({"error": "Signal not found"}, status_code=404)
    votes = _agent_votes_for_signal(signal_id)
    return JSONResponse({
        "signal": signal,
        "votes": votes,
        "tally": {
            "bullish": sum(1 for v in votes if v["vote"] == "bullish"),
            "bearish": sum(1 for v in votes if v["vote"] == "bearish"),
            "neutral": sum(1 for v in votes if v["vote"] == "neutral"),
        },
    })


@router.get("/health")
async def health():
    agents = _get_all_agents()
    return JSONResponse({
        "ok": True,
        "agents": 26,
        "active": sum(1 for a in agents if a["status"] == "active"),
        "scheduler_running": True,
        "timestamp": _now(),
    })
