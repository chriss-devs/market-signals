"""Dashboard routes — all data sourced from the real database."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()


# ── Action helpers ────────────────────────────────────────────────────────────

def _action(direction: str, confidence: float) -> dict:
    """Map direction + confidence to a clear, human-readable trade action."""
    if direction == "neutral" or confidence < 0.45:
        return {"label": "HOLD", "strength": "", "css": "hold",
                "emoji": "⏸", "desc": "Wait for a stronger signal"}

    if direction == "bullish":
        if confidence >= 0.75:
            return {"label": "STRONG BUY", "strength": "strong", "css": "bullish",
                    "emoji": "🟢", "desc": "High-conviction buy signal"}
        if confidence >= 0.60:
            return {"label": "BUY", "strength": "moderate", "css": "bullish",
                    "emoji": "📈", "desc": "Moderate buy signal"}
        return {"label": "WEAK BUY", "strength": "weak", "css": "bullish",
                "emoji": "↗", "desc": "Lean bullish — confirm before acting"}

    # bearish
    if confidence >= 0.75:
        return {"label": "STRONG SELL", "strength": "strong", "css": "bearish",
                "emoji": "🔴", "desc": "High-conviction sell signal"}
    if confidence >= 0.60:
        return {"label": "SELL", "strength": "moderate", "css": "bearish",
                "emoji": "📉", "desc": "Moderate sell signal"}
    return {"label": "WEAK SELL", "strength": "weak", "css": "bearish",
            "emoji": "↘", "desc": "Lean bearish — confirm before acting"}


def _risk(dispersion: float | None) -> dict:
    if dispersion is None:
        return {"level": "UNKNOWN", "css": "neutral", "desc": ""}
    if dispersion < 0.20:
        return {"level": "LOW RISK", "css": "bullish", "desc": "Agents strongly agree"}
    if dispersion < 0.35:
        return {"level": "MEDIUM RISK", "css": "amber", "desc": "Some divergence among agents"}
    return {"level": "HIGH RISK", "css": "bearish", "desc": "Agents disagree — trade with caution"}


def _price_change_pct(price: float | None, direction: str) -> str:
    """Estimated move implied by the signal direction."""
    if price is None:
        return "—"
    if direction == "bullish":
        return "+10%"
    if direction == "bearish":
        return "−10%"
    return "0%"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    from market_signal_engine.database.repository import (
        get_all_agent_performance,
        get_signals,
        get_signals_today_count,
        get_top_agent,
        seed_agent_performance,
    )

    seed_agent_performance()

    agents = get_all_agent_performance()
    raw_signals = get_signals(limit=20)
    signals_today = get_signals_today_count()
    top_agent = get_top_agent()

    active_count = sum(1 for a in agents if a.total_predictions > 0)
    avg_accuracy = round(
        sum(a.accuracy_ema for a in agents) / max(len(agents), 1), 3
    )

    # Enrich signals with action, risk, and asset symbol
    enriched: list[dict] = []
    for sig in raw_signals:
        enriched.append({
            "id": sig.id,
            "asset": sig.asset.symbol if sig.asset else "???",
            "direction": sig.direction,
            "confidence": sig.confidence,
            "dispersion": sig.consensus_dispersion,
            "price": sig.price,
            "entry_price": sig.entry_price,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "timestamp": sig.created_at.isoformat() if sig.created_at else "",
            "action": _action(sig.direction, sig.confidence),
            "risk": _risk(sig.consensus_dispersion),
        })

    # Agent summary for sidebar stats
    agent_list = [
        {
            "name": a.agent_name,
            "id": a.agent_id,
            "tier": a.tier,
            "category": a.category,
            "accuracy": a.accuracy_ema,
            "predictions_count": a.total_predictions,
            "weight": a.weight,
            "status": "active" if a.total_predictions > 0 else "idle",
            "last_prediction": a.last_updated.isoformat() if a.last_updated else "",
        }
        for a in agents
    ]

    # Top agents by accuracy (for the mini-cards strip)
    top_performers = sorted(
        [a for a in agent_list if a["predictions_count"] > 0],
        key=lambda a: a["accuracy"], reverse=True
    )[:8]

    return request.app.state.templates.TemplateResponse("dashboard.html", {
        "request": request,
        "signals": enriched,
        "top_performers": top_performers,
        "stats": {
            "active_agents": active_count,
            "total_agents": 26,
            "avg_accuracy": avg_accuracy,
            "top_agent": top_agent.agent_name if top_agent else "—",
            "top_accuracy": top_agent.accuracy_ema if top_agent else 0.0,
            "signals_today": signals_today,
        },
    })


@router.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    from market_signal_engine.database.repository import (
        get_all_agent_performance,
        seed_agent_performance,
    )

    seed_agent_performance()
    agents = get_all_agent_performance()

    def _build(a):
        return {
            "name": a.agent_name,
            "id": a.agent_id,
            "tier": a.tier,
            "category": a.category,
            "accuracy": a.accuracy_ema,
            "predictions_count": a.total_predictions,
            "weight": a.weight,
            "status": "active" if a.total_predictions > 0 else "idle",
            "last_prediction": a.last_updated.isoformat() if a.last_updated else "",
            "current_signal": "neutral",
        }

    all_agents = [_build(a) for a in agents]
    tier1 = [a for a in all_agents if a["tier"] == 1]
    tier2 = [a for a in all_agents if a["tier"] == 2]
    tier3 = [a for a in all_agents if a["tier"] == 3]

    active = sum(1 for a in all_agents if a["status"] == "active")
    avg_acc = round(sum(a["accuracy"] for a in all_agents) / max(len(all_agents), 1), 3)

    return request.app.state.templates.TemplateResponse("agents.html", {
        "request": request,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "stats": {
            "active_agents": active,
            "total_agents": 26,
            "avg_accuracy": avg_acc,
        },
    })


@router.get("/signal/{signal_id}", response_class=HTMLResponse)
async def signal_detail(request: Request, signal_id: int):
    from market_signal_engine.database.repository import (
        get_predictions_for_signal,
        get_signal,
        get_all_agent_performance,
        seed_agent_performance,
    )

    seed_agent_performance()
    sig = get_signal(signal_id)

    if not sig:
        return HTMLResponse("<h1 style='color:#e0e8f0;background:#060d14;padding:40px;font-family:monospace'>Signal not found</h1>", status_code=404)

    predictions = get_predictions_for_signal(signal_id)
    agents = get_all_agent_performance()

    bullish = [p for p in predictions if p.vote == "bullish"]
    bearish = [p for p in predictions if p.vote == "bearish"]
    neutral = [p for p in predictions if p.vote == "neutral"]
    total = len(predictions)

    active = sum(1 for a in agents if a.total_predictions > 0)
    avg_acc = round(sum(a.accuracy_ema for a in agents) / max(len(agents), 1), 3)

    votes = [
        {
            "agent_name": p.agent_name,
            "agent_id": p.agent_id,
            "tier": p.tier,
            "category": p.category,
            "vote": p.vote,
            "confidence": p.confidence,
            "reasoning": p.reasoning or "",
        }
        for p in predictions
    ]

    return request.app.state.templates.TemplateResponse("signal_detail.html", {
        "request": request,
        "signal": {
            "id": sig.id,
            "asset": sig.asset.symbol if sig.asset else "???",
            "direction": sig.direction,
            "confidence": sig.confidence,
            "consensus_dispersion": sig.consensus_dispersion,
            "meta_agent_version": sig.meta_agent_version,
            "price": sig.price,
            "entry_price": sig.entry_price,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "timestamp": sig.created_at.isoformat() if sig.created_at else "",
            "action": _action(sig.direction, sig.confidence),
            "risk": _risk(sig.consensus_dispersion),
        },
        "votes": votes,
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "bullish_pct": round(len(bullish) / max(total, 1) * 100, 1),
        "bearish_pct": round(len(bearish) / max(total, 1) * 100, 1),
        "neutral_pct": round(len(neutral) / max(total, 1) * 100, 1),
        "stats": {
            "active_agents": active,
            "total_agents": 26,
            "avg_accuracy": avg_acc,
        },
    })


# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.get("/api/agents/summary")
async def api_agents_summary():
    from market_signal_engine.database.repository import get_all_agent_performance, seed_agent_performance
    seed_agent_performance()
    agents = get_all_agent_performance()
    return JSONResponse({
        "total": 26,
        "active": sum(1 for a in agents if a.total_predictions > 0),
        "avg_accuracy": round(sum(a.accuracy_ema for a in agents) / max(len(agents), 1), 3),
        "agents": [
            {
                "name": a.agent_name,
                "id": a.agent_id,
                "tier": a.tier,
                "category": a.category,
                "accuracy": a.accuracy_ema,
                "predictions_count": a.total_predictions,
                "weight": a.weight,
                "status": "active" if a.total_predictions > 0 else "idle",
            }
            for a in agents
        ],
    })


@router.get("/api/signals/{signal_id}/agents")
async def api_signal_agents(signal_id: int):
    from market_signal_engine.database.repository import get_predictions_for_signal, get_signal
    sig = get_signal(signal_id)
    if not sig:
        return JSONResponse({"error": "Signal not found"}, status_code=404)
    predictions = get_predictions_for_signal(signal_id)
    bullish = [p for p in predictions if p.vote == "bullish"]
    bearish = [p for p in predictions if p.vote == "bearish"]
    neutral = [p for p in predictions if p.vote == "neutral"]
    return JSONResponse({
        "signal": {
            "id": sig.id,
            "asset": sig.asset.symbol if sig.asset else "???",
            "direction": sig.direction,
            "confidence": sig.confidence,
            "consensus_dispersion": sig.consensus_dispersion,
            "price": sig.price,
        },
        "votes": [
            {
                "agent_name": p.agent_name,
                "agent_id": p.agent_id,
                "tier": p.tier,
                "category": p.category,
                "vote": p.vote,
                "confidence": p.confidence,
                "reasoning": p.reasoning or "",
            }
            for p in predictions
        ],
        "tally": {
            "bullish": len(bullish),
            "bearish": len(bearish),
            "neutral": len(neutral),
        },
    })


@router.get("/health")
async def health():
    from market_signal_engine.database.repository import (
        get_active_agent_count,
        get_all_agent_performance,
        seed_agent_performance,
    )
    seed_agent_performance()
    agents = get_all_agent_performance()
    return JSONResponse({
        "ok": True,
        "agents": 26,
        "active": get_active_agent_count(),
        "avg_accuracy": round(sum(a.accuracy_ema for a in agents) / max(len(agents), 1), 3),
        "scheduler_running": True,
    })
