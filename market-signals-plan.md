# Multi-Agent Market Prediction Engine — Complete Plan

## Context

Complete rebuild: 26 specialized AI agents analyzing crypto + global stocks, converging into a Meta-Agent that dynamically weights each agent by historical accuracy. Self-improving feedback loop. Telegram push alerts + 15+ commands. Full web dashboard. Optimized Claude Code settings with tiered docs, hooks, Codex QA, and Gemini file reading.

## Agent Architecture

```
Collectors (14+ sources) ──> FeatureBuilder ──> AgentOrchestrator
                                                   │
         ┌─────────────────────────────────────────┤
         │                                         │
    Tier 1 Agents (15)                      Tier 2 Agents (7)
    ┌───────────────────┐                   ┌──────────────────┐
    │ Technical         │                   │ Correlation      │
    │ Sentiment         │                   │ Intermarket      │
    │ On-Chain          │                   │ DeFi             │
    │ Macro             │                   │ L1/L2            │
    │ Fundamental       │                   │ Economic Calendar│
    │ Pattern Recog.    │                   │ Social Media     │
    │ Volume Profile    │                   │ Alert/Reco       │
    │ Market Structure  │                   └──────────────────┘
    │ Momentum          │
    │ Volatility        │         Tier 3 Agents (4)
    │ Fear & Greed      │         ┌──────────────────┐
    │ Whale Tracking    │         │ Sector Rotation  │
    │ Meta-Agent        │         │ Stat Arb         │
    │ Backtesting       │         │ Geopolitical     │
    │ Calibration       │         │ NFT/Gaming       │
    └───────────────────┘         └──────────────────┘
         │                                         │
         └─────────────┬───────────────────────────┘
                       ▼
              AgentPerformance DB
              (per-agent, per-category accuracy)
                       │
                       ▼
              Meta-Agent Consensus
              (weighted voting, dispersion penalty)
                       │
                       ▼
              Signal ──> Alert (Telegram push)
                       │
                       ▼
              OutcomeEvaluator ──> Feedback loop
              (updates agent weights)
```

## All 26 Agents

### Tier 1 — Core Analysis (15 agents, build first)

| # | Agent | Role | Data Sources | Self-Improvement |
|---|-------|------|-------------|------------------|
| 1 | **Technical Analysis** | RSI, MACD, SMA(7/25/50/200), Bollinger Bands, volume trend, divergences, Ichimoku cloud, ATR | FeatureBuilder (existing) | Tunes indicator thresholds based on outcome history |
| 2 | **Sentiment Analysis** | News + social sentiment aggregate, volume spikes, sentiment-price divergence, keyword trends | GDELT, Reddit, X, RSS (existing) | Adjusts source weighting by correlation with outcomes |
| 3 | **On-Chain Analysis** | DEX volume/liquidity, TVL flows, stablecoin market cap, exchange netflows, active addresses | DEX Screener, DeFi Llama, CoinGecko | Learns which on-chain metrics predict moves per token |
| 4 | **Macro Analysis** | Yield curve (T10Y2Y, T10Y3M), DXY, VIX, gold/TLT, CPI/FOMC trajectory, global PMI | FRED, Stooq, ExchangeRate (existing) | Weighs macro factors differently per asset class |
| 5 | **Fundamental Analysis** | PE ratio, EPS, dividend yield, 52w range, market cap, beta, PEG ratio, debt/equity, FCF yield | YFinance, Finnhub (new) | Learns which fundamentals matter per sector |
| 6 | **Pattern Recognition** | Chart patterns (H&S, double top/bottom, triangles, flags, wedges, cup & handle), candlestick patterns (doji, hammer, engulfing, morning star, harami), harmonic patterns | Price history (existing) | Tracks which patterns have highest hit rate per asset |
| 7 | **Volume Profile** | VWAP, volume nodes (POC), volume profile shape, accumulation/distribution zones, volume climax detection | Price + volume data | Adjusts zone sensitivity per market volatility regime |
| 8 | **Market Structure** | Higher highs/lower lows, market phases (accumulation/markup/distribution/markdown), Wyckoff schematics, fair value gaps, order blocks | Price history | Learns phase transition signals per market type |
| 9 | **Momentum** | Time-series momentum (1d/7d/30d/90d), cross-sectional momentum vs peers, momentum crash detection, trend strength (ADX) | Price history | Optimizes lookback periods per market |
| 10 | **Volatility** | Volatility regime detection (low/normal/high/crash), GARCH-style vol forecasting, vol of vol, volatility term structure, Parkinson/Garman-Klass vol | Price history | Calibrates regime thresholds from realized vol |
| 11 | **Fear & Greed** | Composite index from: price momentum, volatility, social sentiment, market dominance, Google Trends, exchange flows | Multiple sources | Learns extreme thresholds that signal reversals |
| 12 | **Whale Tracking** | Large transactions (>$100k), wallet accumulation patterns, exchange whale ratio, smart money flows, new wallet creation spikes | Blockchain APIs, Whale Alert | Tracks which whale behaviors precede moves |
| 13 | **Meta-Agent** | Weighted consensus from all agents, dispersion penalty, confidence calibration, agent performance tracking, final signal emission | All agent outputs | Continuously re-weights agents by EMA accuracy per category |
| 14 | **Backtesting** | Continuously backtests agent strategies, computes Sharpe/Calmar/win rate/profit factor, validates agent predictions against historical data | Historical DB | Discovers which agent combinations had best historical performance |
| 15 | **Calibration** | Ensures predicted confidence matches realized accuracy (70% confidence = 70% correct), Platt scaling, isotonic regression, reliability diagrams | Prediction outcomes | Adjusts each agent's confidence scaling factor |

### Tier 2 — High Value (7 agents, need some new data)

| # | Agent | Role | Data Sources | Self-Improvement |
|---|-------|------|-------------|------------------|
| 16 | **Correlation** | BTC-S&P correlation, crypto sector rotations, stock sector correlations, risk-on/off basket tracking, correlation regime shifts | Multi-asset price data | Detects correlation breakdowns and adjusts weight during regime shifts |
| 17 | **Intermarket** | Commodities→stocks (oil→energy, copper→growth), bonds→equities, DXY→international, gold→inflation expectations | FRED, YFinance, ExchangeRate | Learns lead/lag relationships between markets |
| 18 | **DeFi** | Protocol TVL growth/decline, yield farming APY trends, lending utilization rates, protocol revenue, governance activity | DeFi Llama, DEX Screener | Identifies which DeFi metrics lead token price moves |
| 19 | **L1/L2 Analysis** | Chain adoption (DAU, tx count), gas fees, bridge volumes, developer activity, ecosystem growth, total value settled | Defi Llama, chain explorers | Learns which chain metrics predict ecosystem token performance |
| 20 | **Economic Calendar** | Scheduled releases (CPI, FOMC, GDP, NFP, PMI), expected vs actual surprise, earnings dates, historical reaction patterns | FRED, Finnhub, economic calendars | Learns typical market reactions to each release type |
| 21 | **Social Media** | Influencer tracking (CT, finTwit), viral post detection, coordinated campaign detection, bot-like activity patterns, Reddit/4chan signal | Reddit, X, RSS, 4chan API | Identifies which accounts/sources have predictive value |
| 22 | **Alert/Recommendation** | Converts signals to actionable recommendations (buy/sell/watch), position sizing suggestion, stop-loss/take-profit levels, risk-adjusted entry timing | Meta-Agent output + risk guards | Learns optimal entry/exit timing from historical signals |

### Tier 3 — Specialized (4 agents, limited data but adds edge)

| # | Agent | Role | Data Sources | Self-Improvement |
|---|-------|------|-------------|------------------|
| 23 | **Sector Rotation** | Which stock/crypto sectors leading/lagging, institutional flow rotation, sector momentum ranking, sector correlation matrix | YFinance, Finnhub | Tracks rotation signal reliability across market cycles |
| 24 | **Statistical Arbitrage** | Pairs trading signals (cointegration), mean reversion z-scores, spread divergence detection, basket vs components | Price history | Learns which pairs have stable cointegration relationships |
| 25 | **Geopolitical** | Sanctions tracking, regulatory announcements, trade policy changes, central bank communications, political risk scoring | GDELT, news APIs | Correlates event types with market impact magnitude |
| 26 | **NFT/Gaming** | Blue-chip NFT floors, gaming token economies, metaverse land values, play-to-earn sustainability metrics, NFT volume trends | NFT marketplace APIs, DEX Screener | Identifies leading indicators for gaming/metaverse tokens |

## Self-Improvement System

Each agent has:
- **Internal parameter tuning**: Adjusts own thresholds/weights based on outcome history
- **Confidence calibration**: Compares predicted confidence to realized accuracy
- **Feature importance learning**: Discovers which inputs actually predict outcomes
- **Performance reporting**: Exposes accuracy, calibration error, prediction count

**Feedback loop**:
1. Agent predicts → stored in `agent_predictions`
2. Signal expires → `OutcomeEvaluator` resolves actual direction
3. `AgentPerformanceTracker` updates each agent's accuracy (EMA with decay=0.9)
4. `Meta-Agent` recalibrates weights: `weight_i = accuracy_i² / sum(all accuracy_j²)`
5. Agents self-tune internal parameters based on what worked

**Weight formula (per market category)**:
```
accuracy_ema = prev_accuracy * 0.9 + (was_correct ? 1 : 0) * 0.1
dispersion = abs(up_ratio - 0.5) * 2  # 0=agree, 1=split
penalty = dispersion * 0.2             # max 20% confidence reduction
final_confidence = weighted_confidence * (1 - penalty)
```

## New Collectors

| Collector | Source | Data | API Key |
|-----------|--------|------|---------|
| YFinance | yfinance lib | Global stocks (price, PE, EPS, dividend, 52w, beta, SMA) | None |
| DEX Screener | api.dexscreener.com | DEX pairs (price, volume, liquidity, tx count, price change) | None |
| DeFi Llama | api.llama.fi | TVL per protocol/chain, stablecoins, yields | None |
| Binance | api.binance.com | Order book imbalance, funding rates, futures premium | None |
| Finnhub | finnhub.io | Earnings surprises, analyst targets, insider trading | Free tier |
| Blockchain.com | blockchain.info | On-chain metrics (hash rate, tx count, mempool) | None |
| Whale Alert | whale-alert.io | Large crypto transactions | Free tier |

## Database Changes

**New tables**: `market_assets`, `agent_predictions`, `agent_performance`

**Signal table additions**: `meta_agent_version`, `agent_weights: JSONB`, `consensus_dispersion: float`

**Migration**: `alembic/versions/0002_add_agent_tables.py`

## Telegram Bot — 16 Commands

Existing (11): `/start`, `/help`, `/status`, `/signals`, `/markets`, `/market`, `/crypto`, `/macro`, `/news`, `/forex`, `/system`

New (5): `/agents` (agent status/weights/accuracy), `/reasoning <id>` (per-agent breakdown), `/accuracy [agent]` (history), `/signal <id>` (full detail), `/subscribe` (push alerts)

**Push alerts**: When Meta-Agent consensus confidence ≥ 0.75, auto-push to subscribers with direction, confidence %, top 3 reasons, and agent agreement level.

## Dashboard

- **Agents page** (`/agents`): 26 agent cards with status, weight, accuracy, last prediction
- **Signal Detail** (`/signal/{id}`): Per-agent voting breakdown, consensus explanation
- Updated navigation, dashboard agent summary widget
- New API endpoints: `/api/agents/summary`, `/api/agents/{name}/history`, `/api/signals/{id}/agents`

## Claude Code Optimization (per guide)

### Tier 1: CLAUDE.md (always loaded, <1,000 tokens in first 200 lines)
```
# Market Signal Engine
Multi-agent prediction system: 26 agents → Meta-Agent consensus → signals.
Crypto + global stocks. FastAPI + PostgreSQL + APScheduler + Telegram.

## Critical Rules
- Never commit .env or secrets
- READ_ONLY_SIGNAL_ONLY=true enforced — no trade execution
- Run tests before commit: docker compose run --rm app pytest
- All DB changes require Alembic migration

## Quick Start
- Dev: docker compose up --build
- URL: http://localhost:8000 (admin/debug)
- Health: curl localhost:8000/health
- DB: docker compose exec postgres psql -U market
- Migrations: docker compose exec app alembic upgrade head

## Architecture
agents/ — 26 analysis agents | collectors/ — 14+ data sources | prediction/ — meta-agent consensus
database/ — SQLAlchemy models + repository | jobs/ — APScheduler cycles | telegram/ — bot commands
dashboard/ — FastAPI + Jinja2 | docs/ — detailed docs

## Key Files
- settings: market_signal_engine/config/settings.py
- models: market_signal_engine/database/models.py
- routes: market_signal_engine/dashboard/routes.py
- tasks: market_signal_engine/jobs/tasks.py
- Full index: docs/INDEX.md
```

### Tier 2: Topic-specific docs (loaded on demand)
- `docs/AGENTS.md` — Agent architecture deep dive
- `docs/COLLECTORS.md` — Data sources and schemas
- `docs/DATABASE.md` — Schema, migrations, queries
- `docs/TELEGRAM.md` — Bot commands and push alerts
- `docs/DASHBOARD.md` — Routes, templates, JS

### Tier 3: Reference (as needed)
- `docs/troubleshooting/` — Common errors
- `docs/DEPLOY.md` — Production deployment

### Session Hook (`.claude/hooks/session-start.sh`)
Checks: docker status, git state, recent errors in logs. Shows only relevant context.

## Implementation Order

1. **DB Foundation**: models + migration + repository (1-2 sessions)
2. **Agent ABC + Registry + PerformanceTracker** (1 session)
3. **New Collectors**: YFinance, DEX Screener, DeFi Llama, Binance, Finnhub, Blockchain, Whale Alert (2-3 sessions)
4. **Tier 1 Agents** (15 agents, 4-5 sessions):
   - Batch A: Technical, Sentiment, Macro, Fundamental, Momentum (use existing data)
   - Batch B: Pattern Recognition, Volume Profile, Market Structure, Volatility, Fear & Greed (price data)
   - Batch C: On-Chain, Whale Tracking, Meta-Agent, Backtesting, Calibration
5. **Tier 2 Agents** (7 agents, 2-3 sessions): Correlation, Intermarket, DeFi, L1/L2, Economic Calendar, Social Media, Alert/Recommendation
6. **Tier 3 Agents** (4 agents, 1-2 sessions): Sector Rotation, Stat Arb, Geopolitical, NFT/Gaming
7. **Meta-Agent + Orchestrator**: Consensus engine with all agents (1 session)
8. **Pipeline Integration**: Wire into tasks.py + OutcomeEvaluator + feedback loop (1 session)
9. **Telegram Enhancements**: 5 new commands + push alert system (1 session)
10. **Dashboard**: Agents page, signal detail, API endpoints (1-2 sessions)
11. **Claude Optimization**: CLAUDE.md, docs/ structure, session hook (1 session)
12. **Testing + QA**: Codex review of all agents, Gemini for doc validation (1-2 sessions)

## File Summary

**Create (35+ files)**:
- `market_signal_engine/agents/` — 28 files (base, registry, 26 agent files including meta)
- `market_signal_engine/collectors/` — 7 new collectors
- `alembic/versions/0002_add_agent_tables.py`
- `market_signal_engine/dashboard/templates/agents.html`
- `market_signal_engine/dashboard/templates/signal_detail.html`
- `market_signal_engine/dashboard/static/js/agents.js`
- `docs/AGENTS.md`, `docs/INDEX.md`, `docs/COLLECTORS.md`, `docs/TELEGRAM.md`
- `CLAUDE.md` (rewritten)
- `.claude/hooks/session-start.sh`

**Modify (14 files)**: models.py, repository.py, settings.py, tasks.py, outcomes.py, handlers.py, formatters.py, bot.py, routes.py, dashboard.html, base.html, pyproject.toml, docker-compose.yml, __init__.py files

**Delete (1)**: `prediction/model.py` (BaselineSignalModel replaced by agent system)

## Verification

1. `docker compose up --build` — starts with 26 agents initialized
2. `curl localhost:8000/health` — `{"ok":true, "agents":26, "scheduler_running":true}`
3. `curl localhost:8000/api/agents/summary` — returns all 26 agents with performance
4. Wait 1-2 collection cycles — signals generated with per-agent predictions stored
5. Telegram: `/agents` shows all agents, `/reasoning 123` shows per-agent breakdown
6. Dashboard `/agents` — all agent cards visible with live accuracy
7. After signals resolve: weights shift toward accurate agents, away from inaccurate ones
8. Codex review passes on all agent implementations
9. Claude session context usage reduced 60%+ via tiered docs
