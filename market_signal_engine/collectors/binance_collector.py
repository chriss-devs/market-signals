"""Binance collector — order book, funding rates, futures premium, klines.

Public endpoints, no API key required for market data.
"""

from __future__ import annotations

import json
import urllib.request

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class BinanceCollector(BaseCollector):
    source = "binance"
    base_url = "https://api.binance.com/api/v3"
    fapi_url = "https://fapi.binance.com/fapi/v1"
    rate_limit_sec = 0.2
    cache_ttl_sec = 30.0

    def _fetch(self, symbol: str) -> CollectorResult:
        # Normalize symbol: "BTC/USDT" → "BTCUSDT"
        clean = symbol.replace("/", "").replace("-", "").upper()

        data: dict = {}

        # Spot ticker
        data.update(self._fetch_ticker(clean))

        # Order book depth
        try:
            data.update(self._fetch_orderbook(clean))
        except Exception:
            pass

        # Futures data (funding rate, open interest)
        try:
            data.update(self._fetch_futures(clean))
        except Exception:
            pass

        # Recent klines
        try:
            data.update(self._fetch_klines(clean))
        except Exception:
            pass

        return CollectorResult(source=self.source, symbol=symbol, data=data)

    def _fetch_ticker(self, symbol: str) -> dict:
        url = f"{self.base_url}/ticker/24hr?symbol={symbol}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            ticker = json.loads(resp.read().decode())

        return {
            "price": float(ticker.get("lastPrice", 0)),
            "price_change_pct": float(ticker.get("priceChangePercent", 0)),
            "high_24h": float(ticker.get("highPrice", 0)),
            "low_24h": float(ticker.get("lowPrice", 0)),
            "volume_24h": float(ticker.get("volume", 0)),
            "quote_volume_24h": float(ticker.get("quoteVolume", 0)),
            "trades_24h": int(ticker.get("count", 0)),
        }

    def _fetch_orderbook(self, symbol: str) -> dict:
        url = f"{self.base_url}/depth?symbol={symbol}&limit=100"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            ob = json.loads(resp.read().decode())

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        bid_vol = sum(float(b[1]) for b in bids[:25])
        ask_vol = sum(float(a[1]) for a in asks[:25])
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        # Bid/ask spread
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

        return {
            "orderbook_imbalance": round(imbalance, 4),
            "bid_ask_spread_pct": round(spread_pct, 4),
            "bid_volume_25": round(bid_vol, 2),
            "ask_volume_25": round(ask_vol, 2),
            "best_bid": best_bid,
            "best_ask": best_ask,
        }

    def _fetch_futures(self, symbol: str) -> dict:
        # Funding rate
        fr_url = f"{self.fapi_url}/fundingRate?symbol={symbol}&limit=1"
        fr_req = urllib.request.Request(fr_url, headers=self._headers())
        with urllib.request.urlopen(fr_req, timeout=10) as fr_resp:
            fr_data = json.loads(fr_resp.read().decode())
        funding_rate = float(fr_data[0]["fundingRate"]) if fr_data else 0.0

        # Open interest
        oi_url = f"{self.fapi_url}/openInterest?symbol={symbol}"
        oi_req = urllib.request.Request(oi_url, headers=self._headers())
        with urllib.request.urlopen(oi_req, timeout=10) as oi_resp:
            oi_data = json.loads(oi_resp.read().decode())
        open_interest = float(oi_data.get("openInterest", 0))

        # Premium (futures vs spot)
        try:
            f_price_url = f"{self.fapi_url}/ticker/price?symbol={symbol}"
            fp_req = urllib.request.Request(f_price_url, headers=self._headers())
            with urllib.request.urlopen(fp_req, timeout=10) as fp_resp:
                fp_data = json.loads(fp_resp.read().decode())
            futures_price = float(fp_data.get("price", 0))
            # Spot price should already be in data from _fetch_ticker
        except Exception:
            futures_price = 0

        return {
            "funding_rate": round(funding_rate, 6),
            "futures_open_interest": open_interest,
            "futures_price": futures_price,
        }

    def _fetch_klines(self, symbol: str) -> dict:
        url = f"{self.base_url}/klines?symbol={symbol}&interval=1h&limit=24"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            klines = json.loads(resp.read().decode())

        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        return {
            "hourly_closes_24": closes,
            "hourly_volumes_24": volumes,
        }
