"""YFinance collector — global stock fundamentals and price data.

Data: price, PE ratio, EPS, dividend yield, 52w range, market cap, beta,
      PEG ratio, debt/equity, FCF yield, SMA(50/200).
"""

from __future__ import annotations

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class YFinanceCollector(BaseCollector):
    source = "yfinance"
    rate_limit_sec = 2.0
    cache_ttl_sec = 120.0  # stock data changes slowly

    def _fetch(self, symbol: str) -> CollectorResult:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        history = ticker.history(period="6mo")
        prices = history["Close"].tolist() if not history.empty else []
        volumes = history["Volume"].tolist() if not history.empty else []

        data = {
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "open": info.get("open"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "50d_avg": info.get("fiftyDayAverage"),
            "200d_avg": info.get("twoHundredDayAverage"),
            "peg_ratio": info.get("pegRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "prices_6mo": prices[-90:] if len(prices) > 90 else prices,
            "volumes_6mo": volumes[-90:] if len(volumes) > 90 else volumes,
        }

        return CollectorResult(source=self.source, symbol=symbol, data=data, raw=info)
