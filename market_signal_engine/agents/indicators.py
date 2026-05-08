"""Shared technical indicator computations. Pure functions, no state."""

from __future__ import annotations

import math


def sma(prices: list[float], period: int) -> list[float]:
    """Simple Moving Average. Returns same-length list with NaN for leading window."""
    if len(prices) < period:
        return [math.nan] * len(prices)
    result: list[float] = []
    window_sum = sum(prices[:period])
    for i in range(len(prices)):
        if i < period - 1:
            result.append(math.nan)
        else:
            if i >= period:
                window_sum += prices[i] - prices[i - period]
            result.append(window_sum / period)
    return result


def ema(prices: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if len(prices) < 2:
        return [math.nan] * len(prices)
    multiplier = 2.0 / (period + 1)
    result: list[float] = []
    ema_val = prices[0]
    for i, p in enumerate(prices):
        if i == 0:
            result.append(math.nan)
        else:
            ema_val = (p - ema_val) * multiplier + ema_val
            result.append(ema_val if i >= period - 1 else math.nan)
    return result


def rsi(prices: list[float], period: int = 14) -> list[float]:
    """Relative Strength Index. Values 0-100."""
    if len(prices) < period + 1:
        return [math.nan] * len(prices)
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    result = [math.nan] * (period)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return result


def macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float]]:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line: list[float] = []
    for f, s in zip(ema_fast, ema_slow):
        if math.isnan(f) or math.isnan(s):
            macd_line.append(math.nan)
        else:
            macd_line.append(f - s)

    valid = [v for v in macd_line if not math.isnan(v)]
    signal_line = ema(valid, signal) if len(valid) >= signal else [math.nan] * len(macd_line)

    # Pad signal line to match original length
    nan_prefix = sum(1 for v in macd_line if math.isnan(v))
    signal_padded = [math.nan] * nan_prefix + signal_line

    histogram = []
    for m, s in zip(macd_line, signal_padded):
        if math.isnan(m) or math.isnan(s):
            histogram.append(math.nan)
        else:
            histogram.append(m - s)

    return {"macd": macd_line, "signal": signal_padded, "histogram": histogram}


def bollinger_bands(prices: list[float], period: int = 20, std_dev: float = 2.0) -> dict[str, list[float]]:
    """Bollinger Bands: upper, middle (SMA), lower."""
    middle = sma(prices, period)
    upper: list[float] = []
    lower: list[float] = []
    for i, m in enumerate(middle):
        if math.isnan(m) or i < period - 1:
            upper.append(math.nan)
            lower.append(math.nan)
        else:
            window = prices[i - period + 1 : i + 1]
            mean = sum(window) / period
            variance = sum((x - mean) ** 2 for x in window) / period
            sd = math.sqrt(variance)
            upper.append(m + std_dev * sd)
            lower.append(m - std_dev * sd)
    return {"upper": upper, "middle": middle, "lower": lower}


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    """Average True Range."""
    if len(closes) < 2:
        return [math.nan] * len(closes)
    trs: list[float] = [math.nan]
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    result: list[float] = []
    tr_window = [t for t in trs if not math.isnan(t)]
    if len(tr_window) < period:
        return [math.nan] * len(closes)

    atr_val = sum(tr_window[:period]) / period
    for i in range(len(closes)):
        if i < period:
            result.append(math.nan)
        else:
            atr_val = (atr_val * (period - 1) + (trs[i] or 0)) / period
            result.append(atr_val)
    return result


def adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    """Average Directional Index. Values 0-100, >25 = trending."""
    if len(closes) < period + 1:
        return [math.nan] * len(closes)

    tr: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, len(closes)):
        h_diff = highs[i] - highs[i - 1]
        l_diff = lows[i - 1] - lows[i]
        tr_val = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        tr.append(tr_val)
        plus_dm.append(h_diff if h_diff > l_diff and h_diff > 0 else 0)
        minus_dm.append(l_diff if l_diff > h_diff and l_diff > 0 else 0)

    atr_vals = [sum(tr[:period]) / period]
    for i in range(period, len(tr)):
        atr_vals.append((atr_vals[-1] * (period - 1) + tr[i]) / period)

    plus_di_vals = [100 * sum(plus_dm[:period]) / period / atr_vals[0] if atr_vals[0] else 0]
    minus_di_vals = [100 * sum(minus_dm[:period]) / period / atr_vals[0] if atr_vals[0] else 0]
    for i in range(period, len(tr)):
        p = 100 * ((plus_di_vals[-1] * (period - 1) + plus_dm[i]) / period) / atr_vals[i - period + 1] if atr_vals[i - period + 1] else 0
        m = 100 * ((minus_di_vals[-1] * (period - 1) + minus_dm[i]) / period) / atr_vals[i - period + 1] if atr_vals[i - period + 1] else 0
        plus_di_vals.append(p)
        minus_di_vals.append(m)

    dx_vals: list[float] = []
    for p, m in zip(plus_di_vals, minus_di_vals):
        denom = p + m
        dx_vals.append(100 * abs(p - m) / denom if denom else 0)

    result = [math.nan] * (period + 1)
    if len(dx_vals) > period:
        adx_val = sum(dx_vals[:period]) / period
        result.append(adx_val)
        for i in range(period, len(dx_vals)):
            adx_val = (adx_val * (period - 1) + dx_vals[i]) / period
            result.append(adx_val)
    return result


def slope(values: list[float], lookback: int = 5) -> float:
    """Linear regression slope over the last N valid values."""
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) < lookback:
        return 0.0
    window = valid[-lookback:]
    n = len(window)
    x_mean = (n - 1) / 2.0
    y_mean = sum(window) / n
    num = sum((i - x_mean) * (window[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def last_valid(values: list[float]) -> float:
    """Return the last non-NaN value, or 0.0."""
    for v in reversed(values):
        if not math.isnan(v):
            return v
    return 0.0


def pct_change(prices: list[float], lookback: int = 1) -> float:
    """Percentage change over lookback periods."""
    if len(prices) < lookback + 1:
        return 0.0
    old = prices[-lookback - 1]
    new = prices[-1]
    return (new - old) / old * 100 if old else 0.0
