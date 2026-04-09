import numpy as np
import pandas as pd


def calculate_confluence_score(df: pd.DataFrame) -> tuple:
    """Count independent confirming signals (0-10).

    Each signal tests a different dimension of the market:
    trend, momentum, volume, price structure, volatility,
    Ichimoku, Stochastic RSI, Bollinger, DeMark, Volume Profile.

    Returns (score, list_of_reasons).
    """
    if len(df) < 200:
        return 0, ["Utilstrekkelig data"]

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    # --- 1. TREND: EMA alignment + ADX ---
    ema50 = curr.get('EMA50', 0)
    ema200 = curr.get('EMA200', 0)
    adx = curr.get('ADX', 0)
    price = curr['Close']

    if price > ema50 > ema200:
        if adx > 25:
            score += 1
            reasons.append(f"Sterk trend (ADX {adx:.0f})")
        else:
            # Partial — trend alignment but weak ADX
            reasons.append(f"Trend OK men svak ADX ({adx:.0f})")

    # --- 2. MOMENTUM: RSI + MACD histogram ---
    rsi = curr.get('RSI', 50)
    hist = curr.get('Hist', 0)
    prev_hist = prev.get('Hist', 0)

    if 40 <= rsi <= 60 and hist > prev_hist:
        score += 1
        reasons.append(f"Momentum bekreftet (RSI {rsi:.0f}, MACD stigende)")
    elif 30 <= rsi < 40 and hist > prev_hist:
        score += 1
        reasons.append(f"RSI pullback + MACD snur (RSI {rsi:.0f})")

    # --- 3. VOLUME: RVOL above average ---
    rvol = curr.get('RVOL', 0)
    if rvol > 1.2:
        score += 1
        reasons.append(f"Volum bekreftet (RVOL {rvol:.1f}x)")

    # --- 4. PRICE STRUCTURE: Higher lows ---
    if len(df) >= 10:
        recent_lows = df['Low'].tail(10).values
        if recent_lows[-1] > recent_lows[0] and recent_lows[-1] > np.min(recent_lows[:5]):
            score += 1
            reasons.append("Higher lows (stigende bunner)")

    # --- 5. VOLATILITY: ATR squeeze (contraction before expansion) ---
    atr_now = curr.get('ATR', 0)
    if len(df) >= 20:
        atr_avg = df['ATR'].tail(20).mean()
        bb_width = curr.get('BB_Width', 999)
        bb_avg = df['BB_Width'].tail(50).mean() if 'BB_Width' in df.columns else 999

        if atr_now < atr_avg * 0.8 or bb_width < bb_avg * 0.8:
            score += 1
            reasons.append("Volatilitetssqueeze (kontraksjon)")

    # --- 6. ICHIMOKU: Price above cloud ---
    if curr.get('Ichimoku_Above', 0) == 1:
        tenkan = curr.get('Ichimoku_Tenkan', 0)
        kijun = curr.get('Ichimoku_Kijun', 0)
        if tenkan > kijun:
            score += 1
            reasons.append("Over Ichimoku-sky (bullish)")

    # --- 7. STOCHASTIC RSI: Oversold crossover ---
    stoch_k = curr.get('StochRSI_K', 50)
    stoch_d = curr.get('StochRSI_D', 50)
    prev_k = prev.get('StochRSI_K', 50)
    prev_d = prev.get('StochRSI_D', 50)

    if stoch_k < 30 or (prev_k < prev_d and stoch_k > stoch_d and stoch_k < 50):
        score += 1
        reasons.append(f"StochRSI signal ({stoch_k:.0f})")

    # --- 8. BOLLINGER: Near lower band with rising momentum ---
    bb_lower = curr.get('BB_Lower', 0)
    if bb_lower > 0 and price > 0:
        dist_to_lower = (price - bb_lower) / price
        if dist_to_lower < 0.02 and hist > prev_hist:
            score += 1
            reasons.append("Bollinger bounce")

    # --- 9. DEMARK: Setup 9 or Countdown 13 ---
    td_setup = curr.get('TD_Buy_Setup', 0)
    td_count = curr.get('TD_Buy_Countdown', 0)
    if td_count == 13:
        score += 1
        reasons.append("DeMark Countdown 13")
    elif td_setup == 9:
        score += 1
        reasons.append("DeMark Setup 9")

    # --- 10. VOLUME PROFILE: Price near POC support ---
    poc = curr.get('VPVR_POC', None)
    if poc is not None and not np.isnan(poc) and poc > 0:
        dist_to_poc = abs(price - poc) / price
        if dist_to_poc < 0.03:
            score += 1
            reasons.append(f"Nær POC-støtte ({poc:.2f})")

    return score, reasons


def calculate_entry_quality(df: pd.DataFrame, regime: str) -> tuple:
    """Rate overall entry quality considering market regime.

    Returns (quality_score 0-100, label).
    """
    confluence, _ = calculate_confluence_score(df)
    curr = df.iloc[-1]

    # Base quality from confluence
    quality = confluence * 10

    # Regime adjustment
    if regime == "BULL":
        quality += 10
    elif regime == "FEAR":
        quality -= 30
    elif regime == "BEAR":
        quality -= 20

    # Bonus: EMA9 > EMA21 (short-term bullish)
    if curr.get('EMA9', 0) > curr.get('EMA21', 0):
        quality += 5

    # Bonus: +DI > -DI (directional confirmation)
    if curr.get('+DI', 0) > curr.get('-DI', 0):
        quality += 5

    # Fibonacci support
    fib62 = curr.get('Fib_62', None)
    if fib62 is not None and not np.isnan(fib62):
        dist = abs(curr['Close'] - fib62) / curr['Close']
        if dist < 0.02:
            quality += 10

    quality = max(0, min(100, quality))

    if quality >= 80:
        label = "A+"
    elif quality >= 65:
        label = "A"
    elif quality >= 50:
        label = "B"
    elif quality >= 35:
        label = "C"
    else:
        label = "D"

    return quality, label
