import streamlit as st
import yfinance as yf
import pandas as pd


@st.cache_data(ttl=3600)
def get_market_regime() -> tuple:
    """Determine market regime using SPY trend + VIX level.

    Returns (regime_name, description, color_hex).
    """
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if hasattr(spy.index, 'tz_localize'):
            spy.index = spy.index.tz_localize(None)

        if len(spy) < 200:
            spy_ema50 = spy['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            spy_ema200 = spy['Close'].rolling(min(200, len(spy))).mean().iloc[-1]
        else:
            spy_ema50 = spy['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            spy_ema200 = spy['Close'].ewm(span=200, adjust=False).mean().iloc[-1]

        spy_above = spy_ema50 > spy_ema200

        # SPY recent momentum (20-day ROC)
        spy_roc = (spy['Close'].iloc[-1] / spy['Close'].iloc[-20] - 1) * 100 if len(spy) > 20 else 0

        vix_level = 20.0
        try:
            vix = yf.Ticker("^VIX").history(period="1mo")
            if len(vix) > 0:
                vix_level = vix['Close'].iloc[-1]
        except Exception:
            pass

        # Regime classification
        if spy_above and vix_level < 18:
            return (
                "BULL",
                f"Sterk opptrend. SPY EMA50>200, VIX={vix_level:.1f}. Full posisjon.",
                "#1a7a1a30",
            )
        elif spy_above and vix_level < 25:
            return (
                "BULL (forsiktig)",
                f"Opptrend men noe volatilitet. VIX={vix_level:.1f}. Normal posisjon.",
                "#6b8e2330",
            )
        elif vix_level >= 30:
            return (
                "FEAR",
                f"Hoyt stressniva. VIX={vix_level:.1f}. Halver posisjoner eller sta utenfor.",
                "#ff000030",
            )
        elif not spy_above and spy_roc < -5:
            return (
                "BEAR",
                f"Nedtrend. SPY EMA50<200, ROC={spy_roc:.1f}%. Kun short eller cash.",
                "#8b000030",
            )
        else:
            return (
                "NOYTRAL",
                f"Usikkert marked. VIX={vix_level:.1f}. Reduser posisjon.",
                "#ffa50030",
            )
    except Exception:
        return ("UKJENT", "Kunne ikke hente markedsdata.", "#80808030")
