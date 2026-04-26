"""
wold_scanner.py
---------------
Thomas Wold-inspirert momentumscanner for Oslo Børs.

Filosofi (fra intervju):
  "Du trenger egentlig tre ting: økende volum, stigende kurs og noe fundamentalt."
  "Dagens vinnere blir morgendagens vinnere — taperne fortsetter å falle."
  "Når aksjen først begynner å gå, vil ikke folk selge."

Wold Score (0–100) måler akkurat dette — med VOLUM som tyngst vektet faktor.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OSLO BØRS SEKTORDEFINISJON
# ---------------------------------------------------------------------------
OSLO_SECTORS = {
    "Shipping": [
        "FRO.OL", "GOGL.OL", "HAFNI.OL", "BWLPG.OL", "FLNG.OL", "MPCC.OL",
        "OET.OL", "SIOFF.OL", "2020.OL", "BELCO.OL", "COOL.OL", "AKER.OL",
    ],
    "Olje & Energi": [
        "EQNR.OL", "AKRBP.OL", "PGS.OL", "TGS.OL", "DNO.OL", "OKEA.OL",
        "VAR.OL", "PHEL.OL", "RECSI.OL", "EPR.OL", "ADE.OL",
    ],
    "Laks & Sjømat": [
        "MOWI.OL", "SALM.OL", "LSG.OL", "BAKKA.OL", "NRS.OL", "AUSS.OL",
    ],
    "Finans": [
        "DNB.OL", "STB.OL", "GJF.OL", "SRBNK.OL", "SKUE.OL", "BOUV.OL",
        "ENTRA.OL", "AFG.OL",
    ],
    "Industri & Material": [
        "NHY.OL", "YAR.OL", "SUBC.OL", "KOG.OL", "RANA.OL", "BON.OL",
        "SCHA.OL", "SCHB.OL",
    ],
    "Telecom": [
        "TEL.OL", "TOM.OL", "NAS.OL",
    ],
    "Tech & Software": [
        "ATEA.OL", "KAHOT.OL", "LINK.OL", "VOLUE.OL", "NOD.OL", "IDEX.OL",
        "CLOUD.OL", "ZAP.OL", "NORBT.OL",
    ],
    "Forbruker": [
        "ORK.OL", "XXL.OL", "AUTO.OL", "SATS.OL", "GOLF.OL", "HUNT.OL",
    ],
}

# Omvendt oppslag: ticker → sektor
TICKER_TO_SECTOR = {
    t: sector
    for sector, tickers in OSLO_SECTORS.items()
    for t in tickers
}


# ---------------------------------------------------------------------------
# WOLD SCORE
# ---------------------------------------------------------------------------
def calculate_wold_score(df: pd.DataFrame, ticker: str = "") -> tuple:
    """
    Beregn Thomas Wold-inspirert momentum-score (0–100).

    Retur: (score, reasons, disqualified_reason)
    - score: int 0–100
    - reasons: list[str] — hva som bidro positivt
    - disqualified_reason: str | None — hvis aksjen er diskvalifisert
    """
    if len(df) < 50:
        return 0, [], "Utilstrekkelig data"

    curr = df.iloc[-1]
    score = 0
    reasons = []

    close = df['Close']
    volume = df['Volume']

    # --- DISQUALIFIERS (Wold: "Faller det, vil jeg ikke ha det") ---
    # Sjekk for lavere lavpunkter siste 5 dager (falling stock)
    lows_5d = df['Low'].tail(5).values
    if lows_5d[-1] < lows_5d[0] and lows_5d[-1] < np.min(lows_5d[1:4]):
        return 0, [], "Falling knife — lavere lavpunkter (unngå)"

    # Under EMA50 med negativ momentum (Wold kjøper ikke i nedtrend)
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    last_close = close.iloc[-1]

    roc_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    roc_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(df) > 6 else 0
    roc_20d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100 if len(df) > 21 else 0

    if last_close < ema50 and roc_5d < -5:
        return 0, [], "Under EMA50 med negativ momentum — Wold-filter"

    # -------------------------------------------------------------------
    # 1. VOLUM — TYNGST VEKTET (Wold: "Det jeg ser mest på er volum")
    # -------------------------------------------------------------------
    rvol = curr.get('RVOL', 0)
    vol_avg_5d = volume.tail(5).mean()
    vol_avg_20d = volume.tail(20).mean()
    vol_buildup = vol_avg_5d > vol_avg_20d * 1.3  # Volumet bygger seg opp over 5 dager

    if rvol >= 3.0:
        score += 35
        reasons.append(f"Ekstremt volum (RVOL {rvol:.1f}x) — institusjonelle kjøpere?")
    elif rvol >= 2.0:
        score += 25
        reasons.append(f"Svart høyt volum (RVOL {rvol:.1f}x)")
    elif rvol >= 1.5:
        score += 15
        reasons.append(f"Høyt volum (RVOL {rvol:.1f}x)")
    elif rvol >= 1.2:
        score += 8
        reasons.append(f"Noe over snitt-volum (RVOL {rvol:.1f}x)")

    if vol_buildup and rvol > 1.0:
        score += 5
        reasons.append("Volum bygger seg opp over 5 dager (sustained interest)")

    # -------------------------------------------------------------------
    # 2. KURSBEVEGELSE — "stigende kurs"
    # -------------------------------------------------------------------
    if roc_1d >= 5:
        score += 15
        reasons.append(f"Sterk dagsbevegelse +{roc_1d:.1f}%")
    elif roc_1d >= 2:
        score += 10
        reasons.append(f"God dagsbevegelse +{roc_1d:.1f}%")
    elif roc_1d >= 0.5:
        score += 5
        reasons.append(f"Positiv dag +{roc_1d:.1f}%")
    elif roc_1d < -2:
        score -= 5  # Straff for negativ dag med analyse-trigger

    if roc_5d >= 10:
        score += 10
        reasons.append(f"Sterk 5-dagers trend +{roc_5d:.1f}%")
    elif roc_5d >= 5:
        score += 7
        reasons.append(f"Positiv 5-dagers trend +{roc_5d:.1f}%")
    elif roc_5d >= 2:
        score += 4
        reasons.append(f"5d ROC +{roc_5d:.1f}%")

    if roc_20d >= 20:
        score += 5
        reasons.append(f"Sterk 20-dagers trend +{roc_20d:.1f}%")
    elif roc_20d >= 10:
        score += 3
        reasons.append(f"20d ROC +{roc_20d:.1f}%")

    # -------------------------------------------------------------------
    # 3. TREND-STRUKTUR — "dagens vinnere blir morgendagens vinnere"
    # -------------------------------------------------------------------
    if last_close > ema20 > ema50:
        score += 12
        reasons.append("EMA20 > EMA50 — bullish alignment")
    elif last_close > ema20:
        score += 6
        reasons.append("Over EMA20")

    # Higher highs siste 5 dager (Wold: momentum continuation)
    highs_5d = df['High'].tail(5).values
    if highs_5d[-1] > highs_5d[0] and highs_5d[-2] >= highs_5d[0]:
        score += 8
        reasons.append("Higher highs — momentum continuation")

    # -------------------------------------------------------------------
    # 4. NÆR 52-UKERS TOPP — Wold kjøper styrke, ikke svakhet
    # -------------------------------------------------------------------
    high_52w = df['High'].tail(252).max()
    dist_from_high = (high_52w - last_close) / high_52w

    if dist_from_high <= 0.03:
        score += 15
        reasons.append(f"Nær 52-ukers topp ({dist_from_high*100:.1f}% unna) — breakout-sone")
    elif dist_from_high <= 0.08:
        score += 8
        reasons.append(f"Innen 8% fra 52-ukers topp")
    elif dist_from_high <= 0.15:
        score += 3
        reasons.append(f"Innen 15% fra 52-ukers topp")

    # -------------------------------------------------------------------
    # 5. RSI-MOMENTUM (ikke oversold-søker — Wold vil ha momentum)
    # -------------------------------------------------------------------
    rsi = curr.get('RSI', 50)
    if 55 <= rsi <= 75:
        score += 5
        reasons.append(f"RSI {rsi:.0f} — momentum-sone (ikke overkjøpt)")
    elif 45 <= rsi < 55:
        score += 3
        reasons.append(f"RSI {rsi:.0f} — nøytral men ikke svak")

    # -------------------------------------------------------------------
    # SEKTOR-TAG
    # -------------------------------------------------------------------
    sektor = TICKER_TO_SECTOR.get(ticker, "Ukjent")

    return max(0, min(100, score)), reasons, None


# ---------------------------------------------------------------------------
# WOLD FILTER — "Jeg gidder ikke ta aksjer som faller"
# ---------------------------------------------------------------------------
def passes_wold_filter(df: pd.DataFrame) -> tuple:
    """
    Sjekk om en aksje består Wolds grunnleggende krav.

    Retur: (passed: bool, reason: str)
    """
    if len(df) < 20:
        return False, "For lite data"

    close = df['Close']
    last = close.iloc[-1]

    # Krav 1: Ikke making lower lows siste 5 dager
    lows = df['Low'].tail(5).values
    if lows[-1] < lows[0] and lows[-1] < lows[-2]:
        return False, "Lavere lavpunkter — Wold unngår fallende aksjer"

    # Krav 2: Positiv momentum siste 5 dager
    roc_5d = (last - close.iloc[-6]) / close.iloc[-6] * 100 if len(df) > 6 else 0
    if roc_5d < -10:
        return False, f"5-dagers ROC {roc_5d:.1f}% — for negativ"

    # Krav 3: Ikke i ekstrem nedtrend (pris < 80% av 50-dagers snitt)
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    if last < ema50 * 0.80:
        return False, "Mer enn 20% under EMA50 — unngå"

    return True, "OK"


# ---------------------------------------------------------------------------
# MOMENTUM CONTINUATION SIGNAL
# Wold: "at dagens vinnere ofte blir morgendagens vinnere"
# ---------------------------------------------------------------------------
def detect_momentum_continuation(df: pd.DataFrame) -> dict:
    """
    Sjekk om aksjen viser tegn til momentum-fortsettelse.

    Ser etter: volumøkning + kursstigning over flere dager i strekk.
    """
    if len(df) < 10:
        return {"signal": False, "streak": 0, "vol_trend": False}

    close = df['Close']
    volume = df['Volume']

    # Tell dager med positiv close + volum over snitt
    streak = 0
    for i in range(-1, -6, -1):
        day_positive = close.iloc[i] > close.iloc[i - 1]
        vol_above_avg = volume.iloc[i] > volume.tail(20).mean()
        if day_positive and vol_above_avg:
            streak += 1
        else:
            break

    # Sjekk om volumtrenden er stigende (3-dagers snitt mot 10-dagers snitt)
    vol_3d = volume.tail(3).mean()
    vol_10d = volume.tail(10).mean()
    vol_trend = vol_3d > vol_10d * 1.2

    return {
        "signal": streak >= 2,
        "streak": streak,
        "vol_trend": vol_trend,
        "description": (
            f"{streak} dager på rad med positiv kurs + volum over snitt"
            if streak >= 2
            else "Ingen klar momentum-fortsettelse ennå"
        ),
    }


# ---------------------------------------------------------------------------
# TRIGGER PROXIMITY
# Wold: "Jeg liker å sitte inne før en forventet trigger og selge før den kommer"
# ---------------------------------------------------------------------------
def estimate_trigger_proximity(df: pd.DataFrame) -> dict:
    """
    Enkel heuristikk for å estimere om vi er nær en trigger.

    Ser etter: ATR-kontraksjon + lavt volum (konsolidering = nær breakout).
    """
    if len(df) < 30:
        return {"near_trigger": False, "type": None}

    curr = df.iloc[-1]
    atr_now = curr.get('ATR', 0)
    atr_20d = df['ATR'].tail(20).mean() if 'ATR' in df.columns else atr_now

    # Bollinger Band squeeze
    bb_width = curr.get('BB_Width', None)
    bb_width_20d = df['BB_Width'].tail(20).mean() if 'BB_Width' in df.columns else None

    atr_squeeze = atr_now < atr_20d * 0.75 if atr_20d > 0 else False
    bb_squeeze = (bb_width < bb_width_20d * 0.7) if (bb_width and bb_width_20d) else False

    if atr_squeeze and bb_squeeze:
        return {
            "near_trigger": True,
            "type": "Dobbelt squeeze (ATR + BB) — mulig breakout nært forestående",
        }
    elif atr_squeeze:
        return {
            "near_trigger": True,
            "type": "ATR-kontraksjon — konsolidering",
        }
    elif bb_squeeze:
        return {
            "near_trigger": True,
            "type": "Bollinger squeeze — energi bygges opp",
        }

    return {"near_trigger": False, "type": None}
