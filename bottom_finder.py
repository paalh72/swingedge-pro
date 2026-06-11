"""
Bunnfisker — screener for å identifisere bunner eller rett-etter-bunnen
i aksjesykler for swingtrading.

Kombinerer det beste fra tidligere screener-versjoner:
- RSI-syklus hit-rate ("trekksikkerhet") fra rsi-trend-screener/aksjescreener
- Stigende bunn-streak og bunnprojeksjon fra Ultimate Trend Scanner v44
- DeMark Setup 9 / Countdown 13 bunntiming fra oldapp12+
- Golden Score-elementer: BB-bounce, MACD-vending, RVOL fra oldapp5+
- VPVR/POC-støtte og ATR stop-loss fra oldapp42/43
- Bullish divergens og kapitulasjonsvolum (nytt)

Eksporterer: render_bottom_finder_tab()
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

from data_sources import get_oslo_all_tickers
from data_fetcher import get_stock_data
from indicators import calculate_all_indicators
from risk_manager import calculate_position_size


# ---------------------------------------------------------------------------
# 1. RSI-SYKLUSANALYSE — "trekksikkerhet" (fra rsi-trend-screener)
# ---------------------------------------------------------------------------

def analyze_rsi_cycles(
    df: pd.DataFrame,
    rsi_low: float = 32.0,
    min_gain: float = 0.08,
    max_days: int = 60,
) -> dict:
    """Finn historiske RSI-bunner og mål hvor ofte aksjen steg etterpå.

    En "bunn-hendelse" starter når RSI krysser under rsi_low og slutter
    når RSI krysser over igjen. Bunnpris = laveste close i hendelsen.
    Suksess = kursen steg >= min_gain innen max_days etter bunnen.

    Returnerer hit rate, snittgevinst, median dager til topp og
    bunn-historikk (for streak/divergens/projeksjon).
    """
    out = {
        "events": 0, "hits": 0, "hit_rate": 0.0,
        "avg_gain_pct": 0.0, "median_days": 0,
        "bottoms": [],  # liste av (dato, bunnpris, rsi_ved_bunn)
    }
    if "RSI" not in df.columns or len(df) < 120:
        return out

    rsi = df["RSI"].values
    close = df["Close"].values
    dates = df.index
    n = len(df)

    i = 1
    gains, days_to_peak = [], []
    while i < n:
        if rsi[i] < rsi_low and rsi[i - 1] >= rsi_low:
            # Ny bunn-hendelse: følg til RSI krysser opp igjen
            j = i
            bot_idx = i
            while j < n and rsi[j] < rsi_low:
                if close[j] < close[bot_idx]:
                    bot_idx = j
                j += 1
            bot_price = close[bot_idx]
            out["bottoms"].append((dates[bot_idx], bot_price, rsi[bot_idx]))

            # Fremtidsvindu: ikke tell hendelser uten nok fasit-data,
            # med mindre det er den pågående (siste) hendelsen
            end = min(bot_idx + max_days, n)
            if end - bot_idx >= 10:
                fwd = close[bot_idx + 1:end]
                if len(fwd) > 0:
                    max_gain = (fwd.max() - bot_price) / bot_price
                    out["events"] += 1
                    gains.append(max_gain)
                    if max_gain >= min_gain:
                        out["hits"] += 1
                        days_to_peak.append(int(np.argmax(fwd)) + 1)
            i = j
        else:
            i += 1

    if out["events"] > 0:
        out["hit_rate"] = out["hits"] / out["events"] * 100
        out["avg_gain_pct"] = float(np.mean(gains)) * 100
    if days_to_peak:
        out["median_days"] = int(np.median(days_to_peak))
    return out


def bottom_streak_and_projection(bottoms: list) -> tuple:
    """Stigende bunn-streak + lineær projeksjon av neste bunn (fra v44)."""
    if len(bottoms) < 2:
        return 0, None
    streak = 1
    for k in range(len(bottoms) - 1, 0, -1):
        if bottoms[k][1] > bottoms[k - 1][1]:
            streak += 1
        else:
            break
    proj = None
    if streak >= 2:
        pts = bottoms[-streak:]
        x = [p[0].toordinal() for p in pts]
        y = [p[1] for p in pts]
        try:
            slope, intercept = np.polyfit(x, y, 1)
            proj = slope * pd.Timestamp.now().toordinal() + intercept
        except Exception:
            proj = None
    return streak, proj


def detect_bullish_divergence(bottoms: list, lookback_days: int = 90) -> bool:
    """Pris lavere bunn + RSI høyere bunn = klassisk bunnsignal."""
    if len(bottoms) < 2:
        return False
    d2, p2, r2 = bottoms[-1]
    d1, p1, r1 = bottoms[-2]
    if (pd.Timestamp.now() - pd.Timestamp(d2)).days > 20:
        return False
    if (pd.Timestamp(d2) - pd.Timestamp(d1)).days > lookback_days:
        return False
    return p2 < p1 and r2 > r1 + 2


# ---------------------------------------------------------------------------
# 2. BUNNSCORE (0–100)
# ---------------------------------------------------------------------------

def calculate_bottom_score(df: pd.DataFrame, cycles: dict) -> tuple:
    """Score hvor nær en tradbar bunn aksjen er akkurat nå.

    Del A (0–35): Trekksikkerhet — historisk hit rate og antall sykluser
    Del B (0–40): Bunnsignaler — er vi PÅ eller RETT ETTER bunnen?
    Del C (0–25): Bekreftelse — har snuoperasjonen startet?

    Returnerer (score, reasons, in_bottom_zone).
    """
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    score, reasons = 0, []

    close = df["Close"]
    rsi_series = df["RSI"]
    price = curr["Close"]

    # --- GATE: aksjen må være i bunn-sonen i det hele tatt ---
    low_60d = close.tail(60).min()
    dist_from_low = (price - low_60d) / low_60d if low_60d > 0 else 99
    rsi_recent_min = rsi_series.tail(15).min()
    td_setup_recent = df["TD_Buy_Setup"].tail(7).max() if "TD_Buy_Setup" in df.columns else 0

    in_bottom_zone = (
        dist_from_low <= 0.12
        or rsi_recent_min < 35
        or td_setup_recent >= 8
    )
    if not in_bottom_zone:
        return 0, ["Ikke i bunn-sone"], False

    # ---------- A: TREKKSIKKERHET (0–35) ----------
    hr = cycles["hit_rate"]
    ev = cycles["events"]
    if ev >= 3:
        if hr >= 80:
            score += 30 if ev >= 5 else 25
            reasons.append(f"Hit rate {hr:.0f}% ({ev} sykluser)")
        elif hr >= 60:
            score += 18; reasons.append(f"Hit rate {hr:.0f}% ({ev} sykluser)")
        elif hr >= 40:
            score += 10; reasons.append(f"Hit rate {hr:.0f}%")
    elif ev == 2 and hr == 100:
        score += 12; reasons.append("2/2 sykluser truffet")

    streak, _ = bottom_streak_and_projection(cycles["bottoms"])
    if streak >= 3:
        score += 10; reasons.append(f"{streak} stigende bunner")
    elif streak == 2:
        score += 6; reasons.append("2 stigende bunner")

    # ---------- B: BUNNSIGNALER (0–40) ----------
    b_score = 0

    # RSI var oversolgt og stiger nå (rett etter bunnen)
    rsi_now = curr.get("RSI", 50)
    if rsi_recent_min < 32 and rsi_now > rsi_recent_min + 3:
        b_score += 9; reasons.append(f"RSI snur opp fra {rsi_recent_min:.0f}")
    elif rsi_now < 32:
        b_score += 5; reasons.append(f"RSI oversolgt ({rsi_now:.0f}) — bunn kan pågå")

    # Bullish divergens
    if detect_bullish_divergence(cycles["bottoms"]):
        b_score += 10; reasons.append("Bullish RSI-divergens")

    # DeMark bunntiming
    td_count = curr.get("TD_Buy_Countdown", 0)
    if td_count >= 13:
        b_score += 10; reasons.append("DeMark Countdown 13")
    elif td_setup_recent >= 9:
        b_score += 8; reasons.append("DeMark Setup 9 nylig")
    elif td_count >= 10:
        b_score += 6; reasons.append(f"DeMark Countdown {int(td_count)}")

    # StochRSI krysser opp fra oversolgt
    k, d = curr.get("StochRSI_K", 50), curr.get("StochRSI_D", 50)
    pk, pd_ = prev.get("StochRSI_K", 50), prev.get("StochRSI_D", 50)
    if pk < pd_ and k > d and k < 40:
        b_score += 6; reasons.append("StochRSI bull-kryss")

    # Kapitulasjonsvolum etterfulgt av opp-dag
    tail10 = df.tail(10)
    capit = tail10[(tail10["RVOL"] > 2.0) & (tail10["Close"] < tail10["Close"].shift(1))]
    if not capit.empty and price > prev["Close"]:
        b_score += 5; reasons.append("Kapitulasjon + reversering")

    score += min(b_score, 40)

    # ---------- C: BEKREFTELSE (0–25) ----------
    c_score = 0

    # MACD-histogram stiger
    if curr.get("Hist", 0) > prev.get("Hist", 0):
        c_score += 5; reasons.append("MACD-momentum stiger")

    # Pris reclaimer korte EMA-er
    if price > curr.get("EMA21", np.inf):
        c_score += 7; reasons.append("Over EMA21")
    elif price > curr.get("EMA9", np.inf):
        c_score += 4; reasons.append("Over EMA9")

    # Higher low siste 5 dager
    lows5 = df["Low"].tail(5).values
    if len(lows5) == 5 and lows5[-1] > lows5.min() and np.argmin(lows5) < 3:
        c_score += 4; reasons.append("Higher low siste dager")

    # Nær BB-lower (bounce-sone) eller POC-støtte
    bb_lower = curr.get("BB_Lower", 0)
    if bb_lower > 0 and (price - bb_lower) / price < 0.03:
        c_score += 3; reasons.append("Ved Bollinger-bunn")
    poc = curr.get("VPVR_POC", np.nan)
    if not np.isnan(poc) and poc > 0 and abs(price - poc) / price < 0.03:
        c_score += 3; reasons.append("Ved POC-støtte")

    # Volum på opp-dag
    if curr.get("RVOL", 0) > 1.3 and price > prev["Close"]:
        c_score += 3; reasons.append(f"Volum inn (RVOL {curr['RVOL']:.1f})")

    score += min(c_score, 25)

    return min(int(score), 100), reasons, True


# ---------------------------------------------------------------------------
# 3. RENDER
# ---------------------------------------------------------------------------

def render_bottom_finder_tab(portfolio_value: float = 100_000, risk_pct: float = 1.5):
    st.markdown("## 🎣 Bunnfisker — kjøp bunnen, ikke fallet")
    st.caption(
        "Identifiserer aksjer **på eller rett etter bunnen** i sin sykel. "
        "Kombinerer historisk trekksikkerhet (RSI-sykluser), DeMark-timing, "
        "bullish divergens og volumbekreftelse fra alle tidligere screener-versjoner."
    )

    with st.expander("📖 Slik fungerer Bunnscore (0–100)", expanded=False):
        st.markdown("""
| Del | Maks | Hva måles |
|-----|------|-----------|
| **A. Trekksikkerhet** | 35 | Historisk hit rate: hvor ofte ga RSI-bunn ≥ målgevinst innen 60 dager? + stigende bunn-streak |
| **B. Bunnsignaler** | 40 | RSI snur opp fra oversolgt, bullish divergens, DeMark 9/13, StochRSI-kryss, kapitulasjonsvolum |
| **C. Bekreftelse** | 25 | MACD snur, pris over EMA9/21, higher low, BB/POC-støtte, volum på opp-dag |

**Gate:** Aksjen må være i bunn-sonen (maks 12 % over 60-dagers lav, RSI < 35 nylig,
eller DeMark Setup ≥ 8). Aksjer i etablert opptrend filtreres bort — dette verktøyet
fanger *snupunktet*, ikke trenden.

**Tolkning:**
- **70+** : Sterkt bunnsignal med historisk dokumentert trekksikkerhet
- **50–69**: God kandidat — vent på bekreftelse (grønn dag med volum)
- **< 50** : Bunn-sone, men signalene mangler — observer
        """)

    # ---- Innstillinger ----
    with st.expander("⚙️ Innstillinger", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            universe = st.radio(
                "Univers",
                ["Oslo Børs + Expand", "Alle inkl. Euronext Growth"],
                key="bf_universe",
            )
            min_score = st.slider("Min Bunnscore", 0, 100, 40, 5, key="bf_min_score")
        with c2:
            rsi_low = st.slider("RSI bunn-grense", 20, 40, 32, 1, key="bf_rsi_low")
            min_gain_pct = st.slider("Målgevinst per sykel (%)", 5, 25, 8, 1, key="bf_min_gain")
        with c3:
            min_hit_rate = st.slider("Min hit rate (%) — 0 = av", 0, 100, 0, 10, key="bf_min_hr")
            min_turnover = st.number_input(
                "Min daglig omsetning (mill NOK)", 0.0, 100.0, 1.0, 0.5, key="bf_min_to"
            )

    if st.button("🎣 Scan hele Oslo Børs", key="bf_run", use_container_width=True):
        include_growth = universe == "Alle inkl. Euronext Growth"
        tickers = get_oslo_all_tickers(include_growth=include_growth)
        st.info(f"Henter kursdata for {len(tickers)} aksjer — dette tar 1–2 minutter…")

        data, infos = get_stock_data(tickers, period="2y")

        results = []
        prog = st.progress(0)
        keys = list(data.keys())
        for i, ticker in enumerate(keys):
            prog.progress((i + 1) / len(keys), text=f"Analyserer {ticker}…")
            try:
                df = data[ticker]
                if len(df) < 120:
                    continue

                # Likviditetsfilter: snitt omsetning siste 20 dager
                turnover = (df["Close"] * df["Volume"]).tail(20).mean()
                if turnover < min_turnover * 1e6:
                    continue

                df = calculate_all_indicators(df)
                cycles = analyze_rsi_cycles(
                    df, rsi_low=rsi_low, min_gain=min_gain_pct / 100, max_days=60
                )
                if min_hit_rate > 0 and (
                    cycles["events"] < 2 or cycles["hit_rate"] < min_hit_rate
                ):
                    continue

                score, reasons, in_zone = calculate_bottom_score(df, cycles)
                if not in_zone or score < min_score:
                    continue

                curr = df.iloc[-1]
                price = curr["Close"]
                atr = curr.get("ATR", 0)

                # SL: laveste lav siste 10 dager minus 0.5*ATR (under bunnen)
                sl = df["Low"].tail(10).min() - 0.5 * atr
                # TP: historisk snittgevinst per sykel, ellers 2.5R
                if cycles["avg_gain_pct"] > 3:
                    tp = price * (1 + cycles["avg_gain_pct"] / 100)
                else:
                    tp = price + 2.5 * (price - sl)
                shares, risk_nok = calculate_position_size(
                    portfolio_value, risk_pct, price, sl
                )

                low60 = df["Close"].tail(60).min()
                results.append({
                    "Ticker": ticker,
                    "Navn": infos.get(ticker, {}).get("shortName", ticker),
                    "Bunnscore": score,
                    "Hit rate %": round(cycles["hit_rate"], 0),
                    "Sykluser": cycles["events"],
                    "Snitt gevinst %": round(cycles["avg_gain_pct"], 1),
                    "Dager til topp": cycles["median_days"],
                    "RSI": round(curr.get("RSI", 0), 0),
                    "Fra 60d-lav %": round((price / low60 - 1) * 100, 1),
                    "Pris": round(price, 2),
                    "Stop Loss": round(sl, 2),
                    "Take Profit": round(tp, 2),
                    "Antall": shares,
                    "Signaler": ", ".join(reasons[:4]),
                    "_df": df,
                    "_reasons": reasons,
                })
            except Exception:
                continue
        prog.empty()

        if results:
            st.session_state["bf_results"] = pd.DataFrame(results).sort_values(
                "Bunnscore", ascending=False
            )
        else:
            st.session_state["bf_results"] = pd.DataFrame()
        st.toast(f"Ferdig! {len(results)} bunnkandidater funnet", icon="🎣")

    bf_res = st.session_state.get("bf_results")

    if bf_res is None:
        st.info("Trykk **Scan hele Oslo Børs** for å lete etter bunnkandidater.")
        return
    if bf_res.empty:
        st.warning("Ingen bunnkandidater akkurat nå. Prøv lavere Min Bunnscore.")
        return

    st.success(f"**{len(bf_res)} bunnkandidater** — sortert på Bunnscore")

    col_config = {
        "Bunnscore": st.column_config.ProgressColumn(
            "Bunnscore", format="%d", min_value=0, max_value=100
        ),
        "Hit rate %": st.column_config.ProgressColumn(
            "Hit rate", format="%.0f%%", min_value=0, max_value=100
        ),
        "Fra 60d-lav %": st.column_config.NumberColumn("Fra bunn", format="%+.1f%%"),
        "Snitt gevinst %": st.column_config.NumberColumn("Snitt opptur", format="%.1f%%"),
    }
    display_cols = [
        c for c in [
            "Ticker", "Navn", "Bunnscore", "Hit rate %", "Sykluser",
            "Snitt gevinst %", "Dager til topp", "RSI", "Fra 60d-lav %",
            "Pris", "Stop Loss", "Take Profit", "Antall", "Signaler",
        ] if c in bf_res.columns
    ]

    event = st.dataframe(
        bf_res[display_cols],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=col_config,
        height=400,
    )

    # ---- Detaljvisning ----
    sel_ticker = None
    if event.selection.rows:
        sel_ticker = bf_res.iloc[event.selection.rows[0]]["Ticker"]

    st.markdown("---")
    tickers_list = bf_res["Ticker"].tolist()
    default_idx = tickers_list.index(sel_ticker) if sel_ticker in tickers_list else 0
    g1, g2 = st.columns([1, 3])
    chosen = g1.selectbox("Vis graf for:", tickers_list, index=default_idx, key="bf_sel")
    period = g2.radio(
        "Periode:", ["3 mnd", "6 mnd", "1 år", "2 år"], index=1,
        horizontal=True, key="bf_period",
    )

    if chosen:
        row = bf_res[bf_res["Ticker"] == chosen].iloc[0]
        df_full = row["_df"]
        days_map = {"3 mnd": 90, "6 mnd": 180, "1 år": 365, "2 år": 730}
        df_view = df_full.copy()
        if hasattr(df_view.index, "tz_localize"):
            try:
                df_view.index = df_view.index.tz_localize(None)
            except TypeError:
                pass
        start = df_view.index[-1] - timedelta(days=days_map[period])
        df_view = df_view[df_view.index >= start]

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )
        fig.add_trace(go.Candlestick(
            x=df_view.index, open=df_view["Open"], high=df_view["High"],
            low=df_view["Low"], close=df_view["Close"], name=chosen,
        ), row=1, col=1)
        for col_name, color in [("EMA21", "orange"), ("EMA50", "blue")]:
            if col_name in df_view.columns:
                fig.add_trace(go.Scatter(
                    x=df_view.index, y=df_view[col_name], name=col_name,
                    line=dict(color=color, width=1),
                ), row=1, col=1)
        fig.add_hline(y=row["Stop Loss"], line_dash="dash", line_color="red",
                      annotation_text="SL", row=1, col=1)
        fig.add_hline(y=row["Take Profit"], line_dash="dash", line_color="green",
                      annotation_text="TP", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["RSI"], name="RSI",
            line=dict(color="purple", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=32, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)

        fig.update_layout(
            title=f"{chosen} — Bunnscore {row['Bunnscore']} | "
                  f"Hit rate {row['Hit rate %']:.0f}% over {row['Sykluser']} sykluser",
            xaxis_rangeslider_visible=False, height=600,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Alle signaler:** {', '.join(row['_reasons'])}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historisk snitt-opptur", f"{row['Snitt gevinst %']:.1f}%")
        m2.metric("Median dager til topp", f"{row['Dager til topp']:.0f}")
        m3.metric("Risk/Reward", f"1:{(row['Take Profit']-row['Pris'])/max(row['Pris']-row['Stop Loss'],0.01):.1f}")
        m4.metric("Posisjon", f"{row['Antall']} aksjer")

    st.markdown("---")
    st.caption(
        "**Trekksikkerhet** = andel historiske RSI-bunner som ga målgevinsten innen 60 dager. "
        "Aksjer uten bunn-sone-status filtreres bort uansett score. "
        "Husk: ingen bunn er bekreftet før kursen viser styrke — vurder å vente på en grønn dag med volum."
    )
