"""
Kvartalsslutt Window Dressing Screener — identifiserer aksjer som kan
stige inn mot kvartalsslutt pga. forvalterkjøp.

Strategi basert på tre kriterier:
1. Aksjen har falt i inneværende kvartal (forvaltere vil pynte)
2. Bred fondseierbrøk (mange forvaltere med incentiv til å kjøpe)
3. God annenhåndsomsetning / markedsverdi (exit-mulighet i sluttauksjon)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta


# ---------------------------------------------------------------------------
# Kjente aksjer med bred fondseierbrøk på Oslo Børs
# ---------------------------------------------------------------------------
FUND_HEAVY_TICKERS = {
    "MOWI.OL": {"name": "Mowi", "sector": "Sjømat", "fund_score": 95},
    "SALM.OL": {"name": "SalMar", "sector": "Sjømat", "fund_score": 85},
    "LSG.OL": {"name": "Lerøy Seafood", "sector": "Sjømat", "fund_score": 80},
    "BAKKA.OL": {"name": "Bakkafrost", "sector": "Sjømat", "fund_score": 75},
    "ATEA.OL": {"name": "Atea", "sector": "Teknologi", "fund_score": 85},
    "LINK.OL": {"name": "Link Mobility", "sector": "Teknologi", "fund_score": 70},
    "VEND.OL": {"name": "Vend Marketplaces", "sector": "Teknologi", "fund_score": 80},
    "GJF.OL": {"name": "Gjensidige Forsikring", "sector": "Finans", "fund_score": 90},
    "DNB.OL": {"name": "DNB", "sector": "Finans", "fund_score": 90},
    "STB.OL": {"name": "Storebrand", "sector": "Finans", "fund_score": 85},
    "SRBNK.OL": {"name": "SpareBank 1 SR-Bank", "sector": "Finans", "fund_score": 75},
    "TOM.OL": {"name": "Tomra Systems", "sector": "Industri", "fund_score": 90},
    "EQNR.OL": {"name": "Equinor", "sector": "Energi", "fund_score": 95},
    "AKRBP.OL": {"name": "Aker BP", "sector": "Energi", "fund_score": 85},
    "NHY.OL": {"name": "Norsk Hydro", "sector": "Industri", "fund_score": 90},
    "YAR.OL": {"name": "Yara International", "sector": "Industri", "fund_score": 85},
    "TEL.OL": {"name": "Telenor", "sector": "Telekom", "fund_score": 90},
    "ORK.OL": {"name": "Orkla", "sector": "Konsum", "fund_score": 85},
    "ENTRA.OL": {"name": "Entra", "sector": "Eiendom", "fund_score": 75},
    "SCHA.OL": {"name": "Schibsted A", "sector": "Media", "fund_score": 80},
    "KAHOT.OL": {"name": "Kahoot!", "sector": "Teknologi", "fund_score": 70},
    "SUBC.OL": {"name": "Subsea 7", "sector": "Energi", "fund_score": 80},
    "KOG.OL": {"name": "Kongsberg Gruppen", "sector": "Industri", "fund_score": 85},
    "NAS.OL": {"name": "Norwegian Air Shuttle", "sector": "Transport", "fund_score": 65},
    "RECSI.OL": {"name": "REC Silicon", "sector": "Teknologi", "fund_score": 65},
    "NOD.OL": {"name": "Nordic Semiconductor", "sector": "Teknologi", "fund_score": 80},
    "BOUV.OL": {"name": "Bouvet", "sector": "Teknologi", "fund_score": 70},
    "VOLUE.OL": {"name": "Volue", "sector": "Teknologi", "fund_score": 65},
    "SATS.OL": {"name": "Sats", "sector": "Konsum", "fund_score": 60},
    "AUTO.OL": {"name": "Autostore", "sector": "Teknologi", "fund_score": 75},
    "ELMRA.OL": {"name": "Elmera Group", "sector": "Energi", "fund_score": 65},
    "FRO.OL": {"name": "Frontline", "sector": "Shipping", "fund_score": 70},
    "GOGL.OL": {"name": "Golden Ocean", "sector": "Shipping", "fund_score": 65},
    "HAFNI.OL": {"name": "Hafnia", "sector": "Shipping", "fund_score": 65},
    "MPCC.OL": {"name": "MPC Container Ships", "sector": "Shipping", "fund_score": 60},
    "VAR.OL": {"name": "Vår Energi", "sector": "Energi", "fund_score": 75},
    "PGS.OL": {"name": "PGS", "sector": "Energi", "fund_score": 60},
    "XXL.OL": {"name": "XXL", "sector": "Konsum", "fund_score": 55},
    "PNOR.OL": {"name": "Pexip", "sector": "Teknologi", "fund_score": 55},
}


def _quarter_start(d: date) -> date:
    q_month = ((d.month - 1) // 3) * 3 + 1
    return date(d.year, q_month, 1)


def _quarter_end(d: date) -> date:
    q_month = ((d.month - 1) // 3) * 3 + 3
    if q_month == 12:
        return date(d.year, 12, 31)
    return date(d.year, q_month + 1, 1) - timedelta(days=1)


def _next_quarter_end(d: date) -> date:
    qe = _quarter_end(d)
    if d >= qe:
        if qe.month == 12:
            return _quarter_end(date(d.year + 1, 1, 1))
        return _quarter_end(date(d.year, qe.month + 1, 1))
    return qe


def _last_trading_day_before(d: date) -> date:
    """Approksimer siste børsdag: gå bakover fra dato til ukedag."""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _trading_days_until(target: date, today: date) -> int:
    count = 0
    d = today + timedelta(days=1)
    while d <= target:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_quarter_data(ticker: str, q_start: str, today_str: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        market_cap = info.get("marketCap")
        avg_vol = info.get("averageDailyVolume10Day") or info.get("averageVolume")

        hist = t.history(start=q_start, end=today_str, timeout=15)
        if hist.empty or len(hist) < 3:
            return None
        if hasattr(hist.index, "tz_localize"):
            hist.index = hist.index.tz_localize(None)

        q_open = hist["Close"].iloc[0]
        current = hist["Close"].iloc[-1]
        qtd_return = (current / q_open - 1) * 100

        last_5d = hist.tail(5)
        recent_return = (last_5d["Close"].iloc[-1] / last_5d["Close"].iloc[0] - 1) * 100 if len(last_5d) >= 2 else 0

        avg_daily_vol = hist["Volume"].mean()

        return {
            "ticker": ticker,
            "price": round(current, 2),
            "qtd_return": round(qtd_return, 1),
            "recent_5d": round(recent_return, 1),
            "market_cap": market_cap,
            "avg_volume": avg_daily_vol,
            "avg_vol_10d": avg_vol,
            "hist": hist,
        }
    except Exception:
        return None


def _calculate_dressing_score(qtd_return: float, fund_score: int, market_cap: float | None) -> int:
    score = 0

    if qtd_return <= -20:
        score += 40
    elif qtd_return <= -15:
        score += 35
    elif qtd_return <= -10:
        score += 30
    elif qtd_return <= -5:
        score += 20
    elif qtd_return <= -2:
        score += 10
    elif qtd_return < 0:
        score += 5

    score += int(fund_score * 0.4)

    if market_cap:
        if market_cap >= 100e9:
            score += 20
        elif market_cap >= 50e9:
            score += 15
        elif market_cap >= 10e9:
            score += 10
        elif market_cap >= 5e9:
            score += 5

    return min(score, 100)


def _timing_label(days_left: int) -> str:
    if days_left <= 0:
        return "🔴 Kvartal slutt"
    if days_left <= 3:
        return "🟢 Kjøpsvindu NÅ"
    if days_left <= 7:
        return "🟡 Snart kjøpsvindu"
    if days_left <= 15:
        return "⏳ Forbered posisjon"
    return "📅 Vent med inngang"


def render_quarter_dressing_tab():
    st.markdown("## 🎯 Kvartalsslutt — Window Dressing Screener")
    st.caption(
        "Forvaltere kjøper tapere inn mot kvartalsslutt for å pynte porteføljen. "
        "Posisjoner deg 3–5 dager før og selg i sluttauksjonen siste børsdag."
    )

    today = date.today()
    q_start = _quarter_start(today)
    q_end = _next_quarter_end(today)
    last_trade = _last_trading_day_before(q_end)
    days_left = _trading_days_until(last_trade, today)
    timing = _timing_label(days_left)

    buy_window_start = last_trade - timedelta(days=7)
    while buy_window_start.weekday() >= 5:
        buy_window_start -= timedelta(days=1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kvartalsslutt", last_trade.strftime("%d. %b %Y"))
    col2.metric("Børsdager igjen", days_left)
    col3.metric("Kjøpsvindu åpner", buy_window_start.strftime("%d. %b"))
    col4.metric("Timing", timing)

    with st.expander("📖 Slik fungerer strategien", expanded=False):
        st.markdown("""
**Oppskriften:**
1. **Finn aksjer som har falt i kvartalet** — forvaltere som eier disse vil kjøpe for å heve sluttkursen
2. **Bred fondseierbrøk** — jo flere fond som eier aksjen, jo sterkere blir kjøpspresset
3. **God likviditet** — du trenger å komme deg ut i sluttauksjonen siste børsdag

**Timing:**
- **Kjøp** 3–5 børsdager før kvartalsslutt
- **Selg** i sluttauksjonen siste børsdag i kvartalet
- **Unngå** nyttårseffekten (Q4→Q1) — den er utspilt

**Historisk avkastning:** Artikkelen dokumenterer ~4% gjennomsnittlig gevinst over 3–5 dager
på aksjer som Mowi, Vend, Gjensidige, Atea, Link Mobility og Tomra ved Q1 2026-slutt.
        """)

    st.markdown("---")

    with st.expander("⚙️ Filterinnstillinger", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            min_qtd_fall = st.slider(
                "Min kursfall i kvartalet (%)", -50.0, 0.0, -5.0, 1.0, key="qd_min_fall"
            )
            min_fund_score = st.slider(
                "Min fondseier-score", 0, 100, 50, 5, key="qd_min_fund"
            )
        with f2:
            min_mcap_b = st.slider(
                "Min markedsverdi (mrd NOK)", 0, 200, 5, 1, key="qd_min_mcap"
            )
            min_dressing_score = st.slider(
                "Min dressing-score", 0, 100, 30, 5, key="qd_min_score"
            )

    if st.button("🔍 Scan for kvartalsslutt-kandidater", key="qd_run", use_container_width=True):
        q_start_str = q_start.strftime("%Y-%m-%d")
        today_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        results = []
        tickers = list(FUND_HEAVY_TICKERS.keys())
        bar = st.progress(0, text="Scanner Oslo Børs…")

        for i, ticker in enumerate(tickers):
            bar.progress((i + 1) / len(tickers), text=f"Henter {ticker}…")
            meta = FUND_HEAVY_TICKERS[ticker]
            data = _fetch_quarter_data(ticker, q_start_str, today_str)
            if data is None:
                continue

            qtd = data["qtd_return"]
            mcap = data["market_cap"]

            if qtd > min_qtd_fall:
                continue
            if meta["fund_score"] < min_fund_score:
                continue
            if mcap and mcap / 1e9 < min_mcap_b:
                continue

            d_score = _calculate_dressing_score(qtd, meta["fund_score"], mcap)
            if d_score < min_dressing_score:
                continue

            results.append({
                "Ticker": ticker,
                "Selskap": meta["name"],
                "Sektor": meta["sector"],
                "Dressing Score": d_score,
                "QTD %": qtd,
                "Siste 5d %": data["recent_5d"],
                "Pris": data["price"],
                "Mrd NOK": round(mcap / 1e9, 1) if mcap else None,
                "Fondseier-score": meta["fund_score"],
                "Daglig volum": int(data["avg_volume"]) if data["avg_volume"] else None,
                "_hist": data["hist"],
            })

        bar.empty()

        if results:
            df = pd.DataFrame(results).sort_values("Dressing Score", ascending=False)
            st.session_state["qd_results"] = df
        else:
            st.session_state["qd_results"] = pd.DataFrame()

    qd_res = st.session_state.get("qd_results")

    if qd_res is None:
        st.info("Trykk **Scan** for å finne kvartalsslutt-kandidater.")
    elif qd_res.empty:
        st.warning("Ingen kandidater passerte filtrene. Prøv å justere kriteriene.")
    else:
        st.success(
            f"**{len(qd_res)} kandidater** — {days_left} børsdager til kvartalsslutt ({timing})"
        )

        col_config = {
            "Dressing Score": st.column_config.ProgressColumn(
                "Dressing Score", format="%d", min_value=0, max_value=100
            ),
            "QTD %": st.column_config.NumberColumn("QTD %", format="%+.1f"),
            "Siste 5d %": st.column_config.NumberColumn("5d %", format="%+.1f"),
            "Fondseier-score": st.column_config.ProgressColumn(
                "Fondseier", format="%d", min_value=0, max_value=100
            ),
            "Daglig volum": st.column_config.NumberColumn("Snitt volum", format="%d"),
        }

        display_cols = [
            c for c in [
                "Ticker", "Selskap", "Sektor", "Dressing Score", "QTD %",
                "Siste 5d %", "Pris", "Mrd NOK", "Fondseier-score", "Daglig volum",
            ] if c in qd_res.columns
        ]

        event = st.dataframe(
            qd_res[display_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_config,
            height=400,
        )

        if not qd_res.empty:
            st.markdown("#### 🏆 Topp-kandidater")
            top = qd_res.head(min(4, len(qd_res)))
            cols = st.columns(len(top))
            for col, (_, row) in zip(cols, top.iterrows()):
                with col:
                    delta_color = "inverse" if row["QTD %"] < 0 else "normal"
                    st.metric(
                        row["Ticker"].replace(".OL", ""),
                        f'{row["Pris"]} NOK',
                        f'{row["QTD %"]:+.1f}% QTD',
                        delta_color=delta_color,
                    )
                    st.caption(f'{row["Sektor"]} · Score {row["Dressing Score"]}')

        st.markdown("---")
        st.markdown("#### Kursgraf — QTD utvikling")

        sel_ticker = None
        if event and event.selection.rows:
            sel_idx = event.selection.rows[0]
            sel_ticker = qd_res.iloc[sel_idx]["Ticker"]

        qd_tickers = qd_res["Ticker"].tolist()
        default_idx = 0
        if sel_ticker and sel_ticker in qd_tickers:
            default_idx = qd_tickers.index(sel_ticker)

        chosen = st.selectbox(
            "Vis graf for:", qd_tickers, index=default_idx, key="qd_chart_sel"
        )
        if chosen:
            row = qd_res[qd_res["Ticker"] == chosen].iloc[0]
            hist = row["_hist"]

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name=chosen,
            ))

            if days_left <= 7:
                fig.add_vline(
                    x=last_trade.isoformat(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Kvartalsslutt",
                )

            fig.update_layout(
                title=f"{row['Selskap']} ({chosen}) — QTD: {row['QTD %']:+.1f}%",
                xaxis_title="Dato",
                yaxis_title="Kurs (NOK)",
                xaxis_rangeslider_visible=False,
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📅 Kvartalskalender 2026")
    cal_data = {
        "Kvartal": ["Q1", "Q2", "Q3", "Q4"],
        "Siste børsdag": ["31. mars", "30. juni", "30. sept", "30. des"],
        "Kjøpsvindu": ["25.–31. mars", "24.–30. juni", "24.–30. sept", "23.–30. des"],
        "Merknad": [
            "✅ Historisk sterkt",
            "✅ Halvårsslutt — ekstra press",
            "✅ Historisk sterkt",
            "⚠️ Nyttårseffekt utvanner",
        ],
    }
    st.dataframe(pd.DataFrame(cal_data), use_container_width=True, hide_index=True)

    st.caption(
        "Fondseier-score er et estimat basert på kjente fondsvekter i norske aksjer. "
        "Strategien fungerer best ved Q1-, Q2- og Q3-slutt. "
        "Nyttårseffekten (Q4→Q1) er i stor grad utspilt."
    )
