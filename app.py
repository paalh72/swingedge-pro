import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_sources import get_tickers_for_market, MARKET_OPTIONS
from data_fetcher import get_stock_data
from indicators import calculate_all_indicators
from scoring import calculate_confluence_score, calculate_entry_quality
from backtester import run_backtest
from market_regime import get_market_regime
from risk_manager import calculate_position_size, calculate_trailing_stop
from chart_builder import build_chart
from watchlist import load_watchlist, save_watchlist
from wold_scanner import (
    calculate_wold_score,
    passes_wold_filter,
    detect_momentum_continuation,
    estimate_trigger_proximity,
    TICKER_TO_SECTOR,
)
from sector_tracker import (
    compute_sector_momentum,
    render_sector_heatmap,
    render_shipping_panel,
)
from newsweb import render_newsweb
from commodity_monitor import (
    render_commodity_panel,
    render_psychology_panel,
    detect_slow_price_in,
)

# --- CONFIG ---
st.set_page_config(page_title="SwingEdge Pro v2.0", layout="wide")

# --- AUTH ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Skriv inn passord:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Skriv inn passord:", type="password", on_change=password_entered, key="password")
        st.error("Feil passord")
        return False
    return True

if not check_password():
    st.stop()

# --- SESSION STATE ---
for key, default in [
    ("results", None),
    ("wold_results", None),
    ("watchlist", None),
    ("backtest_results", None),
    ("scan_data", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.watchlist is None:
    st.session_state.watchlist = load_watchlist()

# --- HEADER ---
st.title("SwingEdge Pro v2.0")

# --- MARKET REGIME BANNER ---
regime, regime_desc, regime_color = get_market_regime()
st.markdown(
    f'<div style="background-color:{regime_color};padding:10px;border-radius:5px;margin-bottom:10px;">'
    f'<b>Markedsregime: {regime}</b> — {regime_desc}</div>',
    unsafe_allow_html=True,
)

# --- FANER ---
tab_wold, tab_scan, tab_sektorer, tab_raavarer, tab_nyheter, tab_psykologi, tab_backtest = st.tabs([
    "⚡ Wold-modus",
    "🔬 Teknisk Scan",
    "🗺️ Sektorer & Shipping",
    "🛢️ Råvarer & Rotasjon",
    "📰 Newsweb",
    "🧠 Psykologi",
    "📊 Backtest",
])

# ---------------------------------------------------------------------------
# SIDEBAR (delt mellom alle faner)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Marked")

    with st.expander("Rediger Watchlist", expanded=False):
        new_ticker = st.text_input("Legg til aksje", placeholder="DNB.OL").upper()
        if st.button("Legg til"):
            if new_ticker and new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                save_watchlist(st.session_state.watchlist)
                st.success(f"La til {new_ticker}")
                st.rerun()
        remove_list = st.multiselect("Fjern aksjer:", st.session_state.watchlist)
        if remove_list and st.button("Fjern valgte"):
            for t in remove_list:
                if t in st.session_state.watchlist:
                    st.session_state.watchlist.remove(t)
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    choice = st.radio("Velg Kilde:", list(MARKET_OPTIONS.keys()))

    if choice == "Min Watchlist":
        tickers = st.session_state.watchlist
    elif choice == "Egen liste":
        tickers = [t.strip() for t in st.text_area("Tickers", "EQNR.OL, NHY.OL").split(",")]
    else:
        tickers = get_tickers_for_market(choice)

    if choice not in ("Min Watchlist",) and st.checkbox("Begrens antall", True):
        limit = st.slider("Maks aksjer", 10, min(len(tickers), 500), 80)
        scan_list = tickers[:limit]
    else:
        scan_list = tickers

    st.markdown("---")
    st.header("2. Filtre")
    min_wold_score = st.slider("Min Wold Score", 0, 100, 40)
    min_confluence = st.slider("Min Confluence Score", 0, 10, 3)
    min_adx = st.number_input("Min ADX", 0, 60, 15)
    demark_only = st.checkbox("Kun DeMark 9/13")

    st.markdown("---")
    st.header("3. Risikostyring")
    portfolio_value = st.number_input("Porteføljeverdi (NOK)", 10000, 10_000_000, 100_000, step=10_000)
    risk_per_trade = st.slider("Risiko per trade (%)", 0.5, 5.0, 1.5, 0.5)
    rr_target = st.slider("Risk:Reward mål", 1.5, 5.0, 3.0, 0.5)

    st.markdown("---")
    st.header("4. Visning")
    chart_type = st.radio("Graf:", ("Candlestick", "Linje"))

    st.markdown("---")
    col_scan_btn, col_bt_btn = st.columns(2)
    start_scan = col_scan_btn.button("🚀 Scan", use_container_width=True)
    start_backtest = col_bt_btn.button("📊 Backtest", use_container_width=True)


# ---------------------------------------------------------------------------
# SCAN-LOGIKK (kjøres uansett hvilken fane som er aktiv)
# ---------------------------------------------------------------------------
if start_scan:
    st.toast("Starter analyse...", icon="⚡")
    data, infos = get_stock_data(scan_list)
    st.session_state.scan_data = data  # Lagre for sektoranalyse

    wold_results, tech_results = [], []
    prog = st.progress(0)
    keys = list(data.keys())

    for i, ticker in enumerate(keys):
        prog.progress((i + 1) / len(keys))
        df = data[ticker]
        try:
            df = calculate_all_indicators(df)
            if len(df) < 50:
                continue

            last = df.iloc[-1]
            adx_val = last.get("ADX", 0)

            # ---- WOLD SCORE ----
            wold_score, wold_reasons, disqualified = calculate_wold_score(df, ticker)
            passes, filter_reason = passes_wold_filter(df)
            momentum_cont = detect_momentum_continuation(df)
            trigger_prox = estimate_trigger_proximity(df)

            if passes and wold_score >= min_wold_score and not disqualified:
                atr_val = last.get("ATR", 0)
                sl_price = last["Close"] - (2 * atr_val)
                tp_price = last["Close"] + (2 * atr_val * rr_target)
                shares, risk_amount = calculate_position_size(
                    portfolio_value, risk_per_trade, last["Close"], sl_price
                )
                sektor = TICKER_TO_SECTOR.get(ticker, "Annet")
                slow_pi = detect_slow_price_in(df, ticker)
                wold_results.append({
                    "Ticker": ticker,
                    "Navn": infos.get(ticker, {}).get("shortName", ticker),
                    "Sektor": sektor,
                    "Wold Score": wold_score,
                    "RVOL": round(last.get("RVOL", 0), 1),
                    "1d %": round((last["Close"] / df["Close"].iloc[-2] - 1) * 100, 1),
                    "5d %": round((last["Close"] / df["Close"].iloc[-6] - 1) * 100, 1) if len(df) > 6 else 0,
                    "Momentum-streak": momentum_cont["streak"],
                    "Trigger-nær": "⚡ " + trigger_prox["type"] if trigger_prox["near_trigger"] else "—",
                    "Slow price-in": "🐢 Ja" if slow_pi["signal"] else "—",
                    "Pris": round(last["Close"], 2),
                    "Stop Loss": round(sl_price, 2),
                    "Take Profit": round(tp_price, 2),
                    "Antall": shares,
                    "Wold-signaler": ", ".join(wold_reasons[:3]),
                    "_df": df,
                    "_slow_pi_desc": slow_pi["description"],
                })

            # ---- TEKNISK CONFLUENCE SCORE ----
            if len(df) >= 200:
                if adx_val < min_adx:
                    continue
                confluence, reasons = calculate_confluence_score(df)
                if confluence < min_confluence:
                    continue
                td_setup = int(last.get("TD_Buy_Setup", 0))
                td_count = int(last.get("TD_Buy_Countdown", 0))
                if demark_only and not (td_setup >= 8 or td_count >= 10):
                    continue
                entry_quality, eq_label = calculate_entry_quality(df, regime)
                atr_val = last.get("ATR", 0)
                sl_price = last["Close"] - (2 * atr_val)
                tp_price = last["Close"] + (2 * atr_val * rr_target)
                shares, risk_amount = calculate_position_size(
                    portfolio_value, risk_per_trade, last["Close"], sl_price
                )
                dm_status = ""
                if td_count > 0:
                    dm_status = f"Count {td_count}"
                elif td_setup > 0:
                    dm_status = f"Setup {td_setup}"

                tech_results.append({
                    "Ticker": ticker,
                    "Navn": infos.get(ticker, {}).get("shortName", ticker),
                    "Confluence": confluence,
                    "Entry": eq_label,
                    "DeMark": dm_status,
                    "ADX": int(adx_val),
                    "RVOL": round(last.get("RVOL", 0), 1),
                    "RSI": round(last.get("RSI", 0), 1),
                    "Pris": round(last["Close"], 2),
                    "Stop Loss": round(sl_price, 2),
                    "Take Profit": round(tp_price, 2),
                    "Antall": shares,
                    "Risiko (NOK)": round(risk_amount, 0),
                    "R:R": f"1:{rr_target}",
                    "Signaler": ", ".join(reasons),
                    "_df": df,
                })

        except Exception:
            continue

    prog.empty()

    if wold_results:
        st.session_state.wold_results = pd.DataFrame(wold_results).sort_values(
            ["Momentum-streak", "Wold Score"], ascending=False
        )
    else:
        st.session_state.wold_results = pd.DataFrame()

    if tech_results:
        st.session_state.results = pd.DataFrame(tech_results).sort_values("Confluence", ascending=False)
    else:
        st.session_state.results = pd.DataFrame()

    total = len(wold_results) + len(tech_results)
    st.toast(f"Ferdig! {len(wold_results)} Wold-kandidater, {len(tech_results)} tekniske", icon="✅")


# ===========================================================================
# FANE 1 — WOLD-MODUS
# ===========================================================================
with tab_wold:
    st.markdown("## ⚡ Wold-modus — Momentum Scanner")
    st.caption(
        "Wold: *'Du trenger tre ting: økende volum, stigende kurs og noe fundamentalt. "
        "Kjøper du noe som faller, kjøper du for tidlig og det bare faller videre.'*"
    )

    with st.expander("Slik fungerer Wold Score (0–100)", expanded=False):
        st.markdown("""
| Signal | Vekt | Logikk |
|--------|------|---------|
| **RVOL ≥ 3x** | +35 | Ekstremt volum = institusjonelle kjøpere |
| **RVOL ≥ 2x** | +25 | Svart høyt volum |
| **RVOL ≥ 1.5x** | +15 | Klart over snitt |
| **5-dagers volum-buildup** | +5 | Vedvarende interesse |
| **Dagsbevegelse ≥ 5%** | +15 | Sterk impuls |
| **Dagsbevegelse 2–5%** | +10 | God impuls |
| **5d ROC ≥ 10%** | +10 | Sterk kortsiktig trend |
| **EMA20 > EMA50** | +12 | Bullish EMA-alignment |
| **Higher highs 5 dager** | +8 | Momentum continuation |
| **Nær 52-ukers topp** | +15 | Styrke, ikke svakhet |
| **RSI 55–75** | +5 | Momentum-sone |

**Automatisk diskvalifisert hvis:**
- Lavere lavpunkter siste 5 dager (falling knife)
- Mer enn 20% under EMA50 med negativ momentum
        """)

    wold_res = st.session_state.get("wold_results")

    if wold_res is None:
        st.info("Trykk **Scan** i sidepanelet for å starte analysen.")
    elif wold_res.empty:
        st.warning("Ingen Wold-kandidater funnet med gjeldende filtre.")
    else:
        st.success(f"**{len(wold_res)} Wold-kandidater** — sortert på momentum-streak og Wold Score")

        col_config_wold = {
            "Wold Score": st.column_config.ProgressColumn(
                "Wold Score", format="%d", min_value=0, max_value=100
            ),
            "RVOL": st.column_config.NumberColumn("RVOL", format="%.1f"),
            "1d %": st.column_config.NumberColumn("1d %", format="%+.1f"),
            "5d %": st.column_config.NumberColumn("5d %", format="%+.1f"),
            "Momentum-streak": st.column_config.NumberColumn("Streak 🔥", format="%d"),
        }

        wold_display_cols = [
            "Ticker", "Sektor", "Wold Score", "RVOL", "1d %", "5d %",
            "Momentum-streak", "Trigger-nær", "Slow price-in",
            "Pris", "Stop Loss", "Take Profit", "Antall", "Wold-signaler",
        ]

        wold_event = st.dataframe(
            wold_res[wold_display_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_config_wold,
            height=350,
        )

        # Wold-innsikt boks
        if not wold_res.empty:
            top3 = wold_res.head(3)
            st.markdown("#### Top 3 Wold-kandidater akkurat nå")
            cols = st.columns(3)
            for col, (_, row) in zip(cols, top3.iterrows()):
                with col:
                    st.markdown(
                        f"**{row['Ticker']}** — {row['Sektor']}\n\n"
                        f"Score: **{row['Wold Score']}** | RVOL: **{row['RVOL']}x**\n\n"
                        f"1d: **{row['1d %']:+.1f}%** | 5d: **{row['5d %']:+.1f}%**\n\n"
                        f"_{row['Wold-signaler'][:60]}_"
                    )

        # Graf for valgt Wold-aksje
        sel_wold_ticker = None
        if wold_event.selection.rows:
            sel_idx = wold_event.selection.rows[0]
            sel_wold_ticker = wold_res.iloc[sel_idx]["Ticker"]

        st.markdown("---")
        w1, w2 = st.columns([1, 3])
        default_wold_idx = 0
        wold_tickers = wold_res["Ticker"].unique().tolist()
        if sel_wold_ticker and sel_wold_ticker in wold_tickers:
            default_wold_idx = wold_tickers.index(sel_wold_ticker)

        wold_sel = w1.selectbox("Vis graf for:", wold_tickers, index=default_wold_idx, key="wold_sel")
        wold_period = w2.radio(
            "Periode:", ["1 mnd", "3 mnd", "6 mnd", "1 år"], index=1,
            horizontal=True, key="wold_period"
        )

        if wold_sel:
            row = wold_res[wold_res["Ticker"] == wold_sel].iloc[0]
            df_full = row["_df"]
            days_map = {"1 mnd": 30, "3 mnd": 90, "6 mnd": 180, "1 år": 365}
            delta_days = days_map.get(wold_period, 90)
            df_view = df_full.copy()
            if hasattr(df_view.index, "tz_localize"):
                df_view.index = df_view.index.tz_localize(None)
            start_date = df_view.index[-1] - timedelta(days=delta_days)
            df_view = df_view[df_view.index >= start_date]

            if not df_view.empty:
                try:
                    fig = build_chart(
                        df_view, wold_sel, chart_type,
                        row["Stop Loss"], row["Take Profit"], row["Wold Score"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Graf-feil: {e}")

                # Watchlist-knapp
                if wold_sel not in st.session_state.watchlist:
                    if st.button(f"❤️ Legg {wold_sel} til Watchlist"):
                        st.session_state.watchlist.append(wold_sel)
                        save_watchlist(st.session_state.watchlist)
                        st.toast(f"{wold_sel} lagret!", icon="✅")
                        st.rerun()
                else:
                    st.info(f"✅ {wold_sel} er allerede i Watchlist")


# ===========================================================================
# FANE 2 — TEKNISK SCAN (Confluence)
# ===========================================================================
with tab_scan:
    st.markdown("## 🔬 Teknisk Scan — Confluence-basert")
    st.caption(
        "Teller uavhengige tekniske bekreftelser (0–10). "
        "Minimum 4/10 anbefalt for entry."
    )

    tech_res = st.session_state.get("results")

    if tech_res is None:
        st.info("Trykk **Scan** i sidepanelet for å starte analysen.")
    elif tech_res.empty:
        st.warning("Ingen tekniske kandidater funnet med gjeldende filtre.")
    else:
        st.success(f"**{len(tech_res)} tekniske kandidater**")

        col_config = {
            "Confluence": st.column_config.ProgressColumn(
                "Confluence", format="%d", min_value=0, max_value=10
            ),
            "ADX": st.column_config.NumberColumn("ADX", format="%d"),
            "RVOL": st.column_config.NumberColumn("RVOL", format="%.1f"),
        }

        display_cols = [
            "Ticker", "Confluence", "Entry", "DeMark", "ADX", "RVOL", "RSI",
            "Pris", "Stop Loss", "Take Profit", "Antall", "Risiko (NOK)", "R:R", "Signaler",
        ]

        event = st.dataframe(
            tech_res[display_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_config,
        )

        sel_ticker = None
        if event.selection.rows:
            sel_idx = event.selection.rows[0]
            sel_ticker = tech_res.iloc[sel_idx]["Ticker"]
            if sel_ticker not in st.session_state.watchlist:
                if st.button(f"❤️ Legg {sel_ticker} til Watchlist"):
                    st.session_state.watchlist.append(sel_ticker)
                    save_watchlist(st.session_state.watchlist)
                    st.toast(f"{sel_ticker} lagret!", icon="✅")
                    st.rerun()
            else:
                st.info(f"✅ {sel_ticker} er i Watchlist")

        st.markdown("---")
        c1, c2 = st.columns([1, 3])
        tech_tickers = tech_res["Ticker"].unique().tolist()
        default_idx = 0
        if sel_ticker and sel_ticker in tech_tickers:
            default_idx = tech_tickers.index(sel_ticker)

        tech_sel = c1.selectbox("Vis graf for:", tech_tickers, index=default_idx, key="tech_sel")
        view_period = c2.radio(
            "Periode:", ["3 mnd", "6 mnd", "1 år", "3 år", "5 år"],
            index=2, horizontal=True, key="tech_period"
        )

        if tech_sel:
            row = tech_res[tech_res["Ticker"] == tech_sel].iloc[0]
            df_full = row["_df"]
            days_map = {"3 mnd": 90, "6 mnd": 180, "1 år": 365, "3 år": 1095, "5 år": 1825}
            delta_days = days_map.get(view_period, 365)
            df_view = df_full.copy()
            if hasattr(df_view.index, "tz_localize"):
                df_view.index = df_view.index.tz_localize(None)
            start_date = df_view.index[-1] - timedelta(days=delta_days)
            df_view = df_view[df_view.index >= start_date]

            if not df_view.empty:
                try:
                    fig = build_chart(
                        df_view, tech_sel, chart_type,
                        row["Stop Loss"], row["Take Profit"], row["Confluence"] * 10
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    ts_df = calculate_trailing_stop(df_view, multiplier=2.0)
                    if ts_df is not None:
                        st.markdown("#### Trailing Stop (siste 30 dager)")
                        st.line_chart(ts_df.tail(30)[["Close", "Trailing_Stop"]])
                except Exception as e:
                    st.error(f"Graf-feil: {e}")


# ===========================================================================
# FANE 3 — SEKTORER & SHIPPING
# ===========================================================================
with tab_sektorer:

    st.markdown("## 🗺️ Sektorrotasjon & Shippingrater")

    render_shipping_panel()
    st.markdown("---")

    scan_data = st.session_state.get("scan_data")
    if scan_data:
        sector_df = compute_sector_momentum(scan_data)
        render_sector_heatmap(sector_df)
    else:
        st.info(
            "Kjør en **Scan** fra sidepanelet for å se sektoranalysen. "
            "Sektordata beregnes fra de aksjene som er hentet inn."
        )

    st.markdown("---")
    st.markdown("#### Wold's sektorsyklus")
    st.markdown("""
> *"Nå har det vært mye olje og shipping, men når det begynner å dabbe av,
> blir det mer sånn at man ikke skal røre det."*
>
> *"Det er ikke fordi det har kommet nyheter på Nordic Semiconductor —
> det er fordi den sektoren er hot. Du trenger ikke være først, du må bare være med."*

**Praktisk bruk:**
- Grønn sektor (Score ≥ 70): Vær aggressiv — kjøp de sterkeste aksjene i sektoren
- Gul sektor (Score 30–70): Vær selektiv — kun A+-kandidater
- Rød sektor (Score < 30): Unngå long — Wold ville ikke rørt dette
    """)


# ===========================================================================
# FANE 4 — RÅVARER & ROTASJON
# ===========================================================================
with tab_raavarer:
    st.markdown("## 🛢️ Råvarer, Kapitalrotasjon & Overnight-risiko")
    render_commodity_panel()

    st.markdown("---")
    st.markdown("#### Slow price-in — kandidater fra siste scan")
    st.caption(
        "Wold: *'Kongsberg Gruppen gikk ikke samme dag. Det tok tid. "
        "Selv om mange tror alt er priset inn med en gang, er det ikke alltid slik.'*"
    )
    wold_res = st.session_state.get("wold_results")
    if wold_res is not None and not wold_res.empty and "Slow price-in" in wold_res.columns:
        slow_pi_res = wold_res[wold_res["Slow price-in"] == "🐢 Ja"]
        if not slow_pi_res.empty:
            st.success(f"**{len(slow_pi_res)} kandidater med mulig slow price-in signal:**")
            for _, r in slow_pi_res.iterrows():
                st.markdown(
                    f"**{r['Ticker']}** — {r['_slow_pi_desc']}"
                )
        else:
            st.info("Ingen slow price-in kandidater i dette scannet.")
    else:
        st.info("Kjør en scan for å se slow price-in kandidater.")


# ===========================================================================
# FANE 5 — NEWSWEB
# ===========================================================================
with tab_nyheter:
    result_tickers = []
    if st.session_state.wold_results is not None and not st.session_state.wold_results.empty:
        result_tickers += st.session_state.wold_results["Ticker"].tolist()
    if st.session_state.results is not None and not st.session_state.results.empty:
        result_tickers += st.session_state.results["Ticker"].tolist()

    render_newsweb(result_tickers=result_tickers if result_tickers else None)

    st.markdown("---")
    st.markdown("#### Slik bruker Wold Newsweb")
    st.markdown("""
> *"Jeg går gjennom Newsweb for børsmeldinger, ser på kursmål og justeringer
> fra analytikere, og følger diskusjoner på Xtrainvestor."*

**Hva du ser etter:**
- **⚡ Trigger** — Kontraktsvinning, rekordomsetning, ny finansiering, høyere rater
- **🎯 Analytiker** — Kursmålheving, oppgradering til kjøp — kan drive kurs i dager/uker
- **✅ Grønn rad** — Aksjen er allerede i dine scan-resultater + har en nyhet = Wolds drømmescenario

**Wold-filosofi:** Nyheten trenger ikke å fortelle hele historien med en gang.
En sterk rapport åpner aksjen høyt, men det stopper ikke der — fond skal inn,
estimater justeres opp, og folk våkner litt etter litt.
    """)


# ===========================================================================
# FANE 6 — PSYKOLOGI & DISIPLIN
# ===========================================================================
with tab_psykologi:
    st.markdown("## 🧠 Psykologi & Handelsdisiplin")
    render_psychology_panel()


# ===========================================================================
# FANE 7 — BACKTEST
# ===========================================================================
with tab_backtest:
    st.markdown("## 📊 Backtest — Historisk hitrate")

    if start_backtest:
        data, _ = get_stock_data(scan_list)
        bt_results = run_backtest(data, min_confluence, risk_per_trade, rr_target, regime)
        st.session_state.backtest_results = bt_results

    bt = st.session_state.get("backtest_results")

    if bt is None:
        st.info("Trykk **Backtest** i sidepanelet for å kjøre historisk test.")
        st.markdown("""
**Hva backtesten måler:**
- Kjøper når Confluence Score ≥ din minimumsverdi
- Selger ved Stop Loss (2×ATR) eller Take Profit (ditt R:R-mål)
- Viser faktisk hitrate, profit factor og max drawdown
        """)
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Antall trades", bt["total_trades"])
        c2.metric("Hitrate", f"{bt['win_rate']:.1f}%")
        c3.metric("Profit Factor", f"{bt['profit_factor']:.2f}")
        c4.metric("Snitt R:R", f"{bt['avg_rr']:.2f}R")
        c5.metric("Max Drawdown", f"{bt['max_drawdown']:.1f}%")

        if bt.get("equity_curve") is not None:
            st.markdown("#### Equity Curve")
            st.line_chart(bt["equity_curve"].set_index("date")["equity"])

        with st.expander("Alle trades"):
            if bt.get("trades_df") is not None:
                st.dataframe(bt["trades_df"], use_container_width=True)
