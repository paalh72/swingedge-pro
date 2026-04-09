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

# --- CONFIG ---
st.set_page_config(page_title="SwingEdge Pro v1.0", layout="wide")

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
if 'results' not in st.session_state:
    st.session_state.results = None
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# --- HEADER ---
st.title("SwingEdge Pro v1.0")

# --- MARKET REGIME BANNER ---
regime, regime_desc, regime_color = get_market_regime()
st.markdown(
    f'<div style="background-color:{regime_color};padding:10px;border-radius:5px;margin-bottom:15px;">'
    f'<b>Markedsregime: {regime}</b> — {regime_desc}</div>',
    unsafe_allow_html=True,
)

with st.expander("Slik fungerer SwingEdge Pro"):
    st.markdown("""
**SwingEdge Pro** er et evidensbasert swing trading-verktøy bygget for dager-til-uker-intervall.

#### Kjerneforskjeller fra v44
- **Confluence-basert scoring** — krever at flere uavhengige signaler bekrefter hverandre
- **Markedsregime-filter** — systemet advarer mot long-trades i bear markets (SPY + VIX)
- **Innebygd backtester** — test strategien mot historisk data og se faktisk hitrate
- **Posisjonsstørrelse** — forteller deg hvor mye du bør investere basert på din risikotoleranse
- **Trailing stop + R:R targets** — komplett exit-strategi, ikke bare entry
- **Stochastic RSI + Ichimoku** — flere uavhengige signaler for bedre konvergens

#### Confluence Score (0-10)
Teller antall uavhengige bekreftelser:
1. **Trend** — Pris over EMA50 > EMA200, ADX > 25
2. **Momentum** — RSI 40-60, MACD histogram stigende
3. **Volum** — RVOL > 1.2 (institusjonell aktivitet)
4. **Prisstruktur** — Higher lows siste 10 dager
5. **Volatilitet** — ATR-squeeze (kontraksjon som foregår ekspansjon)
6. **Ichimoku** — Pris over skyen
7. **Stochastic RSI** — Oversold crossover
8. **Bollinger** — Nær nedre bånd med stigende momentum
9. **DeMark** — Setup 9 eller Countdown 13
10. **Volume Profile** — Pris nær POC-støtte

Minimum 4/10 bekreftelser anbefalt for entry.
    """)

# --- SIDEBAR ---
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
        tickers = [t.strip() for t in st.text_area("Tickers", "EQNR.OL, NHY.OL").split(',')]
    else:
        tickers = get_tickers_for_market(choice)

    if choice not in ("Min Watchlist",) and st.checkbox("Begrens antall", True):
        limit = st.slider("Maks aksjer", 10, min(len(tickers), 500), 100)
        scan_list = tickers[:limit]
    else:
        scan_list = tickers

    st.header("2. Filtre")
    min_confluence = st.slider("Min Confluence Score", 0, 10, 4)
    min_adx = st.number_input("Min ADX (trendstyrke)", 0, 60, 20)
    demark_only = st.checkbox("Kun DeMark 9/13")

    st.header("3. Risikostyring")
    portfolio_value = st.number_input("Porteføljeverdi (NOK/USD)", 10000, 10000000, 100000, step=10000)
    risk_per_trade = st.slider("Risiko per trade (%)", 0.5, 5.0, 1.5, 0.5)
    rr_target = st.slider("Risk:Reward mål", 1.5, 5.0, 3.0, 0.5)

    st.header("4. Visning")
    chart_type = st.radio("Graf:", ("Candlestick", "Linje"))

    st.markdown("---")
    col_scan, col_bt = st.columns(2)
    start_scan = col_scan.button("Scan")
    start_backtest = col_bt.button("Backtest")

# --- MAIN SCAN ---
if start_scan:
    st.write(f"Analyserer {len(scan_list)} aksjer...")
    data, infos = get_stock_data(scan_list)
    results = []
    prog = st.progress(0)
    keys = list(data.keys())

    for i, t in enumerate(keys):
        prog.progress((i + 1) / len(keys))
        df = data[t]
        try:
            df = calculate_all_indicators(df)
            if len(df) < 200:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # ADX filter
            adx_val = last.get('ADX', 0)
            if adx_val < min_adx:
                continue

            # Confluence score
            confluence, reasons = calculate_confluence_score(df)
            if confluence < min_confluence:
                continue

            # DeMark filter
            td_setup = int(last.get('TD_Buy_Setup', 0))
            td_count = int(last.get('TD_Buy_Countdown', 0))
            if demark_only and not (td_setup >= 8 or td_count >= 10):
                continue

            # Entry quality
            entry_quality, eq_label = calculate_entry_quality(df, regime)

            # Risk calculations
            atr_val = last.get('ATR', 0)
            sl_price = last['Close'] - (2 * atr_val)
            tp_price = last['Close'] + (2 * atr_val * rr_target)
            shares, risk_amount = calculate_position_size(
                portfolio_value, risk_per_trade, last['Close'], sl_price
            )

            dm_status = ""
            if td_count > 0:
                dm_status = f"Count {td_count}"
            elif td_setup > 0:
                dm_status = f"Setup {td_setup}"

            results.append({
                'Ticker': t,
                'Navn': infos.get(t, {}).get('shortName', t),
                'Confluence': confluence,
                'Entry': eq_label,
                'DeMark': dm_status,
                'ADX': int(adx_val),
                'RVOL': round(last.get('RVOL', 0), 1),
                'RSI': round(last.get('RSI', 0), 1),
                'Pris': round(last['Close'], 2),
                'Stop Loss': round(sl_price, 2),
                'Take Profit': round(tp_price, 2),
                'Antall': shares,
                'Risiko': round(risk_amount, 0),
                'R:R': f"1:{rr_target}",
                'Regime': regime,
                'Signaler': ", ".join(reasons),
                '_df': df,
            })
        except Exception as e:
            continue

    prog.empty()

    if results:
        res = pd.DataFrame(results).sort_values('Confluence', ascending=False)
        st.session_state.results = res
        st.success(f"Fant {len(results)} kandidater!")
    else:
        st.warning("Ingen treff med gjeldende filtre.")

# --- BACKTEST ---
if start_backtest:
    st.write(f"Backtester strategi på {len(scan_list)} aksjer...")
    data, infos = get_stock_data(scan_list)
    bt_results = run_backtest(
        data, min_confluence, risk_per_trade, rr_target, regime
    )
    st.session_state.backtest_results = bt_results

if st.session_state.backtest_results is not None:
    bt = st.session_state.backtest_results
    st.markdown("### Backtest-resultater")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades", bt['total_trades'])
    c2.metric("Win Rate", f"{bt['win_rate']:.1f}%")
    c3.metric("Profit Factor", f"{bt['profit_factor']:.2f}")
    c4.metric("Avg Win/Loss", f"{bt['avg_rr']:.2f}R")
    c5.metric("Max Drawdown", f"{bt['max_drawdown']:.1f}%")

    if bt['equity_curve'] is not None:
        st.line_chart(bt['equity_curve'].set_index('date')['equity'])

    with st.expander("Alle trades"):
        if bt['trades_df'] is not None:
            st.dataframe(bt['trades_df'], use_container_width=True)

# --- RESULTS TABLE ---
if st.session_state.results is not None:
    res = st.session_state.results
    st.markdown("### Resultater")

    col_config = {
        "Confluence": st.column_config.ProgressColumn(
            "Confluence", format="%d", min_value=0, max_value=10
        ),
        "ADX": st.column_config.NumberColumn("ADX", format="%d"),
        "RVOL": st.column_config.NumberColumn("RVOL", format="%.1f"),
    }

    display_cols = [
        'Ticker', 'Confluence', 'Entry', 'DeMark', 'ADX', 'RVOL', 'RSI',
        'Pris', 'Stop Loss', 'Take Profit', 'Antall', 'Risiko', 'R:R', 'Signaler'
    ]

    event = st.dataframe(
        res[display_cols],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=col_config,
    )

    sel_ticker = None
    if event.selection.rows:
        sel_idx = event.selection.rows[0]
        sel_ticker = res.iloc[sel_idx]['Ticker']
        col_btn, col_txt = st.columns([1, 4])
        if sel_ticker not in st.session_state.watchlist:
            if col_btn.button(f"Legg til {sel_ticker} i Watchlist"):
                st.session_state.watchlist.append(sel_ticker)
                save_watchlist(st.session_state.watchlist)
                st.toast(f"{sel_ticker} lagret!")
                st.rerun()
        else:
            col_btn.info(f"{sel_ticker} er i din watchlist")

    st.markdown("---")

    c1, c2 = st.columns([1, 3])
    default_idx = 0
    if sel_ticker:
        tickers_list = res['Ticker'].unique().tolist()
        if sel_ticker in tickers_list:
            default_idx = tickers_list.index(sel_ticker)

    sel = c1.selectbox("Velg aksje for graf:", res['Ticker'].unique(), index=default_idx)
    view_period = c2.radio(
        "Periode:", ["3 mnd", "6 mnd", "1 år", "3 år", "5 år"], index=2, horizontal=True
    )

    if sel:
        sel_rows = res[res['Ticker'] == sel]
        if not sel_rows.empty:
            row = sel_rows.iloc[0]
            try:
                df_full = row['_df']

                # Filter by period
                days_map = {"3 mnd": 90, "6 mnd": 180, "1 år": 365, "3 år": 1095, "5 år": 1825}
                delta_days = days_map.get(view_period, 365)
                df_view = df_full.copy()
                if hasattr(df_view.index, 'tz_localize'):
                    df_view.index = df_view.index.tz_localize(None)
                start_date = df_view.index[-1] - timedelta(days=delta_days)
                df_view = df_view[df_view.index >= start_date]

                if df_view.empty:
                    st.warning("Ingen data for denne perioden.")
                else:
                    sl_price = row['Stop Loss']
                    tp_price = row['Take Profit']
                    fig = build_chart(df_view, sel, chart_type, sl_price, tp_price, row['Confluence'])
                    st.plotly_chart(fig, use_container_width=True)

                    # Trailing stop visual
                    ts_df = calculate_trailing_stop(df_view, multiplier=2.0)
                    if ts_df is not None:
                        st.markdown("#### Trailing Stop (siste 30 dager)")
                        ts_recent = ts_df.tail(30)[['Close', 'Trailing_Stop']]
                        st.line_chart(ts_recent)

            except Exception as e:
                st.error(f"Kunne ikke tegne graf: {e}")
