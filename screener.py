"""
70/30 Leveraged Portfolio Screener — integrert modul for SwingEdge Pro.
Eksporter: render_screener_tab()
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# Region / Currency config
# ---------------------------------------------------------------------------
REGION_CONFIG = {
    "US": {
        "suffix": "",
        "currency": "USD",
        "mcap_large_min": 10_000_000_000,
        "mcap_large_max": 3_000_000_000_000,
        "mcap_small_min": 300_000_000,
        "mcap_small_max": 10_000_000_000,
        "default_tickers_debt": [
            "AAPL","MSFT","JNJ","PG","KO","MO","T","VZ","XOM","CVX",
            "JPM","BAC","WFC","USB","TGT","WMT","MCD","PEP","ABT","MMM",
        ],
        "default_tickers_upside": [
            "NVDA","AMD","TSLA","ENPH","PLUG","RIVN","NFLX","META",
            "AMZN","GOOGL","CRWD","SNOW","DDOG","ZS","PLTR","SOFI",
        ],
    },
    "Norge (OBX)": {
        "suffix": ".OL",
        "currency": "NOK",
        "mcap_large_min": 100_000_000_000,
        "mcap_large_max": 2_000_000_000_000,
        "mcap_small_min": 3_000_000_000,
        "mcap_small_max": 100_000_000_000,
        "default_tickers_debt": [
            "EQNR.OL","DNB.OL","MOWI.OL","ORK.OL","TEL.OL","AKERBP.OL",
            "SALM.OL","SUBC.OL","AKSO.OL","YAR.OL","NHY.OL","RECSI.OL",
        ],
        "default_tickers_upside": [
            "AKER.OL","KAHOT.OL","SRBANK.OL","NEXT.OL",
            "BOUVET.OL","AMSC.OL","LINK.OL","NORDH.OL","NEL.OL",
        ],
    },
    "Sverige (OMXS30)": {
        "suffix": ".ST",
        "currency": "SEK",
        "mcap_large_min": 100_000_000_000,
        "mcap_large_max": 3_000_000_000_000,
        "mcap_small_min": 3_000_000_000,
        "mcap_small_max": 100_000_000_000,
        "default_tickers_debt": [
            "VOLV-B.ST","SAND.ST","ATCO-A.ST","INVE-B.ST","SEB-A.ST",
            "SHB-A.ST","SWED-A.ST","TELIA.ST","ALFA.ST","ERIC-B.ST",
        ],
        "default_tickers_upside": [
            "SINCH.ST","CINT.ST","NIBE-B.ST","HUSQ-B.ST","SAAB-B.ST",
            "BRAVIDA.ST","NCAB.ST","SOBI.ST","BULTEN.ST",
        ],
    },
    "Danmark (C25)": {
        "suffix": ".CO",
        "currency": "DKK",
        "mcap_large_min": 70_000_000_000,
        "mcap_large_max": 2_000_000_000_000,
        "mcap_small_min": 2_000_000_000,
        "mcap_small_max": 70_000_000_000,
        "default_tickers_debt": [
            "NOVO-B.CO","ORSTED.CO","MAERSK-B.CO","DSV.CO","DANSKE.CO",
            "NZYM-B.CO","GN.CO","TRYG.CO","VWS.CO","COLO-B.CO",
        ],
        "default_tickers_upside": [
            "DEMANT.CO","AMBU-B.CO","ROCKWOOL-B.CO","GMAB.CO","ALK-B.CO",
        ],
    },
    "Finland (OMXH25)": {
        "suffix": ".HE",
        "currency": "EUR",
        "mcap_large_min": 5_000_000_000,
        "mcap_large_max": 500_000_000_000,
        "mcap_small_min": 500_000_000,
        "mcap_small_max": 5_000_000_000,
        "default_tickers_debt": [
            "NOKIA.HE","FORTUM.HE","SAMPO.HE","UPM.HE","STERV.HE",
            "NESTE.HE","ELISA.HE","KESKO.HE","TIETO.HE",
        ],
        "default_tickers_upside": [
            "METSB.HE","CGCBV.HE","ORNBV.HE","HARVIA.HE","QTCOM.HE",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff().dropna()
    if len(delta) < period:
        return np.nan
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100 - (100 / (1 + gain / loss))


@st.cache_data(ttl=900, show_spinner=False)
def _fetch(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if not info or price is None:
            return None
        hist = t.history(period="3mo", timeout=15)
        rsi = _calc_rsi(hist["Close"]) if len(hist) >= 15 else np.nan
        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "price": price,
            "market_cap": info.get("marketCap"),
            "beta": info.get("beta"),
            "dividend_yield": (info.get("dividendYield") or 0) * 100,
            "payout_ratio": (info.get("payoutRatio") or 0) * 100,
            "profit_margin": (info.get("profitMargins") or 0) * 100,
            "revenue_growth": (info.get("revenueGrowth") or 0) * 100,
            "gross_margins": (info.get("grossMargins") or 0) * 100,
            "pb_ratio": info.get("priceToBook"),
            "analyst_target": info.get("targetMeanPrice"),
            "rsi": rsi,
            "sector": info.get("sector", "N/A"),
            "currency": info.get("currency", "N/A"),
        }
    except Exception:
        return None


def _passes_debt(d, mcap_min, mcap_max, div_min, div_max, beta_max, payout_max) -> bool:
    mc = d.get("market_cap") or 0
    if not (mcap_min <= mc <= mcap_max):
        return False
    div = d.get("dividend_yield") or 0
    if not (div_min <= div <= div_max):
        return False
    beta = d.get("beta")
    if beta is None or beta > beta_max:
        return False
    payout = d.get("payout_ratio") or 0
    if payout <= 0 or payout > payout_max:
        return False
    return (d.get("profit_margin") or 0) > 0


def _passes_upside(d, mcap_min, mcap_max, strategy, rev_min, gm_min,
                   rsi_max, pb_max, analyst_upside_min) -> bool:
    mc = d.get("market_cap") or 0
    if not (mcap_min <= mc <= mcap_max):
        return False
    price = d.get("price") or 0
    target = d.get("analyst_target") or 0
    upside = ((target - price) / price * 100) if price > 0 else 0
    if upside < analyst_upside_min:
        return False
    if strategy == "Growth / Megatrend":
        return (d.get("revenue_growth") or 0) >= rev_min and (d.get("gross_margins") or 0) >= gm_min
    else:
        rsi = d.get("rsi")
        pb = d.get("pb_ratio")
        if rsi is None or pb is None:
            return False
        return rsi <= rsi_max and pb <= pb_max


def _run_screen(tickers: list[str], filter_fn, label: str) -> pd.DataFrame:
    results = []
    bar = st.progress(0, text=label)
    for i, tkr in enumerate(tickers):
        bar.progress((i + 1) / len(tickers), text=f"Henter {tkr}…")
        data = _fetch(tkr)
        if data and filter_fn(data):
            results.append(data)
    bar.empty()
    return pd.DataFrame(results)


def _show_results(df: pd.DataFrame, cols: list[str], rename: dict):
    if df.empty:
        st.warning("Ingen aksjer passerte filtrene. Prøv å lempe på kriteriene.")
        return
    st.success(f"{len(df)} aksje(r) passerte screenen")
    avail = [c for c in cols if c in df.columns]
    st.dataframe(df[avail].rename(columns=rename).reset_index(drop=True),
                 use_container_width=True)


# ---------------------------------------------------------------------------
# Main render function — kalles fra app.py
# ---------------------------------------------------------------------------

def render_screener_tab():
    st.markdown("## 💼 70/30 Giring — Fundamental Screener")
    st.caption(
        "**70% Debt Engine** — store, stabile utbyttebetalere for å dekke marginrenten  |  "
        "**30% Asymmetrisk oppside** — turnarounds & megatrender"
    )

    # ---- Inline kontrollpanel ----
    with st.expander("⚙️ Innstillinger — region og filtre", expanded=True):
        col_r, col_custom = st.columns([1, 2])

        with col_r:
            region = st.selectbox("Marked", list(REGION_CONFIG.keys()), key="sc_region")
            cfg = REGION_CONFIG[region]
            cur = cfg["currency"]

        with col_custom:
            custom_raw = st.text_area(
                "Egne tickers (komma-separert)",
                placeholder="AKER.OL, NEL.OL, AAPL ...",
                height=68,
                key="sc_custom",
            )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"**Tab 1 – Debt Engine** *(70%)*")
            de_mcap = st.slider(
                f"Market Cap ({cur}, milliarder)",
                0, int(cfg["mcap_large_max"] / 1e9),
                (int(cfg["mcap_large_min"] / 1e9), int(cfg["mcap_large_max"] / 1e9)),
                step=max(1, int(cfg["mcap_large_min"] / 1e9 / 10)),
                key="sc_de_mcap",
            )
            de_div = st.slider("Utbytteyield (%)", 0.0, 15.0, (4.0, 8.0), 0.5, key="sc_de_div")
            de_beta = st.slider("Maks Beta", 0.3, 2.0, 0.85, 0.05, key="sc_de_beta")
            de_payout = st.slider("Maks Payout Ratio (%)", 10, 100, 75, 5, key="sc_de_payout")

        with c2:
            st.markdown(f"**Tab 2 – Asymmetrisk Oppside** *(30%)*")
            up_mcap = st.slider(
                f"Market Cap ({cur}, milliarder)",
                0, int(cfg["mcap_large_max"] / 1e9),
                (int(cfg["mcap_small_min"] / 1e9), int(cfg["mcap_small_max"] / 1e9)),
                step=max(1, int(cfg["mcap_small_min"] / 1e9 / 5)),
                key="sc_up_mcap",
            )
            up_strategy = st.radio(
                "Strategi",
                ["Growth / Megatrend", "Turnaround / Verdi"],
                horizontal=True,
                key="sc_up_strat",
            )
            up_analyst_upside = st.slider("Min analytiker-oppside (%)", 10, 100, 40, 5, key="sc_up_analyst")

            if up_strategy == "Growth / Megatrend":
                up_rev = st.slider("Min omsetningsvekst YoY (%)", 5, 100, 25, 5, key="sc_up_rev")
                up_gm = st.slider("Min bruttomargin (%)", 10, 90, 40, 5, key="sc_up_gm")
                up_rsi_max, up_pb_max = 35, 1.5
            else:
                up_rsi_max = st.slider("Maks RSI (oversolgt)", 15, 50, 35, 1, key="sc_up_rsi")
                up_pb_max = st.slider("Maks P/B", 0.1, 5.0, 1.5, 0.1, key="sc_up_pb")
                up_rev, up_gm = 25, 40

    # ---- Faner ----
    sub1, sub2 = st.tabs(["💰 Debt Engine (70%)", "🚀 Asymmetrisk Oppside (30%)"])

    def parse_custom(raw: str) -> list[str]:
        return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]

    # ---- Debt Engine ----
    with sub1:
        st.markdown(
            f"Filtre aktive: Market Cap **{de_mcap[0]}–{de_mcap[1]}B {cur}** | "
            f"Utbytte **{de_div[0]}–{de_div[1]}%** | Beta **≤{de_beta}** | "
            f"Payout **≤{de_payout}%** | Positiv profittmargin"
        )

        de_tickers = list(dict.fromkeys(cfg["default_tickers_debt"] + parse_custom(custom_raw)))

        if st.button("▶ Kjør Debt Engine Screen", key="sc_run_de"):
            def de_fn(d):
                return _passes_debt(
                    d, de_mcap[0] * 1e9, de_mcap[1] * 1e9,
                    de_div[0], de_div[1], de_beta, de_payout,
                )
            with st.spinner("Screener kjører…"):
                st.session_state["sc_de_results"] = _run_screen(
                    de_tickers, de_fn, "Debt Engine scan…"
                )

        if "sc_de_results" in st.session_state:
            _show_results(
                st.session_state["sc_de_results"],
                cols=["ticker","name","sector","price","currency","market_cap",
                      "dividend_yield","beta","payout_ratio","profit_margin"],
                rename={
                    "ticker":"Ticker","name":"Selskap","sector":"Sektor",
                    "price":"Pris","currency":"Val","market_cap":"Market Cap",
                    "dividend_yield":"Utbytte %","beta":"Beta",
                    "payout_ratio":"Payout %","profit_margin":"Profittmargin %",
                },
            )

    # ---- Asymmetrisk Oppside ----
    with sub2:
        if up_strategy == "Growth / Megatrend":
            filter_desc = f"Omsetningsvekst **≥{up_rev}%** & Bruttomargin **≥{up_gm}%**"
        else:
            filter_desc = f"RSI **≤{up_rsi_max}** (oversolgt) & P/B **≤{up_pb_max}**"

        st.markdown(
            f"Filtre aktive: Market Cap **{up_mcap[0]}–{up_mcap[1]}B {cur}** | "
            f"Analytiker-oppside **≥{up_analyst_upside}%** | {filter_desc}"
        )

        up_tickers = list(dict.fromkeys(cfg["default_tickers_upside"] + parse_custom(custom_raw)))

        if st.button("▶ Kjør Upside Screen", key="sc_run_up"):
            def up_fn(d):
                return _passes_upside(
                    d, up_mcap[0] * 1e9, up_mcap[1] * 1e9,
                    up_strategy, up_rev, up_gm,
                    up_rsi_max, up_pb_max, up_analyst_upside,
                )
            with st.spinner("Screener kjører…"):
                df = _run_screen(up_tickers, up_fn, "Upside scan…")
                if not df.empty:
                    df = df.copy()
                    df["analyst_upside_%"] = (
                        (df["analyst_target"] - df["price"]) / df["price"] * 100
                    ).round(1)
                st.session_state["sc_up_results"] = df

        if "sc_up_results" in st.session_state:
            _show_results(
                st.session_state["sc_up_results"],
                cols=["ticker","name","sector","price","currency","market_cap",
                      "revenue_growth","gross_margins","rsi","pb_ratio",
                      "analyst_target","analyst_upside_%","dividend_yield"],
                rename={
                    "ticker":"Ticker","name":"Selskap","sector":"Sektor",
                    "price":"Pris","currency":"Val","market_cap":"Market Cap",
                    "revenue_growth":"Vekst %","gross_margins":"Bruttomargin %",
                    "rsi":"RSI-14","pb_ratio":"P/B",
                    "analyst_target":"Analytikermål","analyst_upside_%":"Oppside %",
                    "dividend_yield":"Utbytte %",
                },
            )

    st.markdown("---")
    st.caption(
        "Ticker-suffiks: Oslo `.OL` · Stockholm `.ST` · København `.CO` · Helsinki `.HE`  |  "
        "Data caches 15 min · Manglende fundamentaldata hoppes over automatisk"
    )
