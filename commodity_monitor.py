"""
commodity_monitor.py
--------------------
Råvarepriser og kapitalrotasjon relevante for Oslo Børs og swing trading.

Artikkel 2, Wold: "Gassprisen hadde steget" — rett etter intervjuet var han opp
på mobilen. Han følger råvarer tett for å forstå hva energiaksjene vil gjøre.

Tracker:
  - Brent crude (olje) — driver EQNR, AKRBP, PGS, TGS, DNO
  - Natural gas — driver EQNR, olje/gassektoren
  - US Tech (QQQ) vs Oslo Børs (EQNR proxy) — kapitalrotasjon
  - VIX — nattrisiko-barometer
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# RÅVARER OG ROTASJONS-INDIKATORER
# ---------------------------------------------------------------------------
COMMODITIES = {
    "Brent Crude (olje)": {"symbol": "BZ=F", "unit": "USD/fat", "emoji": "🛢️", "sectors": ["Olje & Energi"]},
    "Natural Gas (Henry Hub)": {"symbol": "NG=F", "unit": "USD/MMBtu", "emoji": "🔥", "sectors": ["Olje & Energi"]},
    "Baltic Dry Index": {"symbol": "^BDI", "unit": "poeng", "emoji": "⚓", "sectors": ["Shipping"]},
    # Bruker Equinor (EQNR.OL) som Oslo Børs-proxy siden ^OSEBX/^OSEAX er ustabilt på Yahoo
    "Oslo Børs (EQNR proxy)": {"symbol": "EQNR.OL", "unit": "NOK", "emoji": "🇳🇴", "sectors": ["Alle"]},
    "QQQ (US Tech)": {"symbol": "QQQ", "unit": "USD", "emoji": "💻", "sectors": ["Rotasjon"]},
    "VIX (fryktindeks)": {"symbol": "^VIX", "unit": "poeng", "emoji": "😱", "sectors": ["Risiko"]},
}


@st.cache_data(ttl=1800)  # 30 min cache
def fetch_commodity_data() -> dict:
    """Hent siste råvaredata og beregn momentum."""
    results = {}
    for name, info in COMMODITIES.items():
        try:
            hist = yf.Ticker(info["symbol"]).history(period="3mo", timeout=10)
            if len(hist) < 5:
                results[name] = {**info, "last": None, "chg1d": None, "chg5d": None, "chg20d": None, "hist": None}
                continue

            last = hist["Close"].iloc[-1]
            chg1d = (last / hist["Close"].iloc[-2] - 1) * 100 if len(hist) >= 2 else 0
            chg5d = (last / hist["Close"].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
            chg20d = (last / hist["Close"].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0

            results[name] = {
                **info,
                "last": round(last, 2),
                "chg1d": round(chg1d, 2),
                "chg5d": round(chg5d, 2),
                "chg20d": round(chg20d, 2),
                "hist": hist["Close"],
            }
        except Exception:
            results[name] = {**info, "last": None, "chg1d": None, "chg5d": None, "chg20d": None, "hist": None}

    return results


# ---------------------------------------------------------------------------
# US TECH VS OSLO BØRS ROTASJONSINDIKATOR
# Wold: "Nå er det tech som går igjen. Da begynner Oslo Børs å slite, fordi
#        pengene flytter seg ut."
# ---------------------------------------------------------------------------
def compute_rotation_signal(commodity_data: dict) -> dict:
    """
    Sammenlign QQQ momentum mot OSEBX momentum.

    Returnerer rotasjonssignal med anbefaling.
    """
    qqq = commodity_data.get("QQQ (US Tech)", {})
    osebx = commodity_data.get("Oslo Børs (EQNR proxy)", {})

    qqq_5d = qqq.get("chg5d") or 0
    osebx_5d = osebx.get("chg5d") or 0
    qqq_20d = qqq.get("chg20d") or 0
    osebx_20d = osebx.get("chg20d") or 0

    # Relativ styrke: QQQ vs OSEBX
    rel_5d = qqq_5d - osebx_5d
    rel_20d = qqq_20d - osebx_20d

    if rel_5d > 5 and rel_20d > 5:
        return {
            "signal": "KAPITAL FLYTER TIL US TECH",
            "color": "#cc4400",
            "icon": "⚠️",
            "advice": (
                "US Tech dominerer. Oslo Børs sliter. "
                "Vær selektiv — kun A+-kandidater på OB. "
                "Wold: 'Pengene flytter seg ut av Oslo Børs.'"
            ),
            "qqq_5d": qqq_5d,
            "osebx_5d": osebx_5d,
        }
    elif rel_5d < -3 and osebx_5d > 0:
        return {
            "signal": "OSLO BØRS STERKERE ENN US TECH",
            "color": "#1a7a1a",
            "icon": "✅",
            "advice": (
                "Kapital roterer inn i Oslo Børs. "
                "Vold: 'Det kom mye utenlandsk kapital, og nesten alt gikk.' "
                "Gode betingelser for Oslo Børs swing trades."
            ),
            "qqq_5d": qqq_5d,
            "osebx_5d": osebx_5d,
        }
    else:
        return {
            "signal": "NØYTRAL ROTASJON",
            "color": "#805500",
            "icon": "➡️",
            "advice": "Blandet bilde. Vær selektiv og follow the money.",
            "qqq_5d": qqq_5d,
            "osebx_5d": osebx_5d,
        }


# ---------------------------------------------------------------------------
# OVERNIGHT RISIKO
# Wold: "Skal du sitte over natten, må du nesten analysere Trump. Han skaper
#        ekstreme utslag."
# ---------------------------------------------------------------------------
def compute_overnight_risk(commodity_data: dict) -> dict:
    """
    Beregn risikoscore for å holde posisjoner over natten (0-100).

    Høy score = høy risiko = vurder å lukke før børsens slutt.
    """
    vix_data = commodity_data.get("VIX (fryktindeks)", {})
    vix = vix_data.get("last") or 20.0
    vix_1d_chg = vix_data.get("chg1d") or 0

    rotation = compute_rotation_signal(commodity_data)

    risk_score = 0
    risk_factors = []

    # VIX nivå
    if vix >= 30:
        risk_score += 40
        risk_factors.append(f"VIX {vix:.0f} — ekstremt høy frykt (markedet ustabilt)")
    elif vix >= 25:
        risk_score += 25
        risk_factors.append(f"VIX {vix:.0f} — forhøyet volatilitet")
    elif vix >= 20:
        risk_score += 10
        risk_factors.append(f"VIX {vix:.0f} — lett forhøyet")
    else:
        risk_factors.append(f"VIX {vix:.0f} — rolig marked")

    # VIX-trend (stiger = dårlig)
    if vix_1d_chg > 10:
        risk_score += 20
        risk_factors.append(f"VIX stiger kraftig (+{vix_1d_chg:.1f}% i dag)")
    elif vix_1d_chg > 5:
        risk_score += 10
        risk_factors.append(f"VIX stiger (+{vix_1d_chg:.1f}%)")

    # US Tech-dominans = Oslo Børs kan falle videre over natten
    if rotation["signal"] == "KAPITAL FLYTER TIL US TECH":
        risk_score += 20
        risk_factors.append("Kapitalrotasjon ut av Oslo Børs pågår")

    # Helgenrisiko (fredag)
    import datetime
    weekday = datetime.datetime.now().weekday()
    if weekday == 4:  # Fredag
        risk_score += 15
        risk_factors.append("Fredag — helg-gap-risiko (Trump tweeter i helgen)")

    risk_score = min(100, risk_score)

    if risk_score >= 60:
        label = "HØY RISIKO"
        color = "#8b0000"
        advice = "Vurder å redusere posisjoner eller sikre med trangere stop. Wold: 'Cash er kanskje det viktigste jeg har.'"
    elif risk_score >= 35:
        label = "MODERAT RISIKO"
        color = "#cc6600"
        advice = "Reducer posisjonsstørrelse. Ha plan for gap-down ved åpning i morgen."
    else:
        label = "LAV RISIKO"
        color = "#1a6a1a"
        advice = "Rimelig greit å holde posisjoner over natten."

    return {
        "score": risk_score,
        "label": label,
        "color": color,
        "advice": advice,
        "factors": risk_factors,
    }


# ---------------------------------------------------------------------------
# SLOW PRICE-IN DETEKTOR
# Wold: "Kongsberg Gruppen gikk ikke samme dag. Det tok tid. Det er ikke alt
#        som er priset inn med en gang, selv om mange tror det."
# ---------------------------------------------------------------------------
def detect_slow_price_in(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Sjekk om en aksje kan være midt i en langsom innprisingsfase.

    Tegn: Positiv prisutvikling, men RVOL fortsatt stigende (ikke toppet ut ennå).
    Institusjonelle fond kjøper gradvis inn.
    """
    if len(df) < 15:
        return {"signal": False, "description": "Utilstrekkelig data"}

    close = df["Close"]
    volume = df["Volume"]

    # 1. Positiv 5-dagers trend
    roc_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(df) > 6 else 0

    # 2. RVOL STIGENDE de siste 5 dagene (mer og mer kjøpepress)
    if len(df) >= 10:
        rvol_recent = df["RVOL"].tail(5).values if "RVOL" in df.columns else []
        rvol_trending_up = len(rvol_recent) > 0 and rvol_recent[-1] > rvol_recent[0]
    else:
        rvol_trending_up = False

    # 3. Aksjen steg men fortsatt under 52-ukers høy (rom igjen)
    high_52w = df["High"].tail(252).max()
    dist_from_high = (high_52w - close.iloc[-1]) / high_52w

    # 4. Ikke overkjøpt RSI
    rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50

    slow_price_in = (
        roc_5d > 2
        and rvol_trending_up
        and dist_from_high > 0.05
        and rsi < 70
    )

    if slow_price_in:
        return {
            "signal": True,
            "description": (
                f"Mulig langsom innprisingsfase: +{roc_5d:.1f}% (5d), "
                f"RVOL-trend stigende, "
                f"{dist_from_high*100:.0f}% under 52-ukers topp. "
                "Institusjonelle fond kjøper kanskje gradvis inn."
            ),
        }

    return {"signal": False, "description": "Ingen klar slow-price-in signal"}


# ---------------------------------------------------------------------------
# RENDER I STREAMLIT
# ---------------------------------------------------------------------------
def render_commodity_panel():
    """Vis råvarer, rotasjon og overnight-risiko i Streamlit."""
    st.markdown("### Råvarer & Kapitalrotasjon")
    st.caption(
        "Wold: *'Rett etter intervjuet var det opp med mobilen. Gassprisen hadde steget.'* "
        "Han følger råvarer tett fordi de driver Oslo Børs-sektorer direkte."
    )

    with st.spinner("Henter råvarepriser..."):
        data = fetch_commodity_data()

    # --- Råvarepanel ---
    cols = st.columns(3)
    commodity_list = [
        "Brent Crude (olje)",
        "Natural Gas (Henry Hub)",
        "Baltic Dry Index",
    ]
    for col, name in zip(cols, commodity_list):
        d = data.get(name, {})
        if d.get("last") is not None:
            delta_str = f"{d['chg1d']:+.1f}% (1d) | {d['chg5d']:+.1f}% (5d)"
            delta_color = "normal" if (d["chg1d"] or 0) >= 0 else "inverse"
            col.metric(
                label=f"{d['emoji']} {name}",
                value=f"{d['last']:,.2f} {d['unit']}",
                delta=delta_str,
                delta_color=delta_color,
            )
        else:
            col.metric(label=f"{data[name]['emoji']} {name}", value="N/A")

    st.markdown("---")

    # --- Kapitalrotasjon ---
    rotation = compute_rotation_signal(data)
    st.markdown(
        f'<div style="background:{rotation["color"]}22;border-left:4px solid {rotation["color"]};'
        f'padding:10px;border-radius:4px;margin-bottom:10px;">'
        f'<b>{rotation["icon"]} Kapitalrotasjon: {rotation["signal"]}</b><br>'
        f'QQQ 5d: {rotation["qqq_5d"]:+.1f}% | OSEBX 5d: {rotation["osebx_5d"]:+.1f}%<br>'
        f'<i>{rotation["advice"]}</i></div>',
        unsafe_allow_html=True,
    )

    # --- Overnight-risiko ---
    overnight = compute_overnight_risk(data)
    st.markdown("#### Overnight-risiko (for swing traders)")
    st.caption("Wold: *'Skal du sitte over natten, må du nesten analysere Trump.'*")

    risk_col1, risk_col2 = st.columns([1, 2])
    risk_col1.metric(
        label="Overnight Risk Score",
        value=f"{overnight['score']}/100",
        delta=overnight["label"],
        delta_color="inverse" if overnight["score"] >= 35 else "normal",
    )
    with risk_col2:
        st.markdown(f"**{overnight['advice']}**")
        for f in overnight["factors"]:
            st.markdown(f"- {f}")

    # --- Mini chart: QQQ vs OSEBX ---
    with st.expander("Graf: QQQ vs Oslo Børs (siste 3 mnd)", expanded=False):
        qqq_hist = data.get("QQQ (US Tech)", {}).get("hist")
        osebx_hist = data.get("Oslo Børs (EQNR proxy)", {}).get("hist")

        if qqq_hist is not None and osebx_hist is not None:
            # Normaliser til 100
            qqq_norm = qqq_hist / qqq_hist.iloc[0] * 100
            osebx_norm = osebx_hist / osebx_hist.iloc[0] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=qqq_norm.index, y=qqq_norm.values,
                name="QQQ (US Tech)", line=dict(color="blue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=osebx_norm.index, y=osebx_norm.values,
                name="Oslo Børs", line=dict(color="red", width=2)
            ))
            fig.update_layout(
                height=300, margin=dict(t=20, b=20),
                yaxis_title="Indeksert (= 100 ved start)",
                hovermode="x unified",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rotasjonsdata ikke tilgjengelig.")


# ---------------------------------------------------------------------------
# PSYKOLOGI-PANEL
# Wold: "Hevntrading. Da mister du kontrollen. Det eneste som hjelper er å ta
#        et steg tilbake — roe ned, redusere posisjonene og gradvis bygge opp
#        selvtilliten igjen."
# ---------------------------------------------------------------------------
def render_psychology_panel():
    """Vold-inspirert psykologi-sjekkliste for swing traders."""
    st.markdown("### Psykologi & Disiplin")
    st.caption(
        "Wold: *'Det er mye psykologi. Har du flyt, stoler du på beslutningene. "
        "Har du en dårlig periode, begynner du å tvile på alt.'*"
    )

    with st.expander("⚠️ Hevntrading-sjekk — gjør dette FØR neste trade", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            recent_losses = st.radio(
                "Hvor mange tapshandler har du hatt de siste 3 dagene?",
                ["0 — ingen tap", "1 tap", "2 tap", "3+ tap"],
                index=0,
                key="psych_losses",
            )
            doubting = st.checkbox(
                "Tviler du på beslutningene dine akkurat nå?",
                key="psych_doubt",
            )

        with col2:
            trying_recover = st.checkbox(
                "Prøver du å tjene inn et tidligere tap?",
                key="psych_recover",
            )
            increased_size = st.checkbox(
                "Har du økt posisjonsstørrelsen etter et tap?",
                key="psych_size",
            )

        # Vurdering
        danger_signals = sum([
            "2 tap" in recent_losses or "3+" in recent_losses,
            doubting,
            trying_recover,
            increased_size,
        ])

        if danger_signals >= 2:
            st.error(
                "🚨 **STOPP — Hevntrading-risiko oppdaget**\n\n"
                "Wold: *'Det eneste som hjelper er å ta et steg tilbake — roe ned, "
                "redusere posisjonene og gradvis bygge opp selvtilliten igjen.'*\n\n"
                "**Konkret handling:**\n"
                "- Halver anbefalt posisjonsstørrelse i dag\n"
                "- Kun A+-setup (Wold Score ≥ 70)\n"
                "- Ikke handle de neste 2 timene — gå en tur"
            )
        elif danger_signals == 1:
            st.warning(
                "⚠️ **Vær forsiktig**\n\n"
                "Reduser posisjonsstørrelsen med 25-50%. Krev sterkere setup enn vanlig."
            )
        else:
            st.success(
                "✅ **God psykologisk tilstand**\n\n"
                "Wold: *'Har du flyt, stoler du på beslutningene dine.'* "
                "Handle normalt — men husk stop-nivåer."
            )

    with st.expander("Wolds 5 regler for swing trading", expanded=False):
        st.markdown("""
1. **Kjøp styrke, ikke svakhet** — "Kjøper du noe som faller, kjøper du for tidlig og det bare faller videre."

2. **Cash er en posisjon** — "Cash er kanskje det viktigste jeg har. Hvis det kommer en nyhet, må jeg være klar."

3. **Vær endringsvillig** — "Markedet forandrer seg hele tiden. Det som funker én måned, kan være helt ute neste måned."

4. **Vinn mer enn du taper** — Ikke nødvendigvis ha høy hitrate, men la vinnerne løpe og kutt taperne raskt.

5. **Psykologi er halvparten** — "Har du selvtillit og scorer mål, føles alt enkelt. Går du uten scoring, begynner du å tvile på alt."

> *"Trenger du å tjene penger hver dag, tror jeg du blir en dårligere investor."*
        """)
