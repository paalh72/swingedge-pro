"""
bollinger_position.py
---------------------
Vurderer hvor i Bollinger-båndet aksjen ligger. Nede i båndet = høy score.

KJERNEMÅLET er %B (Bollinger Percent B):

    %B = (Close - Nedre bånd) / (Øvre bånd - Nedre bånd)

    %B = 0.00  -> kursen står presis på NEDRE bånd
    %B = 0.50  -> kursen står på midtbåndet (SMA20)
    %B = 1.00  -> kursen står presis på ØVRE bånd
    %B < 0     -> kursen har brutt UNDER nedre bånd (ekstremt oversolgt)
    %B > 1     -> kursen har brutt OVER øvre bånd

Bollinger-score (0-100) er bygd opp slik at posisjonen dominerer, men at de
to klassiske fellene straffes:

  A. Posisjon i båndet (0-60)   Lineært invertert %B. Nedre bånd gir 60 poeng,
                                midtbåndet 30, øvre bånd 0.
  B. Båndkontekst (0-25)        Er dette en reversering eller et fritt fall?
                                Straffer "banding" (kurs klistret til nedre
                                bånd i dagevis = nedtrend, ikke kjøpssignal) og
                                belønner squeeze + rom opp til midtbåndet.
  C. Vendingsbekreftelse (0-15) Har spretten faktisk startet? %B stiger,
                                grønn dag, RSI snur, volum inn.

VIKTIG FELLE: å ligge på nedre bånd er IKKE i seg selv et kjøpssignal. I en
etablert nedtrend "vandrer" kursen nedover langs nedre bånd i ukevis. Derfor
måler modulen også aksjens EGEN historiske reverseringsevne: hver gang den
før har falt ned i nedre sone, hvor ofte kom den tilbake til midtbåndet
innen X dager? Det skiller range-aksjer (der Bollinger-reversering funker)
fra trend-aksjer (der den ikke gjør det).

Eksporterer: render_bollinger_tab()
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from data_fetcher import get_stock_data
from data_sources import MARKET_OPTIONS, get_tickers_for_market
from indicators import calculate_all_indicators
from risk_manager import calculate_position_size

# ---------------------------------------------------------------------------
# Soner
# ---------------------------------------------------------------------------

ZONES = [
    (-np.inf, 0.0, "🔵 Under båndet"),
    (0.0, 0.20, "🟢 Nedre sone"),
    (0.20, 0.40, "🟡 Nedre-midt"),
    (0.40, 0.60, "⚪ Midt"),
    (0.60, 0.80, "🟠 Øvre-midt"),
    (0.80, 1.00, "🔴 Øvre sone"),
    (1.00, np.inf, "🔥 Over båndet"),
]


def zone_label(pct_b: float) -> str:
    if not np.isfinite(pct_b):
        return ""
    for lo, hi, label in ZONES:
        if lo <= pct_b < hi:
            return label
    return ZONES[-1][2]


def percent_b(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """%B — hvor i båndet kursen ligger. Returnerer NaN der båndet har bredde 0."""
    span = upper - lower
    return ((close - lower) / span.where(span > 0)).astype(float)


# ---------------------------------------------------------------------------
# Historisk reverseringsevne — "funker Bollinger på denne aksjen?"
# ---------------------------------------------------------------------------

def analyze_bb_reversion(
    df: pd.DataFrame,
    zone: float = 0.20,
    max_days: int = 15,
) -> dict:
    """Mål hvor ofte aksjen har returnert til midtbåndet etter et fall ned i
    nedre sone.

    En hendelse starter når %B krysser NED under terskelen `zone`. Treff =
    kursen nådde midtbåndet (SMA20) innen `max_days` handledager etter at
    hendelsen startet. Hendelser som ennå ikke har full fasit (for nær i tid)
    telles ikke, slik at treffraten ikke blir kunstig lav.
    """
    out = {"events": 0, "hits": 0, "hit_rate": np.nan,
           "median_days": np.nan, "avg_gain": np.nan}

    if "BB_Lower" not in df.columns or len(df) < 60:
        return out

    pb = percent_b(df["Close"], df["BB_Lower"], df["BB_Upper"]).to_numpy()
    close = df["Close"].to_numpy(dtype=float)
    mid = df["SMA20"].to_numpy(dtype=float)
    n = len(df)

    days, gains = [], []
    i = 1
    while i < n:
        if not (np.isfinite(pb[i]) and np.isfinite(pb[i - 1])):
            i += 1
            continue
        if pb[i] < zone <= pb[i - 1]:
            # Hendelse: fallet ned i nedre sone starter her
            entry = close[i]
            end = min(i + max_days, n)
            if end - i < 3:
                break  # for nær i tid til å ha fasit
            hit_at = None
            for j in range(i + 1, end):
                if np.isfinite(mid[j]) and close[j] >= mid[j]:
                    hit_at = j
                    break
            out["events"] += 1
            if hit_at is not None:
                out["hits"] += 1
                days.append(hit_at - i)
                gains.append((close[hit_at] / entry - 1.0) * 100.0)
            # Hopp fram til kursen er ute av sonen igjen
            while i < n and np.isfinite(pb[i]) and pb[i] < zone:
                i += 1
        else:
            i += 1

    if out["events"] > 0:
        out["hit_rate"] = out["hits"] / out["events"] * 100.0
    if days:
        out["median_days"] = float(np.median(days))
        out["avg_gain"] = float(np.mean(gains))
    return out


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def calculate_bollinger_score(df: pd.DataFrame) -> tuple:
    """Bollinger-score 0-100. Nede i båndet = høy score.

    Returnerer (score, reasons, metrics). metrics inneholder %B, båndbredde-
    persentil, avstand til midt/øvre bånd og antall dager i nedre sone.
    """
    curr, prev = df.iloc[-1], df.iloc[-2]
    price = float(curr["Close"])
    lower, upper, mid = (
        float(curr.get("BB_Lower", np.nan)),
        float(curr.get("BB_Upper", np.nan)),
        float(curr.get("SMA20", np.nan)),
    )

    metrics = {"pct_b": np.nan, "width_pct": np.nan, "to_mid": np.nan,
               "to_upper": np.nan, "days_low": 0, "banding": False}

    if not all(np.isfinite(v) for v in (price, lower, upper, mid)) or upper <= lower:
        return 0, ["Mangler Bollinger-data"], metrics

    pb_series = percent_b(df["Close"], df["BB_Lower"], df["BB_Upper"])
    pb = float(pb_series.iloc[-1])
    pb_prev = float(pb_series.iloc[-2]) if len(pb_series) > 1 else np.nan
    metrics["pct_b"] = pb

    # Avstand til midtbånd (reverseringsmål) og øvre bånd (full reversering)
    metrics["to_mid"] = (mid / price - 1.0) * 100.0
    metrics["to_upper"] = (upper / price - 1.0) * 100.0

    # Båndbredde-persentil: lav = squeeze (sammentrekning før utbrudd)
    if "BB_Width" in df.columns:
        w = df["BB_Width"].dropna()
        if len(w) >= 60:
            metrics["width_pct"] = float(
                (w.tail(250) < w.iloc[-1]).mean() * 100.0
            )

    # Banding ("walking the band"): kursen er klistret til nedre bånd OG
    # fortsatt synkende. Begge må stemme — ligger kursen i nedre sone mens
    # den bygger base sidelengs, er det ikke et fritt fall, og da skal den
    # ikke straffes.
    days_low = int((pb_series.tail(10) < 0.20).sum())
    metrics["days_low"] = days_low
    still_falling = len(df) > 10 and price < float(df["Close"].iloc[-11])
    metrics["banding"] = days_low >= 6 and still_falling

    score, reasons = 0.0, []

    # ---------- A. POSISJON I BÅNDET (0-60) ----------
    # Lineært invertert %B: nedre bånd = 60 p, midt = 30 p, øvre = 0 p.
    pos = 60.0 * (1.0 - float(np.clip(pb, 0.0, 1.0)))
    score += pos
    reasons.append(f"{zone_label(pb)} (%B {pb * 100:.0f})")

    # ---------- B. BÅNDKONTEKST (0-25) ----------
    ctx = 0.0

    # Straff for banding — kurs klistret til nedre bånd er nedtrend, ikke sprett
    if metrics["banding"]:
        ctx -= 12.0
        reasons.append(
            f"⚠️ Vandrer langs nedre bånd ({days_low}/10 dager, fortsatt fallende)"
        )
    elif days_low <= 2 and pb < 0.25:
        ctx += 8.0
        reasons.append("Ferskt fall i nedre sone")

    # Squeeze — sammentrukket bånd gir større utbruddspotensial
    wp = metrics["width_pct"]
    if np.isfinite(wp):
        if wp <= 20:
            ctx += 9.0
            reasons.append(f"Squeeze (bredde i {wp:.0f}. persentil)")
        elif wp <= 40:
            ctx += 5.0
        elif wp >= 90:
            ctx -= 4.0
            reasons.append("Ekstremt vidt bånd (høy volatilitet)")

    # Rom opp til midtbåndet = reverseringspotensial
    if metrics["to_mid"] >= 6:
        ctx += 8.0
        reasons.append(f"{metrics['to_mid']:.1f}% opp til midtbånd")
    elif metrics["to_mid"] >= 3:
        ctx += 5.0

    # Over EMA200 = reversering i opptrend, langt tryggere enn i nedtrend
    ema200 = curr.get("EMA200", np.nan)
    if np.isfinite(ema200):
        if price > ema200:
            ctx += 6.0
            reasons.append("Over EMA200 (dip i opptrend)")
        else:
            ctx -= 5.0
            reasons.append("Under EMA200 (dip i nedtrend)")

    score += float(np.clip(ctx, -20.0, 25.0))

    # ---------- C. VENDINGSBEKREFTELSE (0-15) ----------
    conf = 0.0
    if np.isfinite(pb_prev) and pb > pb_prev + 0.03:
        conf += 5.0
        reasons.append("%B stiger")
    if price > float(prev["Close"]):
        conf += 3.0
    rsi, rsi_prev = curr.get("RSI", np.nan), prev.get("RSI", np.nan)
    if np.isfinite(rsi) and np.isfinite(rsi_prev) and rsi > rsi_prev and rsi < 50:
        conf += 3.0
        reasons.append(f"RSI snur opp ({rsi:.0f})")
    if curr.get("Hist", 0) > prev.get("Hist", 0):
        conf += 2.0
    rvol = curr.get("RVOL", 0)
    if np.isfinite(rvol) and rvol > 1.3 and price > float(prev["Close"]):
        conf += 2.0
        reasons.append(f"Volum inn (RVOL {rvol:.1f})")

    score += min(conf, 15.0)

    return int(np.clip(round(score), 0, 100)), reasons, metrics


# ---------------------------------------------------------------------------
# Presentasjon
# ---------------------------------------------------------------------------

def _column_config() -> dict:
    return {
        "Score": st.column_config.ProgressColumn(
            "BB-score", format="%d", min_value=0, max_value=100,
            help="0-100. Nede i båndet gir høy score; banding og nedtrend straffes.",
        ),
        "%B": st.column_config.ProgressColumn(
            "%B", format="%.0f", min_value=0, max_value=100,
            help="0 = nedre bånd, 50 = midtbånd, 100 = øvre bånd",
        ),
        "Reversering %": st.column_config.ProgressColumn(
            "Rev.treff", format="%.0f%%", min_value=0, max_value=100,
            help="Hvor ofte aksjen før har nådd midtbåndet etter fall i nedre sone",
        ),
        "Til midt %": st.column_config.NumberColumn(
            "Til midt %", format="%+.1f", help="Oppside til midtbåndet (SMA20)"
        ),
        "Til øvre %": st.column_config.NumberColumn("Til øvre %", format="%+.1f"),
        "Bredde pst": st.column_config.NumberColumn(
            "Bredde pst", format="%.0f",
            help="Båndbreddens persentil. Lav = squeeze.",
        ),
        "Dager i sone": st.column_config.NumberColumn(
            "Dgr i sone", format="%d",
            help="Dager i nedre sone siste 10. Høyt tall kombinert med fallende "
                 "kurs = vandrer langs båndet",
        ),
        "Pris": st.column_config.NumberColumn("Pris", format="%.2f"),
        "Nedre": st.column_config.NumberColumn("Nedre bånd", format="%.2f"),
        "Midt": st.column_config.NumberColumn("Midt (SMA20)", format="%.2f"),
        "Øvre": st.column_config.NumberColumn("Øvre bånd", format="%.2f"),
    }


def _build_chart(df_view: pd.DataFrame, ticker: str, row: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pb = percent_b(df_view["Close"], df_view["BB_Lower"], df_view["BB_Upper"]) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        subplot_titles=(f"{ticker} — Bollinger-bånd (20, 2σ)", "%B — posisjon i båndet"),
    )

    # Båndet som fylt område
    fig.add_trace(go.Scatter(
        x=df_view.index, y=df_view["BB_Upper"], name="Øvre bånd",
        line=dict(color="rgba(200,80,80,0.7)", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_view.index, y=df_view["BB_Lower"], name="Nedre bånd",
        line=dict(color="rgba(80,160,80,0.7)", width=1),
        fill="tonexty", fillcolor="rgba(120,120,160,0.10)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_view.index, y=df_view["SMA20"], name="Midt (SMA20)",
        line=dict(color="orange", width=1.2, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Candlestick(
        x=df_view.index, open=df_view["Open"], high=df_view["High"],
        low=df_view["Low"], close=df_view["Close"], name=ticker,
    ), row=1, col=1)

    # %B-panel med sonene markert
    fig.add_trace(go.Scatter(
        x=df_view.index, y=pb, name="%B",
        line=dict(color="#6a5acd", width=1.6),
    ), row=2, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(80,180,80,0.14)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(200,80,80,0.14)",
                  line_width=0, row=2, col=1)
    for y in (0, 50, 100):
        fig.add_hline(y=y, line_color="gray", line_width=0.8,
                      line_dash="dot", row=2, col=1)

    fig.update_layout(
        height=640, xaxis_rangeslider_visible=False, showlegend=True,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="Kurs", row=1, col=1)
    fig.update_yaxes(title_text="%B", range=[-25, 125], row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Hovedinngang
# ---------------------------------------------------------------------------

def render_bollinger_tab(portfolio_value: float = 100_000, risk_pct: float = 1.5):
    st.markdown("## 🎚️ Bollinger-posisjon — hvor i båndet ligger aksjen?")
    st.caption(
        "Rangerer aksjer etter hvor nede i Bollinger-båndet de ligger. "
        "Nedre bånd gir høy score — men aksjer som *vandrer* langs nedre bånd "
        "i en nedtrend straffes, for det er en felle, ikke et kjøpssignal."
    )

    with st.expander("📖 Slik leses %B og scoren", expanded=False):
        st.markdown("""
**%B** er kjernemålet — hvor i båndet kursen står:

| %B | Betyr | Sone |
|---|---|---|
| under 0 | brutt **under** nedre bånd | 🔵 Under båndet |
| 0 | presis på **nedre** bånd | 🟢 Nedre sone |
| 50 | på **midtbåndet** (SMA20) | ⚪ Midt |
| 100 | presis på **øvre** bånd | 🔴 Øvre sone |
| over 100 | brutt **over** øvre bånd | 🔥 Over båndet |

**Bollinger-score (0–100)** settes sammen slik:

| Del | Maks | Hva som måles |
|---|---|---|
| **A. Posisjon** | 60 | Invertert %B. Nedre bånd = 60 p, midt = 30 p, øvre = 0 p |
| **B. Båndkontekst** | 25 | Squeeze, rom opp til midtbånd, over/under EMA200 — **minus** for banding |
| **C. Bekreftelse** | 15 | %B stiger, grønn dag, RSI snur, volum inn |

**Den viktigste fellen:** nedre bånd er ikke automatisk kjøp. I en etablert
nedtrend glir kursen nedover langs nedre bånd i ukevis. Kolonnen **Dgr i sone**
viser hvor mange av siste 10 dager kursen lå i nedre sone. Ligger den der 6
dager eller mer **og** kursen fortsatt er lavere enn for 10 dager siden, får
aksjen 12 poeng i straff og flagget «vandrer langs nedre bånd». Ligger den i
nedre sone mens den bygger base sidelengs, straffes den ikke — da er det
nettopp setupet du leter etter.

**Rev.treff** er aksjens egen historikk: hvor ofte den tidligere har klart å
komme tilbake til midtbåndet etter et fall ned i nedre sone. Høy treffrate =
en aksje som faktisk reverserer. Lav = en trendaksje der Bollinger-sprett
ikke er å stole på.
        """)

    # ---- Innstillinger ----
    with st.expander("⚙️ Innstillinger", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            markets = [m for m in MARKET_OPTIONS if m != "Egen liste"]
            market = st.selectbox("Marked", markets, key="bb_market")
            limit = st.number_input(
                "Maks antall aksjer", 10, 400, 150, 10, key="bb_limit"
            )
        with c2:
            max_pct_b = st.slider(
                "Maks %B (kun aksjer nede i båndet)", 0, 100, 35, 5, key="bb_maxpb",
                help="35 betyr: vis bare aksjer i nedre tredel av båndet",
            )
            min_score = st.slider("Min BB-score", 0, 100, 0, 5, key="bb_minscore")
        with c3:
            hide_banding = st.checkbox(
                "Skjul aksjer som vandrer langs nedre bånd", value=True,
                key="bb_hideband",
                help="Filtrerer bort aksjer som har ligget i nedre sone 6+ av siste 10 dager",
            )
            min_turnover = st.number_input(
                "Min daglig omsetning (mill)", 0.0, 100.0, 1.0, 0.5, key="bb_minto"
            )

    if st.button("🎚️ Scan Bollinger-posisjon", key="bb_run", use_container_width=True):
        if market == "Min Watchlist":
            tickers = list(st.session_state.get("watchlist") or [])
        else:
            tickers = get_tickers_for_market(market)
        tickers = tickers[: int(limit)]

        if not tickers:
            st.warning("Ingen aksjer i valgt marked.")
            return

        status = st.status(f"Analyserer {len(tickers)} aksjer…", expanded=True)
        with status:
            st.write("Henter kurshistorikk…")
            data, infos = get_stock_data(tickers, period="2y")
            st.write(f"Beregner Bollinger-posisjon for {len(data)} aksjer…")

            rows = []
            skipped_band = 0
            for ticker, raw in data.items():
                try:
                    if len(raw) < 60:
                        continue
                    turnover = (raw["Close"] * raw["Volume"]).tail(20).mean()
                    if turnover < min_turnover * 1e6:
                        continue

                    df = calculate_all_indicators(raw)
                    if "BB_Lower" not in df.columns:
                        continue

                    score, reasons, m = calculate_bollinger_score(df)
                    pb = m["pct_b"]
                    if not np.isfinite(pb):
                        continue
                    if pb * 100 > max_pct_b:
                        continue
                    if hide_banding and m["banding"]:
                        skipped_band += 1
                        continue
                    if score < min_score:
                        continue

                    rev = analyze_bb_reversion(df)
                    curr = df.iloc[-1]
                    price = float(curr["Close"])
                    lower = float(curr["BB_Lower"])
                    atr = float(curr.get("ATR", 0) or 0)
                    # SL under nedre bånd, TP på midtbåndet (reverseringsmålet)
                    sl = lower - 0.5 * atr if atr > 0 else lower * 0.97
                    tp = float(curr["SMA20"])
                    shares, risk_nok = calculate_position_size(
                        portfolio_value, risk_pct, price, sl
                    )

                    rows.append({
                        "Ticker": ticker,
                        "Navn": (infos.get(ticker) or {}).get("shortName", ticker),
                        "Score": score,
                        "Sone": zone_label(pb),
                        "%B": pb * 100,
                        "Til midt %": m["to_mid"],
                        "Til øvre %": m["to_upper"],
                        "Bredde pst": m["width_pct"],
                        "Dager i sone": m["days_low"],
                        "Reversering %": rev["hit_rate"],
                        "Rev. N": rev["events"],
                        "Rev. dager": rev["median_days"],
                        "Pris": price,
                        "Nedre": lower,
                        "Midt": tp,
                        "Øvre": float(curr["BB_Upper"]),
                        "Stop Loss": round(sl, 2),
                        "Take Profit": round(tp, 2),
                        "Antall": shares,
                        "Signaler": ", ".join(reasons[:4]),
                        "_df": df,
                        "_reasons": reasons,
                    })
                except Exception:
                    continue

            st.session_state["bb_results"] = (
                pd.DataFrame(rows).sort_values("Score", ascending=False)
                if rows else pd.DataFrame()
            )
            st.session_state["bb_meta"] = {
                "n_universe": len(tickers), "n_prices": len(data),
                "skipped_band": skipped_band, "market": market,
                "max_pct_b": max_pct_b,
            }
            status.update(
                label=f"Ferdig — {len(rows)} aksjer nede i båndet", state="complete"
            )

    res = st.session_state.get("bb_results")
    meta = st.session_state.get("bb_meta") or {}
    if res is None:
        st.info("Velg marked og trykk **Scan Bollinger-posisjon**.")
        return
    if res.empty:
        st.warning(
            f"Ingen aksjer med %B under {meta.get('max_pct_b')}. "
            "Hev «Maks %B» eller senk minimumskravene."
        )
        if meta.get("skipped_band"):
            st.caption(
                f"{meta['skipped_band']} aksjer ble filtrert bort fordi de vandrer "
                "langs nedre bånd."
            )
        return

    st.success(
        f"**{len(res)} aksjer** med %B under {meta['max_pct_b']} i {meta['market']} "
        f"— sortert på BB-score."
    )
    if meta.get("skipped_band"):
        st.caption(
            f"ℹ️ {meta['skipped_band']} aksjer utelatt fordi de vandrer langs nedre "
            "bånd (nedtrend, ikke reverseringssetup)."
        )

    display = [
        "Ticker", "Navn", "Score", "Sone", "%B", "Til midt %", "Til øvre %",
        "Bredde pst", "Dager i sone", "Reversering %", "Rev. N", "Rev. dager",
        "Pris", "Nedre", "Midt", "Øvre", "Stop Loss", "Take Profit", "Antall",
        "Signaler",
    ]
    cols = [c for c in display if c in res.columns]
    event = st.dataframe(
        res[cols], use_container_width=True, height=440,
        on_select="rerun", selection_mode="single-row",
        column_config=_column_config(), hide_index=True,
    )

    # ---- Topp-kandidater ----
    top = res.head(min(4, len(res)))
    st.markdown("#### 🏆 Lengst nede i båndet")
    for col, (_, r) in zip(st.columns(len(top)), top.iterrows()):
        with col:
            st.metric(
                r["Ticker"].replace(".OL", ""),
                f"{r['Pris']:.2f}",
                f"{r['Til midt %']:+.1f}% til midt",
            )
            st.caption(f"{r['Sone']} · %B {r['%B']:.0f} · score {r['Score']}")

    # ---- Detaljgraf ----
    st.markdown("---")
    tickers_list = res["Ticker"].tolist()
    sel = tickers_list[event.selection.rows[0]] if event.selection.rows else None
    idx = tickers_list.index(sel) if sel in tickers_list else 0

    g1, g2 = st.columns([1, 3])
    chosen = g1.selectbox("Vis graf for:", tickers_list, index=idx, key="bb_sel")
    period = g2.radio(
        "Periode:", ["3 mnd", "6 mnd", "1 år", "2 år"], index=1,
        horizontal=True, key="bb_period",
    )

    if chosen:
        row = res[res["Ticker"] == chosen].iloc[0]
        df_full = row["_df"]
        days = {"3 mnd": 90, "6 mnd": 180, "1 år": 365, "2 år": 730}[period]
        df_view = df_full.copy()
        if getattr(df_view.index, "tz", None) is not None:
            df_view.index = df_view.index.tz_localize(None)
        df_view = df_view[df_view.index >= df_view.index[-1] - timedelta(days=days)]

        if not df_view.empty:
            st.plotly_chart(
                _build_chart(df_view, chosen, row), use_container_width=True
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("%B nå", f"{row['%B']:.0f}", row["Sone"])
        m2.metric("Oppside til midtbånd", f"{row['Til midt %']:+.1f}%")
        rev = row["Reversering %"]
        m3.metric(
            "Historisk reversering",
            "—" if not np.isfinite(rev) else f"{rev:.0f}%",
            f"{int(row['Rev. N'])} hendelser",
        )
        m4.metric(
            "Median dager til midt",
            "—" if not np.isfinite(row["Rev. dager"]) else f"{row['Rev. dager']:.0f}",
        )
        st.markdown(f"**Alle signaler:** {', '.join(row['_reasons'])}")

        st.download_button(
            "⬇ Last ned tabell som CSV",
            res[cols].to_csv(index=False).encode("utf-8"),
            file_name=f"bollinger_posisjon_{meta['market'].replace(' ', '_')}.csv",
            mime="text/csv",
        )

    st.caption(
        "Bollinger-bånd: SMA20 ± 2 standardavvik, som i resten av appen. "
        "Stop loss settes under nedre bånd (nedre − ½ ATR), take profit på "
        "midtbåndet, som er det naturlige reverseringsmålet. "
        "Manglende historikk vises som tomt felt, aldri som 0."
    )
