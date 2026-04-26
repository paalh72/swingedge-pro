"""
sector_tracker.py
-----------------
Sektorrotasjons-tracker for Oslo Børs.

Wold: "Det er perioder hvor alle er i shipping, men når det begynner å dabbe
av, skal du ikke røre det. Du må flytte deg."

Viser hvilke sektorer som er 'hot' akkurat nå basert på:
  - 5-dagers momentum (kortsiktig — swingtrade-relevant)
  - 20-dagers momentum (mellomlang trend)
  - Volumvekst siste 5 vs 20 dager (ny interesse?)
  - Andel av sektorens aksjer som er i opptrend (bredde)
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from wold_scanner import OSLO_SECTORS


# ---------------------------------------------------------------------------
# SHIPPING RATES (Baltic indices)
# Wold: "Jeg er veldig glad i å trade shipping — du kan følge ratene ganske tett"
# ---------------------------------------------------------------------------
SHIPPING_INDICATORS = {
    "Baltic Dry Index (BDI)": "^BDI",
    "Breakwave Dry Bulk ETF (proxy)": "BDRY",
    "Frontline (FRO) — VLCC proxy": "FRO",
    "Golden Ocean (GOGL) — cape proxy": "GOGL.OL",
}


@st.cache_data(ttl=3600)
def get_shipping_rates() -> dict:
    """Hent shippingrate-indikatorer. Returnerer dict med siste verdi og 5d-endring."""
    results = {}
    for name, symbol in SHIPPING_INDICATORS.items():
        try:
            hist = yf.Ticker(symbol).history(period="1mo", timeout=10)
            if len(hist) >= 2:
                last = hist['Close'].iloc[-1]
                prev5 = hist['Close'].iloc[-6] if len(hist) >= 6 else hist['Close'].iloc[0]
                chg5d = (last - prev5) / prev5 * 100
                results[name] = {
                    "symbol": symbol,
                    "last": round(last, 2),
                    "chg5d": round(chg5d, 1),
                    "trend": "opp" if chg5d > 0 else "ned",
                }
        except Exception:
            results[name] = {"symbol": symbol, "last": None, "chg5d": None, "trend": "ukjent"}
    return results


# ---------------------------------------------------------------------------
# SEKTOR MOMENTUM
# ---------------------------------------------------------------------------
def compute_sector_momentum(data: dict) -> pd.DataFrame:
    """
    Beregn momentum for hver Oslo Børs-sektor.

    Parametere:
        data: dict {ticker: DataFrame} — allerede hentet data

    Returnerer DataFrame med kolonner:
        Sektor, Aksjer_i_data, Momentum_5d, Momentum_20d,
        Vol_ratio, Andel_opptrend, Heatmap_score
    """
    rows = []

    for sector_name, sector_tickers in OSLO_SECTORS.items():
        mom5_list, mom20_list, vol_ratio_list, opptrend_list = [], [], [], []

        for ticker in sector_tickers:
            if ticker not in data:
                continue
            df = data[ticker]
            if len(df) < 25:
                continue

            close = df['Close']
            volume = df['Volume']

            # 5-dagers momentum
            if len(close) > 6:
                mom5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100
                mom5_list.append(mom5)

            # 20-dagers momentum
            if len(close) > 21:
                mom20 = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100
                mom20_list.append(mom20)

            # Volumvekst: snitt siste 5d vs snitt siste 20d
            vol_5d = volume.tail(5).mean()
            vol_20d = volume.tail(20).mean()
            if vol_20d > 0:
                vol_ratio_list.append(vol_5d / vol_20d)

            # Opptrend: pris > EMA20?
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            opptrend_list.append(1 if close.iloc[-1] > ema20 else 0)

        n = len(mom5_list)
        if n == 0:
            continue

        avg_mom5 = np.mean(mom5_list)
        avg_mom20 = np.mean(mom20_list) if mom20_list else 0
        avg_vol_ratio = np.mean(vol_ratio_list) if vol_ratio_list else 1.0
        pct_opptrend = np.mean(opptrend_list) * 100 if opptrend_list else 0

        # Heatmap score: kombinert mål (0-100)
        # Vekter: volum 40%, 5d mom 35%, opptrend-bredde 25%
        vol_score = min(100, (avg_vol_ratio - 1.0) / 1.0 * 100)  # 0 = snitt, 100 = 2x snitt
        mom_score = min(100, max(-100, avg_mom5 * 5))             # +/- 20% → +/- 100
        breadth_score = pct_opptrend

        heatmap = 0.40 * max(0, vol_score) + 0.35 * max(0, mom_score) + 0.25 * breadth_score
        heatmap = round(min(100, max(0, heatmap)), 1)

        rows.append({
            "Sektor": sector_name,
            "Aksjer funnet": n,
            "Mom 5d %": round(avg_mom5, 1),
            "Mom 20d %": round(avg_mom20, 1),
            "Vol-ratio": round(avg_vol_ratio, 2),
            "% i opptrend": round(pct_opptrend, 0),
            "Heatmap Score": heatmap,
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values("Heatmap Score", ascending=False)
    return df_out


# ---------------------------------------------------------------------------
# HEATMAP VISUALISERING
# ---------------------------------------------------------------------------
def render_sector_heatmap(sector_df: pd.DataFrame):
    """Vis sektortabell med fargemarkering i Streamlit."""
    if sector_df.empty:
        st.warning("Ingen sektordata tilgjengelig — kjør en scan først.")
        return

    st.markdown("### Sektorrotasjon — Oslo Børs")
    st.caption(
        "Wold: *'Du må flytte deg når sektoren snur. Nå er shipping inn — "
        "men når ratene dabber av, kjøper du ikke.'*"
    )

    def color_heatmap(val):
        if isinstance(val, (int, float)):
            if val >= 70:
                return "background-color: #1a7a1a; color: white"
            elif val >= 50:
                return "background-color: #6b8e23; color: white"
            elif val >= 30:
                return "background-color: #ffa500; color: black"
            else:
                return "background-color: #8b0000; color: white"
        return ""

    def color_mom(val):
        if isinstance(val, (int, float)):
            if val > 5:
                return "color: #00aa00; font-weight: bold"
            elif val > 0:
                return "color: #88cc00"
            elif val > -5:
                return "color: #ff8800"
            else:
                return "color: #cc0000; font-weight: bold"
        return ""

    styled = (
        sector_df.style
        .applymap(color_heatmap, subset=["Heatmap Score"])
        .applymap(color_mom, subset=["Mom 5d %", "Mom 20d %"])
        .format({
            "Mom 5d %": "{:+.1f}%",
            "Mom 20d %": "{:+.1f}%",
            "Vol-ratio": "{:.2f}x",
            "% i opptrend": "{:.0f}%",
            "Heatmap Score": "{:.0f}",
        })
    )

    st.dataframe(styled, use_container_width=True, height=350)

    # Hottest sektor
    if not sector_df.empty:
        best = sector_df.iloc[0]
        worst = sector_df.iloc[-1]
        c1, c2 = st.columns(2)
        c1.success(
            f"**Varmeste sektor:** {best['Sektor']} "
            f"(Score {best['Heatmap Score']:.0f}, 5d: {best['Mom 5d %']:+.1f}%)"
        )
        c2.error(
            f"**Kaldeste sektor:** {worst['Sektor']} "
            f"(Score {worst['Heatmap Score']:.0f}, 5d: {worst['Mom 5d %']:+.1f}%)"
        )


# ---------------------------------------------------------------------------
# SHIPPING RATE PANEL
# ---------------------------------------------------------------------------
def render_shipping_panel():
    """Vis shippingrate-indikatorer i Streamlit."""
    st.markdown("### Shippingrater")
    st.caption(
        "Wold: *'Jeg følger ratene ganske tett. Hvis ratene er på vei opp, "
        "går kursene etter.'*"
    )

    rates = get_shipping_rates()
    cols = st.columns(len(rates))

    for col, (name, info) in zip(cols, rates.items()):
        if info["last"] is not None:
            delta_str = f"{info['chg5d']:+.1f}% (5d)" if info["chg5d"] is not None else ""
            color = "normal" if (info["chg5d"] or 0) >= 0 else "inverse"
            col.metric(
                label=name.split("(")[0].strip(),
                value=f"{info['last']:,.1f}",
                delta=delta_str,
                delta_color=color,
            )
        else:
            col.metric(label=name.split("(")[0].strip(), value="N/A")

    # Wold-interpretasjon
    st.markdown("---")
    all_rates = [v for v in rates.values() if v.get("chg5d") is not None]
    if all_rates:
        positive = sum(1 for r in all_rates if (r["chg5d"] or 0) > 0)
        pct_positive = positive / len(all_rates) * 100
        if pct_positive >= 75:
            st.success(
                "Shippingratene er i OPPGANG — Wold ville vurdert shipping-aksjer nå"
            )
        elif pct_positive >= 50:
            st.info("Blandede shippingsignaler — vær selektiv")
        else:
            st.warning(
                "Shippingratene er svake — Wold: 'Ikke rør det. Det vil bare ned.'"
            )
