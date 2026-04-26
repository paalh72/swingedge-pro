"""
newsweb.py
----------
Oslo Børs nyheter direkte i verktøyet.

Wold: "Før børsen åpner går jeg gjennom Newsweb for børsmeldinger, ser på kursmål
og justeringer fra analytikere."

Henter børsmeldinger fra Newsweb (Oslo Børs offisielle meldingssystem).
Matcher nyheter mot tickers i scan-resultater for å flagge 'fundamentalt'.
"""

import re
import html
from datetime import datetime, timedelta

import requests
import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# NEWSWEB RSS
# ---------------------------------------------------------------------------
NEWSWEB_RSS = "https://newsweb.oslobors.no/search/rss?category=&issuer=&fromDate=&toDate=&market=&messageTitle="

# Fallback: Oslo Børs alle meldinger
NEWSWEB_HTML = "https://newsweb.oslobors.no/search?category=&issuer=&fromDate=&toDate=&market="

# Analytikermeldinger søkeord
ANALYST_KEYWORDS = [
    "kursmål", "anbefaling", "kjøp", "hold", "selg",
    "oppjusterer", "nedjusterer", "target", "upgrade", "downgrade",
    "overweight", "underweight", "buy", "sell",
]

# Fundamental trigger keywords (Wold: "noe fundamentalt")
TRIGGER_KEYWORDS = [
    "kontrakt", "ordre", "kjøper", "oppkjøp", "fusjon", "resultat",
    "inntjening", "rekord", "nye", "vinner", "tildelt", "kapasitet",
    "ratene", "rate", "lån", "refinansiering", "utbytte", "dividend",
    "positiv", "sterk", "vekst", "breakthrough", "breakthrough",
]


@st.cache_data(ttl=900)  # 15 min cache
def fetch_newsweb_news(max_items: int = 40) -> list:
    """
    Hent siste børsmeldinger fra Newsweb RSS.

    Returnerer liste med dict:
      {title, ticker, published, link, category, is_analyst, is_trigger}
    """
    news_items = []

    # Forsøk RSS-feed
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SwingEdgePro/1.0)"
        )
    }

    try:
        resp = requests.get(NEWSWEB_RSS, headers=headers, timeout=10)
        resp.raise_for_status()
        content = resp.text

        # Parse RSS manuelt (unngår feedparser-avhengighet)
        items_raw = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)

        for item_raw in items_raw[:max_items]:
            title_m = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", item_raw)
            link_m = re.search(r"<link>(.*?)</link>", item_raw)
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item_raw)
            desc_m = re.search(r"<description><!\[CDATA\[(.*?)\]\]></description>", item_raw, re.DOTALL)

            title = html.unescape(title_m.group(1).strip()) if title_m else ""
            link = link_m.group(1).strip() if link_m else ""
            pub_str = pub_m.group(1).strip() if pub_m else ""
            description = html.unescape(desc_m.group(1).strip()) if desc_m else ""

            # Trekk ut ticker fra tittel/link
            ticker = _extract_ticker(title, link, description)

            # Klassifiser nyhetene
            combined = (title + " " + description).lower()
            is_analyst = any(kw in combined for kw in ANALYST_KEYWORDS)
            is_trigger = any(kw in combined for kw in TRIGGER_KEYWORDS)

            # Parse dato
            try:
                pub_dt = datetime.strptime(pub_str, "%a, %d %b %Y %H:%M:%S %z")
                pub_local = pub_dt.strftime("%d.%m %H:%M")
                hours_ago = (datetime.now(pub_dt.tzinfo) - pub_dt).total_seconds() / 3600
            except Exception:
                pub_local = pub_str[:16]
                hours_ago = 999

            news_items.append({
                "Tid": pub_local,
                "Ticker": ticker or "—",
                "Overskrift": title,
                "Link": link,
                "Analytiker": "🎯" if is_analyst else "",
                "Trigger": "⚡" if is_trigger else "",
                "_hours_ago": hours_ago,
                "_description": description,
                "_is_analyst": is_analyst,
                "_is_trigger": is_trigger,
                "_ticker_raw": ticker,
            })

    except requests.RequestException as e:
        # Stille fallback — vi viser en advarsel i UI
        return _fallback_news(str(e))

    return news_items


def _extract_ticker(title: str, link: str, description: str) -> str:
    """Prøv å trekke ut ticker-symbol fra tittel/link/beskrivelse."""
    # Newsweb-linker inneholder ofte issuer-parameteren
    m = re.search(r"issuer=([A-Z0-9]+)", link, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Parentes-format: "DNB (DNB.OL)"
    m = re.search(r"\(([A-Z]{2,6}(?:\.OL)?)\)", title)
    if m:
        return m.group(1).upper()

    return ""


def _fallback_news(error_msg: str) -> list:
    """Returner tom liste med info om feilen."""
    return [{
        "Tid": "—",
        "Ticker": "—",
        "Overskrift": f"Newsweb ikke tilgjengelig: {error_msg[:80]}",
        "Link": NEWSWEB_HTML,
        "Analytiker": "",
        "Trigger": "",
        "_hours_ago": 999,
        "_description": "",
        "_is_analyst": False,
        "_is_trigger": False,
        "_ticker_raw": "",
    }]


# ---------------------------------------------------------------------------
# MATCH NYHETER MOT SCAN-RESULTATER
# ---------------------------------------------------------------------------
def match_news_to_results(news_items: list, result_tickers: list) -> dict:
    """
    Finn nyheter for tickers som er i scan-resultatene.

    Returnerer dict: {ticker: [news_items]}
    """
    matches = {}
    result_tickers_upper = [t.upper().replace(".OL", "") for t in result_tickers]

    for item in news_items:
        raw = (item.get("_ticker_raw") or "").upper().replace(".OL", "")
        if raw and raw in result_tickers_upper:
            # Finn original ticker
            orig = next(
                (t for t in result_tickers if t.upper().replace(".OL", "") == raw),
                raw
            )
            matches.setdefault(orig, []).append(item)

    return matches


# ---------------------------------------------------------------------------
# RENDER I STREAMLIT
# ---------------------------------------------------------------------------
def render_newsweb(result_tickers: list = None):
    """
    Vis Newsweb i Streamlit.

    Hvis result_tickers er oppgitt, fremhever vi nyheter for disse.
    """
    st.markdown("### Newsweb — Oslo Børs Børsmeldinger")
    st.caption(
        "Wold: *'Før børsen åpner går jeg gjennom Newsweb for børsmeldinger."
        " Ikke for å finne fasiten, men for å forstå hvor interessen bygger seg opp.'*"
    )

    col1, col2, col3 = st.columns(3)
    filter_analyst = col1.checkbox("Kun analytikermeldinger", False)
    filter_trigger = col2.checkbox("Kun fundamentale triggere", False)
    filter_watchlist = col3.checkbox("Kun mine scan-resultater", False) if result_tickers else False

    with st.spinner("Henter siste børsmeldinger..."):
        news = fetch_newsweb_news(max_items=60)

    if not news or news[0]["Overskrift"].startswith("Newsweb ikke tilgjengelig"):
        st.warning(
            "Newsweb-feed ikke tilgjengelig akkurat nå. "
            f"[Åpne Newsweb direkte]({NEWSWEB_HTML})"
        )
        return

    # Match mot scan-resultater
    matched_tickers = set()
    if result_tickers:
        matches = match_news_to_results(news, result_tickers)
        matched_tickers = set(matches.keys())

    # Filtrer
    display_news = news
    if filter_analyst:
        display_news = [n for n in display_news if n["_is_analyst"]]
    if filter_trigger:
        display_news = [n for n in display_news if n["_is_trigger"]]
    if filter_watchlist and result_tickers:
        result_stripped = [t.upper().replace(".OL", "") for t in result_tickers]
        display_news = [
            n for n in display_news
            if (n.get("_ticker_raw") or "").upper().replace(".OL", "") in result_stripped
        ]

    if not display_news:
        st.info("Ingen meldinger med gjeldende filtre.")
        return

    # Vis som tabell
    df_news = pd.DataFrame([{
        "Tid": n["Tid"],
        "Ticker": n["Ticker"],
        "⚡": n["Trigger"],
        "🎯": n["Analytiker"],
        "Overskrift": n["Overskrift"],
    } for n in display_news])

    # Fremhev rader for scan-kandidater
    def highlight_row(row):
        ticker_clean = row["Ticker"].replace(".OL", "")
        if result_tickers and ticker_clean in [t.replace(".OL", "") for t in (result_tickers or [])]:
            return ["background-color: #1a3a1a; color: #90ee90"] * len(row)
        return [""] * len(row)

    styled = df_news.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True, height=500)

    # Vis linker for topp 5
    st.markdown("#### Siste meldinger (med lenke)")
    for n in display_news[:8]:
        icon = ""
        if n["_is_trigger"]:
            icon += "⚡"
        if n["_is_analyst"]:
            icon += "🎯"
        if result_tickers and (n.get("_ticker_raw") or "").replace(".OL", "") in [
            t.replace(".OL", "") for t in result_tickers
        ]:
            icon += "✅"

        label = f"{icon} [{n['Ticker']}] {n['Overskrift'][:80]}"
        st.markdown(f"- {n['Tid']} — [{label}]({n['Link']})")

    st.caption(
        f"Viser {len(display_news)} av {len(news)} meldinger. "
        f"[Åpne Newsweb]({NEWSWEB_HTML})"
    )
