"""
earnings_reaction.py
--------------------
Analysemodul for kursreaksjon rundt kvartalsrapporter.

Modus A: screening på tvers av et helt marked for ett valgt kvartal.
Modus B: enkeltaksjer brutt ned per kvartal (Q1/Q2/Q3/Q4).


=============================================================================
KURSKONVENSJON — les denne før du tolker tallene
=============================================================================

Problemet: en rapport kan komme før børsåpning (BMO), i løpet av dagen, eller
etter stengetid (AMC). Kommer den etter stengetid, skjer kursreaksjonen først
NESTE børsdag. Måler man da "rapportdagen", måler man dagen FØR nyheten kom.

Løsningen — vi forankrer alt på "reaksjonsdagen" (event day):

  1. Rapporttidspunktet fra datakilden konverteres til børsens lokale tid.
  2. Er klokkeslettet ETTER stengetid  -> reaksjonsdag = neste børsdag.
     Er det før åpning eller i løpet av dagen -> reaksjonsdag = rapportdagen.
     Faller rapportdagen på en ikke-børsdag (helg/helligdag) flyttes den
     fram til første påfølgende børsdag.
  3. BASISKURS = sluttkurs siste børsdag FØR reaksjonsdagen.
  4. Dag 0  = sluttkurs(reaksjonsdag)        / basiskurs - 1
     Dag +n = sluttkurs(reaksjonsdag + n bd) / basiskurs - 1

Dag +1/+3/+5 er altså KUMULATIVE fra basiskursen, ikke dag-for-dag-endringer,
og "n bd" er handledager (helligdager hoppes over automatisk fordi vi teller
posisjoner i den faktiske kursserien).

Hvorfor sluttkurs-til-sluttkurs og ikke rapportdagens åpning som basis:
kommer rapporten før åpning, ligger hele den første reaksjonen i åpningsgapet.
Bruker man åpningskursen som basis, måler man alt UNNTATT selve nyheten.
Vi tar derfor med gapet — men rapporterer det også separat i kolonnen
"Gap %" (= åpning(reaksjonsdag) / basiskurs - 1), slik at man ser hvor mye av
reaksjonen som kom i gapet og hvor mye som kom i løpet av dagen.

MERAVKASTNING = aksjens avkastning minus referanseindeksens avkastning over
nøyaktig samme datointervall. Mangler indeksdata for en av datoene, blir
meravkastningen stående tom — den gjettes aldri.

KURSER er utbytte- og splittjusterte (auto_adjust i datakilden), slik at et
utbytte som går ex på reaksjonsdagen ikke feilaktig ser ut som et kursfall.

MANGLENDE DATA vises alltid som tomt felt, aldri som 0 eller et antatt tall.
=============================================================================

Kvartalsmerking: datakilden oppgir rapportdato, ikke hvilken regnskapsperiode
rapporten gjelder. Vi utleder kvartalet fra rapportmåneden etter nordisk
standard (Q4 rapporteres jan–mar, Q1 apr–jun, Q2 jul–sep, Q3 okt–des).
Selskaper med avvikende regnskapsår (f.eks. Apple) blir merket feil — derfor
vises rapportmåneden i tabellen så mønsteret kan kontrolleres.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from data_fetcher import get_stock_data
from data_sources import MARKET_OPTIONS, get_tickers_for_market, get_oslo_name_map

# ---------------------------------------------------------------------------
# Konfigurasjon
# ---------------------------------------------------------------------------

HORIZONS = (0, 1, 3, 5)

# Marked -> tilgjengelige referanseindekser. Første er standard.
# Historikk-kommentarene er verifisert mot datakilden.
MARKET_BENCHMARKS = {
    "Oslo Bors": [("OBX.OL", "OBX (fra 2004)"), ("OSEBX.OL", "OSEBX (kun fra 2013)")],
    "Oslo Bors (ALLE ~300)": [("OBX.OL", "OBX (fra 2004)"), ("OSEBX.OL", "OSEBX (kun fra 2013)")],
    "USA (Nasdaq 100)": [("^NDX", "Nasdaq 100 (fra 1985)")],
    "USA (S&P 500)": [("^GSPC", "S&P 500 (fra 1927)")],
    "USA (Alle Aksjer)": [("^GSPC", "S&P 500 (fra 1927)")],
    "Stockholm (Large)": [("^OMX", "OMXS30 (fra 2008)"), ("^OMXSPI", "OMXSPI (kun fra 2013)")],
    "Frankfurt (DAX)": [("^GDAXI", "DAX (fra 1987)")],
    "Paris (CAC)": [("^FCHI", "CAC 40 (fra 1990)")],
    "London (FTSE)": [("^FTSE", "FTSE 100 (fra 1984)")],
    "Min Watchlist": [("OBX.OL", "OBX (fra 2004)"), ("^GSPC", "S&P 500"), ("^OMX", "OMXS30")],
}

# Børsregler for å avgjøre før/etter stengetid.
# suffix -> (tidssone, åpning som desimaltime, stengetid som desimaltime)
_EXCHANGE_RULES = {
    ".OL": ("Europe/Oslo", 9.0, 16.33),
    ".ST": ("Europe/Stockholm", 9.0, 17.5),
    ".CO": ("Europe/Copenhagen", 9.0, 17.0),
    ".HE": ("Europe/Helsinki", 10.0, 18.5),
    ".DE": ("Europe/Berlin", 9.0, 17.5),
    ".PA": ("Europe/Paris", 9.0, 17.5),
    ".L": ("Europe/London", 8.0, 16.5),
}
_US_RULE = ("America/New_York", 9.5, 16.0)

# Rapportmåned -> hvilket kvartal rapporten gjelder (nordisk/europeisk standard).
_MONTH_TO_QUARTER = {
    1: "Q4", 2: "Q4", 3: "Q4",
    4: "Q1", 5: "Q1", 6: "Q1",
    7: "Q2", 8: "Q2", 9: "Q2",
    10: "Q3", 11: "Q3", 12: "Q3",
}


def _exchange_rule(ticker: str) -> tuple:
    """Finn tidssone og åpnings-/stengetid for tickerens børs."""
    for suffix, rule in _EXCHANGE_RULES.items():
        if ticker.upper().endswith(suffix):
            return rule
    return _US_RULE


# ---------------------------------------------------------------------------
# Datahenting (cachet — følger samme mønster som resten av appen)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_earnings_calendar(tickers: tuple) -> dict:
    """Hent rapportdatoer + konsensus-EPS for flere tickere parallelt.

    Returnerer {ticker: DataFrame} der DataFrame har rapportdato som
    tz-aware indeks og kolonnene 'EPS Estimate', 'Reported EPS',
    'Surprise(%)'. Tickere uten data utelates helt.

    Krever lxml (yfinance parser en HTML-tabell via pandas.read_html).
    """
    def _one(ticker: str):
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=80)
            if ed is None or len(ed) == 0:
                return ticker, None
            return ticker, ed
        except Exception:
            return ticker, None

    out = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, ed in pool.map(_one, tickers):
            if ed is not None:
                out[ticker] = ed
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(symbol: str) -> pd.Series | None:
    """Hent referanseindeksens sluttkurser. Returnerer tz-naiv dagserie."""
    try:
        hist = yf.Ticker(symbol).history(period="max", timeout=20)
        if hist.empty:
            return None
        idx = hist.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        close = pd.Series(hist["Close"].values, index=pd.DatetimeIndex(idx).normalize())
        return close[~close.index.duplicated(keep="last")]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Kjerneberegning
# ---------------------------------------------------------------------------

def compute_events(
    ticker: str,
    price_df: pd.DataFrame,
    earnings: pd.DataFrame,
    bench: pd.Series | None,
    timing_mode: str = "auto",
) -> list[dict]:
    """Regn ut kursreaksjonen for hver rapport for én aksje.

    Se modulens toppdokumentasjon for kurskonvensjonen. Returnerer én dict
    per rapport med nøklene ret_0/ret_1/ret_3/ret_5 (kursendring i prosent
    fra basiskurs), exc_* (meravkastning mot indeks), gap, surprise,
    kvartal, rapportdato og timing (BMO/Intradag/AMC).

    timing_mode styrer hvordan reaksjonsdagen velges:
      "auto"      — bruk klokkeslettet fra datakilden (standard)
      "same_day"  — anta alltid at reaksjonen kommer på rapportdagen
      "next_day"  — anta alltid at reaksjonen kommer dagen etter
    Datakildens klokkeslett er delvis upålitelig for europeiske aksjer, så
    "same_day"/"next_day" finnes for å teste hvor følsomme tallene er for
    dette valget.
    """
    if price_df is None or len(price_df) < 10 or earnings is None or earnings.empty:
        return []

    # Normaliser kursserien til tz-naive datoer
    pidx = price_df.index
    if getattr(pidx, "tz", None) is not None:
        pidx = pidx.tz_localize(None)
    dates = pd.DatetimeIndex(pidx).normalize()
    closes = price_df["Close"].to_numpy(dtype=float)
    opens = (
        price_df["Open"].to_numpy(dtype=float)
        if "Open" in price_df.columns
        else np.full(len(closes), np.nan)
    )

    # Indeksen justert inn på aksjens handledager. Ingen ffill: mangler
    # indeksdata for en dato, skal meravkastningen stå tom.
    if bench is not None and len(bench) > 0:
        bench_vals = bench.reindex(dates).to_numpy(dtype=float)
    else:
        bench_vals = np.full(len(dates), np.nan)

    tz, open_h, close_h = _exchange_rule(ticker)
    n = len(dates)
    events = []

    for ts, row in earnings.iterrows():
        # --- 1. Rapporttidspunkt i børsens lokale tid ---
        try:
            local = ts.tz_convert(tz) if ts.tzinfo is not None else ts.tz_localize(tz)
        except Exception:
            local = ts
        clock = local.hour + local.minute / 60.0
        report_day = pd.Timestamp(local.date())

        # --- 2. Reaksjonsdag: etter stengetid => neste børsdag ---
        if clock >= close_h:
            timing = "AMC"
        elif clock < open_h:
            timing = "BMO"
        else:
            timing = "Intradag"

        if timing_mode == "same_day":
            after_close = False
        elif timing_mode == "next_day":
            after_close = True
        else:
            after_close = timing == "AMC"
        target = report_day + pd.Timedelta(days=1) if after_close else report_day

        pos = int(dates.searchsorted(target, side="left"))
        if pos <= 0 or pos >= n:
            continue  # ingen basisdag foran, eller rapport i framtiden

        base = closes[pos - 1]
        if not np.isfinite(base) or base <= 0:
            continue

        base_b = bench_vals[pos - 1]

        ev = {
            "ticker": ticker,
            "report_date": report_day,
            "event_date": dates[pos],
            "timing": timing,
            "quarter": _MONTH_TO_QUARTER[report_day.month],
            "year": report_day.year - 1 if report_day.month <= 3 else report_day.year,
            "month": report_day.month,
        }

        # --- 3. Avkastning per horisont, kumulativt fra basiskurs ---
        for h in HORIZONS:
            j = pos + h
            if j >= n or not np.isfinite(closes[j]):
                ev[f"ret_{h}"] = np.nan
                ev[f"exc_{h}"] = np.nan
                continue
            ret = (closes[j] / base - 1.0) * 100.0
            ev[f"ret_{h}"] = ret
            if np.isfinite(base_b) and np.isfinite(bench_vals[j]) and base_b > 0:
                ev[f"exc_{h}"] = ret - (bench_vals[j] / base_b - 1.0) * 100.0
            else:
                ev[f"exc_{h}"] = np.nan

        # Åpningsgapet — hvor mye av reaksjonen kom før første handel
        ev["gap"] = (
            (opens[pos] / base - 1.0) * 100.0 if np.isfinite(opens[pos]) else np.nan
        )

        # Konsensus (kan mangle — da står det tomt)
        est = row.get("EPS Estimate", np.nan)
        rep = row.get("Reported EPS", np.nan)
        sur = row.get("Surprise(%)", np.nan)
        ev["eps_est"] = float(est) if pd.notna(est) else np.nan
        ev["eps_rep"] = float(rep) if pd.notna(rep) else np.nan
        ev["surprise"] = float(sur) if pd.notna(sur) else np.nan

        events.append(ev)

    return events


def aggregate_events(events: list[dict], label: str = "") -> dict | None:
    """Slå sammen enkeltrapporter til median/snitt/treffrate per horisont."""
    if not events:
        return None

    row = {"_label": label}
    d0 = np.array([e["ret_0"] for e in events], dtype=float)
    row["N"] = int(np.isfinite(d0).sum())
    if row["N"] == 0:
        return None

    for h in HORIZONS:
        vals = np.array([e[f"ret_{h}"] for e in events], dtype=float)
        vals = vals[np.isfinite(vals)]
        exc = np.array([e[f"exc_{h}"] for e in events], dtype=float)
        exc = exc[np.isfinite(exc)]

        row[f"D{h} median %"] = float(np.median(vals)) if vals.size else np.nan
        row[f"D{h} snitt %"] = float(np.mean(vals)) if vals.size else np.nan
        row[f"D{h} meravk %"] = float(np.median(exc)) if exc.size else np.nan
        if vals.size:
            wins = int((vals > 0).sum())
            row[f"D{h} treff %"] = wins / vals.size * 100.0
            row[f"D{h} treff"] = f"{wins} av {vals.size}"
        else:
            row[f"D{h} treff %"] = np.nan
            row[f"D{h} treff"] = ""
        row[f"_n{h}"] = int(vals.size)
        row[f"_n_exc{h}"] = int(exc.size)

    gaps = np.array([e["gap"] for e in events], dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    row["Gap median %"] = float(np.median(gaps)) if gaps.size else np.nan

    sur = np.array([e["surprise"] for e in events], dtype=float)
    sur = sur[np.isfinite(sur)]
    row["Overrask. snitt %"] = float(np.mean(sur)) if sur.size else np.nan
    row["Konsensus N"] = int(sur.size)

    months = [e["month"] for e in events]
    row["Rapportmnd"] = int(pd.Series(months).mode().iloc[0]) if months else np.nan
    row["Siste rapport"] = max(e["report_date"] for e in events).date()

    timings = pd.Series([e["timing"] for e in events]).value_counts()
    row["Timing"] = timings.index[0] if len(timings) else ""

    return row


# ---------------------------------------------------------------------------
# Presentasjon
# ---------------------------------------------------------------------------

def _column_config(compact: bool) -> dict:
    cfg = {
        "N": st.column_config.NumberColumn(
            "N", format="%d", help="Antall historiske rapporter snittet bygger på"
        ),
        "Gap median %": st.column_config.NumberColumn(
            "Gap median %", format="%+.2f",
            help="Åpningsgap på reaksjonsdagen: hvor mye av reaksjonen kom før første handel",
        ),
        "Overrask. snitt %": st.column_config.NumberColumn(
            "Overrask. %", format="%+.1f",
            help="Snittlig EPS-overraskelse mot konsensus",
        ),
        "Rapportmnd": st.column_config.NumberColumn(
            "Mnd", format="%d", help="Måneden selskapet vanligvis rapporterer i"
        ),
    }
    for h in HORIZONS:
        cfg[f"D{h} median %"] = st.column_config.NumberColumn(
            f"D{h} median %", format="%+.2f"
        )
        cfg[f"D{h} snitt %"] = st.column_config.NumberColumn(
            f"D{h} snitt %", format="%+.2f"
        )
        cfg[f"D{h} meravk %"] = st.column_config.NumberColumn(
            f"D{h} meravk %", format="%+.2f",
            help="Median meravkastning mot referanseindeksen",
        )
        cfg[f"D{h} treff %"] = st.column_config.ProgressColumn(
            f"D{h} treff", format="%.0f%%", min_value=0, max_value=100
        )
    return cfg


def _visible_columns(df: pd.DataFrame, lead: list[str], compact: bool) -> list[str]:
    cols = list(lead)
    for h in HORIZONS:
        cols.append(f"D{h} median %")
        if not compact:
            cols.append(f"D{h} snitt %")
        cols.append(f"D{h} meravk %")
        cols.append(f"D{h} treff %")
        if not compact:
            cols.append(f"D{h} treff")
    if not compact:
        cols += ["Gap median %", "Overrask. snitt %", "Rapportmnd", "Timing", "Siste rapport"]
    return [c for c in cols if c in df.columns]


TIMING_MODES = {
    "Auto — bruk tidsstempel": "auto",
    "Anta alltid rapportdagen": "same_day",
    "Anta alltid dagen etter": "next_day",
}


def _run_analysis(
    tickers: list[str], bench_symbol: str, progress_label: str,
    timing_mode: str = "auto",
) -> tuple:
    """Hent priser + rapportdatoer og regn ut alle hendelser.

    Returnerer (events_per_ticker, diagnostikk-dict).
    """
    diag = {
        "n_tickers": len(tickers),
        "no_prices": [],
        "no_earnings": [],
        "no_events": [],
        "ok": 0,
        "events_total": 0,
        "events_with_excess": 0,
        "events_with_consensus": 0,
    }

    status = st.status(progress_label, expanded=True)
    with status:
        st.write(f"Henter kurshistorikk for {len(tickers)} aksjer…")
        prices, infos = get_stock_data(tickers, period="max")
        diag["no_prices"] = [t for t in tickers if t not in prices]

        st.write(f"Henter rapportdatoer og konsensus for {len(prices)} aksjer…")
        cal = fetch_earnings_calendar(tuple(sorted(prices.keys())))
        diag["no_earnings"] = [t for t in prices if t not in cal]

        st.write(f"Henter referanseindeks ({bench_symbol})…")
        bench = fetch_benchmark(bench_symbol)
        diag["bench_ok"] = bench is not None
        diag["bench_start"] = bench.index.min().date() if bench is not None else None

        st.write("Beregner kursreaksjoner…")
        per_ticker = {}
        for ticker, earn in cal.items():
            evs = compute_events(ticker, prices[ticker], earn, bench, timing_mode)
            if evs:
                per_ticker[ticker] = evs
                diag["events_total"] += len(evs)
                diag["events_with_excess"] += sum(
                    1 for e in evs if np.isfinite(e["exc_0"])
                )
                diag["events_with_consensus"] += sum(
                    1 for e in evs if np.isfinite(e["eps_est"])
                )
            else:
                diag["no_events"].append(ticker)
        diag["ok"] = len(per_ticker)
        status.update(label=f"Ferdig — {diag['ok']} aksjer med data", state="complete")

    return per_ticker, infos, diag


def _name_lookup(tickers: list[str], infos: dict) -> dict:
    """Selskapsnavn: Euronext-listen for Oslo, ellers shortName fra datakilden."""
    names = {}
    if any(t.endswith(".OL") for t in tickers):
        try:
            names.update(get_oslo_name_map())
        except Exception:
            pass
    for t in tickers:
        if not names.get(t):
            names[t] = (infos.get(t) or {}).get("shortName") or t
    return names


def _render_diagnostics(diag: dict, bench_label: str):
    with st.expander("🔍 Datadekning og hull", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Aksjer med komplett data", f"{diag['ok']} / {diag['n_tickers']}")
        c2.metric("Rapporthendelser", diag["events_total"])
        exc_pct = (
            diag["events_with_excess"] / diag["events_total"] * 100
            if diag["events_total"] else 0
        )
        c3.metric("Med meravkastning", f"{exc_pct:.0f}%")
        cons_pct = (
            diag["events_with_consensus"] / diag["events_total"] * 100
            if diag["events_total"] else 0
        )
        c4.metric("Med konsensus", f"{cons_pct:.0f}%")

        st.markdown(f"**Referanseindeks:** {bench_label}")
        if diag.get("bench_start"):
            st.caption(
                f"Indeksdata starter {diag['bench_start']}. Rapporter før denne "
                "datoen får tom meravkastning — verdien gjettes aldri."
            )
        if not diag.get("bench_ok"):
            st.error("Fant ikke indeksdata — meravkastning er tom for alle rader.")

        for label, key in [
            ("Uten kurshistorikk", "no_prices"),
            ("Uten rapportdatoer i datakilden", "no_earnings"),
            ("Rapportdatoer, men for kort kurshistorikk", "no_events"),
        ]:
            lst = diag.get(key) or []
            if lst:
                st.markdown(f"**{label} ({len(lst)}):**")
                st.caption(", ".join(sorted(lst)[:60]) + (" …" if len(lst) > 60 else ""))


# ---------------------------------------------------------------------------
# Modus A — markedsscreening
# ---------------------------------------------------------------------------

def _render_mode_a():
    st.markdown("### Modus A — screening på tvers av et marked")
    st.caption(
        "Velg marked og kvartal. Appen går gjennom alle aksjene og regner ut "
        "hvordan hver av dem historisk har reagert på akkurat det kvartalets rapport."
    )

    markets = [m for m in MARKET_OPTIONS if m in MARKET_BENCHMARKS]
    c1, c2, c3 = st.columns([2, 1, 2])
    market = c1.selectbox("Marked", markets, key="er_market")
    quarter = c2.selectbox("Kvartal", ["Q1", "Q2", "Q3", "Q4"], key="er_quarter")
    bench_opts = MARKET_BENCHMARKS[market]
    bench_label = c3.selectbox(
        "Referanseindeks", [b[1] for b in bench_opts], key="er_bench"
    )
    bench_symbol = dict((b[1], b[0]) for b in bench_opts)[bench_label]

    c4, c5, c6 = st.columns([1, 1, 2])
    min_obs = c4.number_input(
        "Min. antall observasjoner", 1, 30, 4, 1, key="er_minobs",
        help="Krev minst så mange historiske rapporter for det valgte kvartalet",
    )
    limit = c5.number_input(
        "Maks antall aksjer", 10, 400, 150, 10, key="er_limit",
        help="Begrens universet for raskere kjøring",
    )
    timing_label = c6.selectbox(
        "Rapporttidspunkt", list(TIMING_MODES), key="er_timing",
        help="Datakildens klokkeslett er delvis upålitelig for europeiske aksjer. "
             "Bytt til en fast antagelse for å se hvor følsomme tallene er.",
    )
    compact = st.checkbox("Kompakt tabell", value=False, key="er_compact")

    if st.button("▶ Kjør markedsscreening", key="er_run_a", use_container_width=True):
        if market == "Min Watchlist":
            tickers = list(st.session_state.get("watchlist") or [])
        else:
            tickers = get_tickers_for_market(market)
        tickers = tickers[: int(limit)]

        if not tickers:
            st.warning("Ingen aksjer i valgt marked.")
            return

        per_ticker, infos, diag = _run_analysis(
            tickers, bench_symbol, f"Analyserer {len(tickers)} aksjer for {quarter}…",
            TIMING_MODES[timing_label],
        )
        names = _name_lookup(tickers, infos)

        rows = []
        for ticker, evs in per_ticker.items():
            q_evs = [e for e in evs if e["quarter"] == quarter]
            agg = aggregate_events(q_evs)
            if agg is None or agg["N"] < min_obs:
                continue
            agg["Ticker"] = ticker
            agg["Navn"] = names.get(ticker, ticker)
            rows.append(agg)

        st.session_state["er_a_results"] = pd.DataFrame(rows)
        st.session_state["er_a_meta"] = {
            "quarter": quarter, "market": market, "bench": bench_label,
            "diag": diag, "min_obs": int(min_obs),
        }

    df = st.session_state.get("er_a_results")
    meta = st.session_state.get("er_a_meta") or {}
    if df is None:
        st.info("Velg marked og kvartal, og trykk **Kjør markedsscreening**.")
        return
    if df.empty:
        st.warning(
            f"Ingen aksjer hadde minst {meta.get('min_obs')} historiske "
            f"{meta.get('quarter')}-rapporter. Senk minimumskravet."
        )
        if meta.get("diag"):
            _render_diagnostics(meta["diag"], meta.get("bench", ""))
        return

    _render_diagnostics(meta["diag"], meta["bench"])

    sort_opts = [c for c in df.columns if c.startswith("D") or c == "N"]
    s1, s2 = st.columns([2, 1])
    sort_by = s1.selectbox(
        "Sorter etter", sort_opts,
        index=sort_opts.index("D1 median %") if "D1 median %" in sort_opts else 0,
        key="er_a_sort",
    )
    desc = s2.radio("Retning", ["Synkende", "Stigende"], horizontal=True, key="er_a_dir")

    view = df.sort_values(sort_by, ascending=(desc == "Stigende"), na_position="last")
    st.success(
        f"**{len(view)} aksjer** i {meta['market']} med minst {meta['min_obs']} "
        f"historiske {meta['quarter']}-rapporter. Klikk en kolonneoverskrift for å sortere."
    )
    cols = _visible_columns(view, ["Ticker", "Navn", "N"], compact)
    st.dataframe(
        view[cols], use_container_width=True, height=520,
        column_config=_column_config(compact), hide_index=True,
    )

    st.download_button(
        "⬇ Last ned som CSV",
        view[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"kursreaksjon_{meta['market'].replace(' ', '_')}_{meta['quarter']}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Modus B — enkeltaksjer per kvartal
# ---------------------------------------------------------------------------

def _render_mode_b():
    st.markdown("### Modus B — enkeltaksjer brutt ned per kvartal")
    st.caption(
        "Velg én eller flere aksjer. Hver aksje vises med Q1/Q2/Q3/Q4 hver for seg, "
        "slik at et eventuelt sesongmønster i rapportreaksjonen kommer fram."
    )

    watch = list(st.session_state.get("watchlist") or [])
    c1, c2 = st.columns([2, 2])
    picked = c1.multiselect("Fra watchlist", watch, key="er_b_watch")
    typed = c2.text_input(
        "Egne tickere (komma-separert)", placeholder="MOWI.OL, EQNR.OL, TOM.OL",
        key="er_b_typed",
    )

    c3, c4 = st.columns([2, 2])
    bench_all = sorted({b for opts in MARKET_BENCHMARKS.values() for b in opts})
    bench_label = c3.selectbox(
        "Referanseindeks", [b[1] for b in bench_all], key="er_b_bench",
        help="Velg indeksen som passer aksjenes hjemmemarked",
    )
    bench_symbol = dict((b[1], b[0]) for b in bench_all)[bench_label]
    timing_label_b = c4.selectbox(
        "Rapporttidspunkt", list(TIMING_MODES), key="er_b_timing",
        help="Datakildens klokkeslett er delvis upålitelig for europeiske aksjer.",
    )
    compact = st.checkbox("Kompakt tabell", value=False, key="er_b_compact")

    tickers = list(dict.fromkeys(
        picked + [t.strip().upper() for t in typed.replace("\n", ",").split(",") if t.strip()]
    ))

    if st.button("▶ Analyser valgte aksjer", key="er_run_b", use_container_width=True):
        if not tickers:
            st.warning("Velg minst én aksje.")
            return
        per_ticker, infos, diag = _run_analysis(
            tickers, bench_symbol, f"Analyserer {len(tickers)} aksjer…",
            TIMING_MODES[timing_label_b],
        )
        names = _name_lookup(tickers, infos)

        rows, raw = [], []
        for ticker, evs in per_ticker.items():
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                agg = aggregate_events([e for e in evs if e["quarter"] == q])
                if agg is None:
                    continue
                agg["Ticker"] = ticker
                agg["Navn"] = names.get(ticker, ticker)
                agg["Kvartal"] = q
                rows.append(agg)
            raw.extend(evs)

        st.session_state["er_b_results"] = pd.DataFrame(rows)
        st.session_state["er_b_raw"] = pd.DataFrame(raw)
        st.session_state["er_b_meta"] = {"diag": diag, "bench": bench_label}

    df = st.session_state.get("er_b_results")
    meta = st.session_state.get("er_b_meta") or {}
    if df is None:
        st.info("Velg aksjer og trykk **Analyser valgte aksjer**.")
        return
    if df.empty:
        st.warning("Fant ingen rapporthendelser for de valgte aksjene.")
        if meta.get("diag"):
            _render_diagnostics(meta["diag"], meta.get("bench", ""))
        return

    _render_diagnostics(meta["diag"], meta["bench"])

    cols = _visible_columns(df, ["Ticker", "Navn", "Kvartal", "N"], compact)
    view = df.sort_values(["Ticker", "Kvartal"])
    st.dataframe(
        view[cols], use_container_width=True,
        column_config=_column_config(compact), hide_index=True,
        height=min(520, 60 + 36 * len(view)),
    )

    # --- Sesongmønster visuelt ---
    st.markdown("#### Sesongmønster — median reaksjon per kvartal")
    horizon = st.radio(
        "Horisont", [f"D{h}" for h in HORIZONS], index=1, horizontal=True, key="er_b_h"
    )
    col = f"{horizon} median %"
    if col in view.columns:
        import plotly.graph_objects as go

        fig = go.Figure()
        for ticker in view["Ticker"].unique():
            sub = view[view["Ticker"] == ticker].set_index("Kvartal").reindex(
                ["Q1", "Q2", "Q3", "Q4"]
            )
            fig.add_trace(go.Bar(
                x=["Q1", "Q2", "Q3", "Q4"], y=sub[col], name=ticker,
                customdata=sub["N"],
                hovertemplate="%{x}: %{y:+.2f}% (N=%{customdata})<extra>%{fullData.name}</extra>",
            ))
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.update_layout(
            barmode="group", height=380,
            yaxis_title=f"Median kursendring {horizon} (%)", xaxis_title="Kvartal",
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Enkeltrapporter ---
    raw = st.session_state.get("er_b_raw")
    if raw is not None and not raw.empty:
        with st.expander("📋 Alle enkeltrapporter (rådata)", expanded=False):
            show = raw.copy()
            show["Rapportdato"] = pd.to_datetime(show["report_date"]).dt.date
            show["Reaksjonsdag"] = pd.to_datetime(show["event_date"]).dt.date
            ren = {
                "ticker": "Ticker", "quarter": "Kvartal", "year": "År",
                "timing": "Timing", "gap": "Gap %", "surprise": "Overrask. %",
                "eps_est": "Konsensus EPS", "eps_rep": "Rapportert EPS",
            }
            for h in HORIZONS:
                ren[f"ret_{h}"] = f"D{h} %"
                ren[f"exc_{h}"] = f"D{h} meravk %"
            order = (
                ["Ticker", "Kvartal", "År", "Rapportdato", "Reaksjonsdag", "Timing"]
                + [f"D{h} %" for h in HORIZONS]
                + [f"D{h} meravk %" for h in HORIZONS]
                + ["Gap %", "Konsensus EPS", "Rapportert EPS", "Overrask. %"]
            )
            show = show.rename(columns=ren)
            show = show[[c for c in order if c in show.columns]]
            st.dataframe(
                show.sort_values(["Ticker", "Rapportdato"], ascending=[True, False]),
                use_container_width=True, hide_index=True, height=400,
            )


# ---------------------------------------------------------------------------
# Hovedinngang — kalles fra app.py
# ---------------------------------------------------------------------------

def render_earnings_reaction_tab():
    st.markdown("## 📑 Kursreaksjon på kvartalsrapporter")
    st.caption(
        "Hvordan har aksjen historisk reagert på rapportdagen og dagene etter? "
        "Alle tall er beregnet fra appens ordinære kursdatakilde."
    )

    with st.expander("📖 Metode og kurskonvensjon — les før du tolker tallene", expanded=False):
        st.markdown("""
**Reaksjonsdagen, ikke rapportdagen.** Kommer rapporten etter børsslutt, skjer
reaksjonen først neste børsdag. Vi bruker derfor rapporttidspunktet omregnet til
børsens lokale tid til å finne den *første handelsdagen som kan reagere*:

| Tidspunkt | Merket | Reaksjonsdag |
|---|---|---|
| Før åpning | BMO | Rapportdagen |
| I løpet av dagen | Intradag | Rapportdagen |
| Etter stengetid | AMC | Neste børsdag |

**Basiskurs** = sluttkurs siste børsdag *før* reaksjonsdagen.

- **D0** = sluttkurs reaksjonsdag / basiskurs − 1
- **D1 / D3 / D5** = sluttkurs 1, 3 og 5 *handledager* senere / basiskurs − 1

D1, D3 og D5 er altså **kumulative fra basiskursen** — ikke dag-for-dag-endringer.
Helligdager hoppes over automatisk siden vi teller faktiske handledager.

**Hvorfor sluttkurs og ikke rapportdagens åpning som basis?** Kommer rapporten før
åpning, ligger hele førstereaksjonen i åpningsgapet. Bruker man åpningskursen som
basis, måler man alt *unntatt* selve nyheten. Vi tar med gapet, men viser det også
separat i kolonnen **Gap %** slik at du ser hvor mye som kom i gapet.

**Meravkastning** = aksjens avkastning minus referanseindeksens over nøyaktig samme
datointervall.

**Kurser er utbytte- og splittjustert**, så et utbytte som går ex på reaksjonsdagen
ser ikke ut som et kursfall.

**Manglende data står alltid tomt** — aldri fylt inn med 0 eller antatte verdier.

**Kvartalsmerking:** datakilden oppgir rapportdato, ikke regnskapsperiode. Kvartalet
utledes fra rapportmåneden etter nordisk standard: Q4 rapporteres jan–mar, Q1 apr–jun,
Q2 jul–sep, Q3 okt–des. Selskaper med avvikende regnskapsår merkes feil — kolonnen
**Mnd** viser hvilken måned selskapet faktisk rapporterer i, så du kan kontrollere.
        """)

    mode_a, mode_b = st.tabs(["🔎 Modus A — marked", "🎯 Modus B — enkeltaksjer"])
    with mode_a:
        _render_mode_a()
    with mode_b:
        _render_mode_b()
