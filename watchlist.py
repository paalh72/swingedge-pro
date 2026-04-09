import streamlit as st
import pandas as pd

try:
    from streamlit_gsheets import GSheetsConnection
    HAS_GSHEETS = True
except ImportError:
    HAS_GSHEETS = False

DEFAULT_WATCHLIST = ["EQNR.OL", "NVDA"]


def load_watchlist() -> list:
    """Load watchlist from Google Sheets, falling back to defaults."""
    if not HAS_GSHEETS:
        return list(DEFAULT_WATCHLIST)
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        sheet_url = st.secrets["connections"]["gsheets"].get("spreadsheet", "")
        if sheet_url:
            df = conn.read(spreadsheet=sheet_url, ttl=0)
            if not df.empty and 'Ticker' in df.columns:
                return df['Ticker'].dropna().unique().tolist()
    except Exception:
        pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(tickers: list) -> None:
    """Save watchlist to Google Sheets."""
    if not HAS_GSHEETS:
        return
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        sheet_url = st.secrets["connections"]["gsheets"].get("spreadsheet", "")
        if sheet_url:
            df_new = pd.DataFrame(tickers, columns=['Ticker'])
            conn.update(spreadsheet=sheet_url, data=df_new)
            st.cache_data.clear()
    except Exception as e:
        st.error(f"Lagring feilet: {e}")
