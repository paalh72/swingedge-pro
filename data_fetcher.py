import streamlit as st
import yfinance as yf
import pandas as pd


@st.cache_data(ttl=3600)
def get_stock_data(tickers: list, period: str = "5y") -> tuple:
    """Fetch historical data for a list of tickers.

    Uses bulk download for large lists and individual fetches for small ones.
    Returns (data_dict, info_dict).
    """
    data, infos = {}, {}
    if not tickers:
        return data, infos

    use_bulk = len(tickers) > 60

    if not use_bulk:
        prog = st.progress(0)
        for i, ticker in enumerate(tickers):
            prog.progress((i + 1) / len(tickers))
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, timeout=10)
                if hasattr(hist.index, 'tz_localize'):
                    hist.index = hist.index.tz_localize(None)
                if len(hist) > 50:
                    data[ticker] = hist
                    try:
                        infos[ticker] = stock.info
                    except Exception:
                        infos[ticker] = {'shortName': ticker}
            except Exception:
                continue
        prog.empty()
    else:
        st.info(f"Laster {len(tickers)} aksjer...")
        try:
            df_all = yf.download(
                tickers, period=period, group_by='ticker', threads=True, progress=True,
                timeout=10,
            )
            if df_all.empty:
                return data, infos
            for ticker in tickers:
                try:
                    if isinstance(df_all.columns, pd.MultiIndex):
                        if ticker in df_all.columns.get_level_values(0):
                            df_single = df_all[ticker].dropna()
                        else:
                            continue
                    else:
                        df_single = df_all.dropna()
                    if hasattr(df_single.index, 'tz_localize'):
                        df_single.index = df_single.index.tz_localize(None)
                    if len(df_single) > 50:
                        data[ticker] = df_single
                        infos[ticker] = {'shortName': ticker}
                except Exception:
                    continue
        except Exception as e:
            st.error(str(e))

    return data, infos
