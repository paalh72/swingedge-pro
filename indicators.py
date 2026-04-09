import numpy as np
import pandas as pd


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators on a price DataFrame."""
    df = df.copy()
    if 'Close' not in df.columns or len(df) < 50:
        return df

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # --- ATR & ADX ---
    df = _calculate_adx(df, period=14)

    # --- RSI (Wilder's smoothing) ---
    df['RSI'] = _calculate_rsi(close, period=14)

    # --- Stochastic RSI ---
    df['StochRSI_K'], df['StochRSI_D'] = _calculate_stoch_rsi(df['RSI'], period=14)

    # --- Moving Averages ---
    df['EMA9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA21'] = close.ewm(span=21, adjust=False).mean()
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA200'] = close.ewm(span=200, adjust=False).mean()

    # --- Bollinger Bands ---
    df['SMA20'] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Upper'] = df['SMA20'] + (std20 * 2)
    df['BB_Lower'] = df['SMA20'] - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA20']

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    # --- RVOL (Relative Volume) ---
    vol_sma20 = volume.rolling(20).mean()
    df['RVOL'] = volume / vol_sma20

    # --- Ichimoku Cloud ---
    df = _calculate_ichimoku(df)

    # --- VWAP (rolling intraday proxy — daily reset not possible on daily data) ---
    df['VWAP'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()

    # --- DeMark Sequential ---
    df = _calculate_demark(df)

    # --- Volume Profile POC ---
    df['VPVR_POC'] = _calculate_rolling_poc(df, lookback=50)

    # --- Fibonacci levels from recent swing ---
    df['Fib_38'], df['Fib_50'], df['Fib_62'] = _calculate_fib_levels(df, lookback=50)

    return df


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (alpha=1/period)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calculate_stoch_rsi(rsi: pd.Series, period: int = 14) -> tuple:
    """Stochastic RSI with %K and %D."""
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    k = stoch_rsi.rolling(3).mean() * 100
    d = k.rolling(3).mean()
    return k, d


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX and ATR calculation."""
    if len(df) < period + 1:
        df['ADX'] = 0
        df['ATR'] = 0
        return df

    high, low, close = df['High'], df['Low'], df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = pd.Series(tr, index=df.index).ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / tr_smooth

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.ewm(span=period, adjust=False).mean()
    df['+DI'] = plus_di
    df['-DI'] = minus_di

    return df


def _calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Ichimoku Cloud: Tenkan, Kijun, Senkou A/B, Chikou."""
    high, low, close = df['High'], df['Low'], df['Close']

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)

    df['Ichimoku_Tenkan'] = tenkan
    df['Ichimoku_Kijun'] = kijun
    df['Ichimoku_SenkouA'] = senkou_a
    df['Ichimoku_SenkouB'] = senkou_b
    df['Ichimoku_Chikou'] = chikou

    # Is price above the cloud?
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    df['Ichimoku_Above'] = (close > cloud_top).astype(int)
    df['Ichimoku_Below'] = (close < cloud_bottom).astype(int)

    return df


def _calculate_demark(df: pd.DataFrame) -> pd.DataFrame:
    """DeMark Sequential: Setup (1-9) and Countdown (1-13)."""
    closes = df['Close'].values
    lows = df['Low'].values
    highs = df['High'].values
    n = len(closes)

    buy_setup = np.zeros(n, dtype=int)
    sell_setup = np.zeros(n, dtype=int)

    for i in range(4, n):
        # Buy setup: close < close 4 bars ago
        if closes[i] < closes[i - 4]:
            buy_setup[i] = min(buy_setup[i - 1] + 1, 9)
        else:
            buy_setup[i] = 0

        if closes[i] > closes[i - 4]:
            sell_setup[i] = min(sell_setup[i - 1] + 1, 9)
        else:
            sell_setup[i] = 0

    # Countdown
    buy_countdown = np.zeros(n, dtype=int)
    sell_countdown = np.zeros(n, dtype=int)
    bc_active, bc_count = False, 0
    sc_active, sc_count = False, 0

    for i in range(4, n):
        # Activate buy countdown on perfected setup 9
        if buy_setup[i] == 9:
            # Perfection check: low of bar 8 or 9 < low of bar 6 or 7
            if i >= 2:
                bc_active = True
                bc_count = 0
        if sell_setup[i] == 9:
            bc_active = False
            bc_count = 0

        if bc_active and i >= 2:
            if closes[i] <= lows[i - 2]:
                bc_count += 1
                buy_countdown[i] = bc_count
                if bc_count >= 13:
                    bc_active = False
                    bc_count = 0

        # Sell countdown
        if sell_setup[i] == 9:
            sc_active = True
            sc_count = 0
        if buy_setup[i] == 9:
            sc_active = False
            sc_count = 0

        if sc_active and i >= 2:
            if closes[i] >= highs[i - 2]:
                sc_count += 1
                sell_countdown[i] = sc_count
                if sc_count >= 13:
                    sc_active = False
                    sc_count = 0

    df['TD_Buy_Setup'] = buy_setup
    df['TD_Sell_Setup'] = sell_setup
    df['TD_Buy_Countdown'] = buy_countdown
    df['TD_Sell_Countdown'] = sell_countdown

    return df


def _calculate_rolling_poc(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Rolling Point of Control (price level with most volume)."""
    poc_series = pd.Series(np.nan, index=df.index)
    closes = df['Close'].values
    volumes = df['Volume'].values
    lows = df['Low'].values
    highs = df['High'].values

    for i in range(lookback, len(df)):
        window_low = lows[i - lookback:i].min()
        window_high = highs[i - lookback:i].max()
        if window_high == window_low:
            poc_series.iloc[i] = window_low
            continue
        bins = np.linspace(window_low, window_high, 30)
        idx = np.clip(np.digitize(closes[i - lookback:i], bins) - 1, 0, 28)
        vol_profile = np.bincount(idx, weights=volumes[i - lookback:i], minlength=29)
        poc_idx = np.argmax(vol_profile)
        poc_series.iloc[i] = bins[poc_idx]

    return poc_series


def _calculate_fib_levels(df: pd.DataFrame, lookback: int = 50) -> tuple:
    """Fibonacci retracement levels from recent swing high/low."""
    fib38 = pd.Series(np.nan, index=df.index)
    fib50 = pd.Series(np.nan, index=df.index)
    fib62 = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        swing_high = window['High'].max()
        swing_low = window['Low'].min()
        diff = swing_high - swing_low
        fib38.iloc[i] = swing_high - 0.382 * diff
        fib50.iloc[i] = swing_high - 0.500 * diff
        fib62.iloc[i] = swing_high - 0.618 * diff

    return fib38, fib50, fib62
