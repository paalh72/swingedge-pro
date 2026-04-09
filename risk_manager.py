import pandas as pd
import numpy as np


def calculate_position_size(
    portfolio_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
) -> tuple:
    """Calculate position size based on fixed-risk model.

    Args:
        portfolio_value: Total portfolio value.
        risk_pct: Max risk per trade as percentage (e.g. 1.5).
        entry_price: Planned entry price.
        stop_loss_price: Stop loss price level.

    Returns:
        (number_of_shares, risk_amount_in_currency)
    """
    if entry_price <= 0 or stop_loss_price <= 0 or entry_price <= stop_loss_price:
        return 0, 0.0

    risk_per_share = entry_price - stop_loss_price
    risk_amount = portfolio_value * (risk_pct / 100)
    shares = int(risk_amount / risk_per_share)
    return max(shares, 0), round(risk_amount, 2)


def calculate_trailing_stop(
    df: pd.DataFrame,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Calculate ATR-based trailing stop.

    The trailing stop moves up with price but never moves down.
    Stop = Highest Close since entry - multiplier * ATR.

    Returns DataFrame with Trailing_Stop column.
    """
    if 'ATR' not in df.columns or len(df) < 2:
        return None

    df = df.copy()
    atr = df['ATR'].values
    close = df['Close'].values
    trailing = np.zeros(len(df))
    trailing[0] = close[0] - multiplier * atr[0]

    for i in range(1, len(df)):
        new_stop = close[i] - multiplier * atr[i]
        if close[i] > close[i - 1]:
            trailing[i] = max(trailing[i - 1], new_stop)
        else:
            trailing[i] = trailing[i - 1]

    df['Trailing_Stop'] = trailing
    return df


def calculate_rr_targets(entry: float, stop_loss: float, ratios: list = None) -> dict:
    """Calculate take-profit targets at multiple R:R ratios."""
    if ratios is None:
        ratios = [1.5, 2.0, 3.0]

    risk = entry - stop_loss
    if risk <= 0:
        return {}

    return {f"{r}R": round(entry + risk * r, 2) for r in ratios}
