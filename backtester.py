import pandas as pd
import numpy as np
from indicators import calculate_all_indicators
from scoring import calculate_confluence_score


def run_backtest(
    data: dict,
    min_confluence: int = 4,
    risk_pct: float = 1.5,
    rr_target: float = 3.0,
    regime: str = "BULL",
    initial_capital: float = 100000,
) -> dict:
    """Backtest the confluence-based swing strategy across all tickers.

    For each ticker, walk through history and simulate trades:
    - Entry: when confluence score >= min_confluence
    - Stop loss: 2 x ATR below entry
    - Take profit: rr_target x risk above entry
    - Time stop: exit after 20 trading days if neither SL nor TP hit
    - No new entry while a trade is open

    Returns a dict with summary stats and trade log.
    """
    all_trades = []
    portfolio = initial_capital
    equity_history = []

    for ticker, df in data.items():
        try:
            df = calculate_all_indicators(df)
        except Exception:
            continue

        if len(df) < 250:
            continue

        # Walk forward from day 200 (need history for indicators)
        in_trade = False
        entry_price = 0
        sl_price = 0
        tp_price = 0
        entry_date = None
        entry_idx = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            date = df.index[i]

            if in_trade:
                # Check stop loss
                if row['Low'] <= sl_price:
                    pnl_r = -1.0
                    all_trades.append({
                        'ticker': ticker,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(sl_price, 2),
                        'pnl_r': pnl_r,
                        'result': 'SL',
                        'days': i - entry_idx,
                    })
                    in_trade = False
                    continue

                # Check take profit
                if row['High'] >= tp_price:
                    pnl_r = rr_target
                    all_trades.append({
                        'ticker': ticker,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(tp_price, 2),
                        'pnl_r': pnl_r,
                        'result': 'TP',
                        'days': i - entry_idx,
                    })
                    in_trade = False
                    continue

                # Time stop: 20 days
                if i - entry_idx >= 20:
                    exit_price = row['Close']
                    risk = entry_price - sl_price
                    pnl_r = (exit_price - entry_price) / risk if risk > 0 else 0
                    result = 'Time-W' if pnl_r > 0 else 'Time-L'
                    all_trades.append({
                        'ticker': ticker,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'pnl_r': round(pnl_r, 2),
                        'result': result,
                        'days': 20,
                    })
                    in_trade = False
                    continue
            else:
                # Check entry signal
                # Build a temporary slice for confluence calculation
                window = df.iloc[max(0, i - 200):i + 1]
                if len(window) < 200:
                    continue

                conf, _ = calculate_confluence_score(window)
                if conf >= min_confluence:
                    atr = row.get('ATR', 0)
                    if atr <= 0:
                        continue
                    entry_price = row['Close']
                    sl_price = entry_price - 2 * atr
                    tp_price = entry_price + 2 * atr * rr_target
                    entry_date = date
                    entry_idx = i
                    in_trade = True

    # Compute stats
    if not all_trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_rr': 0,
            'max_drawdown': 0,
            'equity_curve': None,
            'trades_df': None,
        }

    trades_df = pd.DataFrame(all_trades)
    total = len(trades_df)
    wins = trades_df[trades_df['pnl_r'] > 0]
    losses = trades_df[trades_df['pnl_r'] <= 0]

    win_rate = (len(wins) / total) * 100 if total > 0 else 0
    gross_profit = wins['pnl_r'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_r'].sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss
    avg_rr = trades_df['pnl_r'].mean()

    # Equity curve
    risk_amount = initial_capital * (risk_pct / 100)
    equity = initial_capital
    eq_list = [{'date': trades_df.iloc[0]['entry_date'], 'equity': equity}]

    peak = equity
    max_dd = 0
    for _, trade in trades_df.iterrows():
        equity += trade['pnl_r'] * risk_amount
        eq_list.append({'date': trade['exit_date'], 'equity': equity})
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    equity_curve = pd.DataFrame(eq_list)

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'profit_factor': round(profit_factor, 2),
        'avg_rr': round(avg_rr, 2),
        'max_drawdown': round(max_dd, 1),
        'equity_curve': equity_curve,
        'trades_df': trades_df,
    }
