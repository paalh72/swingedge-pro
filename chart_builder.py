import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta


def build_chart(
    df: pd.DataFrame,
    ticker: str,
    chart_type: str,
    sl_price: float,
    tp_price: float,
    confluence: int,
) -> go.Figure:
    """Build a comprehensive multi-panel chart.

    Panels:
    1. Price + Ichimoku cloud + Bollinger + EMAs + VPVR + DeMark + Fib levels
    2. Volume (color-coded by RVOL)
    3. MACD
    4. RSI + Stochastic RSI
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.50, 0.15, 0.20, 0.15],
        subplot_titles=(
            f"{ticker} (Confluence: {confluence}/10)",
            "Volum",
            "MACD",
            "RSI / StochRSI",
        ),
    )

    # --- VPVR background shapes ---
    _add_vpvr_shapes(fig, df)

    # --- Ichimoku Cloud ---
    _add_ichimoku(fig, df)

    # --- Bollinger Bands ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(width=0), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['BB_Lower'],
            fill='tonexty', fillcolor='rgba(173, 216, 230, 0.15)',
            line=dict(width=0), name='Bollinger',
        ),
        row=1, col=1,
    )

    # --- Price ---
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='OHLC',
            ),
            row=1, col=1,
        )
        fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Kurs', line=dict(color='black')),
            row=1, col=1,
        )

    # --- EMAs ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='#ff9800', width=1, dash='dot')),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='#ff5722', width=1, dash='dot')),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='orange', width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA200'], name='EMA200', line=dict(color='blue', dash='dot', width=1.5)),
        row=1, col=1,
    )

    # --- Fibonacci levels ---
    if 'Fib_38' in df.columns:
        for col, label, color in [
            ('Fib_38', '38.2%', '#9c27b0'),
            ('Fib_50', '50%', '#673ab7'),
            ('Fib_62', '61.8%', '#3f51b5'),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col], name=f'Fib {label}',
                    line=dict(color=color, width=0.8, dash='dash'),
                    visible='legendonly',
                ),
                row=1, col=1,
            )

    # --- Stop Loss & Take Profit lines ---
    fig.add_hline(
        y=sl_price, line_dash="dash", line_color="red",
        annotation_text=f"Stop Loss ({sl_price:.2f})", row=1, col=1,
    )
    fig.add_hline(
        y=tp_price, line_dash="dash", line_color="green",
        annotation_text=f"Take Profit ({tp_price:.2f})", row=1, col=1,
    )

    # --- DeMark annotations ---
    _add_demark(fig, df)

    # --- Volume panel (RVOL color-coded) ---
    vol_colors = []
    for _, r in df.iterrows():
        if r.get('RVOL', 0) > 1.5:
            vol_colors.append('purple')
        elif r['Close'] >= r['Open']:
            vol_colors.append('green')
        else:
            vol_colors.append('red')

    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='Volum'),
        row=2, col=1,
    )

    # --- MACD panel ---
    hist_colors = ['green' if v >= 0 else 'red' for v in df['Hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Hist'], marker_color=hist_colors, name='Histogram'),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Signal'], line=dict(color='orange'), name='Signal'),
        row=3, col=1,
    )

    # --- RSI + StochRSI panel ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['StochRSI_K'],
            line=dict(color='#00bcd4', width=1, dash='dot'), name='StochRSI %K',
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['StochRSI_D'],
            line=dict(color='#ff9800', width=1, dash='dot'), name='StochRSI %D',
        ),
        row=4, col=1,
    )
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)

    # --- Layout ---
    fig.update_layout(
        height=1100,
        margin=dict(t=30, b=20),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showspikes=True, spikethickness=1, spikedash='solid',
        spikecolor="gray", spikemode='across', spikesnap='cursor',
    )

    return fig


def _add_vpvr_shapes(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add Volume Profile (VPVR) as background shapes."""
    num_bins = 50
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_range = price_max - price_min
    if price_range == 0:
        return

    bins = np.linspace(price_min, price_max, num_bins + 1)
    indices = np.clip(np.digitize(df['Close'].values, bins) - 1, 0, num_bins - 1)
    vol_profile = np.bincount(indices, weights=df['Volume'].values, minlength=num_bins)

    max_vol = vol_profile.max()
    if max_vol == 0:
        return

    start_time = df.index[0]
    total_ms = (df.index[-1] - start_time).total_seconds() * 1000

    for i in range(num_bins):
        vol = vol_profile[i]
        if vol <= 0:
            continue
        bar_width_ms = (vol / max_vol) * (total_ms * 0.25)
        color = "rgba(255, 215, 0, 0.3)" if vol == max_vol else "rgba(128, 128, 128, 0.12)"
        fig.add_shape(
            type="rect",
            x0=start_time, x1=start_time + timedelta(milliseconds=bar_width_ms),
            y0=bins[i], y1=bins[i + 1],
            fillcolor=color, line=dict(width=0), layer="below",
            row=1, col=1,
        )


def _add_ichimoku(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add Ichimoku cloud shading."""
    if 'Ichimoku_SenkouA' not in df.columns:
        return

    sa = df['Ichimoku_SenkouA']
    sb = df['Ichimoku_SenkouB']

    fig.add_trace(
        go.Scatter(
            x=df.index, y=sa, line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=sb,
            fill='tonexty', fillcolor='rgba(144, 238, 144, 0.1)',
            line=dict(width=0), name='Ichimoku Cloud',
            hoverinfo='skip',
        ),
        row=1, col=1,
    )


def _add_demark(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add DeMark Sequential numbers to chart."""
    if 'TD_Buy_Setup' not in df.columns:
        return

    # Buy setup
    mask_b = df['TD_Buy_Setup'] > 0
    if mask_b.any():
        fig.add_trace(
            go.Scatter(
                x=df.index[mask_b], y=df.loc[mask_b, 'Low'] * 0.99,
                mode='text', text=df.loc[mask_b, 'TD_Buy_Setup'].astype(int).astype(str),
                textfont=dict(color='green', size=8), name='DM Buy Setup',
            ),
            row=1, col=1,
        )

    # Sell setup
    mask_s = df['TD_Sell_Setup'] > 0
    if mask_s.any():
        fig.add_trace(
            go.Scatter(
                x=df.index[mask_s], y=df.loc[mask_s, 'High'] * 1.01,
                mode='text', text=df.loc[mask_s, 'TD_Sell_Setup'].astype(int).astype(str),
                textfont=dict(color='red', size=8), name='DM Sell Setup',
            ),
            row=1, col=1,
        )

    # Buy countdown 13 highlight
    mask_bc = df['TD_Buy_Countdown'] > 0
    if mask_bc.any():
        texts = df.loc[mask_bc, 'TD_Buy_Countdown'].apply(
            lambda x: "13" if x == 13 else str(int(x))
        )
        fig.add_trace(
            go.Scatter(
                x=df.index[mask_bc], y=df.loc[mask_bc, 'Low'] * 0.97,
                mode='text', text=texts,
                textfont=dict(color='darkgreen', size=11), name='DM Buy Count',
            ),
            row=1, col=1,
        )

    # Sell countdown
    mask_sc = df['TD_Sell_Countdown'] > 0
    if mask_sc.any():
        texts = df.loc[mask_sc, 'TD_Sell_Countdown'].apply(
            lambda x: "13" if x == 13 else str(int(x))
        )
        fig.add_trace(
            go.Scatter(
                x=df.index[mask_sc], y=df.loc[mask_sc, 'High'] * 1.03,
                mode='text', text=texts,
                textfont=dict(color='darkred', size=11), name='DM Sell Count',
            ),
            row=1, col=1,
        )
