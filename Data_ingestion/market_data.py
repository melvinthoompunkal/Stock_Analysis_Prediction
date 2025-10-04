# ==============================================================================
# IMPORTS & SETUP
# ==============================================================================
import pandas as pd
import time
from polygon import RESTClient
import os
from dotenv import load_dotenv
# Simple market classification function
def classify_market(macd_result, rsi_val, atr_val, adx_val):
    """Simple market classification based on technical indicators."""
    if rsi_val is None:
        return "Neutral"
    
    if rsi_val > 70:
        return "Overbought"
    elif rsi_val < 30:
        return "Oversold"
    elif rsi_val > 50:
        return "Bullish"
    else:
        return "Bearish"

# Load API key
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
client = RESTClient(api_key=POLYGON_API_KEY)


# ==============================================================================
# DATA FETCHING HELPER (Polygon.io)
# ==============================================================================
def _get_polygon_data(ticker, start_date, end_date, timespan="day", multiplier=1):
    """Fetch OHLCV data from Polygon.io and normalize into a DataFrame."""
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
        )

        # Normalize Polygon return types
        if hasattr(aggs, "results"):
            results = aggs.results or []
        elif isinstance(aggs, (list, tuple)):
            results = list(aggs)
        elif aggs is None:
            results = []
        else:
            results = [aggs]

        if not results:
            print(f"⚠️ No data returned for {ticker}.")
            return pd.DataFrame()

        rows = []
        for item in results:
            if isinstance(item, dict):
                t, o, h, l, c, v = item.get("t"), item.get("o"), item.get("h"), item.get("l"), item.get("c"), item.get("v")
            else:
                t = getattr(item, "t", None) or getattr(item, "timestamp", None)
                o = getattr(item, "o", None) or getattr(item, "open", None)
                h = getattr(item, "h", None) or getattr(item, "high", None)
                l = getattr(item, "l", None) or getattr(item, "low", None)
                c = getattr(item, "c", None) or getattr(item, "close", None)
                v = getattr(item, "v", None) or getattr(item, "volume", None)

            if t is None:
                continue

            try:
                timestamp = pd.to_datetime(t, unit="ms") if isinstance(t, (int, float)) else pd.to_datetime(t)
            except Exception:
                continue

            rows.append({"timestamp": timestamp, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp").sort_index()
        return df
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


# ==============================================================================
# TECHNICAL INDICATORS
# ==============================================================================
def RSI(data_df=None, window=14):
    """Relative Strength Index (momentum)."""
    if data_df is None or data_df.empty:
        return pd.Series()

    delta = data_df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = (100 - (100 / (1 + rs))).dropna()
    rsi.index = rsi.index.date

    return rsi


def MACD(data_df=None, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence (trend indicator)."""
    if data_df is None or data_df.empty:
        return {"MACD": pd.Series(), "Signal": pd.Series(), "Histogram": pd.Series()}

    close = data_df["Close"]

    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram
    macd_hist = macd_line - signal_line

    # Align indexes to dates
    macd_line.index = macd_line.index.date
    signal_line.index = signal_line.index.date
    macd_hist.index = macd_hist.index.date

    return {"MACD": macd_line, "Signal": signal_line, "Histogram": macd_hist}

def ATR(data_df, window=14):
    """Average True Range (volatility)."""
    if data_df.empty: return pd.Series()
    high, low, close = data_df['High'], data_df['Low'], data_df['Close']
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    atr.index = atr.index.date
    return atr.dropna()


def ADX(data_df, window=14):
    """Average Directional Index (trend strength)."""
    if data_df.empty: return pd.Series()
    high, low, close = data_df['High'], data_df['Low'], data_df['Close']

    plus_dm = high.diff()
    minus_dm = low.shift() - low
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = (high - low).combine((high - close.shift()).abs(), max).combine((low - close.shift()).abs(), max)
    atr = tr.rolling(window=window).mean()

    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    adx.index = adx.index.date
    return adx.dropna()


def ROC(data_df, period=12):
    """Rate of Change (momentum)."""
    if data_df.empty: return pd.Series()
    close = data_df['Close']
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    roc.index = roc.index.date
    return roc.dropna()


def CCI(data_df, window=20):
    """Commodity Channel Index (overbought/oversold)."""
    if data_df.empty: return pd.Series()
    tp = (data_df['High'] + data_df['Low'] + data_df['Close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: (x - x.mean()).abs().mean())
    cci = (tp - sma) / (0.015 * mad)
    cci.index = cci.index.date
    return cci.dropna()


def CMF(data_df, window=20):
    """Chaikin Money Flow (volume + price)."""
    if data_df.empty: return pd.Series()
    mfm = ((data_df['Close'] - data_df['Low']) - (data_df['High'] - data_df['Close'])) / (data_df['High'] - data_df['Low'])
    mfv = mfm * data_df['Volume']
    cmf = mfv.rolling(window=window).sum() / data_df['Volume'].rolling(window=window).sum()
    cmf.index = cmf.index.date
    return cmf.dropna()
import plotly.graph_objects as go

def plot_candlestick_with_indicators(data, rsi=None, macd=None, atr=None):
    """
    Plot candlestick chart with optional RSI, MACD, ATR overlays using Plotly.
    
    Args:
        data: DataFrame with Open, High, Low, Close (datetime index)
        rsi: pandas Series (optional)
        macd: dict from MACD() function (optional)
        atr: pandas Series (optional)
    """
    fig = go.Figure()

    # --- Candlestick ---
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    # --- RSI Overlay ---
    if rsi is not None and not rsi.empty:
        fig.add_trace(go.Scatter(
            x=rsi.index,
            y=rsi,
            line=dict(color="purple"),
            name="RSI"
        ))

    # --- MACD Overlay ---
    if macd is not None and not macd["MACD"].empty:
        fig.add_trace(go.Scatter(
            x=macd["MACD"].index,
            y=macd["MACD"],
            line=dict(color="blue"),
            name="MACD"
        ))
        fig.add_trace(go.Scatter(
            x=macd["Signal"].index,
            y=macd["Signal"],
            line=dict(color="orange"),
            name="Signal"
        ))

    # --- ATR Overlay ---
    if atr is not None and not atr.empty:
        fig.add_trace(go.Scatter(
            x=atr.index,
            y=atr,
            line=dict(color="red"),
            name="ATR"
        ))

    fig.update_layout(
        title="Candlestick with Indicators",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700
    )
    fig.show()



# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("\nWhat stock are you looking to research about?")
    stockTicker = input().upper()

    endDate = pd.Timestamp.now().strftime('%Y-%m-%d')
    startDate = (pd.Timestamp.now() - pd.DateOffset(months=2)).strftime('%Y-%m-%d')

    print(f"\n--- Analyzing {stockTicker} from {startDate} to {endDate} ---")

    main_historical_data = _get_polygon_data(stockTicker, startDate, endDate)
    print(len(main_historical_data), main_historical_data.head(1))

    if not main_historical_data.empty:
        print("✅ Data fetched successfully. Running analysis...")

        # Indicators
        atr = ATR(main_historical_data)
        adx = ADX(main_historical_data)
        roc = ROC(main_historical_data)
        cci = CCI(main_historical_data)
        cmf = CMF(main_historical_data)
        rsi_series = RSI(main_historical_data)
        macd_result = MACD(main_historical_data)

        # --- CLASSIFICATION STEP ---
        try:
            atr_val = atr.iloc[-1] if not atr.empty else None
            adx_val = adx.iloc[-1] if not adx.empty else None
            rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else None

            category = classify_market(macd_result, rsi_val, atr_val, adx_val)
            print(f"📊 Market Classification: {category}")
        except Exception as e:
            print(f"❌ Error when classifying market: {e}")

        # --- Print results ---
        print("RSI (last 5):", rsi_series.tail())
        print("MACD (last 5):", macd_result["MACD"].tail())
        print("Signal (last 5):", macd_result["Signal"].tail())
        print("ATR (last 5):", atr.tail())
        print("ADX (last 5):", adx.tail())
        print("ROC (last 5):", roc.tail())
        print("CCI (last 5):", cci.tail())
        print("CMF (last 5):", cmf.tail())

    else:
        print(f"❌ Could not fetch historical data for {stockTicker}. Analysis aborted.")
        
    plot_candlestick_with_indicators(
    main_historical_data,
    rsi=rsi_series,
    macd=macd_result,
    atr=atr
)
