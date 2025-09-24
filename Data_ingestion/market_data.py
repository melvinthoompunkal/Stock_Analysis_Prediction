import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import time

# --- Third-party libraries ---
from polygon import RESTClient
from googleapiclient.discovery import build
from google.oauth2 import service_account
from dotenv import load_dotenv

# ==============================================================================
# SETUP: API Keys and Service Initialization
# ==============================================================================

# Load environment variables from .env file
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# --- Initialize Polygon.io Client ---
try:
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not found in .env file.")
    client = RESTClient(api_key=POLYGON_API_KEY)
    print("✅ Polygon.io client initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Polygon client: {e}")
    exit()

#=========================================================
# DATA FETCHING HELPER (Polygon.io)
# ==============================================================================

def _get_polygon_data(ticker, start_date, end_date, timespan="day", multiplier=1):
    """Fetch OHLCV data from Polygon.io and normalize into a DataFrame.

    Handles both SDK return shapes (object with .results or list of dict/model),
    and is resilient to minor API changes. Returns a DataFrame indexed by timestamp
    with columns: Open, High, Low, Close, Volume. Returns empty DataFrame on failure.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000
            )

            # Extract results regardless of SDK version/shape
            results = getattr(aggs, "results", aggs)

            # If the SDK returns a generator, listify it
            if results is None:
                results = []
            if not isinstance(results, (list, tuple)):
                try:
                    results = list(results)
                except Exception:
                    # Unknown shape
                    results = []

            if len(results) == 0:
                print(f"Warning: No data returned for {ticker} from {start_date} to {end_date}.")
                return pd.DataFrame()

            # Convert model objects to dicts if needed
            normalized = []
            for item in results:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    # Handle Agg objects from Polygon SDK
                    try:
                        # Extract attributes directly from the Agg object
                        item_dict = {
                            "t": getattr(item, "timestamp", None),
                            "o": getattr(item, "open", None),
                            "h": getattr(item, "high", None),
                            "l": getattr(item, "low", None),
                            "c": getattr(item, "close", None),
                            "v": getattr(item, "volume", None)
                        }
                        # Only add if we have the essential data
                        if item_dict["t"] is not None and item_dict["c"] is not None:
                            normalized.append(item_dict)
                    except Exception as e:
                        print(f"Warning: Could not parse item: {e}")
                        continue

            if len(normalized) == 0:
                print(f"Warning: Unable to parse data for {ticker}.")
                return pd.DataFrame()

            df = pd.DataFrame(normalized)

            # Determine timestamp column name
            ts_col = None
            for key in ("t", "timestamp", "T", "time"):
                if key in df.columns:
                    ts_col = key
                    break
            if ts_col is None:
                print("Warning: No timestamp column found in results.")
                return pd.DataFrame()

            # Convert timestamp; Polygon returns ms since epoch in 't' for aggs
            try:
                df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms', errors='coerce')
            except Exception:
                df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
            df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

            # Rename price/volume columns to a consistent schema
            rename_map = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
            df = df.rename(columns=rename_map)

            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"Warning: Missing expected columns: {missing}")
                # Keep only what we have; return empty if core columns missing
                if {"Close", "Volume"}.difference(df.columns):
                    return pd.DataFrame()
                # Fill missing with NaN if non-core
                for col in missing:
                    df[col] = pd.NA

            return df[required_cols]

        except Exception as e:
            # Handle rate limiting with simple backoff
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg:
                wait_s = 2 ** attempt
                print(f"Rate limited by API (attempt {attempt+1}/{max_retries}). Waiting {wait_s}s...")
                time.sleep(wait_s)
                continue
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    # If we exhausted retries
    return pd.DataFrame()

# ==============================================================================
# FINANCIAL INDICATOR FUNCTIONS (Refactored for Polygon.io)
# ==============================================================================

def simpleData(data_df, ticker, startDate, endDate):
    """Calculate simple data from existing DataFrame to avoid extra API calls."""
    simpleValues = {}
    
    if not data_df.empty:
        # Use the data we already have
        simpleValues["Open_Price"] = data_df['Open'].iloc[0]
        simpleValues["High_Price"] = data_df['High'].max()
        simpleValues["Previous_Close_Price"] = data_df['Close'].iloc[-1]  # Last close as proxy for previous close
        simpleValues["Ten_Day_Volume"] = data_df['Volume'].tail(10).mean() if len(data_df) >= 10 else data_df['Volume'].mean()
    else:
        simpleValues["Open_Price"] = 0
        simpleValues["High_Price"] = 0
        simpleValues["Previous_Close_Price"] = 0
        simpleValues["Ten_Day_Volume"] = 0
    
    return simpleValues

def SMA(ticker, startDate, endDate, window):
    data = _get_polygon_data(ticker, startDate, endDate)
    if data.empty: return pd.Series()
    sma_series = data["Close"].rolling(window=window).mean()
    # Plotting logic can be uncommented if needed
    # data["Close"].plot(label="Close Price")
    # sma_series.plot(label=f"{window}-Day SMA")
    # plt.legend(); plt.show()
    return sma_series

def EMA(ticker, startDate, endDate, window):
    data = _get_polygon_data(ticker, startDate, endDate)
    if data.empty: return pd.Series()
    return data["Close"].ewm(span=window, adjust=False).mean()

def RSI(data_df=None, ticker=None, start_date=None, end_date=None, window=14, for_graphing=False):
    # If no DataFrame is provided, fetch the data. Otherwise, use the one passed in.
    if data_df is None:
        if ticker and start_date and end_date:
            data_df = _get_polygon_data(ticker, start_date, end_date)
        else:
            return "Error: Ticker and dates required."

    if data_df.empty: return "No data" if not for_graphing else []
    
    delta = data_df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = (100 - (100 / (1 + rs))).dropna()
    rsi.index = rsi.index.date
    
    if for_graphing:
        return [[float(val)] for val in rsi.values]
    return rsi.to_string(index=False, header=False)

def MACD(data_df=None, ticker=None, startDate=None, endDate=None, fast=12, slow=26, signal=9):
    # If no DataFrame is provided, fetch the data. Otherwise, use the one passed in.
    if data_df is None:
        if ticker and startDate and endDate:
            data_df = _get_polygon_data(ticker, startDate, endDate)
        else:
            return {"MACD": "", "Signal": "", "Histogram": pd.Series()}

    if data_df.empty: return {"MACD": "", "Signal": "", "Histogram": pd.Series()}

    ema_fast = data_df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data_df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    macd_line.index = macd_line.index.date
    signal_line.index = signal_line.index.date

    return {
        "MACD": macd_line.to_string(index=True, header=False),
        "Signal": signal_line.to_string(index=True, header=False),
        "Histogram": macd_histogram
    }

def BollingerBands(ticker, startDate, endDate, window=20):
    data = _get_polygon_data(ticker, startDate, endDate)
    if data.empty: return {"SMA": pd.Series(), "Upper Band": pd.Series(), "Lower Band": pd.Series()}

    sma = data["Close"].rolling(window=window).mean()
    std = data["Close"].rolling(window=window).std()
    return {
        "SMA": sma,
        "Upper Band": sma + (2 * std),
        "Lower Band": sma - (2 * std)
    }

def VWAP(ticker, startDate, endDate):
    # Note: Polygon free tier may limit intraday data to recent dates.
    data = _get_polygon_data(ticker, startDate, endDate, timespan="minute")
    if data.empty:
        print("No intraday data for VWAP. Choose a recent date or check API subscription.")
        return pd.Series()
    
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    vwap = (typical_price * data["Volume"]).cumsum() / data["Volume"].cumsum()
    return vwap.to_string(index=True, header=False)



# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================



if __name__ == "__main__":
    print("\nWhat stock are you looking to research about?")
    stockTicker = input().upper()
    
    # Use current date for more reliable data fetching
    endDate = pd.Timestamp.now().strftime('%Y-%m-%d')
    startDate = (pd.Timestamp.now() - pd.DateOffset(months=2)).strftime('%Y-%m-%d')
    
    print(f"\n--- Analyzing {stockTicker} from {startDate} to {endDate} ---")

    # --- Step 1: Fetch data EFFICIENTLY ---
    # Fetch the main historical data ONCE.
    main_historical_data = _get_polygon_data(stockTicker, startDate, endDate)
    print(len(main_historical_data), main_historical_data.head(1))
    # Calculate simple data from the data we already fetched
    stockSimpleData = simpleData(main_historical_data, stockTicker, startDate, endDate)

    # --- Step 2: Run analysis on the data you already have ---
    # Check if the data was fetched successfully before analyzing
    if not main_historical_data.empty:
        print("✅ Data fetched successfully. Running analysis...")
        # Pass the DataFrame to the functions instead of making them re-fetch it.
        rsi_string = RSI(data_df=main_historical_data)
        rsi_for_graphing = RSI(data_df=main_historical_data, for_graphing=True)
        macd_result = MACD(data_df=main_historical_data)
        
        print("Simple Data:", stockSimpleData)
        print("RSI:", rsi_string)
        print("MACD:", macd_result)

    else:
        print(f"❌ Could not fetch historical data for {stockTicker}. Analysis aborted.")