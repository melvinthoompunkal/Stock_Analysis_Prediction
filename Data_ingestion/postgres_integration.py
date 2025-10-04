# ==============================================================================
# POSTGRESQL INTEGRATION FOR STOCK ANALYSIS SYSTEM
# ==============================================================================
"""
PostgreSQL integration module for storing and retrieving stock market data
and sentiment analysis results. Provides a clean interface between the
market_data.py and News_analysis.py modules and PostgreSQL database.
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDatabaseManager:
    """
    Manages PostgreSQL database operations for stock analysis data.
    Handles connection management, table creation, data insertion, and retrieval.
    """
    
    def __init__(self, dbname: str = "mydb", user: str = "melvint", 
                 password: str = "MelvinGeorgi", host: str = "localhost", port: str = "5432"):
        """
        Initialize database connection parameters.
        
        Args:
            dbname: Database name
            user: Username
            password: Password
            host: Host address
            port: Port number
        """
        self.connection_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures proper connection handling and cleanup.
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def create_tables(self):
        """
        Create all necessary tables for stock analysis data.
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                # Stock symbols table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS stock_symbols (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) UNIQUE NOT NULL,
                        company_name TEXT,
                        sector VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Market data table (OHLCV + technical indicators)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) NOT NULL,
                        date DATE NOT NULL,
                        open_price DECIMAL(10,4),
                        high_price DECIMAL(10,4),
                        low_price DECIMAL(10,4),
                        close_price DECIMAL(10,4),
                        volume BIGINT,
                        rsi DECIMAL(5,2),
                        macd DECIMAL(10,6),
                        macd_signal DECIMAL(10,6),
                        macd_histogram DECIMAL(10,6),
                        atr DECIMAL(10,4),
                        adx DECIMAL(5,2),
                        roc DECIMAL(10,6),
                        cci DECIMAL(10,6),
                        cmf DECIMAL(10,6),
                        market_classification VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, date)
                    );
                """)
                
                # News sentiment analysis table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_sentiment (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) NOT NULL,
                        analysis_date DATE NOT NULL,
                        short_term_score DECIMAL(5,2),
                        long_term_score DECIMAL(5,2),
                        overall_score DECIMAL(5,2),
                        sentiment_label VARCHAR(50),
                        confidence_level VARCHAR(20),
                        article_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, analysis_date)
                    );
                """)
                
                # Detailed sentiment breakdown table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_breakdown (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) NOT NULL,
                        analysis_date DATE NOT NULL,
                        timeframe VARCHAR(20) NOT NULL, -- 'short_term' or 'long_term'
                        momentum_score DECIMAL(5,2),
                        momentum_confidence VARCHAR(20),
                        news_score DECIMAL(5,2),
                        news_confidence VARCHAR(20),
                        news_article_count INTEGER,
                        analyst_score DECIMAL(5,2),
                        analyst_confidence VARCHAR(20),
                        analyst_count INTEGER,
                        financial_health_score DECIMAL(5,2),
                        financial_health_confidence VARCHAR(20),
                        relative_strength_score DECIMAL(5,2),
                        relative_strength_confidence VARCHAR(20),
                        additional_factors JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, analysis_date, timeframe)
                    );
                """)
                
                # Create indexes for better performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date 
                    ON market_data(ticker, date);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date 
                    ON news_sentiment(ticker, analysis_date);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_breakdown_ticker_date 
                    ON sentiment_breakdown(ticker, analysis_date);
                """)
                
                conn.commit()
                logger.info("All tables created successfully")
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error creating tables: {e}")
                raise
            finally:
                cur.close()
    
    def insert_stock_symbol(self, ticker: str, company_name: str = None, sector: str = None):
        """
        Insert or update stock symbol information.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            sector: Company sector
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                cur.execute("""
                    INSERT INTO stock_symbols (ticker, company_name, sector)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker) 
                    DO UPDATE SET 
                        company_name = EXCLUDED.company_name,
                        sector = EXCLUDED.sector,
                        updated_at = CURRENT_TIMESTAMP;
                """, (ticker, company_name, sector))
                
                conn.commit()
                logger.info(f"Stock symbol {ticker} inserted/updated successfully")
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error inserting stock symbol {ticker}: {e}")
                raise
            finally:
                cur.close()
    
    def insert_market_data(self, ticker: str, market_data_df: pd.DataFrame, 
                          indicators: Dict[str, Any], market_classification: str = None):
        """
        Insert market data and technical indicators into the database.
        
        Args:
            ticker: Stock ticker symbol
            market_data_df: DataFrame with OHLCV data (datetime index)
            indicators: Dictionary containing technical indicators
            market_classification: Market classification result
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                # Prepare data for insertion
                data_rows = []
                
                for date_idx, row in market_data_df.iterrows():
                    # Convert datetime index to date
                    if hasattr(date_idx, 'date'):
                        date_val = date_idx.date()
                    else:
                        date_val = pd.to_datetime(date_idx).date()
                    
                    # Get indicator values for this date
                    rsi_val = indicators.get('rsi', pd.Series()).get(date_val, None)
                    macd_data = indicators.get('macd', {})
                    macd_val = macd_data.get('MACD', pd.Series()).get(date_val, None)
                    macd_signal_val = macd_data.get('Signal', pd.Series()).get(date_val, None)
                    macd_hist_val = macd_data.get('Histogram', pd.Series()).get(date_val, None)
                    atr_val = indicators.get('atr', pd.Series()).get(date_val, None)
                    adx_val = indicators.get('adx', pd.Series()).get(date_val, None)
                    roc_val = indicators.get('roc', pd.Series()).get(date_val, None)
                    cci_val = indicators.get('cci', pd.Series()).get(date_val, None)
                    cmf_val = indicators.get('cmf', pd.Series()).get(date_val, None)
                    
                    data_rows.append((
                        ticker, date_val, row['Open'], row['High'], row['Low'], 
                        row['Close'], row['Volume'], rsi_val, macd_val, macd_signal_val,
                        macd_hist_val, atr_val, adx_val, roc_val, cci_val, cmf_val,
                        market_classification
                    ))
                
                # Insert data using executemany for better performance
                cur.executemany("""
                    INSERT INTO market_data (
                        ticker, date, open_price, high_price, low_price, close_price, volume,
                        rsi, macd, macd_signal, macd_histogram, atr, adx, roc, cci, cmf,
                        market_classification
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) 
                    DO UPDATE SET 
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        rsi = EXCLUDED.rsi,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram,
                        atr = EXCLUDED.atr,
                        adx = EXCLUDED.adx,
                        roc = EXCLUDED.roc,
                        cci = EXCLUDED.cci,
                        cmf = EXCLUDED.cmf,
                        market_classification = EXCLUDED.market_classification,
                        updated_at = CURRENT_TIMESTAMP;
                """, data_rows)
                
                conn.commit()
                logger.info(f"Inserted {len(data_rows)} market data records for {ticker}")
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error inserting market data for {ticker}: {e}")
                raise
            finally:
                cur.close()
    
    def insert_sentiment_analysis(self, ticker: str, sentiment_report: Dict[str, Any]):
        """
        Insert sentiment analysis results into the database.
        
        Args:
            ticker: Stock ticker symbol
            sentiment_report: Complete sentiment analysis report
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                analysis_date = date.today()
                
                # Insert main sentiment summary
                short_term = sentiment_report.get('short_term', {})
                long_term = sentiment_report.get('long_term', {})
                
                cur.execute("""
                    INSERT INTO news_sentiment (
                        ticker, analysis_date, short_term_score, long_term_score,
                        overall_score, sentiment_label, confidence_level, article_count
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, analysis_date) 
                    DO UPDATE SET 
                        short_term_score = EXCLUDED.short_term_score,
                        long_term_score = EXCLUDED.long_term_score,
                        overall_score = EXCLUDED.overall_score,
                        sentiment_label = EXCLUDED.sentiment_label,
                        confidence_level = EXCLUDED.confidence_level,
                        article_count = EXCLUDED.article_count;
                """, (
                    ticker, analysis_date,
                    short_term.get('overall_score'),
                    long_term.get('overall_score'),
                    sentiment_report.get('avg_score'),
                    short_term.get('sentiment'),
                    short_term.get('confidence'),
                    short_term.get('breakdown', {}).get('news', {}).get('article_count', 0)
                ))
                
                # Insert detailed breakdown for short-term
                short_breakdown = short_term.get('breakdown', {})
                cur.execute("""
                    INSERT INTO sentiment_breakdown (
                        ticker, analysis_date, timeframe, momentum_score, momentum_confidence,
                        news_score, news_confidence, news_article_count, analyst_score,
                        analyst_confidence, analyst_count, relative_strength_score,
                        relative_strength_confidence, additional_factors
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, analysis_date, timeframe) 
                    DO UPDATE SET 
                        momentum_score = EXCLUDED.momentum_score,
                        momentum_confidence = EXCLUDED.momentum_confidence,
                        news_score = EXCLUDED.news_score,
                        news_confidence = EXCLUDED.news_confidence,
                        news_article_count = EXCLUDED.news_article_count,
                        analyst_score = EXCLUDED.analyst_score,
                        analyst_confidence = EXCLUDED.analyst_confidence,
                        analyst_count = EXCLUDED.analyst_count,
                        relative_strength_score = EXCLUDED.relative_strength_score,
                        relative_strength_confidence = EXCLUDED.relative_strength_confidence,
                        additional_factors = EXCLUDED.additional_factors;
                """, (
                    ticker, analysis_date, 'short_term',
                    short_breakdown.get('momentum', {}).get('score'),
                    short_breakdown.get('momentum', {}).get('confidence'),
                    short_breakdown.get('news', {}).get('score'),
                    short_breakdown.get('news', {}).get('confidence'),
                    short_breakdown.get('news', {}).get('article_count', 0),
                    short_breakdown.get('analyst', {}).get('score'),
                    short_breakdown.get('analyst', {}).get('confidence'),
                    short_breakdown.get('analyst', {}).get('count', 0),
                    short_breakdown.get('relative_strength', {}).get('score'),
                    short_breakdown.get('relative_strength', {}).get('confidence'),
                    json.dumps(short_breakdown.get('relative_strength', {}).get('stock_return', 0))
                ))
                
                # Insert detailed breakdown for long-term
                long_breakdown = long_term.get('breakdown', {})
                cur.execute("""
                    INSERT INTO sentiment_breakdown (
                        ticker, analysis_date, timeframe, financial_health_score,
                        financial_health_confidence, analyst_score, analyst_confidence,
                        analyst_count, momentum_score, momentum_confidence, news_score,
                        news_confidence, news_article_count, additional_factors
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, analysis_date, timeframe) 
                    DO UPDATE SET 
                        financial_health_score = EXCLUDED.financial_health_score,
                        financial_health_confidence = EXCLUDED.financial_health_confidence,
                        analyst_score = EXCLUDED.analyst_score,
                        analyst_confidence = EXCLUDED.analyst_confidence,
                        analyst_count = EXCLUDED.analyst_count,
                        momentum_score = EXCLUDED.momentum_score,
                        momentum_confidence = EXCLUDED.momentum_confidence,
                        news_score = EXCLUDED.news_score,
                        news_confidence = EXCLUDED.news_confidence,
                        news_article_count = EXCLUDED.news_article_count,
                        additional_factors = EXCLUDED.additional_factors;
                """, (
                    ticker, analysis_date, 'long_term',
                    long_breakdown.get('financials', {}).get('score'),
                    long_breakdown.get('financials', {}).get('confidence'),
                    long_breakdown.get('analyst', {}).get('score'),
                    long_breakdown.get('analyst', {}).get('confidence'),
                    long_breakdown.get('analyst', {}).get('count', 0),
                    long_breakdown.get('momentum', {}).get('score'),
                    long_breakdown.get('momentum', {}).get('confidence'),
                    long_breakdown.get('news', {}).get('score'),
                    long_breakdown.get('news', {}).get('confidence'),
                    long_breakdown.get('news', {}).get('article_count', 0),
                    json.dumps(long_breakdown.get('financials', {}).get('factors', []))
                ))
                
                conn.commit()
                logger.info(f"Inserted sentiment analysis for {ticker}")
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error inserting sentiment analysis for {ticker}: {e}")
                raise
            finally:
                cur.close()
    
    def get_market_data(self, ticker: str, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """
        Retrieve market data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with market data and indicators
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                query = """
                    SELECT date, open_price, high_price, low_price, close_price, volume,
                           rsi, macd, macd_signal, macd_histogram, atr, adx, roc, cci, cmf,
                           market_classification
                    FROM market_data 
                    WHERE ticker = %s
                """
                params = [ticker]
                
                if start_date:
                    query += " AND date >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= %s"
                    params.append(end_date)
                
                query += " ORDER BY date"
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                df = pd.DataFrame(rows, columns=[
                    'date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ATR', 'ADX', 'ROC', 'CCI', 'CMF',
                    'market_classification'
                ])
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                return df
                
            except psycopg2.Error as e:
                logger.error(f"Error retrieving market data for {ticker}: {e}")
                raise
            finally:
                cur.close()
    
    def get_latest_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get the latest sentiment analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment analysis results
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                # Get main sentiment summary
                cur.execute("""
                    SELECT analysis_date, short_term_score, long_term_score, overall_score,
                           sentiment_label, confidence_level, article_count
                    FROM news_sentiment 
                    WHERE ticker = %s 
                    ORDER BY analysis_date DESC 
                    LIMIT 1
                """, (ticker,))
                
                main_result = cur.fetchone()
                if not main_result:
                    return {}
                
                # Get detailed breakdowns
                cur.execute("""
                    SELECT timeframe, momentum_score, momentum_confidence, news_score, news_confidence,
                           news_article_count, analyst_score, analyst_confidence, analyst_count,
                           financial_health_score, financial_health_confidence, relative_strength_score,
                           relative_strength_confidence, additional_factors
                    FROM sentiment_breakdown 
                    WHERE ticker = %s AND analysis_date = %s
                    ORDER BY timeframe
                """, (ticker, main_result[0]))
                
                breakdown_results = cur.fetchall()
                
                # Format results
                result = {
                    'analysis_date': main_result[0].isoformat(),
                    'short_term_score': main_result[1],
                    'long_term_score': main_result[2],
                    'overall_score': main_result[3],
                    'sentiment_label': main_result[4],
                    'confidence_level': main_result[5],
                    'article_count': main_result[6],
                    'breakdowns': {}
                }
                
                for breakdown in breakdown_results:
                    timeframe = breakdown[0]
                    result['breakdowns'][timeframe] = {
                        'momentum': {'score': breakdown[1], 'confidence': breakdown[2]},
                        'news': {'score': breakdown[3], 'confidence': breakdown[4], 'article_count': breakdown[5]},
                        'analyst': {'score': breakdown[6], 'confidence': breakdown[7], 'count': breakdown[8]},
                        'financial_health': {'score': breakdown[9], 'confidence': breakdown[10]},
                        'relative_strength': {'score': breakdown[11], 'confidence': breakdown[12]},
                        'additional_factors': json.loads(breakdown[13]) if breakdown[13] else {}
                    }
                
                return result
                
            except psycopg2.Error as e:
                logger.error(f"Error retrieving sentiment for {ticker}: {e}")
                raise
            finally:
                cur.close()
    
    def get_all_tickers(self) -> List[str]:
        """
        Get list of all tickers in the database.
        
        Returns:
            List of ticker symbols
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                cur.execute("SELECT DISTINCT ticker FROM stock_symbols ORDER BY ticker")
                return [row[0] for row in cur.fetchall()]
                
            except psycopg2.Error as e:
                logger.error(f"Error retrieving tickers: {e}")
                raise
            finally:
                cur.close()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Clean up old data to keep database size manageable.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            try:
                cutoff_date = date.today() - pd.Timedelta(days=days_to_keep)
                
                # Clean up old market data
                cur.execute("""
                    DELETE FROM market_data 
                    WHERE date < %s
                """, (cutoff_date,))
                
                # Clean up old sentiment data
                cur.execute("""
                    DELETE FROM news_sentiment 
                    WHERE analysis_date < %s
                """, (cutoff_date,))
                
                cur.execute("""
                    DELETE FROM sentiment_breakdown 
                    WHERE analysis_date < %s
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error cleaning up old data: {e}")
                raise
            finally:
                cur.close()


# ==============================================================================
# CONVENIENCE FUNCTIONS FOR EASY INTEGRATION
# ==============================================================================

def format_market_data_for_postgres(market_data_df: pd.DataFrame, 
                                  indicators: Dict[str, Any], 
                                  ticker: str) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Format market data DataFrame and indicators for PostgreSQL insertion.
    
    Args:
        market_data_df: DataFrame with OHLCV data
        indicators: Dictionary containing technical indicators
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (formatted_data_rows, metadata)
    """
    data_rows = []
    
    for date_idx, row in market_data_df.iterrows():
        # Convert datetime index to date
        if hasattr(date_idx, 'date'):
            date_val = date_idx.date()
        else:
            date_val = pd.to_datetime(date_idx).date()
        
        # Get indicator values for this date
        rsi_val = indicators.get('rsi', pd.Series()).get(date_val, None)
        macd_data = indicators.get('macd', {})
        macd_val = macd_data.get('MACD', pd.Series()).get(date_val, None)
        macd_signal_val = macd_data.get('Signal', pd.Series()).get(date_val, None)
        macd_hist_val = macd_data.get('Histogram', pd.Series()).get(date_val, None)
        atr_val = indicators.get('atr', pd.Series()).get(date_val, None)
        adx_val = indicators.get('adx', pd.Series()).get(date_val, None)
        roc_val = indicators.get('roc', pd.Series()).get(date_val, None)
        cci_val = indicators.get('cci', pd.Series()).get(date_val, None)
        cmf_val = indicators.get('cmf', pd.Series()).get(date_val, None)
        
        data_rows.append((
            ticker, date_val, row['Open'], row['High'], row['Low'], 
            row['Close'], row['Volume'], rsi_val, macd_val, macd_signal_val,
            macd_hist_val, atr_val, adx_val, roc_val, cci_val, cmf_val
        ))
    
    metadata = {
        'total_records': len(data_rows),
        'date_range': {
            'start': data_rows[0][1] if data_rows else None,
            'end': data_rows[-1][1] if data_rows else None
        },
        'indicators_present': list(indicators.keys())
    }
    
    return data_rows, metadata


def format_sentiment_for_postgres(sentiment_report: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """
    Format sentiment analysis report for PostgreSQL insertion.
    
    Args:
        sentiment_report: Complete sentiment analysis report
        ticker: Stock ticker symbol
        
    Returns:
        Formatted data dictionary
    """
    return {
        'ticker': ticker,
        'analysis_date': date.today(),
        'report': sentiment_report,
        'formatted_at': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test the database integration
    db_manager = StockDatabaseManager()
    
    try:
        # Create tables
        db_manager.create_tables()
        print("✅ Database tables created successfully")
        
        # Test connection
        tickers = db_manager.get_all_tickers()
        print(f"✅ Database connection successful. Found {len(tickers)} tickers")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
