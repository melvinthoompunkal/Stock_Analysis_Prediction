# ==============================================================================
# DATA API: SIMPLE INTERFACE FOR RETRIEVING STOCK DATA FROM POSTGRESQL
# ==============================================================================
"""
Simple API-like interface for retrieving stock analysis data from PostgreSQL.
This module provides easy-to-use functions that can be imported by other
Python modules or used by frontend applications.
"""

import pandas as pd
import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from postgres_integration import StockDatabaseManager

class StockDataAPI:
    """
    Simple API class for retrieving stock data from PostgreSQL database.
    Provides clean, easy-to-use methods for frontend and other modules.
    """
    
    def __init__(self, dbname: str = "mydb", user: str = "melvint", 
                 password: str = "MelvinGeorgi", host: str = "localhost", port: str = "5432"):
        """
        Initialize the data API with database connection parameters.
        """
        self.db_manager = StockDatabaseManager(dbname, user, password, host, port)
    
    def get_stock_overview(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive overview of a stock including latest price and sentiment.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock overview data
        """
        try:
            # Get latest market data
            market_data = self.db_manager.get_market_data(ticker)
            if market_data.empty:
                return {"error": f"No market data found for {ticker}"}
            
            # Get latest sentiment
            sentiment = self.db_manager.get_latest_sentiment(ticker)
            
            # Get company info
            with self.db_manager.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT company_name, sector 
                    FROM stock_symbols 
                    WHERE ticker = %s
                """, (ticker,))
                company_info = cur.fetchone()
                cur.close()
            
            latest_data = market_data.iloc[-1]
            
            overview = {
                "ticker": ticker,
                "company_name": company_info[0] if company_info else ticker,
                "sector": company_info[1] if company_info else "Unknown",
                "latest_price": float(latest_data['Close']),
                "latest_volume": int(latest_data['Volume']),
                "latest_rsi": float(latest_data['RSI']) if pd.notna(latest_data['RSI']) else None,
                "market_classification": latest_data.get('market_classification', 'Unknown'),
                "data_points": len(market_data),
                "last_updated": market_data.index[-1].isoformat(),
                "sentiment": {
                    "overall_score": sentiment.get('overall_score', 50),
                    "sentiment_label": sentiment.get('sentiment_label', 'Neutral'),
                    "confidence_level": sentiment.get('confidence_level', 'low'),
                    "analysis_date": sentiment.get('analysis_date'),
                    "short_term_score": sentiment.get('short_term_score', 50),
                    "long_term_score": sentiment.get('long_term_score', 50)
                } if sentiment else None
            }
            
            return overview
            
        except Exception as e:
            return {"error": f"Error retrieving overview for {ticker}: {str(e)}"}
    
    def get_price_history(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """
        Get price history for a stock.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of history to retrieve
            
        Returns:
            Dictionary with price history data
        """
        try:
            start_date = date.today() - timedelta(days=days)
            market_data = self.db_manager.get_market_data(ticker, start_date=start_date)
            
            if market_data.empty:
                return {"error": f"No price data found for {ticker}"}
            
            # Format data for easy consumption
            price_history = []
            for date_idx, row in market_data.iterrows():
                price_history.append({
                    "date": date_idx.strftime('%Y-%m-%d'),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "rsi": float(row['RSI']) if pd.notna(row['RSI']) else None,
                    "macd": float(row['MACD']) if pd.notna(row['MACD']) else None,
                    "atr": float(row['ATR']) if pd.notna(row['ATR']) else None
                })
            
            return {
                "ticker": ticker,
                "period_days": days,
                "data_points": len(price_history),
                "price_history": price_history
            }
            
        except Exception as e:
            return {"error": f"Error retrieving price history for {ticker}: {str(e)}"}
    
    def get_technical_indicators(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """
        Get technical indicators for a stock.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of indicators to retrieve
            
        Returns:
            Dictionary with technical indicators data
        """
        try:
            start_date = date.today() - timedelta(days=days)
            market_data = self.db_manager.get_market_data(ticker, start_date=start_date)
            
            if market_data.empty:
                return {"error": f"No indicator data found for {ticker}"}
            
            indicators = []
            for date_idx, row in market_data.iterrows():
                indicators.append({
                    "date": date_idx.strftime('%Y-%m-%d'),
                    "rsi": float(row['RSI']) if pd.notna(row['RSI']) else None,
                    "macd": float(row['MACD']) if pd.notna(row['MACD']) else None,
                    "macd_signal": float(row['MACD_Signal']) if pd.notna(row['MACD_Signal']) else None,
                    "macd_histogram": float(row['MACD_Histogram']) if pd.notna(row['MACD_Histogram']) else None,
                    "atr": float(row['ATR']) if pd.notna(row['ATR']) else None,
                    "adx": float(row['ADX']) if pd.notna(row['ADX']) else None,
                    "roc": float(row['ROC']) if pd.notna(row['ROC']) else None,
                    "cci": float(row['CCI']) if pd.notna(row['CCI']) else None,
                    "cmf": float(row['CMF']) if pd.notna(row['CMF']) else None
                })
            
            return {
                "ticker": ticker,
                "period_days": days,
                "data_points": len(indicators),
                "indicators": indicators
            }
            
        except Exception as e:
            return {"error": f"Error retrieving indicators for {ticker}: {str(e)}"}
    
    def get_sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Get detailed sentiment analysis for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment analysis data
        """
        try:
            sentiment = self.db_manager.get_latest_sentiment(ticker)
            
            if not sentiment:
                return {"error": f"No sentiment data found for {ticker}"}
            
            return {
                "ticker": ticker,
                "analysis_date": sentiment['analysis_date'],
                "overall_score": sentiment['overall_score'],
                "sentiment_label": sentiment['sentiment_label'],
                "confidence_level": sentiment['confidence_level'],
                "article_count": sentiment['article_count'],
                "short_term": {
                    "score": sentiment['short_term_score'],
                    "breakdown": sentiment.get('breakdowns', {}).get('short_term', {})
                },
                "long_term": {
                    "score": sentiment['long_term_score'],
                    "breakdown": sentiment.get('breakdowns', {}).get('long_term', {})
                }
            }
            
        except Exception as e:
            return {"error": f"Error retrieving sentiment for {ticker}: {str(e)}"}
    
    def get_all_stocks(self) -> Dict[str, Any]:
        """
        Get list of all stocks in the database with basic info.
        
        Returns:
            Dictionary with list of all stocks
        """
        try:
            tickers = self.db_manager.get_all_tickers()
            
            stocks = []
            for ticker in tickers:
                # Get basic info for each ticker
                overview = self.get_stock_overview(ticker)
                if "error" not in overview:
                    stocks.append({
                        "ticker": ticker,
                        "company_name": overview.get("company_name", ticker),
                        "sector": overview.get("sector", "Unknown"),
                        "latest_price": overview.get("latest_price"),
                        "latest_sentiment": overview.get("sentiment", {}).get("sentiment_label", "Unknown"),
                        "latest_score": overview.get("sentiment", {}).get("overall_score", 50)
                    })
            
            return {
                "total_stocks": len(stocks),
                "stocks": stocks
            }
            
        except Exception as e:
            return {"error": f"Error retrieving stock list: {str(e)}"}
    
    def search_stocks(self, query: str) -> Dict[str, Any]:
        """
        Search for stocks by ticker or company name.
        
        Args:
            query: Search query (ticker or company name)
            
        Returns:
            Dictionary with matching stocks
        """
        try:
            with self.db_manager.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT ticker, company_name, sector 
                    FROM stock_symbols 
                    WHERE LOWER(ticker) LIKE LOWER(%s) 
                       OR LOWER(company_name) LIKE LOWER(%s)
                    ORDER BY ticker
                """, (f"%{query}%", f"%{query}%"))
                
                results = cur.fetchall()
                cur.close()
            
            matches = []
            for ticker, company_name, sector in results:
                matches.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "sector": sector
                })
            
            return {
                "query": query,
                "matches_found": len(matches),
                "results": matches
            }
            
        except Exception as e:
            return {"error": f"Error searching stocks: {str(e)}"}
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get overall market summary with statistics.
        
        Returns:
            Dictionary with market summary
        """
        try:
            all_stocks = self.get_all_stocks()
            if "error" in all_stocks:
                return all_stocks
            
            stocks = all_stocks["stocks"]
            
            # Calculate statistics
            sentiment_counts = {}
            sector_counts = {}
            total_stocks = len(stocks)
            
            for stock in stocks:
                sentiment = stock.get("latest_sentiment", "Unknown")
                sector = stock.get("sector", "Unknown")
                
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Calculate average sentiment score
            scores = [stock.get("latest_score", 50) for stock in stocks if stock.get("latest_score")]
            avg_score = sum(scores) / len(scores) if scores else 50
            
            return {
                "total_stocks": total_stocks,
                "average_sentiment_score": round(avg_score, 2),
                "sentiment_distribution": sentiment_counts,
                "sector_distribution": sector_counts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error generating market summary: {str(e)}"}


# ==============================================================================
# CONVENIENCE FUNCTIONS FOR EASY IMPORT
# ==============================================================================

def get_stock_overview(ticker: str) -> Dict[str, Any]:
    """Quick function to get stock overview."""
    api = StockDataAPI()
    return api.get_stock_overview(ticker)

def get_price_history(ticker: str, days: int = 30) -> Dict[str, Any]:
    """Quick function to get price history."""
    api = StockDataAPI()
    return api.get_price_history(ticker, days)

def get_sentiment_analysis(ticker: str) -> Dict[str, Any]:
    """Quick function to get sentiment analysis."""
    api = StockDataAPI()
    return api.get_sentiment_analysis(ticker)

def get_all_stocks() -> Dict[str, Any]:
    """Quick function to get all stocks."""
    api = StockDataAPI()
    return api.get_all_stocks()

def search_stocks(query: str) -> Dict[str, Any]:
    """Quick function to search stocks."""
    api = StockDataAPI()
    return api.search_stocks(query)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Stock Data API Example")
    print("=" * 40)
    
    # Initialize API
    api = StockDataAPI()
    
    # Example 1: Get stock overview
    print("\n1. Stock Overview:")
    overview = api.get_stock_overview("AAPL")
    if "error" not in overview:
        print(f"  {overview['ticker']} - {overview['company_name']}")
        print(f"  Price: ${overview['latest_price']}")
        print(f"  Sentiment: {overview['sentiment']['sentiment_label']}")
    else:
        print(f"  Error: {overview['error']}")
    
    # Example 2: Get price history
    print("\n2. Price History (last 7 days):")
    history = api.get_price_history("AAPL", days=7)
    if "error" not in history:
        print(f"  Retrieved {history['data_points']} data points")
        print(f"  Latest close: ${history['price_history'][-1]['close']}")
    else:
        print(f"  Error: {history['error']}")
    
    # Example 3: Get all stocks
    print("\n3. All Stocks:")
    all_stocks = api.get_all_stocks()
    if "error" not in all_stocks:
        print(f"  Total stocks: {all_stocks['total_stocks']}")
        for stock in all_stocks['stocks'][:3]:  # Show first 3
            print(f"  • {stock['ticker']}: {stock['latest_sentiment']}")
    else:
        print(f"  Error: {all_stocks['error']}")
    
    # Example 4: Search stocks
    print("\n4. Search Results:")
    search_results = api.search_stocks("apple")
    if "error" not in search_results:
        print(f"  Found {search_results['matches_found']} matches")
        for result in search_results['results']:
            print(f"  • {result['ticker']}: {result['company_name']}")
    else:
        print(f"  Error: {search_results['error']}")
    
    print("\n✅ API examples completed!")
