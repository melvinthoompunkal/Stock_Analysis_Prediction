import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import statistics
import yfinance as yf
from collections import Counter
import psycopg2


load_dotenv()

class StockSentimentAnalyzer:
    """
    Analyzes stock sentiment with real, working data sources.
    Provides accurate bullish/bearish predictions.
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.newsapi = NewsApiClient(api_key=self.news_api_key) if self.news_api_key else None
        
        # Get company info from yfinance
        self.stock = yf.Ticker(self.ticker)
        try:
            self.company_name = self.stock.info.get('longName', self.ticker)
            self.sector = self.stock.info.get('sector', 'Unknown')
        except:
            self.company_name = self.ticker
            self.sector = 'Unknown'
    
    def get_news_sentiment(self, days_back=7):
        """
        Analyzes recent news sentiment using NewsAPI.
        Returns: Score 0-100 with confidence level
        """
        if not self.newsapi:
            return {'score': 50, 'confidence': 'low', 'article_count': 0}
        
        try:
            date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            all_articles = self.newsapi.get_everything(
                q=f'{self.ticker} OR "{self.company_name}"',
                language='en',
                from_param=date_from,
                sort_by='publishedAt',
                page_size=100
            )
            
            if all_articles['status'] == 'ok' and all_articles['totalResults'] > 0:
                sentiments = []
                keywords_positive = []
                keywords_negative = []
                top_headlines = []
                
                for idx, article in enumerate(all_articles['articles'][:50]):  # Analyze top 50
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}".lower()
                    
                    if text and self.ticker.lower() in text or self.company_name.lower() in text:
                        blob = TextBlob(text)
                        polarity = blob.sentiment.polarity
                        sentiments.append(polarity)
                        
                        # Track keywords
                        if polarity > 0.1:
                            keywords_positive.extend([w for w in ['growth', 'profit', 'beat', 'surge', 'rally', 'upgrade', 'buy'] if w in text])
                        elif polarity < -0.1:
                            keywords_negative.extend([w for w in ['loss', 'decline', 'miss', 'downgrade', 'sell', 'cut', 'warning'] if w in text])

                        # Capture a few top headlines for reasoning context
                        if len(top_headlines) < 3:
                            top_headlines.append({
                                'title': title[:160] if title else '(no title)',
                                'source': (article.get('source') or {}).get('name', 'Unknown'),
                                'url': article.get('url', ''),
                                'publishedAt': article.get('publishedAt', '')
                            })
                
                if sentiments:
                    avg_sentiment = statistics.mean(sentiments)
                    std_dev = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
                    
                    # Adjust for keyword frequency
                    positive_boost = min(len(keywords_positive) * 2, 15)
                    negative_drag = min(len(keywords_negative) * 2, 15)
                    
                    # Convert to 0-100 scale with adjustments
                    base_score = (avg_sentiment + 1) * 50
                    adjusted_score = base_score + positive_boost - negative_drag
                    final_score = max(0, min(100, adjusted_score))
                    
                    # Confidence based on article count and consistency
                    confidence = 'high' if len(sentiments) >= 10 and std_dev < 0.5 else 'medium' if len(sentiments) >= 5 else 'low'
                    
                    return {
                        'score': round(final_score, 2),
                        'confidence': confidence,
                        'article_count': len(sentiments),
                        'avg_sentiment': round(avg_sentiment, 3),
                        'top_headlines': top_headlines
                    }
            
            return {'score': 50, 'confidence': 'low', 'article_count': 0}
            
        except Exception as e:
            print(f"News sentiment error: {e}")
            return {'score': 50, 'confidence': 'low', 'article_count': 0}
    
    def get_price_momentum(self, short_period=14, long_period=50):
        """
        Analyzes price momentum and trend strength.
        Returns: Score 0-100
        """
        try:
            # Get historical data
            hist = self.stock.history(period='6mo')
            
            if len(hist) < long_period:
                return {'score': 50, 'confidence': 'low'}
            
            # Calculate moving averages
            hist['SMA_short'] = hist['Close'].rolling(window=short_period).mean()
            hist['SMA_long'] = hist['Close'].rolling(window=long_period).mean()
            
            current_price = hist['Close'].iloc[-1]
            sma_short = hist['SMA_short'].iloc[-1]
            sma_long = hist['SMA_long'].iloc[-1]
            
            # Price position relative to MAs
            score = 50
            
            # Golden cross / Death cross
            if sma_short > sma_long:
                score += 20
            else:
                score -= 20
            
            # Price above/below short-term MA
            if current_price > sma_short:
                score += 15
            else:
                score -= 15
            
            # Recent momentum (last 5 days vs previous 5 days)
            recent_5 = hist['Close'].iloc[-5:].mean()
            previous_5 = hist['Close'].iloc[-10:-5].mean()
            
            if recent_5 > previous_5:
                momentum_strength = ((recent_5 - previous_5) / previous_5) * 100
                score += min(momentum_strength * 10, 15)
            else:
                momentum_strength = ((recent_5 - previous_5) / previous_5) * 100
                score += max(momentum_strength * 10, -15)
            
            return {
                'score': round(max(0, min(100, score)), 2),
                'confidence': 'high',
                'trend': 'bullish' if score > 50 else 'bearish'
            }
            
        except Exception as e:
            print(f"Price momentum error: {e}")
            return {'score': 50, 'confidence': 'low'}
    
    def get_analyst_recommendations(self):
        """
        Gets analyst recommendations from yfinance.
        Returns: Score 0-100
        """
        try:
            recommendations = self.stock.recommendations
            
            if recommendations is None or len(recommendations) == 0:
                return {'score': 50, 'confidence': 'low', 'count': 0}
            
            # Get last 3 months of recommendations
            recent = recommendations.tail(20)
            
            # Count recommendation types
            rec_counts = Counter(recent['To Grade'].str.lower())
            
            # Score mapping
            scores = {
                'strong buy': 95,
                'buy': 75,
                'outperform': 70,
                'overweight': 70,
                'hold': 50,
                'neutral': 50,
                'underweight': 30,
                'underperform': 30,
                'sell': 20,
                'strong sell': 10
            }
            
            total_score = 0
            total_count = 0
            
            for rec, count in rec_counts.items():
                for key, value in scores.items():
                    if key in rec:
                        total_score += value * count
                        total_count += count
                        break
            
            if total_count > 0:
                avg_score = total_score / total_count
                confidence = 'high' if total_count >= 10 else 'medium' if total_count >= 5 else 'low'
                
                return {
                    'score': round(avg_score, 2),
                    'confidence': confidence,
                    'count': total_count
                }
            
            return {'score': 50, 'confidence': 'low', 'count': 0}
            
        except Exception as e:
            print(f"Analyst recommendations error: {e}")
            return {'score': 50, 'confidence': 'low', 'count': 0}
    
    def get_financial_health(self):
        """
        Analyzes financial metrics from company data.
        Returns: Score 0-100
        """
        try:
            info = self.stock.info
            score = 50
            factors = []
            
            # Revenue growth
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth:
                if revenue_growth > 0.15:  # 15%+ growth
                    score += 15
                    factors.append(f"Strong revenue growth: {revenue_growth*100:.1f}%")
                elif revenue_growth > 0.05:
                    score += 8
                    factors.append(f"Positive revenue growth: {revenue_growth*100:.1f}%")
                elif revenue_growth < 0:
                    score -= 10
                    factors.append(f"Declining revenue: {revenue_growth*100:.1f}%")
            
            # Profit margins
            profit_margin = info.get('profitMargins', 0)
            if profit_margin:
                if profit_margin > 0.20:  # 20%+ margin
                    score += 10
                    factors.append(f"High profit margin: {profit_margin*100:.1f}%")
                elif profit_margin > 0.10:
                    score += 5
                elif profit_margin < 0:
                    score -= 15
                    factors.append("Negative profit margin")
            
            # Debt to equity
            debt_to_equity = info.get('debtToEquity', 100)
            if debt_to_equity:
                if debt_to_equity < 50:
                    score += 10
                    factors.append("Low debt levels")
                elif debt_to_equity > 150:
                    score -= 10
                    factors.append("High debt levels")
            
            # EPS growth
            earnings_growth = info.get('earningsGrowth', 0)
            if earnings_growth:
                if earnings_growth > 0.15:
                    score += 10
                    factors.append(f"Strong earnings growth: {earnings_growth*100:.1f}%")
                elif earnings_growth < 0:
                    score -= 10
                    factors.append(f"Declining earnings: {earnings_growth*100:.1f}%")
            
            # Current ratio (liquidity)
            current_ratio = info.get('currentRatio', 1)
            if current_ratio:
                if current_ratio > 2:
                    score += 5
                elif current_ratio < 1:
                    score -= 10
                    factors.append("Liquidity concerns")
            
            return {
                'score': round(max(0, min(100, score)), 2),
                'confidence': 'high' if len(factors) >= 3 else 'medium',
                'factors': factors
            }
            
        except Exception as e:
            print(f"Financial health error: {e}")
            return {'score': 50, 'confidence': 'low', 'factors': []}
    
    def get_relative_strength(self):
        """
        Compares stock performance to market (SPY) and sector.
        Returns: Score 0-100
        """
        try:
            # Get stock performance
            stock_hist = self.stock.history(period='3mo')
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='3mo')
            
            if len(stock_hist) < 20 or len(spy_hist) < 20:
                return {'score': 50, 'confidence': 'low'}
            
            # Calculate returns
            stock_return = (stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0] - 1) * 100
            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
            
            # Relative strength
            relative_strength = stock_return - spy_return
            
            # Score based on outperformance
            if relative_strength > 10:
                score = 85
            elif relative_strength > 5:
                score = 70
            elif relative_strength > 0:
                score = 60
            elif relative_strength > -5:
                score = 45
            elif relative_strength > -10:
                score = 30
            else:
                score = 20
            
            return {
                'score': score,
                'confidence': 'high',
                'stock_return': round(stock_return, 2),
                'market_return': round(spy_return, 2),
                'relative_strength': round(relative_strength, 2)
            }
            
        except Exception as e:
            print(f"Relative strength error: {e}")
            return {'score': 50, 'confidence': 'low'}
    
    def calculate_short_term_sentiment(self):
        """
        Short-term (1-3 months) with emphasis on momentum and news.
        """
        print("Analyzing short-term indicators...")
        
        momentum = self.get_price_momentum()
        news = self.get_news_sentiment(days_back=7)
        analyst = self.get_analyst_recommendations()
        relative = self.get_relative_strength()
        
        # Weighted scoring - momentum matters most for short-term
        weights = {
            'momentum': (momentum['score'], 0.35),
            'news': (news['score'], 0.30),
            'relative_strength': (relative['score'], 0.20),
            'analyst': (analyst['score'], 0.15)
        }
        
        weighted_score = sum(score * weight for score, weight in weights.values())
        
        return {
            'overall_score': round(weighted_score, 2),
            'sentiment': self._interpret_score(weighted_score),
            'confidence': self._calculate_confidence([momentum, news, analyst, relative]),
            'breakdown': {
                'momentum': momentum,
                'news': news,
                'analyst': analyst,
                'relative_strength': relative
            }
        }
    
    def calculate_long_term_sentiment(self):
        """
        Long-term (6-12+ months) with emphasis on fundamentals.
        """
        print("Analyzing long-term indicators...")
        
        financials = self.get_financial_health()
        analyst = self.get_analyst_recommendations()
        news = self.get_news_sentiment(days_back=30)
        momentum = self.get_price_momentum(short_period=50, long_period=200)
        
        # Weighted scoring - fundamentals matter most long-term
        weights = {
            'financials': (financials['score'], 0.40),
            'analyst': (analyst['score'], 0.25),
            'momentum': (momentum['score'], 0.20),
            'news': (news['score'], 0.15)
        }
        
        weighted_score = sum(score * weight for score, weight in weights.values())
        
        return {
            'overall_score': round(weighted_score, 2),
            'sentiment': self._interpret_score(weighted_score),
            'confidence': self._calculate_confidence([financials, analyst, news, momentum]),
            'breakdown': {
                'financials': financials,
                'analyst': analyst,
                'momentum': momentum,
                'news': news
            }
        }
    
    def _calculate_confidence(self, indicators):
        """Calculate overall confidence from multiple indicators"""
        confidence_scores = {'high': 3, 'medium': 2, 'low': 1}
        total = sum(confidence_scores.get(ind.get('confidence', 'low'), 1) for ind in indicators)
        avg = total / len(indicators)
        
        if avg >= 2.5:
            return 'high'
        elif avg >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _interpret_score(self, score):
        """Interprets numerical score into sentiment label"""
        if score >= 75:
            return "🚀 Strong Bullish"
        elif score >= 60:
            return "📈 Bullish"
        elif score >= 45:
            return "➡️ Neutral"
        elif score >= 30:
            return "📉 Bearish"
        else:
            return "⚠️ Strong Bearish"
    
    def generate_report(self):
        """Generates comprehensive sentiment report"""
        print(f"\n{'='*70}")
        print(f"SENTIMENT ANALYSIS: {self.ticker} - {self.company_name}")
        print(f"Sector: {self.sector}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Short-term
        short_term = self.calculate_short_term_sentiment()
        print("📊 SHORT-TERM OUTLOOK (1-3 months)")
        print(f"Score: {short_term['overall_score']}/100")
        print(f"Sentiment: {short_term['sentiment']}")
        print(f"Confidence: {short_term['confidence'].upper()}")
        print("\nKey Factors:")
        for name, data in short_term['breakdown'].items():
            score = data['score']
            conf = data.get('confidence', 'N/A')
            print(f"  • {name.replace('_', ' ').title()}: {score}/100 (confidence: {conf})")
            if 'article_count' in data:
                print(f"    └─ Based on {data['article_count']} articles")
            if 'factors' in data and data['factors']:
                for factor in data['factors'][:3]:
                    print(f"    └─ {factor}")
        
        print(f"\n{'-'*70}\n")
        
        # Long-term
        long_term = self.calculate_long_term_sentiment()
        print("📈 LONG-TERM OUTLOOK (6-12+ months)")
        print(f"Score: {long_term['overall_score']}/100")
        print(f"Sentiment: {long_term['sentiment']}")
        print(f"Confidence: {long_term['confidence'].upper()}")
        print("\nKey Factors:")
        for name, data in long_term['breakdown'].items():
            score = data['score']
            conf = data.get('confidence', 'N/A')
            print(f"  • {name.replace('_', ' ').title()}: {score}/100 (confidence: {conf})")
            if 'factors' in data and data['factors']:
                for factor in data['factors'][:3]:
                    print(f"    └─ {factor}")
        
        # Summary recommendation
        print(f"\n{'='*70}")
        print("💡 RECOMMENDATION")
        print(f"{'='*70}")
        
        avg_score = (short_term['overall_score'] + long_term['overall_score']) / 2
        
        if avg_score >= 65:
            print("✅ STRONG BUY - Both short and long-term indicators are positive")
        elif avg_score >= 55:
            print("✅ BUY - Overall positive sentiment")
        elif avg_score >= 45:
            print("⚠️ HOLD - Mixed signals, wait for clearer trend")
        elif avg_score >= 35:
            print("❌ SELL - Overall negative sentiment")
        else:
            print("❌ STRONG SELL - Multiple bearish indicators")
        
        print(f"\n{'='*70}\n")
        
        return {
            'short_term': short_term,
            'long_term': long_term,
            'avg_score': round(avg_score, 2)
        }


# Example usage
if __name__ == "__main__":
    # Test with different stocks
    tickers = ["OPEN"]  # Add more: ["AAPL", "TSLA", "NVDA"]
    
    for ticker in tickers:
        try:
            analyzer = StockSentimentAnalyzer(ticker)
            report = analyzer.generate_report()
            print("\n" + "="*70 + "\n")
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}\n")