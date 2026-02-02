# Add this to your tools.py file
import os
import requests
from datetime import datetime, timedelta
from crewai.tools import tool
import feedparser

# Rest of the code...
# ============================================================================
# NEWS SCRAPING TOOL
# ============================================================================

@tool("Get Recent News")
def get_recent_news(ticker: str) -> str:
    """
    Fetch recent news about the company from multiple sources.
    Uses NewsAPI (free tier: 100 requests/day)
    Sign up: https://newsapi.org/
    """
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        # Fallback: Use free news aggregator or return basic info
        return f"📰 News API not configured. Add NEWS_API_KEY to environment for real-time news."
    
    # Clean ticker for search
    company_name = ticker.replace('.NS', '').replace('.BO', '')
    
    # Search for news from last 7 days
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        url = f'https://newsapi.org/v2/everything?q={company_name}&from={from_date}&sortBy=publishedAt&language=en&apiKey={api_key}'
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('status') == 'ok' and data.get('articles'):
            articles = data['articles'][:5]  # Top 5 news
            
            news_summary = f"📰 Recent News for {ticker}:\n\n"
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]
                description = article.get('description', '')[:150]
                
                news_summary += f"{i}. [{published}] {title}\n"
                news_summary += f"   Source: {source}\n"
                news_summary += f"   {description}...\n\n"
            
            return news_summary
        else:
            return f"📰 No recent news found for {ticker}"
    
    except Exception as e:
        return f"📰 Could not fetch news: {str(e)}"


# Alternative: Free news scraper using Google News RSS (no API key needed)
@tool("Get Google News")
def get_google_news(ticker: str) -> str:
    """
    Fetch recent news using Google News RSS feed (free, no API key needed).
    """
    import feedparser
    
    company_name = ticker.replace('.NS', '').replace('.BO', '')
    
    try:
        # Google News RSS feed
        url = f'https://news.google.com/rss/search?q={company_name}+stock+OR+{company_name}+shares&hl=en-IN&gl=IN&ceid=IN:en'
        feed = feedparser.parse(url)
        
        if feed.entries:
            news_summary = f"📰 Recent Google News for {ticker}:\n\n"
            for i, entry in enumerate(feed.entries[:5], 1):
                title = entry.get('title', 'No title')
                published = entry.get('published', 'Unknown date')
                link = entry.get('link', '')
                
                news_summary += f"{i}. {title}\n"
                news_summary += f"   Published: {published}\n"
                news_summary += f"   Link: {link}\n\n"
            
            return news_summary
        else:
            return f"📰 No recent Google News found for {ticker}"
    
    except Exception as e:
        return f"📰 Could not fetch Google News: {str(e)}"