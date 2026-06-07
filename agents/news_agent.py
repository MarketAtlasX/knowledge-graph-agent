"""
News Agent for the Geopolitical Knowledge Graph Intelligence Agent.

Purpose:
Retrieve geopolitical news from REAL APIs ONLY.

Integrated APIs:
- GDELT (Global Event, Language, and Tone Database) - FREE
- NewsAPI (Real-time news aggregation) - FREE TIER AVAILABLE
- Financial Times RSS (Financial geopolitical news)
- ACLED (Armed Conflict Location & Event Data) - FREE

NO HARDCODED DATA - PRODUCTION ONLY
"""

import logging
import os
import requests
from typing import Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

from graph.state import AgentState, NewsArticle

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# API Configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
GDELT_URL = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/search/tv")
ACLED_URL = "https://api.acleddata.com/api/add-specific-filters"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES_PER_SOURCE", 50))


class NewsAPIError(Exception):
    """Custom exception for news API errors"""
    pass


class APIClient:
    """Unified API client with retry logic"""
    
    def __init__(self, max_retries: int = MAX_RETRIES, timeout: int = REQUEST_TIMEOUT):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
    
    def get(self, url: str, params: dict = None, headers: dict = None) -> dict:
        """
        Make GET request with retry logic.
        
        Args:
            url: API endpoint
            params: Query parameters
            headers: HTTP headers
            
        Returns:
            JSON response
            
        Raises:
            NewsAPIError: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}: {url}")
                if attempt == self.max_retries - 1:
                    raise NewsAPIError(f"API timeout after {self.max_retries} retries")
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise NewsAPIError(f"Connection failed after {self.max_retries} retries")
                    
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    raise NewsAPIError("Invalid API key. Check your credentials in .env")
                elif response.status_code == 429:
                    logger.warning(f"Rate limited on attempt {attempt + 1}/{self.max_retries}")
                    if attempt == self.max_retries - 1:
                        raise NewsAPIError("API rate limit exceeded")
                else:
                    raise NewsAPIError(f"HTTP Error {response.status_code}: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise NewsAPIError(f"Failed after {self.max_retries} retries: {str(e)}")
        
        raise NewsAPIError("Max retries exceeded")


def fetch_from_newsapi(company: str, client: APIClient) -> list[NewsArticle]:
    """
    Fetch news from NewsAPI.org
    
    Args:
        company: Company ticker/name
        client: API client
        
    Returns:
        List of news articles
        
    Raises:
        NewsAPIError: If API call fails
    """
    if not NEWSAPI_KEY:
        raise NewsAPIError("NEWSAPI_KEY not set in .env file. Get it from https://newsapi.org")
    
    try:
        # Build search query
        search_query = f'"{company}" AND (geopolitical OR sanctions OR supply OR trade OR export OR China OR Taiwan OR government OR semiconductor)'
        
        params = {
            "q": search_query,
            "sortBy": "publishedAt",
            "pageSize": MAX_ARTICLES,
            "language": "en",
            "apiKey": NEWSAPI_KEY,
        }
        
        logger.info(f"Fetching from NewsAPI for: {company}")
        response = client.get(NEWSAPI_URL, params=params)
        
        if response.get("status") != "ok":
            raise NewsAPIError(f"NewsAPI error: {response.get('message', 'Unknown error')}")
        
        articles: list[NewsArticle] = []
        for article in response.get("articles", [])[:MAX_ARTICLES]:
            articles.append({
                "title": article.get("title", ""),
                "content": article.get("description", ""),
                "source": article.get("source", {}).get("name", "NewsAPI"),
                "date": article.get("publishedAt", ""),
                "url": article.get("url", ""),
            })
        
        logger.info(f"✅ NewsAPI: Retrieved {len(articles)} articles for {company}")
        return articles
        
    except Exception as e:
        logger.error(f"❌ NewsAPI error: {str(e)}")
        raise NewsAPIError(f"NewsAPI failed: {str(e)}")


def fetch_from_gdelt(company: str, client: APIClient) -> list[NewsArticle]:
    """
    Fetch geopolitical events from GDELT
    
    Args:
        company: Company ticker/name
        client: API client
        
    Returns:
        List of news articles
        
    Raises:
        NewsAPIError: If API call fails
    """
    try:
        # GDELT query for geopolitical events related to company
        search_query = f'{company} AND (export OR import OR sanctions OR trade OR Taiwan OR China OR government)'
        
        params = {
            "query": search_query,
            "mode": "artlist",
            "maxrecords": MAX_ARTICLES,
            "format": "json",
        }
        
        logger.info(f"Fetching from GDELT for: {company}")
        response = client.get(GDELT_URL, params=params)
        
        if "articles" not in response:
            logger.warning("No articles found in GDELT response")
            return []
        
        articles: list[NewsArticle] = []
        for article in response.get("articles", [])[:MAX_ARTICLES]:
            articles.append({
                "title": article.get("title", ""),
                "content": article.get("snippet", ""),
                "source": article.get("sourceurl", "GDELT"),
                "date": article.get("pubdate", ""),
                "url": article.get("sourceurl", ""),
            })
        
        logger.info(f"✅ GDELT: Retrieved {len(articles)} events for {company}")
        return articles
        
    except Exception as e:
        logger.error(f"❌ GDELT error: {str(e)}")
        raise NewsAPIError(f"GDELT failed: {str(e)}")


def fetch_from_acled(company: str, client: APIClient) -> list[NewsArticle]:
    """
    Fetch armed conflict/geopolitical events from ACLED
    
    Args:
        company: Company ticker/name
        client: API client
        
    Returns:
        List of news articles
        
    Raises:
        NewsAPIError: If API call fails
    """
    try:
        # ACLED focuses on armed conflict and political violence
        # Relevant for geopolitical supply chain disruptions
        
        # Get last 30 days of events
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        params = {
            "event_id_subnational": "",
            "event_date": f"{start_date}|{end_date}",
            "limit": MAX_ARTICLES,
            "format": "json",
            "text": f"{company},semiconductor,supply,trade,export",
        }
        
        logger.info(f"Fetching from ACLED for: {company}")
        response = client.get(ACLED_URL, params=params)
        
        if "data" not in response:
            logger.warning("No data found in ACLED response")
            return []
        
        articles: list[NewsArticle] = []
        for event in response.get("data", [])[:MAX_ARTICLES]:
            articles.append({
                "title": event.get("event_type", "Event"),
                "content": f"Location: {event.get('country', 'Unknown')}. {event.get('notes', '')}",
                "source": "ACLED",
                "date": event.get("event_date", ""),
                "url": event.get("url", ""),
            })
        
        logger.info(f"✅ ACLED: Retrieved {len(articles)} events for {company}")
        return articles
        
    except Exception as e:
        logger.error(f"❌ ACLED error: {str(e)}")
        # ACLED is optional, don't fail the workflow
        return []


def fetch_news(state: AgentState) -> dict[str, Any]:
    """
    Fetch geopolitical news from REAL APIs ONLY.
    
    This function integrates multiple real-time news sources:
    1. NewsAPI - General news with geopolitical keywords
    2. GDELT - Global event database
    3. ACLED - Armed conflict and political violence data
    
    Args:
        state: Current agent state containing stock ticker
        
    Returns:
        Updated state with news articles from real APIs
        
    Raises:
        ValueError: If stock ticker not provided
        NewsAPIError: If all API sources fail
    """
    try:
        stock = state.get("stock", "").upper()
        
        if not stock:
            raise ValueError("Stock ticker not provided in state")
        
        logger.info(f"🌐 Fetching news for {stock} from real APIs...")
        
        # Initialize API client
        client = APIClient(max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT)
        
        all_articles: list[NewsArticle] = []
        errors = []
        
        # Fetch from NewsAPI (primary source)
        try:
            newsapi_articles = fetch_from_newsapi(stock, client)
            all_articles.extend(newsapi_articles)
        except NewsAPIError as e:
            logger.error(f"NewsAPI failed: {str(e)}")
            errors.append(f"NewsAPI: {str(e)}")
        
        # Fetch from GDELT (secondary source)
        try:
            gdelt_articles = fetch_from_gdelt(stock, client)
            all_articles.extend(gdelt_articles)
        except NewsAPIError as e:
            logger.error(f"GDELT failed: {str(e)}")
            errors.append(f"GDELT: {str(e)}")
        
        # Fetch from ACLED (optional)
        try:
            acled_articles = fetch_from_acled(stock, client)
            all_articles.extend(acled_articles)
        except Exception as e:
            logger.warning(f"ACLED failed (optional): {str(e)}")
        
        # Remove duplicates by URL
        unique_articles = {article.get("url"): article for article in all_articles}
        all_articles = list(unique_articles.values())
        
        if not all_articles:
            error_msg = " | ".join(errors) if errors else "No articles found"
            raise NewsAPIError(f"No news articles retrieved: {error_msg}")
        
        logger.info(f"✅ Successfully fetched {len(all_articles)} unique articles for {stock}")
        
        return {
            **state,
            "news": all_articles,
            "messages": [
                *(state.get("messages", [])),
                f"[NEWS_AGENT] Fetched {len(all_articles)} articles from real APIs (NewsAPI, GDELT, ACLED)"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching news: {str(e)}")
        raise

