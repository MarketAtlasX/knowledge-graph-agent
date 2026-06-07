"""
Event Intelligence Agent for the Geopolitical Knowledge Graph Intelligence Agent.

Purpose:
Transform unstructured news articles into structured geopolitical events.

Responsibilities:
- Entity extraction using spaCy
- Sentiment analysis
- Event detection
- Theme classification

Models Used:
- spaCy en_core_web_sm for NER
- TextBlob or VADER for sentiment analysis (optional FinBERT later)
"""

import logging
from typing import Any
import spacy
from textblob import TextBlob

from graph.state import AgentState, Entity

# Configure logging
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment label: "Positive", "Negative", or "Neutral"
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return "Neutral"


def extract_entities(text: str) -> list[dict[str, str]]:
    """
    Extract entities from text using spaCy NER.
    
    Args:
        text: Text to process
        
    Returns:
        List of entities with their types
        
    Example:
        >>> entities = extract_entities("China imposes restrictions on semiconductors.")
        >>> any(e["entity"] == "China" for e in entities)
        True
    """
    if nlp is None:
        logger.warning("spaCy model not loaded. Installing required packages...")
        return []
    
    try:
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            # Focus on relevant entity types
            if ent.label_ in ["GPE", "ORG", "PERSON", "PRODUCT", "EVENT", "FAC", "LOC"]:
                entities.append({
                    "entity": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        
        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return []


def process_event_intelligence(state: AgentState) -> dict[str, Any]:
    """
    Convert unstructured news articles into structured events.
    
    Args:
        state: Current agent state containing news articles
        
    Returns:
        Updated state with extracted entities
        
    Raises:
        ValueError: If no news articles found in state
    """
    try:
        news_articles = state.get("news", [])
        
        if not news_articles:
            raise ValueError("No news articles found in state")
        
        all_entities: list[Entity] = []
        entity_seen = set()  # Track unique entities
        
        for article in news_articles:
            title = article.get("title", "")
            content = article.get("content", "")
            
            # Combine title and content for analysis
            full_text = f"{title}. {content}"
            
            # Extract entities
            extracted = extract_entities(full_text)
            
            # Analyze sentiment
            sentiment = analyze_sentiment(full_text)
            
            # Process each entity
            for entity_data in extracted:
                entity_key = (entity_data["entity"].lower(), entity_data["type"])
                
                # Avoid duplicates
                if entity_key not in entity_seen:
                    entity_seen.add(entity_key)
                    
                    entity: Entity = {
                        "entity": entity_data["entity"],
                        "type": entity_data["type"],
                        "sentiment": sentiment,
                    }
                    all_entities.append(entity)
        
        logger.info(f"Extracted {len(all_entities)} unique entities from {len(news_articles)} articles")
        
        return {
            **state,
            "entities": all_entities,
            "messages": [
                *(state.get("messages", [])),
                f"[EVENT_AGENT] Extracted {len(all_entities)} entities with sentiment analysis"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error processing event intelligence: {str(e)}")
        raise


def get_geopolitical_keywords(entity_text: str) -> list[str]:
    """
    Extract geopolitical keywords from entity text.
    
    Args:
        entity_text: Entity text to analyze
        
    Returns:
        List of geopolitical keywords
    """
    geopolitical_keywords = [
        "export", "import", "trade", "sanctions", "tensions",
        "restrict", "ban", "war", "conflict", "policy",
        "regulation", "control", "supply", "chain", "security",
        "alliance", "cooperation", "dispute", "agreement"
    ]
    
    text_lower = entity_text.lower()
    return [kw for kw in geopolitical_keywords if kw in text_lower]
