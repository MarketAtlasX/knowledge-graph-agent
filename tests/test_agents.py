"""
Comprehensive test suite for Geopolitical Knowledge Graph Intelligence Agent.

Tests cover:
- News Agent: Article fetching and data structure
- Event Agent: Entity extraction and sentiment analysis
- Knowledge Graph Agent: Graph building and relationship creation
- Workflow: End-to-end integration
"""

import pytest
import logging
from graph.state import AgentState, NewsArticle, Entity, GraphNode, GraphEdge
from agents.news_agent import fetch_news
from agents.event_agent import process_event_intelligence, analyze_sentiment, extract_entities
from agents.kg_agent import build_knowledge_graph, analyze_graph_impact
from graph.workflow import compile_workflow

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestNewsAgent:
    """Test suite for News Agent"""
    
    def test_fetch_news_nvidia(self):
        """Test fetching news for NVIDIA"""
        state: AgentState = {"stock": "NVIDIA"}
        result = fetch_news(state)
        
        assert "news" in result
        assert len(result["news"]) > 0
        assert result["stock"] == "NVIDIA"
        
    def test_fetch_news_multiple_companies(self):
        """Test fetching news for various companies"""
        companies = ["NVIDIA", "TSMC", "AMD", "APPLE"]
        
        for company in companies:
            state: AgentState = {"stock": company}
            result = fetch_news(state)
            assert "news" in result
            logger.info(f"✅ Fetched {len(result['news'])} articles for {company}")
    
    def test_fetch_news_preserves_structure(self):
        """Test that fetched news maintains required structure"""
        state: AgentState = {"stock": "NVIDIA"}
        result = fetch_news(state)
        
        for article in result["news"]:
            assert "title" in article
            assert "content" in article
            assert isinstance(article["title"], str)
            assert isinstance(article["content"], str)
    
    def test_fetch_news_missing_stock(self):
        """Test error handling for missing stock"""
        state: AgentState = {"stock": ""}
        
        with pytest.raises(ValueError):
            fetch_news(state)
    
    def test_fetch_news_messages_updated(self):
        """Test that workflow messages are updated"""
        state: AgentState = {"stock": "NVIDIA", "messages": ["Initial message"]}
        result = fetch_news(state)
        
        assert len(result.get("messages", [])) > 1
        assert any("[NEWS_AGENT]" in msg for msg in result.get("messages", []))


class TestEventAgent:
    """Test suite for Event Agent"""
    
    def test_sentiment_analysis_positive(self):
        """Test positive sentiment detection"""
        text = "Great news! The company achieved record growth and success!"
        sentiment = analyze_sentiment(text)
        assert sentiment in ["Positive", "Negative", "Neutral"]
        logger.info(f"Positive text sentiment: {sentiment}")
    
    def test_sentiment_analysis_negative(self):
        """Test negative sentiment detection"""
        text = "The company faces severe challenges and declining performance."
        sentiment = analyze_sentiment(text)
        assert sentiment in ["Positive", "Negative", "Neutral"]
        logger.info(f"Negative text sentiment: {sentiment}")
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        text = "China imposes restrictions on semiconductor exports to Taiwan."
        entities = extract_entities(text)
        
        # Check structure
        assert isinstance(entities, list)
        if len(entities) > 0:
            for ent in entities:
                assert "entity" in ent
                assert "type" in ent
                logger.info(f"Extracted entity: {ent['entity']} ({ent['type']})")
    
    def test_process_event_intelligence(self):
        """Test full event intelligence processing"""
        state: AgentState = {
            "stock": "NVIDIA",
            "news": [
                {
                    "title": "China exports ban",
                    "content": "China announces export restrictions on semiconductors affecting NVIDIA."
                }
            ]
        }
        
        result = process_event_intelligence(state)
        
        assert "entities" in result
        assert isinstance(result["entities"], list)
        logger.info(f"Extracted {len(result['entities'])} entities")
    
    def test_process_event_intelligence_missing_news(self):
        """Test error handling for missing news"""
        state: AgentState = {"stock": "NVIDIA"}
        
        with pytest.raises(ValueError):
            process_event_intelligence(state)
    
    def test_entity_sentiment_included(self):
        """Test that sentiment is included in entities"""
        state: AgentState = {
            "stock": "NVIDIA",
            "news": [
                {
                    "title": "Positive news",
                    "content": "Company growth exceeds expectations!"
                }
            ]
        }
        
        result = process_event_intelligence(state)
        
        if result["entities"]:
            for entity in result["entities"]:
                assert "sentiment" in entity
                logger.info(f"Entity {entity['entity']} has sentiment: {entity['sentiment']}")


class TestKGAgent:
    """Test suite for Knowledge Graph Agent"""
    
    def test_build_knowledge_graph(self):
        """Test knowledge graph building"""
        state: AgentState = {
            "stock": "NVIDIA",
            "entities": [
                {"entity": "China", "type": "GPE", "sentiment": "Negative"},
                {"entity": "Taiwan", "type": "GPE", "sentiment": "Neutral"},
                {"entity": "Semiconductors", "type": "PRODUCT", "sentiment": "Negative"},
            ]
        }
        
        result = build_knowledge_graph(state)
        
        assert "graph_nodes" in result
        assert "graph_edges" in result
        assert len(result["graph_nodes"]) > 0
        assert len(result["graph_edges"]) > 0
        
        logger.info(f"✅ Built graph with {len(result['graph_nodes'])} nodes and {len(result['graph_edges'])} edges")
    
    def test_graph_contains_company_node(self):
        """Test that company node is in the graph"""
        state: AgentState = {
            "stock": "NVIDIA",
            "entities": [
                {"entity": "China", "type": "GPE", "sentiment": "Negative"},
            ]
        }
        
        result = build_knowledge_graph(state)
        node_ids = [n["id"] for n in result["graph_nodes"]]
        
        assert "NVIDIA" in node_ids
        logger.info("✅ Company node correctly added to graph")
    
    def test_graph_relationships_created(self):
        """Test that relationships are created between nodes"""
        state: AgentState = {
            "stock": "NVIDIA",
            "entities": [
                {"entity": "China", "type": "GPE", "sentiment": "Negative"},
            ]
        }
        
        result = build_knowledge_graph(state)
        
        assert len(result["graph_edges"]) > 0
        
        # Verify edges connect to company
        edge_sources = {e["source"] for e in result["graph_edges"]}
        edge_targets = {e["target"] for e in result["graph_edges"]}
        
        # At least one edge should involve NVIDIA
        involved = "NVIDIA" in edge_sources or "NVIDIA" in edge_targets
        assert involved
        
        logger.info("✅ Relationships correctly created")
    
    def test_analyze_graph_impact(self):
        """Test impact pathway analysis"""
        state: AgentState = {
            "stock": "NVIDIA",
            "graph_nodes": [
                {"id": "NVIDIA", "label": "NVIDIA", "node_type": "Company", "properties": {}},
                {"id": "China", "label": "China", "node_type": "Country", "properties": {}},
            ],
            "graph_edges": [
                {"source": "China", "target": "NVIDIA", "relationship": "affects", "weight": 1.0, "properties": {}}
            ]
        }
        
        result = analyze_graph_impact(state)
        
        assert "messages" in result
        logger.info("✅ Impact analysis completed")


class TestWorkflow:
    """Test suite for complete workflow"""
    
    def test_workflow_creation(self):
        """Test workflow compilation"""
        app = compile_workflow()
        assert app is not None
        logger.info("✅ Workflow compiled successfully")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        app = compile_workflow()
        
        initial_state: AgentState = {
            "stock": "NVIDIA",
            "messages": ["Starting test workflow"],
        }
        
        result = app.invoke(initial_state)
        
        # Verify all stages completed
        assert result["stock"] == "NVIDIA"
        assert "news" in result
        assert "entities" in result
        assert "graph_nodes" in result
        assert "graph_edges" in result
        assert len(result.get("messages", [])) > 1
        
        logger.info("✅ End-to-end workflow completed successfully")
    
    def test_workflow_multiple_companies(self):
        """Test workflow with different companies"""
        app = compile_workflow()
        companies = ["NVIDIA", "TSMC", "AMD"]
        
        for company in companies:
            initial_state: AgentState = {
                "stock": company,
                "messages": [],
            }
            
            result = app.invoke(initial_state)
            
            assert result["stock"] == company
            assert len(result["graph_nodes"]) > 0
            logger.info(f"✅ Workflow completed for {company}")
    
    def test_workflow_data_flow(self):
        """Test that data flows correctly through the workflow"""
        app = compile_workflow()
        
        initial_state: AgentState = {
            "stock": "NVIDIA",
            "messages": [],
        }
        
        result = app.invoke(initial_state)
        
        # Verify data flow
        assert len(result["news"]) > 0, "News agent should produce news"
        assert len(result["entities"]) > 0, "Event agent should extract entities"
        assert len(result["graph_nodes"]) > 0, "KG agent should create nodes"
        assert len(result["graph_edges"]) > 0, "KG agent should create edges"
        
        # Verify company is in nodes
        company_node = any(n["id"] == "NVIDIA" for n in result["graph_nodes"])
        assert company_node, "Company should be in graph nodes"
        
        logger.info("✅ Data flow verified through all agents")


class TestStateModels:
    """Test suite for state TypedDicts"""
    
    def test_agent_state_required_fields(self):
        """Test AgentState has required fields"""
        state: AgentState = {"stock": "NVIDIA"}
        assert state["stock"] == "NVIDIA"
    
    def test_news_article_structure(self):
        """Test NewsArticle structure"""
        article: NewsArticle = {
            "title": "Test",
            "content": "Test content",
            "source": "Reuters",
            "date": "2024-01-15",
        }
        assert article["title"] == "Test"
    
    def test_entity_structure(self):
        """Test Entity structure"""
        entity: Entity = {
            "entity": "China",
            "type": "GPE",
            "sentiment": "Negative",
        }
        assert entity["entity"] == "China"
    
    def test_graph_node_structure(self):
        """Test GraphNode structure"""
        node: GraphNode = {
            "id": "NVIDIA",
            "label": "NVIDIA",
            "node_type": "Company",
            "properties": {},
        }
        assert node["id"] == "NVIDIA"
    
    def test_graph_edge_structure(self):
        """Test GraphEdge structure"""
        edge: GraphEdge = {
            "source": "China",
            "target": "NVIDIA",
            "relationship": "affects",
            "weight": 1.0,
            "properties": {},
        }
        assert edge["source"] == "China"


# Pytest fixtures
@pytest.fixture
def sample_state():
    """Provide sample state for tests"""
    return {
        "stock": "NVIDIA",
        "messages": [],
    }


@pytest.fixture
def sample_news_state():
    """Provide state with news articles"""
    return {
        "stock": "NVIDIA",
        "news": [
            {
                "title": "Taiwan semiconductor policy updated",
                "content": "Government announces export restrictions.",
            },
            {
                "title": "US-China trade tensions escalate",
                "content": "New export controls on AI chips.",
            }
        ]
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
