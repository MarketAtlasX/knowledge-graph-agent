"""
LangGraph Workflow for the Geopolitical Knowledge Graph Intelligence Agent.

Orchestrates the multi-agent workflow:
1. News Agent - Fetches geopolitical news
2. Event Agent - Extracts entities and analyzes sentiment
3. Knowledge Graph Agent - Builds relationships and identifies impact pathways
"""

import logging
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from agents.news_agent import fetch_news
from agents.event_agent import process_event_intelligence
from agents.kg_agent import build_knowledge_graph, analyze_graph_impact

# Configure logging
logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """
    Create and compile the LangGraph workflow.
    
    Workflow Structure:
        Company Input
            ↓
        News Agent (fetch news articles)
            ↓
        Event Agent (extract entities and sentiment)
            ↓
        Knowledge Graph Agent (build relationships)
            ↓
        Impact Analysis
            ↓
        END
    
    Returns:
        Compiled StateGraph application
        
    Example:
        >>> workflow = create_workflow()
        >>> result = workflow.invoke({"stock": "NVIDIA"})
        >>> "graph_nodes" in result
        True
    """
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("news_agent", fetch_news)
    workflow.add_node("event_agent", process_event_intelligence)
    workflow.add_node("kg_agent", build_knowledge_graph)
    workflow.add_node("impact_analysis", analyze_graph_impact)
    
    # Define edges (sequential workflow)
    workflow.set_entry_point("news_agent")
    workflow.add_edge("news_agent", "event_agent")
    workflow.add_edge("event_agent", "kg_agent")
    workflow.add_edge("kg_agent", "impact_analysis")
    workflow.add_edge("impact_analysis", END)
    
    logger.info("Workflow created successfully")
    
    return workflow


def compile_workflow() -> "StateGraph":
    """
    Create and compile the workflow application.
    
    Returns:
        Compiled workflow app ready for invocation
    """
    workflow = create_workflow()
    app = workflow.compile()
    logger.info("Workflow compiled successfully")
    return app


if __name__ == "__main__":
    # For testing workflow creation
    app = compile_workflow()
    print("Workflow compiled successfully!")
