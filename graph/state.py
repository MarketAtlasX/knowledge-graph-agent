"""
State management for the Geopolitical Knowledge Graph Intelligence Agent.

This module defines the TypedDict schema for the agent state,
ensuring type safety and consistency across the workflow.
"""

from typing import TypedDict, Any
from typing_extensions import NotRequired


class NewsArticle(TypedDict):
    """Schema for a news article."""
    title: str
    content: str
    source: NotRequired[str]
    date: NotRequired[str]
    url: NotRequired[str]


class Entity(TypedDict):
    """Schema for an extracted entity."""
    entity: str
    type: str  # GPE, ORG, PERSON, PRODUCT, etc.
    sentiment: NotRequired[str]  # Positive, Negative, Neutral


class GraphNode(TypedDict):
    """Schema for a graph node."""
    id: str
    label: str
    node_type: str  # Company, Country, Organization, etc.
    properties: NotRequired[dict[str, Any]]


class GraphEdge(TypedDict):
    """Schema for a graph edge."""
    source: str
    target: str
    relationship: str
    weight: NotRequired[float]
    properties: NotRequired[dict[str, Any]]


class AgentState(TypedDict):
    """
    Main state schema for the Geopolitical Knowledge Graph Intelligence Agent.
    
    This TypedDict ensures type safety and consistency across all agents
    in the workflow.
    """
    stock: str
    """The company ticker or name to analyze (e.g., 'NVIDIA', 'TSMC')"""
    
    news: NotRequired[list[NewsArticle]]
    """List of geopolitical news articles relevant to the stock"""
    
    entities: NotRequired[list[Entity]]
    """Extracted entities from news articles with their types and sentiment"""
    
    graph_nodes: NotRequired[list[GraphNode]]
    """Nodes in the knowledge graph (companies, countries, organizations)"""
    
    graph_edges: NotRequired[list[GraphEdge]]
    """Edges in the knowledge graph (relationships between entities)"""
    
    messages: NotRequired[list[str]]
    """Debug messages and workflow information"""
