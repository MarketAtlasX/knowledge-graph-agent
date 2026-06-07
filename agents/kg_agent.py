"""
Knowledge Graph Agent for the Geopolitical Knowledge Graph Intelligence Agent.

Purpose:
Identify indirect relationships between geopolitical events and stocks.

Responsibilities:
- Create company node
- Create entity nodes
- Create relationships based on extracted entities
- Connect company to relevant entities
- Store graph metadata using NetworkX

Example:
    China
        │
    Tensions
        ▼
    Taiwan
        │
    Semiconductors
        ▼
    NVIDIA
"""

import logging
from typing import Any
import networkx as nx

from graph.state import AgentState, GraphNode, GraphEdge, Entity

# Configure logging
logger = logging.getLogger(__name__)


def build_knowledge_graph(state: AgentState) -> dict[str, Any]:
    """
    Build a knowledge graph from extracted entities and company.
    
    Args:
        state: Current agent state containing stock and entities
        
    Returns:
        Updated state with graph nodes and edges
        
    Raises:
        ValueError: If no entities found in state
    """
    try:
        stock = state.get("stock", "")
        entities = state.get("entities", [])
        
        if not stock:
            raise ValueError("Stock ticker not found in state")
        
        if not entities:
            logger.warning("No entities found in state. Creating minimal graph.")
            entities = []
        
        # Initialize graph
        graph: nx.DiGraph = nx.DiGraph()
        
        # Create company node
        company_node: GraphNode = {
            "id": stock,
            "label": stock,
            "node_type": "Company",
            "properties": {
                "ticker": stock,
            }
        }
        graph.add_node(stock, **company_node)
        
        nodes_list: list[GraphNode] = [company_node]
        edges_list: list[GraphEdge] = []
        
        # Add entity nodes and relationships
        entity_relationships = _create_relationships(stock, entities)
        
        for relationship in entity_relationships:
            source_id = relationship["source"]
            target_id = relationship["target"]
            relationship_type = relationship["relationship"]
            
            # Add source node if not exists
            if source_id not in graph:
                source_node: GraphNode = {
                    "id": source_id,
                    "label": source_id,
                    "node_type": _classify_entity_type(source_id, entities),
                    "properties": {}
                }
                graph.add_node(source_id, **source_node)
                nodes_list.append(source_node)
            
            # Add target node if not exists
            if target_id not in graph:
                target_node: GraphNode = {
                    "id": target_id,
                    "label": target_id,
                    "node_type": _classify_entity_type(target_id, entities),
                    "properties": {}
                }
                graph.add_node(target_id, **target_node)
                nodes_list.append(target_node)
            
            # Add edge
            graph.add_edge(source_id, target_id, relationship=relationship_type)
            
            edge: GraphEdge = {
                "source": source_id,
                "target": target_id,
                "relationship": relationship_type,
                "weight": 1.0,
                "properties": {}
            }
            edges_list.append(edge)
        
        logger.info(f"Built knowledge graph with {len(nodes_list)} nodes and {len(edges_list)} edges")
        
        return {
            **state,
            "graph_nodes": nodes_list,
            "graph_edges": edges_list,
            "messages": [
                *(state.get("messages", [])),
                f"[KG_AGENT] Built graph with {len(nodes_list)} nodes and {len(edges_list)} edges"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}")
        raise


def _create_relationships(company: str, entities: list[Entity]) -> list[dict[str, str]]:
    """
    Create relationships between company and entities.
    
    Args:
        company: Company ticker/name
        entities: List of extracted entities
        
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    
    # Map entity types to relationship types
    relationship_mapping = {
        "GPE": "affects",           # Geographic location
        "ORG": "interacts_with",    # Organization
        "PRODUCT": "produces",       # Product
        "FAC": "located_in",         # Facility
        "PERSON": "involves",        # Person
        "EVENT": "impacted_by",      # Event
    }
    
    for entity in entities:
        entity_name = entity.get("entity", "")
        entity_type = entity.get("type", "")
        
        if not entity_name:
            continue
        
        # Create bidirectional relationships
        rel_type = relationship_mapping.get(entity_type, "related_to")
        
        # Company -> Entity
        relationships.append({
            "source": company,
            "target": entity_name,
            "relationship": f"{rel_type}_by"
        })
        
        # Entity -> Company (reverse)
        relationships.append({
            "source": entity_name,
            "target": company,
            "relationship": rel_type
        })
    
    return relationships


def _classify_entity_type(entity: str, entities: list[Entity]) -> str:
    """
    Classify entity type from the entities list.
    
    Args:
        entity: Entity name to classify
        entities: List of entities with their types
        
    Returns:
        Entity type classification
    """
    for ent in entities:
        if ent["entity"].lower() == entity.lower():
            entity_type_map = {
                "GPE": "Country",
                "ORG": "Organization",
                "PRODUCT": "Product",
                "FAC": "Facility",
                "PERSON": "Individual",
                "EVENT": "Event",
                "LOC": "Location",
            }
            return entity_type_map.get(ent["type"], "Entity")
    
    return "Entity"


def analyze_graph_impact(state: AgentState) -> dict[str, Any]:
    """
    Analyze impact pathways in the knowledge graph.
    
    Args:
        state: Current agent state with graph
        
    Returns:
        Updated state with impact analysis
    """
    try:
        nodes = state.get("graph_nodes", [])
        edges = state.get("graph_edges", [])
        stock = state.get("stock", "")
        
        if not nodes or not edges:
            logger.warning("No graph data available for impact analysis")
            return state
        
        # Build networkx graph for analysis
        graph = nx.DiGraph()
        
        for node in nodes:
            graph.add_node(node["id"], **node)
        
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"], relationship=edge["relationship"])
        
        # Find all paths from entities to company
        impact_paths = []
        for node in graph.nodes():
            if node != stock:
                try:
                    paths = list(nx.all_simple_paths(graph, node, stock, cutoff=4))
                    impact_paths.extend(paths)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        
        impact_summary = f"Identified {len(impact_paths)} potential impact pathways to {stock}"
        logger.info(impact_summary)
        
        return {
            **state,
            "messages": [
                *(state.get("messages", [])),
                f"[KG_AGENT] {impact_summary}"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing graph impact: {str(e)}")
        return state
