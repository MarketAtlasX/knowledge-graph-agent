"""
Main entry point for the Geopolitical Knowledge Graph Intelligence Agent.

This script demonstrates the complete workflow:
1. Takes a company ticker as input
2. Fetches geopolitical news
3. Extracts entities and analyzes sentiment
4. Builds a knowledge graph
5. Identifies impact pathways
6. Returns structured results

Example:
    python main.py --stock NVIDIA
    
    or programmatically:
    
    result = run_agent("NVIDIA")
    print(result)
"""

import logging
import json
from typing import Any
from dotenv import load_dotenv

from graph.workflow import compile_workflow
from graph.state import AgentState

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_agent(stock: str) -> dict[str, Any]:
    """
    Run the complete geopolitical intelligence workflow.
    
    Args:
        stock: Company ticker or name (e.g., 'NVIDIA', 'TSMC', 'AMD', 'APPLE')
        
    Returns:
        Complete workflow result with news, entities, and knowledge graph
        
    Raises:
        ValueError: If stock parameter is invalid
        
    Example:
        >>> result = run_agent("NVIDIA")
        >>> len(result["graph_nodes"]) > 0
        True
    """
    
    if not stock or not isinstance(stock, str):
        raise ValueError("Stock parameter must be a non-empty string")
    
    try:
        logger.info(f"Starting workflow for stock: {stock}")
        
        # Compile workflow
        app = compile_workflow()
        
        # Initialize state
        initial_state: AgentState = {
            "stock": stock.upper(),
            "messages": [f"Starting analysis for {stock}"],
        }
        
        logger.info(f"Invoking workflow with stock: {stock}")
        
        # Execute workflow
        result = app.invoke(initial_state)
        
        logger.info(f"Workflow completed successfully for {stock}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running agent for {stock}: {str(e)}")
        raise


def print_results(result: dict[str, Any]) -> None:
    """
    Pretty print the workflow results.
    
    Args:
        result: Workflow result dictionary
    """
    
    stock = result.get("stock", "Unknown")
    print("\n" + "="*80)
    print(f"GEOPOLITICAL KNOWLEDGE GRAPH ANALYSIS: {stock}")
    print("="*80)
    
    # Print messages
    messages = result.get("messages", [])
    if messages:
        print("\n📋 WORKFLOW MESSAGES:")
        for msg in messages:
            print(f"  {msg}")
    
    # Print news articles
    news = result.get("news", [])
    if news:
        print(f"\n📰 NEWS ARTICLES ({len(news)} found):")
        for i, article in enumerate(news, 1):
            print(f"\n  {i}. {article.get('title', 'N/A')}")
            print(f"     Source: {article.get('source', 'N/A')} | {article.get('date', 'N/A')}")
            print(f"     {article.get('content', 'N/A')[:100]}...")
    
    # Print entities
    entities = result.get("entities", [])
    if entities:
        print(f"\n🏷️  EXTRACTED ENTITIES ({len(entities)} found):")
        for entity in entities:
            print(f"  • {entity.get('entity', 'N/A')} ({entity.get('type', 'N/A')}) - {entity.get('sentiment', 'Neutral')}")
    
    # Print graph nodes
    nodes = result.get("graph_nodes", [])
    if nodes:
        print(f"\n🔵 GRAPH NODES ({len(nodes)} found):")
        for node in nodes:
            print(f"  • {node.get('label', 'N/A')} - Type: {node.get('node_type', 'N/A')}")
    
    # Print graph edges
    edges = result.get("graph_edges", [])
    if edges:
        print(f"\n🔗 RELATIONSHIPS ({len(edges)} found):")
        for edge in edges:
            print(f"  • {edge.get('source', 'N/A')} --[{edge.get('relationship', 'N/A')}]--> {edge.get('target', 'N/A')}")
    
    print("\n" + "="*80 + "\n")


def export_results(result: dict[str, Any], filename: str = None) -> str:
    """
    Export results to JSON file.
    
    Args:
        result: Workflow result dictionary
        filename: Output filename (default: {stock}_results.json)
        
    Returns:
        Path to exported file
    """
    
    if filename is None:
        stock = result.get("stock", "unknown").lower()
        filename = f"{stock}_analysis_results.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results exported to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Geopolitical Knowledge Graph Intelligence Agent"
    )
    parser.add_argument(
        "--stock",
        type=str,
        default="NVIDIA",
        help="Company ticker or name (e.g., NVIDIA, TSMC, AMD, APPLE)"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        # Run the agent
        result = run_agent(args.stock)
        
        # Print results
        print_results(result)
        
        # Export if requested
        if args.export:
            filepath = export_results(result, args.export)
            print(f"Results exported to: {filepath}")
            
    except Exception as e:
        logger.error(f"Failed to run agent: {str(e)}", exc_info=True)
        sys.exit(1)
