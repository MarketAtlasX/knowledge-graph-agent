# Geopolitical Knowledge Graph Intelligence Agent — PRODUCTION VERSION 2.0

A multi-agent system that analyzes geopolitical events and their impact on companies using real APIs. Extracts entities, performs sentiment analysis, builds a knowledge graph, and identifies impact pathways.

```
         Input: Stock Ticker (e.g. NVIDIA)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  News Agent                   │
    │  NewsAPI + GDELT + ACLED      │  → state["news"]
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Event Agent                  │
    │  spaCy NER + TextBlob         │  → state["entities"]
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Knowledge Graph Agent        │
    │  NetworkX graph builder       │  → state["graph_nodes"]
    │                               │    state["graph_edges"]
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Impact Analysis              │
    │  Pathway detection            │  → state["messages"]
    └───────────────┬───────────────┘
                    │
                    ▼
         Output: Graph + Impact Pathways
```

---

## Table of Contents

1. [Database Schema (Data Models)](#1-database-schema-data-models)
2. [Backend Connections (APIs & Services)](#2-backend-connections-apis--services)
3. [Setup Instructions](#3-setup-instructions)
4. [Usage](#4-usage)
5. [Architecture](#5-architecture)
6. [Deployment](#6-deployment)
7. [Testing](#7-testing)
8. [Error Handling](#8-error-handling)
9. [Project Structure](#9-project-structure)

---

## 1. Database Schema (Data Models)

All data flows through the agent as **in-memory TypedDicts** defined in `graph/state.py`. Every agent reads from and writes to these schemas.

### AgentState — The Main Container

This is the single object passed through the entire workflow. Every agent reads from it and returns an updated copy.

| Field | Type | Required | Description | Written By |
|---|---|---|---|---|
| `stock` | `str` | **Yes** | Company ticker (e.g. `NVIDIA`, `TSMC`) | User input |
| `news` | `list<NewsArticle>` | No | Geopolitical news articles | News Agent |
| `entities` | `list<Entity>` | No | Named entities with sentiment | Event Agent |
| `graph_nodes` | `list<GraphNode>` | No | Knowledge graph nodes | KG Agent |
| `graph_edges` | `list<GraphEdge>` | No | Knowledge graph relationships | KG Agent |
| `messages` | `list<str>` | No | Workflow audit trail | All agents |

### NewsArticle — A Single News Item

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | `str` | **Yes** | Headline |
| `content` | `str` | **Yes** | Body or description |
| `source` | `str` | No | Publisher name (e.g. `Reuters`) |
| `date` | `str` | No | Publication date |
| `url` | `str` | No | Link to original article |

### Entity — A Named Entity with Sentiment

| Field | Type | Required | Description |
|---|---|---|---|
| `entity` | `str` | **Yes** | Extracted entity name (e.g. `China`) |
| `type` | `str` | **Yes** | NER label: `GPE`, `ORG`, `PERSON`, `PRODUCT`, `EVENT`, `FAC`, `LOC` |
| `sentiment` | `str` | No | `Positive`, `Negative`, or `Neutral` |

### GraphNode — A Vertex in the Knowledge Graph

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `str` | **Yes** | Unique ID (e.g. `NVIDIA`) |
| `label` | `str` | **Yes** | Display name |
| `node_type` | `str` | **Yes** | `Company`, `Country`, `Organization`, `Product`, `Facility`, `Individual`, `Event`, `Location`, `Entity` |
| `properties` | `dict` | No | Extra metadata (key-value pairs) |

### GraphEdge — A Relationship Between Two Nodes

| Field | Type | Required | Description |
|---|---|---|---|
| `source` | `str` | **Yes** | Source node ID |
| `target` | `str` | **Yes** | Target node ID |
| `relationship` | `str` | **Yes** | `affects`, `interacts_with`, `produces`, `located_in`, `involves`, `impacted_by`, `related_to` (plus inverse forms suffixed `_by`) |
| `weight` | `float` | No | Edge weight (default `1.0`) |
| `properties` | `dict` | No | Extra metadata |

### Neo4j Schema (Planned Migration)

When the graph is migrated to persistent storage:

```
(:Company {id, label, ticker})
(:Country  {id, label, region})
(:Organization {id, label, type})
(:Product  {id, label, category})

(:Country)-[:AFFECTS]->(:Company)
(:Company)-[:PRODUCES]->(:Product)
(:Organization)-[:INTERACTS_WITH]->(:Company)
(:Event)-[:IMPACTED_BY]->(:Company)
```

---

## 2. Backend Connections (APIs & Services)

### 2.1 External News APIs

| API | URL | Auth | Rate Limit | On Failure |
|---|---|---|---|---|
| **NewsAPI** | `https://newsapi.org/v2/everything` | `NEWSAPI_KEY` | 1,000 req/day free | Blocks workflow |
| **GDELT** | `https://api.gdeltproject.org/api/v2/search/tv` | None | 100+ req/hour | Logged, continues |
| **ACLED** | `https://api.acleddata.com/api/add-specific-filters` | None | 10 req/min | Silent, continues |

All three are called via `APIClient` in `agents/news_agent.py` which implements:
- Retry logic — 3 attempts with exponential backoff
- Timeout handling — configurable via `REQUEST_TIMEOUT` (default 30s)
- HTTP error handling — 401 (bad key), 429 (rate limited), 5xx (server error)

### 2.2 Optional Backend Services

| Service | Env Variable | Purpose |
|---|---|---|
| Finnhub | `FINNHUB_API_KEY` | Financial market data |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Stock market data |
| OpenAI | `OPENAI_API_KEY` | LLM reasoning (future) |
| Claude | `CLAUDE_API_KEY` | LLM reasoning (future) |
| Neo4j | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Persistent graph DB (future) |

### 2.3 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEWSAPI_KEY` | **Yes** | — | NewsAPI key (get from https://newsapi.org/register) |
| `GDELT_BASE_URL` | No | `https://api.gdeltproject.org/api/v2/search/tv` | GDELT API endpoint |
| `REQUEST_TIMEOUT` | No | `30` | HTTP request timeout (seconds) |
| `MAX_RETRIES` | No | `3` | API retry attempts |
| `MAX_ARTICLES_PER_SOURCE` | No | `50` | Max articles per API |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 3. Setup Instructions

### Prerequisites
- Python 3.11+
- [NewsAPI key](https://newsapi.org/register) (free)

### Quick Start

```bash
# 1. Clone
git clone https://github.com/MarketAtlasX/knowledge-graph-agent.git
cd knowledge-graph-agent

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows
source venv/bin/activate        # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Configure environment
copy .env.prod .env
# Edit .env — set NEWSAPI_KEY=your_actual_key

# 5. Verify
python -m pytest tests/ -v
```

---

## 4. Usage

### CLI

```bash
python main.py --stock NVIDIA
python main.py --stock TSMC --export results.json
```

### Programmatic

```python
from main import run_agent

result = run_agent("NVIDIA")

print(f"News articles:  {len(result['news'])}")
print(f"Entities:       {len(result['entities'])}")
print(f"Graph nodes:    {len(result['graph_nodes'])}")
print(f"Relationships:  {len(result['graph_edges'])}")
```

---

## 5. Architecture

### Agents

| Agent | File | Input | Output | Technology |
|---|---|---|---|---|
| News Agent | `agents/news_agent.py` | Stock ticker | `list<NewsArticle>` | NewsAPI, GDELT, ACLED, `requests` |
| Event Agent | `agents/event_agent.py` | `list<NewsArticle>` | `list<Entity>` | spaCy NER, TextBlob sentiment |
| KG Agent | `agents/kg_agent.py` | `list<Entity>` | `list<GraphNode>` + `list<GraphEdge>` | NetworkX |

### Workflow

Defined in `graph/workflow.py` using LangGraph's `StateGraph`:

```
news_agent → event_agent → kg_agent → impact_analysis → END
```

---

## 6. Deployment

### Local

```bash
python main.py --stock NVIDIA --export output.json
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
COPY . .
ENTRYPOINT ["python", "main.py"]
```

```bash
docker build -t kg-agent .
docker run -e NEWSAPI_KEY=your_key kg-agent --stock TSMC
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python -m spacy download en_core_web_sm
      - run: python -m pytest tests/ -v --cov=agents --cov=graph
```

### Cloud Options

| Platform | Method |
|---|---|
| AWS Lambda | Package as Lambda function + API Gateway trigger |
| Google Cloud Run | Deploy Docker image, set env vars, expose HTTP |
| Azure Container Instances | `az container create --image kg-agent --env NEWSAPI_KEY=...` |

### Production Checklist

- [ ] Set `LOG_LEVEL=WARNING` in production
- [ ] Use a secret manager (not `.env`) for API keys
- [ ] Configure health checks and log aggregation
- [ ] Add process manager (supervisor / systemd) for long-running instances
- [ ] Implement Neo4j for persistent graph storage

---

## 7. Testing

```bash
# Unit tests
python -m pytest tests/test_agents.py -v

# Mocked API tests
python -m pytest tests/test_production_api.py -v

# Coverage
python -m pytest --cov=agents --cov=graph tests/
```

### Test Coverage Areas
- NewsAgent: article fetching, deduplication, error handling
- Event Agent: NER extraction, sentiment analysis
- KG Agent: graph building, relationship creation, impact analysis
- Workflow: end-to-end pipeline, data flow, multi-company runs
- API Client: timeout retries, connection errors, auth errors

---

## 8. Error Handling

| Error | Cause | Handling |
|---|---|---|
| `NEWSAPI_KEY not set` | Missing API key in `.env` | Raises `NewsAPIError` with fix instructions |
| `API rate limit exceeded` | Exceeded free tier | Retries 3 times, then raises error |
| `Connection timeout` | Network issue | Automatic retry (3 attempts) |
| `Invalid API key` | 401 from NewsAPI | Helpful error message pointing to `.env` |
| No articles found | All APIs returned empty | Raises error combining per-API failure messages |
| Missing stock ticker | Empty `stock` field | Raises `ValueError` |
| No news in state | Event agent called before news agent | Raises `ValueError` |
| ACLED failure | Optional API down | Silently continues (does not block workflow) |

---

## 9. Project Structure

```
Knowledge_graph_agent/
├── agents/
│   ├── __init__.py          # Agent function exports
│   ├── news_agent.py        # NewsAPI + GDELT + ACLED fetcher
│   ├── event_agent.py       # spaCy NER + TextBlob sentiment
│   └── kg_agent.py          # NetworkX graph builder + impact paths
├── graph/
│   ├── __init__.py          # Graph exports
│   ├── state.py             # TypedDict schemas (data models)
│   └── workflow.py          # LangGraph pipeline orchestration
├── tests/
│   ├── test_agents.py       # Unit tests for all agents
│   └── test_production_api.py  # Mocked real-API integration tests
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.prod                # Environment variable template
├── .gitignore
└── README.md
```

---

## Future Enhancements

- [ ] Neo4j database integration (persistent graph storage)
- [ ] LLM-powered reasoning (OpenAI / Claude)
- [ ] FinBERT for financial sentiment
- [ ] Real-time websocket updates
- [ ] Web dashboard
- [ ] Market impact prediction

---

## License

MIT License — free to use and modify.
