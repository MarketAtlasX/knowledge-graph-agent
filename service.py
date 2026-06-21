"""FastAPI service wrapper around the KG agent workflow."""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import run_agent

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kg_service")

app = FastAPI(title="Knowledge Graph Agent Service", version="2.0.0")


class AnalyzeRequest(BaseModel):
    stock: str


class AnalyzeCountryRequest(BaseModel):
    country: str


class HealthResponse(BaseModel):
    status: str
    service: str
    gnews_key_set: bool


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="knowledge-graph-agent",
        gnews_key_set=bool(os.getenv("GNEWS_KEY") or os.getenv("NEWSAPI_KEY")),
    )


@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    stock = body.stock.strip().upper()
    if not stock:
        raise HTTPException(status_code=400, detail="stock must be a non-empty string")

    try:
        result = run_agent(stock)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow failed for {stock}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-country")
def analyze_country(body: AnalyzeCountryRequest):
    country = body.country.strip()
    if not country:
        raise HTTPException(status_code=400, detail="country must be a non-empty string")

    try:
        result = run_agent(country)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow failed for country {country}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
