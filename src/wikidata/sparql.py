"""Wikidata SPARQL client for resolving Freebase MIDs to labels/descriptions."""

import logging
import time
from pathlib import Path

import requests

from src.config import get_settings
from src.wikidata.cache import JSONCache

logger = logging.getLogger(__name__)

# Rate limiting
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 1.0  # seconds between SPARQL requests


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _sparql_query(
    query: str, endpoint: str | None = None, user_agent: str | None = None
) -> dict:
    """Execute a SPARQL query against Wikidata with retry logic."""
    settings = get_settings()
    endpoint = endpoint or settings.wikidata_sparql_url
    user_agent = user_agent or settings.wikidata_user_agent

    _rate_limit()

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/sparql-results+json",
    }
    params = {"query": query, "format": "json"}

    for attempt in range(3):
        try:
            resp = requests.get(
                endpoint, headers=headers, params=params, timeout=30
            )
            if resp.status_code == 429:
                wait = 2**attempt * 5
                logger.warning(
                    f"Rate limited, waiting {wait}s (attempt {attempt + 1})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"SPARQL request failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                raise
    return {}


class WikidataResolver:
    """Resolves Freebase MIDs to Wikidata labels and descriptions."""

    def __init__(self, cache_dir: Path | None = None):
        settings = get_settings()
        cache_dir = cache_dir or settings.cache_dir
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.mid2qid_cache = JSONCache(cache_dir / "mid2qid.json")
        self.entity_text_cache = JSONCache(cache_dir / "entity_text.json")

    def mid_to_qid(self, mid: str) -> str | None:
        """Map a Freebase MID (e.g., /m/02mjmr) to a Wikidata QID (e.g., Q76)."""
        if self.mid2qid_cache.has(mid):
            return self.mid2qid_cache.get(mid)

        query = f"""
        SELECT ?item WHERE {{
            ?item wdt:P646 "{mid}" .
        }} LIMIT 1
        """
        try:
            result = _sparql_query(query)
            bindings = result.get("results", {}).get("bindings", [])
            if bindings:
                uri = bindings[0]["item"]["value"]
                qid = uri.split("/")[-1]
            else:
                qid = None
            self.mid2qid_cache.set(mid, qid)
            self.mid2qid_cache.save()
            return qid
        except Exception as e:
            logger.error(f"Failed to resolve MID {mid}: {e}")
            return None

    def get_entity_text(self, qid: str) -> dict[str, str]:
        """Fetch English label and description for a Wikidata QID."""
        if self.entity_text_cache.has(qid):
            return self.entity_text_cache.get(qid)

        query = f"""
        SELECT ?label ?description WHERE {{
            wd:{qid} rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            OPTIONAL {{
                wd:{qid} schema:description ?description .
                FILTER(LANG(?description) = "en")
            }}
        }} LIMIT 1
        """
        try:
            result = _sparql_query(query)
            bindings = result.get("results", {}).get("bindings", [])
            if bindings:
                text = {
                    "label": bindings[0].get("label", {}).get("value", qid),
                    "description": bindings[0]
                    .get("description", {})
                    .get("value", ""),
                }
            else:
                text = {"label": qid, "description": ""}
            self.entity_text_cache.set(qid, text)
            self.entity_text_cache.save()
            return text
        except Exception as e:
            logger.error(f"Failed to get text for {qid}: {e}")
            return {"label": qid, "description": ""}

    def mid_to_text(self, mid: str) -> dict[str, str]:
        """Full pipeline: MID -> QID -> label + description.

        Returns dict with keys: mid, qid, label, description.
        Falls back to using MID as label if resolution fails.
        """
        qid = self.mid_to_qid(mid)
        if qid:
            text = self.get_entity_text(qid)
            return {
                "mid": mid,
                "qid": qid,
                "label": text["label"],
                "description": text["description"],
            }
        else:
            return {
                "mid": mid,
                "qid": None,
                "label": mid,
                "description": "",
            }

    def resolve_batch(
        self, mids: list[str], progress: bool = True
    ) -> dict[str, dict[str, str]]:
        """Resolve a batch of MIDs with optional progress bar."""
        results = {}
        iterator = mids
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(mids, desc="Resolving MIDs")
            except ImportError:
                pass
        for mid in iterator:
            results[mid] = self.mid_to_text(mid)
        return results

    def cache_stats(self) -> dict[str, int]:
        return {
            "mid2qid_cached": len(self.mid2qid_cache),
            "entity_text_cached": len(self.entity_text_cache),
        }
