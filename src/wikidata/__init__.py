"""Wikidata SPARQL client and caching utilities."""

from src.wikidata.cache import JSONCache
from src.wikidata.sparql import WikidataResolver

__all__ = ["JSONCache", "WikidataResolver"]
