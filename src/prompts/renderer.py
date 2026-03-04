"""Prompt template loading and rendering with Jinja2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

logger = logging.getLogger(__name__)

_TEMPLATES_FILE = Path(__file__).parent / "templates.yaml"


class PromptTemplate:
    """A single prompt template backed by Jinja2."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        system: str,
        user: str,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self._system_raw = system
        self._user_raw = user
        self.system_template = Template(system)
        self.user_template = Template(user)

    def render(
        self,
        head_label: str,
        relation: str,
        tail_label: str,
        head_description: str = "",
        tail_description: str = "",
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Render the template with entity data.

        Returns a list of message dicts ready for the LLM chat API:
        ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``.
        """
        context: dict[str, Any] = {
            "head_label": head_label,
            "relation": relation,
            "tail_label": tail_label,
            "head_description": head_description,
            "tail_description": tail_description,
            **kwargs,
        }
        system_content = self.system_template.render(**context)
        user_content = self.user_template.render(**context)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


class TemplateRegistry:
    """Loads and manages all prompt templates from a YAML file."""

    def __init__(self, templates_file: Path = _TEMPLATES_FILE) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._load(templates_file)

    def _load(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for key, tmpl in data.get("templates", {}).items():
            self._templates[key] = PromptTemplate(
                id=tmpl["id"],
                name=tmpl["name"],
                description=tmpl["description"],
                system=tmpl["system"],
                user=tmpl["user"],
            )
        logger.debug("Loaded %d prompt templates from %s", len(self._templates), path)

    def get(self, template_id: str) -> PromptTemplate:
        """Return the template with the given ID.

        Raises ``KeyError`` if the template does not exist.
        """
        if template_id not in self._templates:
            raise KeyError(
                f"Unknown template '{template_id}'. "
                f"Available: {list(self._templates)}"
            )
        return self._templates[template_id]

    def list_ids(self) -> list[str]:
        """Return a sorted list of all template IDs."""
        return sorted(self._templates)

    def all(self) -> list[PromptTemplate]:
        """Return all templates in sorted order."""
        return [self._templates[k] for k in self.list_ids()]

    def __len__(self) -> int:
        return len(self._templates)
