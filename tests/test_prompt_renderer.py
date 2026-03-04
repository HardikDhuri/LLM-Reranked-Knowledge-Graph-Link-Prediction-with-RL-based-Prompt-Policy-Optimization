"""Unit tests for src/prompts/renderer.py."""

from pathlib import Path

import pytest
import yaml

from src.prompts.renderer import PromptTemplate, TemplateRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATES_FILE = Path(__file__).parent.parent / "src" / "prompts" / "templates.yaml"


# ---------------------------------------------------------------------------
# templates.yaml – structural checks
# ---------------------------------------------------------------------------


def test_templates_yaml_exists():
    assert _TEMPLATES_FILE.is_file(), "templates.yaml must exist"


def test_templates_yaml_has_at_least_five_templates():
    data = yaml.safe_load(_TEMPLATES_FILE.read_text(encoding="utf-8"))
    assert len(data.get("templates", {})) >= 5


def test_templates_yaml_each_template_has_required_keys():
    data = yaml.safe_load(_TEMPLATES_FILE.read_text(encoding="utf-8"))
    for key, tmpl in data["templates"].items():
        for field in ("id", "name", "description", "system", "user"):
            assert field in tmpl, f"Template '{key}' missing field '{field}'"


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


def _make_template(**overrides) -> PromptTemplate:
    defaults = {
        "id": "test",
        "name": "Test Template",
        "description": "For testing.",
        "system": "You are an expert.",
        "user": (
            "Head: {{ head_label }}, Relation: {{ relation }}, Tail: {{ tail_label }}"
        ),
    }
    defaults.update(overrides)
    return PromptTemplate(**defaults)


def test_prompt_template_stores_metadata():
    tmpl = _make_template()
    assert tmpl.id == "test"
    assert tmpl.name == "Test Template"
    assert tmpl.description == "For testing."


def test_render_returns_two_messages():
    tmpl = _make_template()
    messages = tmpl.render(
        head_label="Paris", relation="capital of", tail_label="France"
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_render_interpolates_labels():
    tmpl = _make_template()
    messages = tmpl.render(
        head_label="Paris", relation="capital of", tail_label="France"
    )
    content = messages[1]["content"]
    assert "Paris" in content
    assert "capital of" in content
    assert "France" in content


def test_render_with_descriptions_included():
    jinja_user = (
        "{{ head_label }}"
        "{% if head_description %} — {{ head_description }}{% endif %}"
    )
    tmpl = _make_template(user=jinja_user)
    messages = tmpl.render(
        head_label="Paris",
        relation="located in",
        tail_label="France",
        head_description="capital city of France",
    )
    content = messages[1]["content"]
    assert "capital city of France" in content


def test_render_with_empty_description_omitted():
    jinja_user = (
        "{{ head_label }}"
        "{% if head_description %} — {{ head_description }}{% endif %}"
    )
    tmpl = _make_template(user=jinja_user)
    messages = tmpl.render(
        head_label="Paris",
        relation="located in",
        tail_label="France",
        head_description="",
    )
    content = messages[1]["content"]
    assert "—" not in content


def test_render_passes_extra_kwargs():
    tmpl = _make_template(user="{{ head_label }} {{ extra }}")
    messages = tmpl.render(head_label="X", relation="R", tail_label="Y", extra="bonus")
    assert "bonus" in messages[1]["content"]


def test_render_system_message_rendered():
    tmpl = _make_template(system="System: {{ head_label }}")
    messages = tmpl.render(head_label="A", relation="R", tail_label="B")
    assert messages[0]["content"] == "System: A"


# ---------------------------------------------------------------------------
# TemplateRegistry
# ---------------------------------------------------------------------------


def test_registry_loads_default_templates():
    registry = TemplateRegistry()
    assert len(registry) >= 5


def test_registry_list_ids_sorted():
    registry = TemplateRegistry()
    ids = registry.list_ids()
    assert ids == sorted(ids)


def test_registry_get_known_template():
    registry = TemplateRegistry()
    tmpl = registry.get("minimal")
    assert tmpl.id == "minimal"
    assert isinstance(tmpl, PromptTemplate)


def test_registry_get_unknown_raises_key_error():
    registry = TemplateRegistry()
    with pytest.raises(KeyError, match="unknown_template"):
        registry.get("unknown_template")


def test_registry_all_returns_all_templates():
    registry = TemplateRegistry()
    all_tmpls = registry.all()
    assert len(all_tmpls) == len(registry)
    assert all(isinstance(t, PromptTemplate) for t in all_tmpls)


def test_registry_load_custom_yaml(tmp_path):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        "templates:\n"
        "  simple:\n"
        "    id: simple\n"
        "    name: Simple\n"
        "    description: A simple template.\n"
        "    system: System prompt.\n"
        "    user: 'Head: {{ head_label }}'\n",
        encoding="utf-8",
    )
    registry = TemplateRegistry(templates_file=custom_yaml)
    assert len(registry) == 1
    tmpl = registry.get("simple")
    assert tmpl.name == "Simple"


# ---------------------------------------------------------------------------
# End-to-end: all default templates render without errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template_id", TemplateRegistry().list_ids())
def test_all_templates_render_without_error(template_id):
    registry = TemplateRegistry()
    tmpl = registry.get(template_id)
    messages = tmpl.render(
        head_label="Albert Einstein",
        relation="educated at",
        tail_label="ETH Zurich",
        head_description="theoretical physicist",
        tail_description="Swiss federal institute of technology",
    )
    assert len(messages) == 2
    assert messages[1]["content"].strip()


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


def test_public_api_importable():
    from src.prompts import PromptTemplate as PT
    from src.prompts import TemplateRegistry as TR

    assert PT is PromptTemplate
    assert TR is TemplateRegistry
