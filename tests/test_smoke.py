def test_import():
    import src

    assert src is not None


def test_submodule_imports():
    import src.data
    import src.eval
    import src.models
    import src.prompts
    import src.rl
    import src.wikidata

    assert src.data is not None
    assert src.eval is not None
    assert src.models is not None
    assert src.prompts is not None
    assert src.rl is not None
    assert src.wikidata is not None
