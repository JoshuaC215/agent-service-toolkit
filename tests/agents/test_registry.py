from agents.agents import DEFAULT_AGENT, agents, get_all_agent_info


def test_default_agent_registered() -> None:
    assert DEFAULT_AGENT in agents


def test_registry_contains_product_tracks() -> None:
    info = get_all_agent_info()
    tracks = {a.track for a in info}
    packs = {a.pack for a in info}

    assert "core" in tracks
    assert "product" in tracks
    assert "skill" in packs
    assert "dwh" in packs


def test_registry_keys_unique() -> None:
    keys = [a.key for a in get_all_agent_info()]
    assert len(keys) == len(set(keys))
