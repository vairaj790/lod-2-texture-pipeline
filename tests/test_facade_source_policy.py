import pytest

from lod2_texture_pipeline.pipeline import _resolve_facade_source_selection_policy


def _item(fragment_count):
    return {"records": [{} for _ in range(fragment_count)]}


def test_auto_policy_uses_legacy_prism_for_singleton_only_building():
    policy, singleton_only = _resolve_facade_source_selection_policy(
        [_item(1), _item(1), _item(1)],
        configured_mode="auto",
    )

    assert policy == "legacy_wall_prism"
    assert singleton_only is True


def test_auto_policy_preserves_grouped_coverage_mode_for_mixed_building():
    policy, singleton_only = _resolve_facade_source_selection_policy(
        [_item(1), _item(4), _item(1)],
        configured_mode="auto",
    )

    assert policy == "projected_coverage"
    assert singleton_only is False


def test_explicit_source_policy_override_and_validation():
    policy, _singleton_only = _resolve_facade_source_selection_policy(
        [_item(1)],
        configured_mode="projected_coverage",
    )
    assert policy == "projected_coverage"

    with pytest.raises(ValueError):
        _resolve_facade_source_selection_policy([_item(1)], configured_mode="unknown")
