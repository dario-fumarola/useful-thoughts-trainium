from pathlib import Path
import pytest
from weekend_thinking.legacy_repair import FILES,patched_sources,verify_target_repair


def fixture_sources():
    cls=(Path(__file__).parent/'fixtures'/'legacy_ogeo.py').read_text()
    return {FILES[0]:cls+'\n# geo.buffer(x, anchors, Z, lens)\n',
            FILES[1]:'# sgeo.buffer(x, anchors, Z[sl], lens[sl])\n# ggeo.buffer(x, anchors, Zsel, lsel)\n',
            FILES[2]:'# geo.buffer(x, anchors, Z, lens)\n# hgeo.buffer(x, anchors, Zh, lens_h)\n'}


def test_actual_legacy_geometry_label_regression():
    original=fixture_sources();patched=patched_sources(original)
    result=verify_target_repair(original[FILES[0]],patched[FILES[0]])
    assert result['passed_cases']==12


def test_patcher_refuses_changed_signatures():
    sources=fixture_sources();sources[FILES[0]]=sources[FILES[0]].replace('lens):','something_else):')
    with pytest.raises(ValueError):patched_sources(sources)


def test_patcher_is_not_silently_reapplied():
    with pytest.raises(ValueError):patched_sources(patched_sources(fixture_sources()))
