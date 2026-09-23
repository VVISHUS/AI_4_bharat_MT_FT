"""Tests for the data filter funnel.

Runs without a GPU, without network, and without `datasets` installed -- the
module-level `datasets` import is stubbed so the pure-python filtering logic can
be checked on any machine. Run with:

    python -m pytest tests/ -v
    python tests/test_filters.py        # also works standalone
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub `datasets` before importing src.data so these tests run in a bare env.
if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    _stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = _stub

from src.data import _devanagari_ratio, filter_stream  # noqa: E402

FILTERS = {
    "min_words": 3,
    "max_words": 80,
    "max_word_ratio": 2.5,
    "min_devanagari_ratio": 0.55,
    "drop_copies": True,
    "dedup": True,
    "dedup_on_source": True,
}

GOOD_A = {"src": "The weather is pleasant today.", "tgt": "आज हवामान आल्हाददायक आहे."}
GOOD_B = {"src": "She submitted the report on time.", "tgt": "तिने वेळेवर अहवाल सादर केला."}


def _run(rows, filters=None, target_size=100):
    return filter_stream(rows, "src", "tgt", filters or FILTERS, target_size)


def test_devanagari_ratio():
    assert _devanagari_ratio("आज हवामान आहे.") > 0.8
    assert _devanagari_ratio("This is English") == 0.0
    assert _devanagari_ratio("") == 0.0
    # Mixed script: a Marathi sentence with an embedded English brand name
    # should still read as predominantly Devanagari.
    assert 0.55 < _devanagari_ratio("मी Google वापरतो आणि तो उपयुक्त आहे.") < 1.0


def test_clean_pairs_survive():
    kept, report = _run([GOOD_A, GOOD_B])
    assert report.kept == 2
    assert [p["src"] for p in kept] == [GOOD_A["src"], GOOD_B["src"]]


def test_each_stage_catches_its_row():
    """One deliberately broken row per stage; every stage should fire once."""
    rows = [
        GOOD_A,
        GOOD_B,
        {"src": "", "tgt": "काहीतरी मजकूर इथे आहे."},                      # non_empty
        {"src": "Hello", "tgt": "नमस्कार"},                                 # length_bounds
        {"src": "Short one here.",                                          # length_ratio
         "tgt": "हे एक खूप खूप खूप खूप खूप खूप खूप खूप खूप लांब वाक्य आहे बरं का."},
        {"src": "A properly aligned English line.",                         # script_check
         "tgt": "http://example.com/12345 99 100"},
        dict(GOOD_A),                                                       # dedup
        {"src": GOOD_A["src"], "tgt": "आजचे हवामान छान आहे नक्कीच."},        # dedup_source
    ]
    kept, report = _run(rows)

    dropped = {stage.name: stage.dropped for stage in report.stages}
    assert dropped["non_empty"] == 1
    assert dropped["length_bounds"] == 1
    assert dropped["length_ratio"] == 1
    assert dropped["script_check"] == 1
    assert dropped["dedup"] == 1
    assert dropped["dedup_source"] == 1
    assert report.kept == 2
    assert report.raw == len(rows)
    # The funnel must balance: raw - sum(dropped) == kept
    assert report.raw - sum(dropped.values()) == report.kept


def test_copy_pairs_dropped():
    identical = "This sentence is in English only and appears on both sides."
    kept, report = _run([{"src": identical, "tgt": identical}])
    # script_check fires before copy_pairs (cheaper checks run first), so we
    # only assert the row does not survive, not which stage claimed it.
    assert report.kept == 0
    assert kept == []


def test_early_exit_respects_target_size():
    rows = [{"src": f"This is sentence number {i} here.", "tgt": f"हे वाक्य क्रमांक {i} आहे."}
            for i in range(50)]
    kept, report = _run(rows, target_size=10)
    assert len(kept) == 10
    # Early exit means we stop streaming rather than scanning all 50.
    assert report.raw < 50


def test_report_markdown_is_well_formed():
    _, report = _run([GOOD_A, GOOD_B])
    markdown = report.to_markdown()
    assert markdown.startswith("| Stage |")
    assert "**final**" in markdown
    # header + separator + raw row + one row per stage + final
    assert len(markdown.splitlines()) == 4 + len(report.stages)


def test_report_dict_roundtrips():
    _, report = _run([GOOD_A])
    payload = report.to_dict()
    assert payload["kept"] == 1
    assert {"name", "reason", "dropped"} <= set(payload["stages"][0])


if __name__ == "__main__":
    import io

    # Windows consoles default to cp1252 and cannot print Devanagari.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
