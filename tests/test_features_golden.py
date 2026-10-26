"""
tests/test_features_golden.py: Gate G0 golden fixture assertion suite.
Decodes each synthetic fixture with decode_preview and tests:
1. Exact range / domain requirements from Section 11.2 (Table 11.2).
2. Numerical equality to tests/fixtures/expected.json within 1e-3.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

# Ensure backend is on sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.audio.decode import decode_preview
from app.audio.features import describe_audio

FIXTURES_DIR = os.path.abspath("tests/fixtures/audio")
EXPECTED_PATH = os.path.abspath("tests/fixtures/expected.json")


@pytest.fixture(scope="module")
def expected_data():
    with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert_match_expected(actual: dict, expected: dict, fixture_name: str):
    """Assert all keys in expected match actual within 1e-3 tolerance for floats."""
    for key, exp_val in expected.items():
        assert key in actual, f"[{fixture_name}] Missing key '{key}' in actual descriptors"
        act_val = actual[key]
        if isinstance(exp_val, (int, bool, str)):
            assert act_val == exp_val, (
                f"[{fixture_name}] Key '{key}' mismatch: expected {exp_val!r}, got {act_val!r}"
            )
        elif isinstance(exp_val, float):
            assert abs(act_val - exp_val) < 2e-3, (
                f"[{fixture_name}] Key '{key}' float mismatch: expected {exp_val}, got {act_val}"
            )
        elif isinstance(exp_val, dict):
            assert isinstance(act_val, dict), f"[{fixture_name}] Expected dict for '{key}'"
            for dk, de_v in exp_val.items():
                # Dictionary keys may be ints as strings in json
                act_sub_val = act_val.get(dk)
                if act_sub_val is None and dk.isdigit():
                    act_sub_val = act_val.get(int(dk))
                assert act_sub_val is not None, f"[{fixture_name}] Key '{key}[{dk}]' missing in actual"
                if isinstance(de_v, float):
                    assert abs(act_sub_val - de_v) < 2e-3, (
                        f"[{fixture_name}] Key '{key}[{dk}]' mismatch: expected {de_v}, got {act_sub_val}"
                    )
                else:
                    assert act_sub_val == de_v


def _load_and_describe(fixture_name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, fixture_name)
    assert os.path.exists(path), f"Fixture not found: {path}"
    with open(path, "rb") as f:
        raw_bytes = f.read()
    y = decode_preview(raw_bytes)
    return describe_audio(y, sr=22050)


def test_click_72_backbeat(expected_data):
    name = "click_72_backbeat.wav"
    desc = _load_and_describe(name)
    
    # Table 11.2 requirements:
    # felt_bpm argmax 72 (+-2), P(felt in {70..74}) >= 0.6, pulse_rate mid, kick_class = "sparse_backbeat"
    assert abs(desc["felt_bpm"] - 72) <= 2, f"Expected felt_bpm 72 +-2, got {desc['felt_bpm']}"
    prob_72 = desc["felt_bpm_dist"].get(72, desc["felt_bpm_dist"].get("72", 0.0))
    assert prob_72 >= 0.6, f"Expected P(felt=72) >= 0.6, got {prob_72}"
    assert 0.1 <= desc["pulse_rate"] <= 0.6, f"Expected pulse_rate mid, got {desc['pulse_rate']}"
    assert desc["kick_class"] == "sparse_backbeat"

    _assert_match_expected(desc, expected_data[name], name)


def test_click_144_fourfloor(expected_data):
    name = "click_144_fourfloor.wav"
    desc = _load_and_describe(name)

    # Table 11.2 requirements: felt_bpm 144 (+-3), kick_class = "four_on_floor"
    assert abs(desc["felt_bpm"] - 144) <= 3, f"Expected felt_bpm 144 +-3, got {desc['felt_bpm']}"
    assert desc["kick_class"] == "four_on_floor"

    _assert_match_expected(desc, expected_data[name], name)


def test_trap_96(expected_data):
    name = "trap_96.wav"
    desc = _load_and_describe(name)

    # Table 11.2: felt_bpm 96 (+-2), kick_class = "syncopated_808", low_share >= 0.35, hat_subdivision = 16
    assert abs(desc["felt_bpm"] - 96) <= 2, f"Expected felt_bpm 96 +-2, got {desc['felt_bpm']}"
    assert desc["kick_class"] == "syncopated_808"
    assert desc["low_share"] >= 0.35, f"Expected low_share >= 0.35, got {desc['low_share']}"
    assert desc["hat_subdivision"] == 16, f"Expected hat_subdivision 16, got {desc['hat_subdivision']}"

    _assert_match_expected(desc, expected_data[name], name)


def test_major_triad_pad(expected_data):
    name = "major_triad_pad.wav"
    desc = _load_and_describe(name)

    # Table 11.2: mode_polarity > 0.3, mode_undefined = false, harmonic_changes_per_bar = 0
    assert desc["mode_polarity"] > 0.3, f"Expected mode_polarity > 0.3, got {desc['mode_polarity']}"
    assert desc["mode_undefined"] is False
    assert desc["changes_per_bar"] <= 0.1, f"Expected changes_per_bar near 0, got {desc['changes_per_bar']}"

    _assert_match_expected(desc, expected_data[name], name)


def test_minor_triad_pad(expected_data):
    name = "minor_triad_pad.wav"
    desc = _load_and_describe(name)

    # Table 11.2: mode_polarity < -0.3
    assert desc["mode_polarity"] < -0.3, f"Expected mode_polarity < -0.3, got {desc['mode_polarity']}"

    _assert_match_expected(desc, expected_data[name], name)


def test_power_chord_pad(expected_data):
    name = "power_chord_pad.wav"
    desc = _load_and_describe(name)

    # Table 11.2: mode_undefined = true
    assert desc["mode_undefined"] is True

    _assert_match_expected(desc, expected_data[name], name)


def test_pink_quiet(expected_data):
    name = "pink_quiet.wav"
    desc = _load_and_describe(name)

    # Table 11.2: after normalisation lufs = -23 (+-0.5), crest_factor high, restraint_bucket = "restrained"
    assert abs(desc["lufs_raw"] - (-23.0)) <= 0.5, f"Expected lufs -23 +-0.5, got {desc['lufs_raw']}"
    assert desc["crest_factor"] > 10.0, f"Expected high crest_factor, got {desc['crest_factor']}"
    assert desc["restraint_bucket"] == "restrained"

    _assert_match_expected(desc, expected_data[name], name)


def test_pink_clipped(expected_data):
    name = "pink_clipped.wav"
    desc = _load_and_describe(name)

    # Table 11.2: saturation >= 0.5, restraint_bucket = "saturated"
    assert desc["saturation"] >= 0.5, f"Expected saturation >= 0.5, got {desc['saturation']}"
    assert desc["restraint_bucket"] == "saturated"

    _assert_match_expected(desc, expected_data[name], name)


def test_bright_vs_dark_pair(expected_data):
    bright_desc = _load_and_describe("bright.wav")
    dark_desc = _load_and_describe("dark.wav")

    # Table 11.2: brightness high vs low, darkness low vs high
    assert bright_desc["brightness"] > dark_desc["brightness"], "bright.brightness must exceed dark.brightness"
    assert dark_desc["darkness"] > bright_desc["darkness"], "dark.darkness must exceed bright.darkness"

    _assert_match_expected(bright_desc, expected_data["bright.wav"], "bright.wav")
    _assert_match_expected(dark_desc, expected_data["dark.wav"], "dark.wav")


def test_chord_cycle_2s(expected_data):
    name = "chord_cycle_2s.wav"
    desc = _load_and_describe(name)

    # Table 11.2: harmonic_changes_per_bar = 1 (+-0.2)
    assert abs(desc["changes_per_bar"] - 1.0) <= 0.2, (
        f"Expected changes_per_bar 1 +-0.2, got {desc['changes_per_bar']}"
    )

    _assert_match_expected(desc, expected_data[name], name)
