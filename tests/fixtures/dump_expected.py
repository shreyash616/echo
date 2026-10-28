"""
Computes actual features on the 10 golden fixtures and writes tests/fixtures/expected.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.audio.decode import decode_preview
from app.audio.features import describe_audio


def main():
    fixtures = [
        "click_72_backbeat.wav",
        "click_144_fourfloor.wav",
        "trap_96.wav",
        "major_triad_pad.wav",
        "minor_triad_pad.wav",
        "power_chord_pad.wav",
        "pink_quiet.wav",
        "pink_clipped.wav",
        "bright.wav",
        "dark.wav",
        "chord_cycle_2s.wav",
        "fingerpicked_90.wav",
    ]

    expected = {}
    for fx in fixtures:
        path = os.path.join("tests/fixtures/audio", fx)
        with open(path, "rb") as f:
            y = decode_preview(f.read())
        desc = describe_audio(y)

        # Keep relevant feature numbers to 1e-3
        record = {}
        for k, v in desc.items():
            if isinstance(v, (int, bool, str)):
                record[k] = v
            elif isinstance(v, float):
                record[k] = round(v, 3)
            elif isinstance(v, dict):
                record[k] = {dk: round(dv, 3) if isinstance(dv, float) else dv for dk, dv in v.items()}
        expected[fx] = record
        print(f"Recorded expected for {fx}: felt_bpm={record.get('felt_bpm')}, kick_class={record.get('kick_class')}")

    out_path = "tests/fixtures/expected.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)
    print(f"Written to {out_path} successfully!")


if __name__ == "__main__":
    main()
