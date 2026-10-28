"""
tests/test_describe_parity.py: Parity test between cold describe and stored catalogue descriptors (S5).
Asserts:
- Describes 20 random catalogue previews through the cold path (recommender.describe).
- Compares kick_class, restraint_bucket, and felt_bpm (within one octave bin) to the stored row.
- At least 18 of 20 equal on each descriptor.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import pytest

sys.path.insert(0, os.path.abspath("backend"))

import app.services.recommender as rec_module


def is_octave_equivalent(bpm1: float, bpm2: float, tolerance: float = 0.12) -> bool:
    """Checks if bpm1 and bpm2 are within tolerance of 1x, 2x, or 0.5x octave ratios."""
    if bpm1 <= 0 or bpm2 <= 0:
        return False
    ratio = bpm1 / bpm2
    for mult in [1.0, 2.0, 0.5, 4.0, 0.25]:
        if abs(math.log2(ratio / mult)) <= tolerance:
            return True
    return False


def test_describe_parity():
    rec_module.load()
    recommender = rec_module.recommender
    catalogue = recommender._metadata
    if not catalogue:
        pytest.skip("Catalogue artifacts absent (CI checkout has no track_metadata.json)")
    assert len(catalogue) >= 1000, f"Catalogue too small: {len(catalogue)}"

    # Find catalogue tracks with local preview files
    previews_dir = os.path.abspath("ml/data/previews")
    valid_pairs = []
    for track in catalogue:
        tid = track.get("id")
        itunes_id = track.get("itunes_id")
        candidates = []
        if tid:
            candidates.append(os.path.join(previews_dir, f"{tid}.m4a"))
        if itunes_id:
            candidates.append(os.path.join(previews_dir, f"{itunes_id}.m4a"))
            candidates.append(os.path.join(previews_dir, f"it_{itunes_id}.m4a"))
        for p in candidates:
            if os.path.exists(p):
                valid_pairs.append((track, p))
                break

    if len(valid_pairs) < 20:
        pytest.skip(f"Found only {len(valid_pairs)} local previews for catalogue tracks (previews directory empty or absent)")

    # Sample 20 random tracks deterministically
    random.seed(42)
    sample = random.sample(valid_pairs, 20)

    kc_matches = 0
    rb_matches = 0
    bpm_matches = 0

    print("\n" + "=" * 90)
    print(f"{'Title':28s} | {'Kick Class (Cold / Stored)':25s} | {'Restraint (Cold / Stored)':23s} | {'BPM (Cold / Stored)':18s}")
    print("-" * 90)

    for track, audio_path in sample:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        meta = {"title": track.get("title"), "artist": track.get("artist"), "genre": track.get("genre")}
        cold_desc = recommender.describe(audio_bytes, meta=meta)

        # 1. kick_class parity
        stored_kc = track.get("kick_class")
        cold_kc = cold_desc.get("kick_class")
        kc_eq = (stored_kc == cold_kc)
        if kc_eq:
            kc_matches += 1

        # 2. restraint_bucket parity
        stored_rb = track.get("restraint_bucket")
        cold_rb = cold_desc.get("restraint_bucket")
        rb_eq = (stored_rb == cold_rb)
        if rb_eq:
            rb_matches += 1

        # 3. felt_bpm octave parity
        stored_bpm = track.get("felt_bpm", 120)
        cold_bpm = cold_desc.get("felt_bpm", 120)
        bpm_eq = is_octave_equivalent(float(cold_bpm), float(stored_bpm))
        if bpm_eq:
            bpm_matches += 1

        print(
            f"{track.get('title', '')[:28]:28s} | "
            f"{(str(cold_kc) + ' / ' + str(stored_kc)):25s} | "
            f"{(str(cold_rb) + ' / ' + str(stored_rb)):23s} | "
            f"{(str(cold_bpm) + ' / ' + str(stored_bpm)):18s}"
        )

    print("=" * 90)
    print(f"Parity Results over 20 tracks:")
    print(f"  kick_class:       {kc_matches}/20 equal ({kc_matches/20*100:.1f}%) [Required >= 18]")
    print(f"  restraint_bucket: {rb_matches}/20 equal ({rb_matches/20*100:.1f}%) [Required >= 18]")
    print(f"  felt_bpm (oct):   {bpm_matches}/20 equal ({bpm_matches/20*100:.1f}%) [Required >= 18]")
    print("=" * 90)

    assert kc_matches >= 18, f"kick_class parity failed: {kc_matches}/20 matches (required >= 18)"
    assert rb_matches >= 18, f"restraint_bucket parity failed: {rb_matches}/20 matches (required >= 18)"
    assert bpm_matches >= 18, f"felt_bpm octave parity failed: {bpm_matches}/20 matches (required >= 18)"
