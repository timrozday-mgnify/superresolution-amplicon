"""Presence evidence: greedy claims, calibrated spill-over, leftover test, drop-one LRT."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import presence_evidence as pe  # noqa: E402  (needs local bin directory)


def test_presence_evidence_separates_real_members_from_spill_over() -> None:
    """An abundant member (label 0) spills ~0.5% onto its one-edit neighbours. A rare
    member far above that spill is present; one at spill level is not detected; two
    identical strains whose own amplicon holds only spill are one group, not detected,
    and lose the label they share with the abundant member to it."""
    table = pd.DataFrame({
        "genome_id": ["big", "strainA", "strainA", "strainB", "strainB", "rare", "absent"],
        "source": ["sb", "s1", "sb", "s1", "sb", "sr", "sa"],
        "weight": [1.0, 0.75, 0.25, 0.75, 0.25, 1.0, 1.0],
    })
    home = {"sb": 0, "s1": 1, "sr": 2, "sa": 3}
    source_ids = ["sb", "s1", "sr", "sa"]
    calibration = list(range(4, 14))                   # non-panel one-edit neighbours of sb
    labels = [1, 2, 3] + calibration
    neighbours = (source_ids, np.zeros(len(labels), int), np.array(labels), np.ones(len(labels), int))
    counts = np.zeros(14)
    counts[0], counts[1], counts[2], counts[3] = 100_000, 300, 3_000, 400
    counts[calibration] = [100, 200, 300, 400, 500, 150, 250, 350, 450, 50]

    ev, cal = pe.evaluate("S", counts, table, home, neighbours)
    ev = ev.set_index("genome_id")
    assert cal["spill_pairs_d1"] == 10 and 0.004 < cal["spill_rate_d1"] < 0.005
    assert ev.at["big", "status"] == "present" and ev.at["big", "presence_posterior"] > 0.99
    assert ev.at["rare", "status"] == "present" and ev.at["rare", "presence_posterior"] > 0.99
    assert ev.at["absent", "status"] == "not_detected"
    assert ev.at["absent", "detection_limit_reads"] > ev.at["absent", "own_reads"]
    # The strains: one indistinguishable group, credited only their own amplicon's reads.
    assert ev.at["strainA", "group"] == ev.at["strainB", "group"] == "strainA+strainB"
    assert ev.at["strainA", "status"] == "not_detected"
    assert ev.at["strainA", "free_labels"] == 1 and ev.at["strainA", "own_reads"] == 300
    assert ev.at["strainA", "presence_posterior"] < 0.5
