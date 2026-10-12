"""Home labels when MAPseq's hit is a database entry the primers do not amplify."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import build_panel_kernel as bpk  # noqa: E402  (needs local bin directory)


def _home(tmp_path, rows):
    mseq = tmp_path / "home.mseq"
    mseq.write_text("".join(f"{q}\t{hit}\t0\t{ident}\n" for q, hit, ident in rows))
    src = {"in_db": 0, "not_in_db": 1, "relabelled": 2}
    label_of_hit = {"full|0": 0, "other|0": 1}        # amplifiable entries; "trunc|0" is not
    own = {"in_db": 0, "relabelled": 0}               # sources whose sequence is a label
    return bpk._home_labels(mseq, src, label_of_hit, own)


def test_hit_without_a_label_keeps_the_sources_own_group(tmp_path):
    home, identity = _home(tmp_path, [("in_db", "trunc|0", "0.99"),
                                      ("not_in_db", "trunc|0", "0.99"),
                                      ("relabelled", "other|0", "0.996")])
    assert home[0] == 0 and identity[0] == 1.0     # its own group, though MAPseq hit "trunc"
    assert home[1] == -1                           # nothing to fall back to
    assert home[2] == 1                            # a real relabel below identity 1 stands
    assert np.isnan(identity[1])


def test_unhit_source_keeps_its_own_group(tmp_path):
    home, _ = _home(tmp_path, [("in_db", "", "")])
    assert home[0] == 0
