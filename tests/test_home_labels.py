"""Home labels: unamplified hits, exact ties, and primer-flanked home queries."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def _tie(tmp_path, hit, label_of_hit):
    """One source, `src|0`'s own group (label 0), tied by MAPseq onto `hit` at identity 1."""
    mseq = tmp_path / "tie.mseq"
    mseq.write_text(f"src\t{hit}\t250\t1.0\n")
    return bpk._home_labels(mseq, {"src": 0}, label_of_hit, {"src": 0})


def test_an_exact_tie_goes_to_the_better_supported_label(tmp_path):
    """A tie means the mapper cannot separate the two labels, so real reads from this
    source land wherever its references do. The bigger group takes them.

    Keeping the source's own group regardless is what modelled half of V. parvula's
    reads onto a label that then drew 0.6% of them.
    """
    # The source's own group has 2 references, MAPseq's tied label has 4.
    label_of_hit = {"own|0": 0, "own|1": 0, "big|0": 1, "big|1": 1, "big|2": 1, "big|3": 1}
    home, identity = _tie(tmp_path, "big|0", label_of_hit)
    assert home[0] == 1 and identity[0] == 1.0       # merges into the bigger group


def test_an_exact_tie_on_a_thinner_label_keeps_the_sources_own_group(tmp_path):
    """The S. salivarius case: a malformed singleton scores 1 against everything it
    covers and must not win the home of the references real reads land on."""
    label_of_hit = {"own|0": 0, "own|1": 0, "own|2": 0, "singleton|0": 1}
    home, _ = _tie(tmp_path, "singleton|0", label_of_hit)
    assert home[0] == 0

    # A draw keeps the source's own group too: nothing argues for moving it.
    home, _ = _tie(tmp_path, "other|0", {"own|0": 0, "other|0": 1})
    assert home[0] == 0


def _prepare(tmp_path, **flags):
    """One genome whose amplicon ACGTACGTAC sits between primers AAGGY and GGTTW."""
    panel = tmp_path / "panel"
    panel.mkdir(exist_ok=True)
    (panel / "g.amplicons.fasta").write_text(">g|0\nTTAAGGCACGTACGTACAAACCTT\n")
    db = tmp_path / "db.fasta"
    db.write_text(">ref|0\nACGTACGTAC\n")
    out = tmp_path / f"prep{len(flags)}"
    bpk.prepare(SimpleNamespace(panel_amplicons=panel, panel_taxa=None, db_amplicons=db,
                                fwd_primer="AAGGY", rev_primer="GGTTW", max_mismatch=0,
                                alias=[], out=out, **flags))
    return dict(bpk.si.read_fasta(out / "home_queries.fasta"))


def test_home_queries_carry_the_primers_only_when_asked(tmp_path):
    """AAP's merged reads keep both primers (the OTU path trims none before MAPseq),
    and MAPseq places them differently: V. parvula's 2-copy allele lands on its 1-copy
    allele's references only with the primers attached. The home query must match."""
    source = bpk.v4g("ACGTACGTAC")
    assert _prepare(tmp_path) == {source: "ACGTACGTAC"}             # = sources.fasta
    assert _prepare(tmp_path, home_query_primers=True) == {
        source: "AAGGY" + "ACGTACGTAC" + "WAACC"}                     # primers as given


def test_a_primer_flanked_home_on_a_neighbour_merges_the_source(tmp_path):
    """V. parvula 90a5 with the AAP primers attached hits b9af's KF842187 at 0.983. It is
    below identity 1, so it is MAPseq's relabel and stands: 90a5 shares b9af's home."""
    mseq = tmp_path / "home.mseq"
    mseq.write_text("v4g_90a5\tKF842187\t286\t0.9828\nv4g_b9af\tLG086562\t289\t0.9931\n")
    label_of_hit = {"KF842187": 0, "LG086562": 0, "JQ460182": 1}
    home, _ = bpk._home_labels(mseq, {"v4g_90a5": 0, "v4g_b9af": 1}, label_of_hit,
                               {"v4g_b9af": 0, "v4g_90a5": 1})
    assert home.tolist() == [0, 0]
