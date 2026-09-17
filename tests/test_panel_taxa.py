"""Taxon panel entries: resolving a taxon to database V4 groups, and reporting per entry."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import build_panel_kernel as bpk  # noqa: E402  (needs local bin directory)
import infer_composition as ic  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)

BAC = "Bacteria;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae"
LINEAGE = {                                   # header -> (V4 sequence, lineage)
    "frag|0|frag": ("AAAA", f"{BAC};Bacteroides"),
    "unif|0|unif": ("CCCC", f"{BAC};Bacteroides"),
    "ovat|0|ovat": ("GGGG", f"{BAC};Bacteroides"),
    "ovat|1|ovat": ("GGGG", f"{BAC};Bacteroides"),          # a second copy, same group
    "ambi|0|ambi": ("GNGG", f"{BAC};Bacteroides"),           # MAPseq cannot place it
    "phoc|0|phoc": ("TTTT", f"{BAC};Phocaeicola"),
    "shar|0|shar": ("CCCC", f"{BAC};Phocaeicola"),           # shares unif's group
    "stre|0|stre": ("ACAC", "Bacteria;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus"),
    "euk|0|euk":   ("AGAG", "Eukaryota;Opisthokonta;Streptococcus"),   # a homonym
}


def _db() -> tuple[list[str], list[str], np.ndarray, dict[str, str]]:
    headers = list(LINEAGE)
    ids, label_of_ref = np.unique([bpk.v4g(s) for s, _ in LINEAGE.values()], return_inverse=True)
    return headers, ids.tolist(), label_of_ref, {h: t for h, (_, t) in LINEAGE.items()}


def _members(taxa, genome_sources=(), max_sources=200):
    genome_sources = {*genome_sources, g("GNGG")}           # as prepare excludes it
    headers, ids, label_of_ref, lineage_of = _db()
    return bpk.taxon_members(taxa, headers, ids, label_of_ref, lineage_of,
                             set(genome_sources), max_sources)


def g(seq: str) -> str:
    return bpk.v4g(seq)


def test_lineage_prefix_and_unique_bare_name_resolve() -> None:
    got = _members([("bact", f"{BAC};Bacteroides"), ("phoc", "Phocaeicola")])
    assert got == {"bact": sorted([g("AAAA"), g("CCCC"), g("GGGG")]),
                   "phoc": sorted([g("TTTT"), g("CCCC")])}      # CCCC shared, unnested


def test_ambiguous_or_unknown_taxon_is_an_error_listing_candidates() -> None:
    with pytest.raises(SystemExit, match="Eukaryota;Opisthokonta;Streptococcus"):
        _members([("strep", "Streptococcus")])
    with pytest.raises(SystemExit, match="no database lineage prefix"):
        _members([("x", "Bacteria;Bacteroidota;Nope")])


def test_nested_taxa_go_to_the_most_specific_and_genomes_keep_their_sources() -> None:
    got = _members([("family", BAC), ("bact", f"{BAC};Bacteroides")],
                   genome_sources=[g("AAAA")])
    assert got == {"bact": sorted([g("CCCC"), g("GGGG")]),       # AAAA is the genome's
                   "family": sorted([g("TTTT"), g("CCCC")])}     # other Bacteroidaceae


def test_a_taxon_over_the_cap_fails_with_its_count() -> None:
    with pytest.raises(SystemExit, match="resolves to 3 V4 groups"):
        _members([("bact", "Bacteroides")], max_sources=2)


def test_prepare_writes_taxon_members_beside_genomes(tmp_path: Path) -> None:
    db = tmp_path / "amplicons.fasta"
    db.write_text("".join(f">{h}\n{s}\n" for h, (s, _) in LINEAGE.items()))
    tax = tmp_path / "db.tax"
    tax.write_text("#levels: Kingdom Genus\n"
                   + "".join(f"{h}\t{t}\n" for h, (_, t) in LINEAGE.items()))
    taxa = tmp_path / "taxa.tsv"
    taxa.write_text("id\ttaxon\nbact\tBacteroides\n")
    bad = tmp_path / "bad.tsv"
    bad.write_text("background\tBacteroides\n")
    args = dict(panel_amplicons=None, db_amplicons=db, db_taxonomy=tax, alias=[],
                max_taxon_sources=200, fwd_primer="", rev_primer="", max_mismatch=2)
    with pytest.raises(SystemExit, match="background"):
        bpk.prepare(SimpleNamespace(**args, panel_taxa=bad, out=tmp_path / "bad"))
    prep = tmp_path / "prep"
    bpk.prepare(SimpleNamespace(**args, panel_taxa=taxa, out=prep))

    tr = pd.read_csv(prep / "panel_translation.tsv", sep="\t")
    groups = sorted([g("AAAA"), g("CCCC"), g("GGGG")])
    assert tr.genome_id.tolist() == [f"bact::{x}" for x in groups]
    assert (tr.weight == 1.0).all() and tr.source.tolist() == groups
    assert pd.read_csv(prep / "sources.tsv", sep="\t").in_db.all()
    assert dict(bpk.si.read_fasta(prep / "sources.fasta"))[g("GGGG")] == "GGGG"


def test_prepare_whole_database_reuses_groups_and_copy_weights(tmp_path: Path) -> None:
    amplicon_dir = tmp_path / "database_amplicons"
    amplicon_dir.mkdir()
    records = {
        "g1|0|a": "AAAA",
        "g1|1|b": "AAAA",
        "g2|0|c": "CCCC",
    }
    (amplicon_dir / "amplicons.fasta").write_text(
        "".join(f">{header}\n{sequence}\n" for header, sequence in records.items()))
    pd.DataFrame({
        "genome_id": ["g1", "g1", "g2"],
        "refseq": list(records),
        "weight": [0.25, 0.75, 1.0],
    }).to_csv(amplicon_dir / "translation_table.tsv", sep="\t", index=False)

    prep = tmp_path / "prepared"
    bpk.prepare(SimpleNamespace(
        panel_amplicons=None, panel_taxa=None, whole_database=amplicon_dir,
        db_amplicons=None, db_taxonomy=None, alias=[], max_taxon_sources=200,
        fwd_primer="not-used", rev_primer="not-used", max_mismatch=2, out=prep,
    ))

    translation = pd.read_csv(prep / "panel_translation.tsv", sep="\t")
    assert translation.to_dict("records") == [
        {"genome_id": "g1", "source": g("AAAA"), "weight": 1.0},
        {"genome_id": "g2", "source": g("CCCC"), "weight": 1.0},
    ]
    sources = pd.read_csv(prep / "sources.tsv", sep="\t")
    assert sources.in_db.all()
    assert dict(bpk.si.read_fasta(prep / "sources.fasta")) == {
        g("AAAA"): "AAAA",
        g("CCCC"): "CCCC",
    }


def _panel_run(tmp_path: Path, genome_ids: list[str], K: np.ndarray, per_label: dict,
               mode: str, **kw) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    K = np.column_stack([K, np.zeros(len(K))])       # the last label no source reaches
    headers = [f"h{i}|0|h{i}" for i in range(K.shape[1])]
    kernel = tmp_path / "kernel.npz"
    sources = [f"s{i}" for i in range(K.shape[0])]
    sm.write_kernel(kernel, sparse.csr_array(K), sources, [f"l{i}" for i in range(K.shape[1])],
                    K.argmax(axis=1), ref_headers=headers, label_of_ref=np.arange(K.shape[1]))
    prep = tmp_path / "prep"
    prep.mkdir()
    pd.DataFrame({"genome_id": genome_ids, "source": sources, "weight": 1.0}).to_csv(
        prep / "panel_translation.tsv", sep="\t", index=False)
    obs = tmp_path / "obs.mseq"
    obs.write_text("".join(f"r{h}{i}\t{headers[h]}\t250\t1.0\n"
                           for h, n in per_label.items() for i in range(n)))
    out = tmp_path / "out"
    ic.run(SimpleNamespace(
        amplicon_dir=prep, mismapping_matrix=kernel, sim_mseq=None, obs_mseq=[obs],
        min_identity=None, sample_id="S", mode=mode, alpha=1.0, steps=1500, lr=0.02,
        num_samples=400, warmup=0, no_mismapping=False, no_presence=True,
        presence_prior=0.5, presence_temp=0.5, seed=0, min_infer_reads=1, output_dir=out, **kw))
    return out


def test_entry_is_reported_from_summed_member_draws(tmp_path: Path) -> None:
    """Two members of one taxon with identical rows trade mass freely; their entry does not."""
    K = np.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.0, 0.1, 0.9]])
    out = _panel_run(tmp_path, ["tax::a", "tax::b", "gen"], K,
                     {0: 3000, 1: 1000, 2: 2700, 3: 300}, "vi")
    comp = pd.read_csv(out / "inferred_composition.csv", index_col="genome_id")
    members = pd.read_csv(out / "inferred_panel_members.csv", index_col="genome_id")
    assert comp.index.tolist() == ["gen", "tax", "background"]
    assert members.entry.tolist() == ["gen", "tax", "tax", "background"]
    assert comp.resolution.tax == "identifiable"                 # its own members may collide

    with np.load(out / "posterior_draws.npz") as npz:
        draws, ids = npz["theta_eff"], npz["genome_ids"].tolist()
    summed = draws[:, ids.index("tax::a")] + draws[:, ids.index("tax::b")]
    assert np.isclose(comp.inferred_mean.tax, summed.mean())
    assert np.allclose([comp.inferred_lo.tax, comp.inferred_hi.tax],
                       np.quantile(summed, [0.05, 0.95]))
    width = comp.inferred_hi.tax - comp.inferred_lo.tax
    member_width = (members.inferred_hi - members.inferred_lo)[["tax::a", "tax::b"]].sum()
    assert width < 0.5 * member_width, (width, member_width)
    assert abs(comp.inferred_mean.tax - 0.5) < 0.03, comp

    # The fit check reads the entry table against member draws.
    import json
    import check_composition_fit as ccf
    from test_composition_fit import _fit_args
    report = tmp_path / "fit.json"
    ccf.run(_fit_args(tmp_path / "prep", tmp_path / "kernel.npz", tmp_path / "obs.mseq",
                      out / "inferred_composition.csv", out / "posterior_draws.npz", report))
    assert json.loads(report.read_text())["kernel_format"] == "rectangular"


def test_a_one_group_taxon_fits_like_the_genome_it_equals(tmp_path: Path) -> None:
    K = np.array([[0.8, 0.2, 0.0], [0.0, 0.3, 0.7]])
    counts = {0: 2400, 1: 1500, 2: 1400, 3: 700}
    genome = _panel_run(tmp_path / "g", ["x", "y"], K, counts, "mle")
    taxon = _panel_run(tmp_path / "t", ["x::v4g_1", "y"], K, counts, "mle")
    a = pd.read_csv(genome / "inferred_composition.csv").inferred_mean.to_numpy()
    b = pd.read_csv(taxon / "inferred_composition.csv").inferred_mean.to_numpy()
    assert np.allclose(a, b, atol=1e-6), (a, b)
    assert not (genome / "inferred_panel_members.csv").exists()
