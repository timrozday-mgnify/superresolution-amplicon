"""What the square path did, carried over to panel inference: the presence gate, the
horseshoe, the --no-mismapping baseline, and the low-depth and no-hit results.

The resolution column and the latent distance decay are covered in
test_measured_mismapping.py (panel_inference_recovers..., panel_align_kernel_redecays...).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import check_composition_fit as ccf  # noqa: E402  (needs local bin directory)
import infer_composition as ic  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)
import subspecies_infer as si  # noqa: E402  (needs local bin directory)
from test_composition_fit import _fit_args  # noqa: E402  (needs tests on sys.path)


def _run(tmp_path: Path, K: np.ndarray, counts: list[int], genomes: list[str] | None = None,
         home: np.ndarray | None = None, **kw) -> Path:
    """One source per genome, one database label per column, ``counts`` reads per label."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_sources, n_labels = K.shape
    genomes = genomes or [f"g{i}" for i in range(n_sources)]
    sources = [f"s{i}" for i in range(n_sources)]
    headers = [f"h{i}|0|h{i}" for i in range(n_labels)]
    sm.write_kernel(tmp_path / "kernel.npz", sparse.csr_array(K), sources,
                    [f"l{i}" for i in range(n_labels)],
                    K.argmax(axis=1) if home is None else home,
                    ref_headers=headers, label_of_ref=np.arange(n_labels))
    prep = tmp_path / "prep"
    prep.mkdir()
    pd.DataFrame({"genome_id": genomes, "source": sources, "weight": 1.0}).to_csv(
        prep / "panel_translation.tsv", sep="\t", index=False)
    (tmp_path / "obs.mseq").write_text("".join(
        f"r{h}_{i}\t{headers[h]}\t250\t1.0\n" for h, n in enumerate(counts) for i in range(n)))
    args = dict(
        amplicon_dir=prep, mismapping_matrix=tmp_path / "kernel.npz",
        obs_mseq=[tmp_path / "obs.mseq"], min_identity=None, sample_id="S", mode="vi",
        alpha=0.5, steps=1500, lr=0.05, num_samples=200, warmup=0, no_mismapping=False,
        no_presence=True, presence_prior=si.DEFAULT_PRESENCE_PRIOR,
        presence_temp=si.DEFAULT_PRESENCE_TEMP, seed=0, min_infer_reads=1,
        output_dir=tmp_path / "out")
    args.update(kw)
    ic.run(SimpleNamespace(**args))
    return tmp_path / "out"


def _composition(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "inferred_composition.csv", index_col="genome_id")


def _diagnostics(out: Path) -> pd.Series:
    return pd.read_csv(out / "inference_diagnostics.csv").iloc[0]


def test_presence_gate_calls_an_unobserved_genome_absent(tmp_path: Path) -> None:
    K = np.array([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0], [0.0, 0.05, 0.95]])
    comp = _composition(_run(tmp_path, K, [6000, 4000, 0], no_presence=False))
    assert (comp.presence_prob[["g0", "g1"]] > si.PRESENCE_THRESHOLD).all(), comp
    assert comp.presence_prob.g2 < si.PRESENCE_THRESHOLD, comp
    assert comp.inferred_mean.g2 < 0.01, comp

    off = _composition(_run(tmp_path / "off", K, [6000, 4000, 0]))
    assert off.presence_prob.isna().all(), off                  # no call, not "absent"


def test_horseshoe_recovers_a_relabelled_source_from_its_secondary_leak(tmp_path: Path) -> None:
    """MAPseq never labels a on its own label (90% -> b, 10% -> c), so only c's reads betray
    it. c as a source would also leak onto e's label, which drew no reads, so a is the only
    consistent explanation; the horseshoe must keep a and shrink c. Twenty pure sources at
    exact counts pin the overdispersion, as a real sample's many labels do."""
    n_pure = 20
    K = np.eye(5 + n_pure)
    K[:5, :5] = [[0, .9, .1, 0, 0], [0, 1, 0, 0, 0], [0, 0, .5, 0, .5], [0, 0, 0, 1, 0],
                 [0, 0, 0, .1, .9]]
    out = _run(tmp_path, K, [0, 5700, 300, 4000, 0] + [500] * n_pure,
               home=np.arange(5 + n_pure), horseshoe=True, steps=8000, num_samples=100)
    comp = _composition(out)
    # Truth: a 0.15, b 0.15, d 0.20, pure sources 0.50 (a's 3,000 reads of 20,000).
    assert abs(comp.inferred_mean.g0 - 0.15) < 0.03, comp
    assert comp.inferred_mean.g2 < 0.01, comp
    assert bool(_diagnostics(out).horseshoe)


def test_no_mismapping_labels_every_read_with_its_source_home(tmp_path: Path) -> None:
    """Truth 0.5 / 0.5 through K lands 0.4 / 0.6 on the labels. The corrected fit recovers
    the truth; the uncorrected baseline reports the labels."""
    K = np.array([[0.8, 0.2], [0.0, 1.0]])
    corrected = _composition(_run(tmp_path / "k", K, [4000, 6000], mode="mle"))
    baseline_out = _run(tmp_path / "b", K, [4000, 6000], mode="mle", no_mismapping=True)
    baseline = _composition(baseline_out)
    assert abs(corrected.inferred_mean.g0 - 0.5) < 0.02, corrected
    assert abs(baseline.inferred_mean.g0 - 0.4) < 0.02, baseline
    assert not _diagnostics(baseline_out).use_mismapping


def test_low_depth_writes_a_gated_zero_composition(tmp_path: Path) -> None:
    out = _run(tmp_path, np.eye(2), [15, 15], min_infer_reads=1000)
    comp = _composition(out)
    assert set(comp.fit_status) == {"low_depth"} and (comp.inferred_mean == 0).all(), comp
    diag = _diagnostics(out)
    assert diag.status == "low_depth" and diag.fit_status == "low_depth", diag
    with np.load(out / "posterior_draws.npz") as posterior:
        assert posterior["theta_eff"].shape == (0, 3)            # g0, g1, background

    report = tmp_path / "fit.json"
    ccf.run(_fit_args(tmp_path / "prep", tmp_path / "kernel.npz", tmp_path / "obs.mseq",
                      out / "inferred_composition.csv", out / "posterior_draws.npz", report))
    assert json.loads(report.read_text())["fit_status"] == "low_depth"


def test_no_reference_hits_is_a_result_not_a_crash(tmp_path: Path) -> None:
    out = _run(tmp_path, np.eye(2), [0, 0], min_infer_reads=1000)
    comp = _composition(out)
    assert comp.index.tolist() == ["g0", "g1", "background"], comp
    assert (comp.inferred_mean == 0).all() and (comp.observed_rel_abundance == 0).all(), comp
    diag = _diagnostics(out)
    assert diag.status == "no_reference_hits" and diag.n_reads == 0, diag
