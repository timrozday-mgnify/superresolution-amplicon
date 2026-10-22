"""numpy_vi, the torch-free ``--mode vi`` fit: its gradient is torch's, and its fit is pyro's.

Nov2025's 53 samples over 10 seeds each (dev/numpy_vi.md) are the real check. These are the
quick ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy_vi as nv  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)
from test_panel_inference import _composition, _run  # noqa: E402  (needs tests on sys.path)

# Three sources leaking into each other's labels, the third genome with two copies.
K = sparse.csr_array(np.array([[0.90, 0.08, 0.02, 0.0],
                               [0.10, 0.85, 0.05, 0.0],
                               [0.00, 0.10, 0.90, 0.0],
                               [0.00, 0.05, 0.85, 0.1],
                               [0.00, 0.00, 0.00, 1.0]]))      # background row onto U
HOME = np.array([0, 1, 2, 2, 3])


def _problem(use_presence: bool) -> nv.Problem:
    return nv.Problem(K, HOME, sm.home_entries(K, HOME), np.array([0, 1, 2, 2, 3]),
                      np.array([1.0, 1.0, 0.5, 0.5, 1.0]), 4, np.array([3000, 2500, 1200, 40]),
                      use_presence=use_presence)


def _torch_loss(problem: nv.Problem, params: np.ndarray, noise: np.ndarray):
    """``-ELBO`` built from sramp's own torch model code and pyro's distributions."""
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("pyro.distributions")
    import subspecies_infer as si
    from torch.distributions.transforms import StickBreakingTransform

    f64 = torch.float64
    p, eps = torch.tensor(params, requires_grad=True), torch.tensor(noise)
    sl, g1, sp = problem.slices, problem.G - 1, torch.nn.functional.softplus
    loc, sc = p[sl["loc_theta"]], sp(p[sl["raw_theta"]])
    ut = loc + sc * eps[:g1]
    sb = StickBreakingTransform()
    theta = sb(ut)
    loss = (dist.Normal(loc, sc).log_prob(ut).sum() - sb.log_abs_det_jacobian(ut, theta)
            - dist.Dirichlet(problem.alpha * torch.ones(problem.G, dtype=f64)).log_prob(theta))
    latent = {}
    for i, (name, mu, sigma) in enumerate((("s", 0.0, problem.s_sigma), ("c", nv.OD_LOC, nv.OD_SCALE))):
        loc_, sc_ = p[sl[f"loc_{name}"]][0], sp(p[sl[f"raw_{name}"]][0])
        u = loc_ + sc_ * eps[g1 + i]
        latent[name] = u.exp()
        loss = (loss + dist.Normal(loc_, sc_).log_prob(u) - u
                - dist.LogNormal(torch.tensor(mu, dtype=f64), torch.tensor(sigma, dtype=f64))
                .log_prob(latent[name]))
    if problem.use_presence:
        lq, u = p[sl["z_logits"]], eps[g1 + 2:]
        temp = torch.tensor(problem.temp, dtype=f64)
        q = dist.RelaxedBernoulli(temp, logits=lq)
        z = q.transforms[0]((u.log() - (-u).log1p() + lq) / temp)
        prior = dist.RelaxedBernoulli(temp, probs=torch.full((problem.G,), 0.01, dtype=f64))
        loss = loss + q.log_prob(z).sum() - prior.log_prob(z).sum()
        theta = z * theta / (z * theta).sum()
    coo = K.tocoo()
    M = (torch.sparse_coo_tensor(torch.tensor(np.vstack([coo.row, coo.col])),
                                 torch.tensor(coo.data), size=K.shape).coalesce(),
         torch.tensor(problem.diagonal), None, torch.tensor(HOME))
    r = si._reference_distribution(theta, (torch.tensor(problem.gor), torch.tensor(problem.weight),
                                           problem.G))
    r_obs = si._apply_mismapping(r / r.sum(), M, latent["s"])
    conc = latent["c"] * problem.n * r_obs / r_obs.sum() + 1e-6
    loss = loss - dist.DirichletMultinomial(concentration=conc, total_count=int(problem.n)).log_prob(
        torch.tensor(problem.y))
    loss.backward()
    return float(loss), p.grad.numpy()


@pytest.mark.parametrize("use_presence", [True, False])
def test_gradient_is_torchs_on_the_same_noise(use_presence: bool) -> None:
    problem = _problem(use_presence)
    rng = np.random.default_rng(0)
    for noise in problem.noise(rng, 5):
        params = problem.init_params([0.4, 0.3, 0.2, 0.1]) + 0.3 * rng.standard_normal(problem.n_params)
        loss, grad = problem.loss_and_grad(params, noise)
        ref_loss, ref_grad = _torch_loss(problem, params, noise)
        assert loss == pytest.approx(ref_loss, rel=1e-10)
        assert np.allclose(grad, ref_grad, rtol=1e-9, atol=1e-9), (grad, ref_grad)


def test_numpy_backend_matches_pyro_through_infer_composition(tmp_path: Path) -> None:
    """Same run, both backends: the means agree to within VI's seed noise, the calls exactly.

    One seed's g0 has a spread of about 0.012 here for either backend (40 seeds each: pyro
    0.614, numpy 0.619), so a difference of two fits has a spread of about 0.017."""
    pytest.importorskip("pyro")
    Kd = np.array([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0], [0.0, 0.05, 0.95]])
    kw = dict(no_presence=False, steps=1500)
    pyro = _composition(_run(tmp_path / "pyro", Kd, [6000, 4000, 0], **kw))
    numpy = _composition(_run(tmp_path / "numpy", Kd, [6000, 4000, 0], backend="numpy", **kw))
    assert np.allclose(numpy.inferred_mean, pyro.inferred_mean, atol=0.05), (numpy, pyro)
    assert ((numpy.presence_prob >= 0.5) == (pyro.presence_prob >= 0.5)).all(), (numpy, pyro)
    assert numpy.inferred_lo.notna().all()


def test_numpy_backend_refuses_what_it_does_not_fit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--mode vi only"):
        _run(tmp_path, np.eye(2), [500, 500], backend="numpy", mode="mle")


def test_loss_falls() -> None:
    problem = _problem(True)
    _, point, diag, losses = nv.fit(problem, [0.25] * 4, steps=1500, lr=0.05, num_samples=200)
    assert np.mean(losses[-100:]) < np.mean(losses[:100])
    assert pd.Series(diag["presence_prob"]).between(0, 1).all()
    assert point.sum() == pytest.approx(1.0)
