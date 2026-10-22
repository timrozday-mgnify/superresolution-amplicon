"""The default ``--mode vi`` panel fit in numpy alone, for where torch and pyro do not run.

Pyodide (CPython in the browser) ships numpy and scipy but neither torch nor pyro. This is
``subspecies_infer._fit`` for what ``infer_composition.run`` gives it by default: the
rectangular ``(K, home)`` kernel, the compact translation ``T``, the Dirichlet-Multinomial
likelihood, the mis-mapping scale ``s`` and, optionally, the presence gate. It is the same
ELBO as pyro's ``Trace_ELBO`` over the same guide, ``AutoNormal`` (stick-breaking for
``theta``, ``exp`` for ``s`` and ``conc_frac``, softplus scales initialised at 0.1) plus the
learnable ``RelaxedBernoulli`` presence guide, with the same clamps, optimised by the same
Adam. The gradient is written out by hand: the graph is a few dozen vector operations, so an
autodiff library would cost more than the arithmetic.

The noise comes from numpy's generator, not torch's, so a fit equals pyro's in distribution,
not draw for draw (``tests/test_numpy_vi.py`` checks the gradient against torch's autograd
on the same noise, and the fit against pyro's over seeds).
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, expit, gammaln

EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny
LOG_2PI = np.log(2.0 * np.pi)
INIT_SCALE = 0.1        # AutoNormal's default
INIT_LOGIT = 2.0        # subspecies_infer._presence_guide's
OD_LOC, OD_SCALE = 1.1, 1.0   # composition_model's conc_frac prior
ADAM_BETAS, ADAM_EPS = (0.9, 0.999), 1e-8


def _softplus(x):
    return np.logaddexp(0.0, x)


def _stick_breaking(u, offset_log):
    """torch's ``StickBreakingTransform``: ``(theta, log|det J|, z, clamp mask, x)``."""
    x = u - offset_log
    z = expit(x)
    inside = (z >= TINY) & (z <= 1.0 - EPS)
    z = np.clip(z, TINY, 1.0 - EPS)
    remaining = np.cumprod(1.0 - z)
    theta = np.append(z, 1.0) * np.append(1.0, remaining)
    ladj = float(np.sum(-_softplus(x) + np.log(theta[:-1])))   # -x + logsigmoid(x) = -softplus(x)
    return theta, ladj, z, inside, x


def _stick_breaking_inv(theta, offset_log):
    sf = np.clip(1.0 - np.cumsum(theta[:-1]), TINY, None)
    return np.log(theta[:-1]) - np.log(sf) + offset_log


def _stick_breaking_vjp(h, z, inside):
    """``dF/du`` given ``h[m] = dF/dtheta[m] * theta[m]`` (the log-space adjoint)."""
    after = np.cumsum(h[::-1])[::-1]          # after[k] = sum_{m >= k} h[m]
    return inside * ((1.0 - z) * h[:-1] - z * after[1:])


class Problem:
    """One sample's fit inputs, as ``infer_composition.run`` hands them to ``_fit``.

    ``K`` is the fitted rows x labels kernel (scipy sparse), ``home`` each row's home label,
    ``diagonal`` ``K[a, home[a]]``, ``genome_of_row``/``weight`` the compact ``T`` over
    ``G`` genomes, ``counts`` the reads per label.
    """

    def __init__(self, K, home, diagonal, genome_of_row, weight, G, counts, alpha=0.5,
                 s_sigma=0.3, use_presence=True, presence_prior=0.01, presence_temp=1.0):
        coo = K.tocoo()
        self.row, self.col, self.val = coo.row, coo.col, coo.data.astype(np.float64)
        self.R, self.L = K.shape
        self.home = np.asarray(home)
        self.diagonal = np.asarray(diagonal, dtype=np.float64)
        self.gor = np.asarray(genome_of_row)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.G = int(G)
        # composition_model rounds y_obs * N back to counts; ours are counts already.
        self.y = np.round(np.asarray(counts, dtype=np.float64))
        self.n = float(self.y.sum())
        self.alpha, self.s_sigma = float(alpha), float(s_sigma)
        self.use_presence, self.temp = use_presence, float(presence_temp)
        p = np.clip(presence_prior, EPS, 1.0 - EPS)
        self.prior_logit = float(np.log(p) - np.log1p(-p))
        self.offset_log = np.log(np.arange(self.G - 1, 0, -1, dtype=np.float64))
        # Constants of log p(y) and log p(theta); they shift the loss, not its gradient.
        self.const = (float(gammaln(self.n + 1.0) - gammaln(self.y + 1.0).sum())
                      + float(gammaln(self.G * self.alpha) - self.G * gammaln(self.alpha)))
        # Parameter layout: [loc_theta, raw_theta, loc_s, raw_s, loc_c, raw_c, z_logits].
        g1 = self.G - 1
        self.slices = {"loc_theta": slice(0, g1), "raw_theta": slice(g1, 2 * g1)}
        for i, name in enumerate(("loc_s", "raw_s", "loc_c", "raw_c")):
            self.slices[name] = slice(2 * g1 + i, 2 * g1 + i + 1)
        self.slices["z_logits"] = slice(2 * g1 + 4, 2 * g1 + 4 + (self.G if use_presence else 0))
        self.n_params = self.slices["z_logits"].stop
        self.noise_dim = g1 + 2 + (self.G if use_presence else 0)

    def init_params(self, theta_init):
        p = np.empty(self.n_params)
        raw = np.log(np.expm1(INIT_SCALE))
        sl = self.slices
        p[sl["loc_theta"]] = _stick_breaking_inv(np.asarray(theta_init, np.float64), self.offset_log)
        p[sl["raw_theta"]] = raw
        p[sl["loc_s"]], p[sl["raw_s"]] = 0.0, raw          # s starts at 1
        p[sl["loc_c"]], p[sl["raw_c"]] = np.log(3.0), raw  # conc_frac at the prior median
        p[sl["z_logits"]] = INIT_LOGIT
        return p

    def noise(self, rng, n):
        """``n`` draws of ``[eps_theta, eps_s, eps_c, u]``, ``u`` clamped as torch's."""
        g1 = self.G - 1
        eps = rng.standard_normal((n, g1 + 2))
        if not self.use_presence:
            return eps
        return np.hstack([eps, np.clip(rng.random((n, self.G)), EPS, 1.0 - EPS)])

    def _observe(self, theta_eff, s):
        """``r_obs`` (normalised) and what the backward pass needs."""
        r = theta_eff[self.gor] * self.weight
        r_sum = r.sum()
        rn = r / r_sum
        d = self.diagonal
        de_raw = 1.0 - s + s * d
        de = np.maximum(de_raw, 0.0)
        rs = de + s * (1.0 - d)
        a = rn * s / rs
        corr = rn * (de - s * d) / rs
        robs = (np.bincount(self.col, a[self.row] * self.val, minlength=self.L)
                + np.bincount(self.home, corr, minlength=self.L))
        ro_sum = robs.sum()
        return robs / ro_sum, (r_sum, rn, de_raw, de, rs, a, corr, ro_sum)

    def loss_and_grad(self, params, noise):
        """One-particle ``-ELBO`` and its gradient, as ``Trace_ELBO`` computes them."""
        sl, G, t = self.slices, self.G, self.temp
        g1 = G - 1
        grad = np.zeros_like(params)

        # Guide draws.
        loc_t, raw_t = params[sl["loc_theta"]], params[sl["raw_theta"]]
        sc_t = _softplus(raw_t)
        eps_t = noise[:g1]
        theta, ladj, zb, inside, x_t = _stick_breaking(loc_t + sc_t * eps_t, self.offset_log)
        loc_s, raw_s, loc_c, raw_c = (float(params[sl[k]][0]) for k in
                                      ("loc_s", "raw_s", "loc_c", "raw_c"))
        sc_s, sc_c = float(_softplus(raw_s)), float(_softplus(raw_c))
        eps_s, eps_c = noise[g1], noise[g1 + 1]
        us, uc = loc_s + sc_s * eps_s, loc_c + sc_c * eps_c
        s, cf = np.exp(us), np.exp(uc)

        # log q - log p, term by term. The Normal densities enter as -0.5 eps^2 - log sc.
        loss = (float(np.sum(-0.5 * eps_t ** 2 - np.log(sc_t))) - g1 * 0.5 * LOG_2PI - ladj
                - (self.alpha - 1.0) * float(np.log(theta).sum()))
        # s: q's -us and p's +us (LogNormal) cancel.
        loss += (-0.5 * eps_s ** 2 - np.log(sc_s) + np.log(self.s_sigma)
                 + us ** 2 / (2.0 * self.s_sigma ** 2))
        loss += (-0.5 * eps_c ** 2 - np.log(sc_c) + np.log(OD_SCALE)
                 + (uc - OD_LOC) ** 2 / (2.0 * OD_SCALE ** 2))

        if self.use_presence:
            lq = params[sl["z_logits"]]
            u = noise[g1 + 2:]
            x = (np.log(u) - np.log1p(-u) + lq) / t
            z_raw = expit(x)
            free = (z_raw >= EPS) & (z_raw <= 1.0 - EPS)
            z = np.clip(z_raw, EPS, 1.0 - EPS)
            xv = np.log(z) - np.log1p(-z)       # the value both densities are scored at
            dq, dp = lq - xv * t, self.prior_logit - xv * t
            # The sigmoid's Jacobian is common to q and p and cancels.
            loss += float(np.sum((dq - 2.0 * _softplus(dq)) - (dp - 2.0 * _softplus(dp))))
            w = z * theta
        else:
            w = theta
        w_sum = w.sum()
        theta_eff = w / w_sum

        # Likelihood.
        ro, (r_sum, rn, de_raw, de, rs, a, corr, ro_sum) = self._observe(theta_eff, s)
        conc = cf * self.n * ro + 1e-6
        A = conc.sum()
        loss -= (float(gammaln(A) - gammaln(self.n + A))
                 + float(np.sum(gammaln(self.y + conc) - gammaln(conc))) + self.const)

        # Backward.
        g_conc = -(digamma(A) - digamma(self.n + A) + digamma(self.y + conc) - digamma(conc))
        g_cf = float(np.dot(g_conc, ro)) * self.n
        g_ro = g_conc * (cf * self.n)
        g_robs = (g_ro - np.dot(g_ro, ro)) / ro_sum
        g_a = np.bincount(self.row, g_robs[self.col] * self.val, minlength=self.R)
        g_corr = g_robs[self.home]
        d = self.diagonal
        g_rn = (g_a * s + g_corr * (de - s * d)) / rs
        g_rs = -(g_a * a + g_corr * corr) / rs
        g_de = g_corr * rn / rs + g_rs
        g_s = float(np.sum(g_a * rn / rs - g_corr * rn * d / rs + g_rs * (1.0 - d)
                           + g_de * (d - 1.0) * (de_raw >= 0.0)))
        g_r = (g_rn - np.dot(g_rn, rn)) / r_sum
        g_te = np.bincount(self.gor, g_r * self.weight, minlength=G)
        g_w = (g_te - np.dot(g_te, theta_eff)) / w_sum
        if self.use_presence:
            g_theta = g_w * z
            g_z = g_w * theta
            fq, fp = 1.0 - 2.0 * expit(dq), 1.0 - 2.0 * expit(dp)
            grad[sl["z_logits"]] = np.where(free, fp + g_z * z * (1.0 - z) / t, fq)
        else:
            g_theta = g_w
        # Through the stick-breaking: the Dirichlet's -(alpha-1) log theta, the likelihood,
        # and -log|det J| = sum softplus(x) - sum log theta[:-1].
        h = g_theta * theta - (self.alpha - 1.0)
        h[:-1] -= 1.0
        g_ut = _stick_breaking_vjp(h, zb, inside) + expit(x_t)
        grad[sl["loc_theta"]] = g_ut
        grad[sl["raw_theta"]] = (g_ut * eps_t - 1.0 / sc_t) * expit(raw_t)
        g_us = us / self.s_sigma ** 2 + g_s * s
        g_uc = (uc - OD_LOC) / OD_SCALE ** 2 + g_cf * cf
        grad[sl["loc_s"]], grad[sl["raw_s"]] = g_us, (g_us * eps_s - 1.0 / sc_s) * expit(raw_s)
        grad[sl["loc_c"]], grad[sl["raw_c"]] = g_uc, (g_uc * eps_c - 1.0 / sc_c) * expit(raw_c)
        return float(loss), grad

    def draws(self, params, rng, num_samples):
        """Posterior draws of ``theta_eff``, ``s`` and ``conc_frac`` from the fitted guide."""
        sl, g1 = self.slices, self.G - 1
        noise = self.noise(rng, num_samples)
        u_t = params[sl["loc_theta"]] + _softplus(params[sl["raw_theta"]]) * noise[:, :g1]
        x = u_t - self.offset_log
        zb = np.clip(expit(x), TINY, 1.0 - EPS)
        rem = np.cumprod(1.0 - zb, axis=1)
        ones = np.ones((num_samples, 1))
        theta = np.hstack([zb, ones]) * np.hstack([ones, rem])
        if self.use_presence:
            u = noise[:, g1 + 2:]
            lq = params[sl["z_logits"]]
            z = np.clip(expit((np.log(u) - np.log1p(-u) + lq) / self.temp), EPS, 1.0 - EPS)
            theta = z * theta
        theta = theta / theta.sum(axis=1, keepdims=True)
        out = {"theta_eff": theta}
        for site, loc, raw, col in (("s", "loc_s", "raw_s", g1), ("conc_frac", "loc_c", "raw_c", g1 + 1)):
            out[site] = np.exp(params[sl[loc]][0] + _softplus(params[sl[raw]][0]) * noise[:, col])
        return out


def fit(problem: Problem, theta_init, *, steps=3000, lr=0.02, num_samples=500, seed=0):
    """SVI with Adam, then guide draws: ``(theta_draws, point, diag, losses)`` as ``_fit``."""
    rng = np.random.default_rng(seed)
    params = problem.init_params(theta_init)
    m, v = np.zeros_like(params), np.zeros_like(params)
    b1, b2 = ADAM_BETAS
    losses = np.empty(steps)
    noise = problem.noise(rng, steps)
    for step in range(steps):
        losses[step], g = problem.loss_and_grad(params, noise[step])
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        k = step + 1
        params = params - (lr / (1.0 - b1 ** k)) * m / (np.sqrt(v / (1.0 - b2 ** k)) + ADAM_EPS)
    drawn = problem.draws(params, rng, num_samples)
    theta = drawn["theta_eff"]
    diag: dict = {"final_loss": float(losses[-1])}
    if problem.use_presence:
        p = expit(params[problem.slices["z_logits"]])
        diag["presence_prob"] = p.tolist()
        diag["n_present"] = int((p >= 0.5).sum())
    diag["_posterior_draws"] = drawn
    return theta, theta.mean(axis=0), diag, losses.tolist()
