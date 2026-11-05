"""Fisher / Cramér-Rao uncertainty machinery for the differentiable-MC pipeline.

Single audited implementation (series 25) extracted from the series-23/24
notebooks and ``agent-guided-hypopt/experiment_*/ag_hypopt.py``, where it had
drifted across copies (e.g. 23a used inflated bandwidths, 23b plain Scott).

The likelihood is the 2D KDE over the simulated ``(FWHM, sigma_fit)`` cloud,
evaluated at the target (data) points. The Fisher information is the empirical
outer product of the per-point scores, averaged over the target points::

    J = (1/N) sum_i s(x_i) s(x_i)^T ,   s = grad_theta log p(x_i | theta)

The per-scan CRB is ``J^-1``; the dataset CRB (uncertainty on the estimate from
all N scans) is ``J^-1 / N`` (= per-scan / sqrt(N)).

Bandwidths follow Scott's rule from the TARGET data spread, with two scale knobs
``h_f_scale`` / ``h_s_scale`` and a floor ``h_s_min`` on the sigma kernel.

The gamma score has two forms: the RAW ``1/h^2`` chain (``gamma_scale=False``) is
the TRUE log-likelihood score and is what the Fisher uses; the z-form ``1/h``
(``gamma_scale=True``) is only an optimizer step-size rescaling.
"""

import math
import os

import numpy as np
import torch

from src.fitting import fit_profile, fwhm_from_theta, nll
from src.implicit import compute_fwhm_and_dgamma
from src.samplers import draw_fixed_noise

H_REF = 1.0  # reference scale for the z-form gamma score (optimizer only)


def _fit_fn(ph):
    return fit_profile(ph, n_iters=80, model="lorentzian", uniform_bg=False)


def _fwhm_fn(th):
    return fwhm_from_theta(th, model="lorentzian")


def _nll_fn(th, ph):
    return nll(th, ph, model="lorentzian", uniform_bg=False)


def kde_scores(sim_f, sim_s, sim_n, sim_df, sim_ds, data_f, data_s,
               h_f, h_s, mu, sigma_prop, gamma_scale=False, sigma_weight=1.0):
    """2D KDE log-likelihood + per-data-point scores.

    ``sim_*`` is the simulated cloud at the current parameters (one entry per
    simulated scan); ``data_*`` is the target. Returns
    ``(s_mu, s_gamma, nll, w, W)``.
    """
    sw = float(sigma_weight)
    d_f = data_f[:, None] - sim_f[None, :]
    d_s = data_s[:, None] - sim_s[None, :]
    W = torch.exp(-0.5 * (d_f / h_f) ** 2 - 0.5 * sw * (d_s / h_s) ** 2)
    w = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)
    score = (sim_n[None, :] - mu) / sigma_prop ** 2
    s_mu = (w * score).sum(dim=1)
    if gamma_scale:
        # z-form: ONE power of bandwidth (unitless distances), fixed reference.
        dlogG = ((d_f * sim_df[None, :]) / h_f + sw * (d_s * sim_ds[None, :]) / h_s) / H_REF
    else:
        # raw form: the true log-likelihood score.
        dlogG = (d_f * sim_df[None, :]) / h_f ** 2 + sw * (d_s * sim_ds[None, :]) / h_s ** 2
    s_gamma = (w * dlogG).sum(dim=1)
    logp = torch.log((W.sum(dim=1) / len(sim_f)).clamp_min(1e-30))
    return s_mu, s_gamma, -logp.mean(), w, W


def bandwidths(target_f, target_s, h_f_scale=1.0, h_s_scale=1.0, h_s_min=0.05):
    """Scott's rule from the target spread, scaled (fixed w.r.t. mu/gamma)."""
    scott = len(target_f) ** (-1.0 / 6.0)
    H_F = h_f_scale * float(target_f.std()) * scott
    H_S = max(h_s_scale * float(target_s.std()) * scott, h_s_min)
    return H_F, H_S


def sim_cloud(mu, sigma_prop, lam, gamma, n_runs, seed):
    """Simulate one cloud of ``(FWHM, sigma_fit, n, dFWHM/dgamma, dsigma/dgamma)``
    at ``(mu, gamma)`` (frozen noise, seed-controlled)."""
    rng = np.random.default_rng(seed)
    ns, res = [], []
    for _ in range(n_runs):
        u, b, n = draw_fixed_noise(mu, sigma_prop, lam, rng)
        r = compute_fwhm_and_dgamma(gamma, u.numpy(), b.numpy(),
                                    _fit_fn, _fwhm_fn, _nll_fn, n_params=2)
        res.append(r)
        ns.append(n)
    return (torch.tensor([r[0] for r in res], dtype=torch.float32),
            torch.tensor([r[1] for r in res], dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor([r[2] for r in res], dtype=torch.float32),
            torch.tensor([r[3] for r in res], dtype=torch.float32))


def fisher_at(cloud, target_f, target_s, h_f, h_s, mu, sigma_prop,
              gamma_scale=False, sigma_weight=1.0):
    """Fisher -> CRB of the KDE likelihood at ``(mu, gamma)`` for a pre-simulated cloud.

    Returns per-scan (``*_single``) and dataset (``*_data`` = /sqrt(N)) sigmas.
    """
    ft, si_t, nt, dg_t, ds_t = cloud
    N = int(len(target_f))
    s_mu, s_gamma, _, _, _ = kde_scores(ft, si_t, nt, dg_t, ds_t,
                                        target_f, target_s, h_f, h_s, mu, sigma_prop,
                                        gamma_scale=gamma_scale, sigma_weight=sigma_weight)
    s = torch.stack([s_mu, s_gamma], dim=1)
    J = s.T @ s / N
    inv = torch.linalg.inv(J + 1e-12 * torch.eye(2))
    inv_data = inv / N
    return dict(N=N, J=J.tolist(),
                sig_mu_single=float(math.sqrt(inv[0, 0])),
                sig_g_single=float(math.sqrt(inv[1, 1])),
                corr_single=float(inv[0, 1] / math.sqrt(inv[0, 0] * inv[1, 1])),
                sig_mu_data=float(math.sqrt(inv_data[0, 0])),
                sig_g_data=float(math.sqrt(inv_data[1, 1])),
                corr_data=float(inv_data[0, 1] / math.sqrt(inv_data[0, 0] * inv_data[1, 1])))


# ---------------------------------------------------------------------------
# Real-data target + experiment truth table (15-series / data_explanation.md)
# ---------------------------------------------------------------------------

REAL_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "processed", "fwhm_linewidths.csv")

# (power, trans, mu_true, sigma_prop, lam, gamma_true)
EXPERIMENTS = [
    ("1nW", 5, 9.393, 2.576, 2.232, 8.5),
    ("1nW", 10, 12.372, 3.445, 2.122, 8.5),
    ("1nW", 20, 17.316, 4.141, 2.286, 8.5),
    ("1nW", 40, 38.405, 7.198, 2.351, 8.5),
    ("1nW", 60, 61.374, 9.851, 2.593, 8.5),
    ("1nW", 80, 79.365, 12.627, 2.758, 8.5),
    ("1nW", 100, 70.817, 17.221, 2.636, 8.5),
    ("3nW", 5, 13.204, 3.724, 2.186, 14.1),
    ("3nW", 10, 24.476, 5.639, 2.158, 14.1),
    ("3nW", 20, 34.279, 8.319, 2.264, 14.1),
    ("3nW", 40, 84.892, 24.013, 2.475, 14.1),
    ("3nW", 60, 103.203, 23.95, 2.741, 14.1),
    ("3nW", 80, 137.537, 32.107, 2.911, 14.1),
    ("3nW", 100, 175.707, 40.975, 3.087, 14.1),
]


def load_real_target(power_nW, trans, csv_path=REAL_CSV):
    """Real measured target (FWHM, fit_error) in MHz for one experiment.

    Scales raw -> MHz (x1000) and keeps valid rows with ``err/fwhm < 10``
    (the 15-series / _load_real_target convention).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    sub = df[(df["power_nW"] == power_nW) & (df["transmission"] == trans)]
    f = sub["fwhm"].to_numpy(dtype=float) * 1000.0
    e = sub["fit_error"].to_numpy(dtype=float) * 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(f) & np.isfinite(e) & (f > 0)
        filt = ok & ((e / f) < 10.0)
    return (torch.tensor(f[filt], dtype=torch.float32),
            torch.tensor(e[filt], dtype=torch.float32))
