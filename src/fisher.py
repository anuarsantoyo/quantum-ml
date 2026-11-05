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
import multiprocessing as _mp
import os
from concurrent.futures import ProcessPoolExecutor as _PPE

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


def _run_one(args):
    """One simulated scan -> (FWHM, sigma_fit, dFWHM/dgamma, dsigma/dgamma).

    Module-level so ProcessPoolExecutor can pickle it by reference. The frozen
    noise (u, b) is drawn in the PARENT and shipped in ``args``; only the fit
    runs here (single-threaded via ``_init_worker``).
    """
    gamma_val, u, b = args
    return compute_fwhm_and_dgamma(gamma_val, u, b, _fit_fn, _fwhm_fn, _nll_fn, n_params=2)


def _init_worker():
    torch.set_num_threads(1)


def _parallel_map(pool, tasks):
    return list(pool.map(_run_one, tasks, chunksize=8))


def make_pool(n_workers=None):
    """fork-based ProcessPoolExecutor (one process per core, each single-threaded).

    Matches the series-24 pattern. Works from a plain script or an nbconvert/papermill
    driver; do NOT use from an interactive ipykernel (fork livelocks there).
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    return _PPE(max_workers=n_workers, mp_context=_mp.get_context('fork'),
                initializer=_init_worker)


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


def sim_cloud(mu, sigma_prop, lam, gamma, n_runs, seed, pool=None):
    """Simulate one cloud of ``(FWHM, sigma_fit, n, dFWHM/dgamma, dsigma/dgamma)``
    at ``(mu, gamma)`` (frozen noise, seed-controlled).

    With ``pool`` (a ProcessPoolExecutor from :func:`make_pool`) the fits run
    across cores; the frozen noise (u, b) is always drawn in the PARENT so the
    result is seed-reproducible regardless of worker count.
    """
    rng = np.random.default_rng(seed)
    if pool is not None:
        tasks, ns = [], []
        for _ in range(n_runs):
            u, b, n = draw_fixed_noise(mu, sigma_prop, lam, rng)
            tasks.append((gamma, u.numpy(), b.numpy()))
            ns.append(n)
        res = _parallel_map(pool, tasks)
    else:
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


def fisher_gaussian(cloud, n_data):
    """Bandwidth-free Gaussian-summary linewidth (gamma) uncertainty.

    Assumes the fitted FWHM is ~N(m, sigma_fwhm^2) across scans (the CLT
    approximation for a fitted statistic). The sensitivity of the summary to
    gamma is dFWHM/dgamma (already computed per scan), so

        sigma_gamma = sigma_fwhm / (mean|dFWHM/dgamma| * sqrt(N_data))

    i.e. the measured scatter divided by the sensitivity. No bandwidth.
    Returns per-scan and dataset sigma_gamma plus the FWHM cloud's skewness and
    excess kurtosis, so the Gaussian assumption can be checked directly.
    """
    ft, _si, _nt, dg, _ds = cloud
    f = np.asarray(ft.numpy(), dtype=float)
    sigma_fwhm = float(f.std())
    j = float(np.asarray(dg.numpy(), dtype=float).mean())
    j = j if abs(j) > 1e-9 else 1.0
    sig_g_single = sigma_fwhm / abs(j)
    sig_g_data = sig_g_single / math.sqrt(max(int(n_data), 1))
    c = f - f.mean()
    m2 = float((c ** 2).mean())
    m3 = float((c ** 3).mean())
    m4 = float((c ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 - 3.0 if m2 > 0 else 0.0
    return dict(sigma_fwhm=sigma_fwhm, j=j, sig_g_single=sig_g_single,
                sig_g_data=sig_g_data, skew=skew, kurt=kurt)


def gaussian_summary(cloud, target_f, target_s, mu, sigma_prop):
    """2D Gaussian-summary loss (mean + spread) and its gradients.

    Assumes the summary x = (FWHM, sigma_fit) is ~ N(m(theta), Sigma(theta)) across
    scans. The per-data-point negative log-likelihood under this Gaussian is

        L = 0.5 [ log|Sigma| + tr(Sigma^-1 S) + (xbar - m)^T Sigma^-1 (xbar - m) ]

    where (xbar, S) are the data's mean/covariance and (m, Sigma) the simulated
    cloud's. gamma's gradient is PATHWISE (through the per-scan implicit derivatives
    dFWHM/dgamma, dsigma/dgamma); mu's gradient is the SCORE IDENTITY (count score
    weighted by 2D-Gaussian responsibilities). No bandwidth anywhere.

    Returns dict(loss, grad_mu, grad_gamma, m, Sigma, xbar, S, ...).
    """
    ft, si, nt, dg, ds = [t.float() for t in cloud]
    M = len(ft)
    data = torch.stack([target_f.float(), target_s.float()], dim=1)  # (N, 2)
    N = len(data)

    xbar = data.mean(dim=0)                                  # (2,)
    dc = data - xbar                                         # (N, 2)
    S = (dc.t() @ dc) / N + 1e-6 * torch.eye(2)              # biased data covariance

    sim = torch.stack([ft, si], dim=1)                       # (M, 2)
    m = sim.mean(dim=0)                                      # (2,)
    sc = sim - m                                             # (M, 2)
    Sigma = (sc.t() @ sc) / M + 1e-6 * torch.eye(2)          # biased sim covariance

    Sigma_inv = torch.linalg.inv(Sigma)
    logdet = torch.logdet(Sigma)
    res = xbar - m
    loss = 0.5 * (logdet + (Sigma_inv * S).sum() + res @ Sigma_inv @ res)

    # --- gamma gradient (pathwise) ---
    dl_dm = -(Sigma_inv @ res)                               # (2,)
    dl_dSigma = 0.5 * (Sigma_inv - Sigma_inv @ S @ Sigma_inv
                       - Sigma_inv @ torch.outer(res, res) @ Sigma_inv)  # (2,2)

    def _cov(a, b):  # biased covariance (1/M), matching Sigma's normalisation
        a = a - a.mean(); b = b - b.mean()
        return (a * b).sum() / M

    dm_dg = torch.stack([dg.mean(), ds.mean()])              # (2,)
    dSigma_dg = torch.zeros(2, 2)
    dSigma_dg[0, 0] = 2.0 * _cov(ft, dg)
    dSigma_dg[1, 1] = 2.0 * _cov(si, ds)
    dSigma_dg[0, 1] = _cov(ft, ds) + _cov(si, dg)
    dSigma_dg[1, 0] = dSigma_dg[0, 1]
    grad_gamma = float((dl_dm @ dm_dg).item() + (dl_dSigma * dSigma_dg).sum().item())

    # --- mu gradient (score identity, 2D Gaussian responsibilities) ---
    diff = data[:, None, :] - sim[None, :, :]                # (N, M, 2)
    mahal = ((diff @ Sigma_inv) * diff).sum(dim=-1)          # (N, M) Mahalanobis^2
    W = torch.exp(-0.5 * mahal)
    w = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)      # (N, M) responsibilities
    score = (nt - mu) / sigma_prop ** 2                      # (M,)
    grad_mu = -float((w @ score).sum().item())

    return dict(loss=float(loss), grad_mu=grad_mu, grad_gamma=grad_gamma,
                m=m.detach(), Sigma=Sigma.detach(), xbar=xbar.detach(), S=S.detach(),
                fwhm_mean=float(m[0]), sf_mean=float(m[1]),
                fwhm_data_mean=float(xbar[0]), sf_data_mean=float(xbar[1]))


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
