"""Series 19 — shared machinery (worker functions must live in a real module so
ProcessPoolExecutor can pickle them by reference; works with fork from a plain
single-threaded driver — DO NOT fork from inside ipykernel, it livelocks).

Trial a (19a): closed-loop Voigt-target sweep. The ONLY change vs 18c is on the
DATA side (Voigt-broadened targets). The simulator/optimizer stays pure Lorentzian.
"""
import math

import numpy as np
import torch

from src.fitting import nll, fwhm_from_theta, fit_profile
from src.samplers import sample_cauchy_truncated
from src.implicit import compute_fwhm_and_dgamma

torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------------------
# Simulation-side worker (identical to 17g/18c): pure Lorentzian + implicit diff
# ---------------------------------------------------------------------------
def _fit_fn(ph):
    return fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)

def _fwhm_fn(th):
    return fwhm_from_theta(th, model='lorentzian')

def _nll_fn(th, ph):
    return nll(th, ph, model='lorentzian', uniform_bg=False)

def run_one(args):
    """(FWHM, sigma_fit, dFWHM/dgamma, dsigma/dgamma) of one simulated Lorentzian scan."""
    gamma_val, u, b = args
    return compute_fwhm_and_dgamma(gamma_val, u, b, _fit_fn, _fwhm_fn, _nll_fn, n_params=2)

def init_worker():
    torch.set_num_threads(1)

def parallel_map(pool, tasks):
    return list(pool.map(run_one, tasks, chunksize=8))

# ---------------------------------------------------------------------------
# Data-side worker (19a): Voigt targets — Cauchy line + Gaussian broadening,
# window-truncated, fitted with the SAME Lorentzian MLE as the real pipeline.
# ---------------------------------------------------------------------------
def voigt_photons(gamma, nbar, sigma_prop, lam, sigma_g, rng, half=75.0):
    n = int(max(round(nbar + sigma_prop * rng.standard_normal()), 0))
    x = sample_cauchy_truncated(gamma, n, window=2 * half, rng=rng)  # Cauchy detunings
    if sigma_g > 0 and n > 0:
        x = x + sigma_g * torch.tensor(rng.standard_normal(n), dtype=torch.float32)
        keep = (x >= -half) & (x <= half)     # out-of-window photons are lost
        x = x[keep]
    m = int(rng.poisson(lam))
    b = torch.tensor(rng.uniform(-half, half, size=m), dtype=torch.float32)
    return torch.cat([x, b])

def crlb_sigma(theta, photons, reg=1e-4):
    """Per-scan fit uncertainty: sqrt(dF^T H^-1 dF / N) at the fitted theta (as implicit.py)."""
    n_params = len(theta)
    t = theta.detach().clone().requires_grad_()
    gt = torch.autograd.grad(
        nll(t, photons, model='lorentzian', uniform_bg=False), t, create_graph=True)[0]
    H = torch.zeros(n_params, n_params, dtype=torch.float32)
    for i in range(n_params):
        gi = gt[i]
        for j in range(n_params):
            retain = not (i == n_params - 1 and j == n_params - 1)
            H[i, j] = torch.autograd.grad(gi, t, retain_graph=retain, create_graph=False)[0][j].detach()
    H_reg = H + reg * torch.eye(n_params, dtype=torch.float32)
    dF = torch.autograd.grad(fwhm_from_theta(t, model='lorentzian'), t, create_graph=False)[0]
    sig = torch.sqrt((dF @ torch.linalg.solve(H_reg, dF)) / max(len(photons), 1))
    return float(sig.clamp_min(1e-30))

def scan_features(ph, gamma_ref):
    """(FWHM, sigma_fit) of one scan; fallback identical to implicit.py's (2*gamma, 0.5*gamma)."""
    if len(ph) < 3:
        return 2.0 * gamma_ref, 0.5 * gamma_ref
    th = fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)
    if th is None:
        return 2.0 * gamma_ref, 0.5 * gamma_ref
    return float(fwhm_from_theta(th, model='lorentzian')), crlb_sigma(th, ph)

def target_one(args):
    """One Voigt target scan -> (FWHM, sigma_fit). Deterministic per (exp, sigma_g, idx, seed_base)."""
    exp, sigma_g, idx, seed_base = args
    rng = np.random.default_rng(seed_base + int(round(sigma_g * 100)) + idx * 1000003)
    ph = voigt_photons(exp['gamma_true'], exp['mu_true'], exp['sigma_prop'], exp['lam'], sigma_g, rng)
    return scan_features(ph, exp['gamma_true'])

def target_map(pool, tasks):
    return list(pool.map(target_one, tasks, chunksize=8))
