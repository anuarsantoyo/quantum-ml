"""
Implicit differentiation through the L-BFGS fit.

Uses the implicit function theorem to compute dFWHM/dgamma:
    dFWHM/dgamma = -(dFWHM/dtheta) @ H_theta_theta^{-1} @ H_theta_gamma

where H_theta_theta is the Hessian of the NLL w.r.t. fit parameters at the optimum,
and H_theta_gamma is the mixed derivative w.r.t. fit parameters and gamma.
"""

import math
import torch
import numpy as np


def compute_fwhm_and_dgamma(gamma_val, u, b, fit_fn, fwhm_fn, nll_fn,
                             n_params=3, reg=1e-4, eps_g=1e-3):
    """
    Compute FWHM and dFWHM/dgamma for a single PLE scan.

    Args:
        gamma_val: float — Lorentzian HWHM
        u: np.array — frozen uniform noise for signal photons
        b: np.array — frozen uniform noise for background photons
        fit_fn: callable(photons) -> theta*
        fwhm_fn: callable(theta) -> FWHM
        nll_fn: callable(theta, photons) -> scalar loss
        n_params: int — number of fitted parameters (3 for Lorentzian)
        reg: float — Tikhonov regularization strength
        eps_g: float — FD step for gamma

    Returns:
        fwhm: float — fitted FWHM
        sigma_fwhm: float — uncertainty estimate from Hessian
        df_dgamma: float — dFWHM/dgamma
    """
    n_sig, n_bg = len(u), len(b)
    if n_sig + n_bg < 3:
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0

    from src.samplers import build_photons
    # u, b are numpy arrays -> convert to tensors
    u_t = torch.as_tensor(u, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)
    photons_t = build_photons(
        torch.tensor(gamma_val, dtype=torch.float32),
        u_t, b_t)

    # ---- Step 1: L-BFGS fit ----
    theta = fit_fn(photons_t)
    if theta is None:
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0

    theta_star = theta.detach().clone()
    fwhm_star = float(fwhm_fn(theta.detach()))

    # ---- Step 2: Gradient of NLL w.r.t theta at (theta_star, gamma±eps) ----
    def _grad_theta_at_gamma(g_val):
        t = theta_star.detach().clone().requires_grad_()
        p = build_photons(
            torch.tensor(g_val, dtype=torch.float32),
            u_t, b_t)
        loss = nll_fn(t, p)
        return torch.autograd.grad(loss, t, create_graph=False)[0]

    grad_hi = _grad_theta_at_gamma(gamma_val + eps_g)
    grad_lo = _grad_theta_at_gamma(gamma_val - eps_g)
    H_theta_gamma = (grad_hi - grad_lo) / (2 * eps_g)  # (n_params,)

    # ---- Step 3: Hessian H_theta_theta at theta_star ----
    t_for_hess = theta_star.detach().clone().requires_grad_()
    loss_for_hess = nll_fn(t_for_hess, photons_t)
    H_theta_theta = torch.zeros(n_params, n_params, dtype=torch.float32)
    grad_t = torch.autograd.grad(loss_for_hess, t_for_hess, create_graph=True)[0]
    for i in range(n_params):
        gi = grad_t[i]
        for j in range(n_params):
            retain = not (i == n_params - 1 and j == n_params - 1)
            H_theta_theta[i, j] = torch.autograd.grad(
                gi, t_for_hess, retain_graph=retain, create_graph=False
            )[0][j].detach()

    # ---- Step 4: dFWHM / dtheta at theta_star ----
    t_for_df = theta_star.detach().clone().requires_grad_()
    fwhm_at_theta = fwhm_fn(t_for_df)
    dF_dtheta = torch.autograd.grad(fwhm_at_theta, t_for_df, create_graph=False)[0]

    # ---- Step 5: Solve: dFWHM/dgamma = -dF_dtheta @ H_inv @ H_theta_gamma ----
    H_reg = H_theta_theta + reg * torch.eye(n_params, dtype=torch.float32)
    try:
        H_inv = torch.linalg.solve(H_reg, torch.eye(n_params, dtype=torch.float32))
    except RuntimeError:
        H_inv = torch.linalg.pinv(H_reg)

    df_dgamma = -(dF_dtheta.unsqueeze(0) @ H_inv @ H_theta_gamma.unsqueeze(1)).squeeze().item()
    if not math.isfinite(df_dgamma):
        df_dgamma = 2.0

    # Sigma from CRLB: sqrt(dF/dtheta @ H_inv @ dF/dtheta^T / N_data)
    sigma_fwhm = torch.sqrt(
        ((dF_dtheta.unsqueeze(0) @ H_inv @ dF_dtheta.unsqueeze(1)).squeeze().clamp_min(1e-30))
        / max(len(photons_t), 1)
    ).item()
    if not math.isfinite(sigma_fwhm):
        sigma_fwhm = 0.5 * gamma_val

    return fwhm_star, sigma_fwhm, df_dgamma
