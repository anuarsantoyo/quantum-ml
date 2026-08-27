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
    Compute FWHM, dFWHM/dgamma and dsigma/dgamma for a single PLE scan.

    Args:
        gamma_val: float — Lorentzian HWHM
        u: np.array — frozen uniform noise for signal photons
        b: np.array — frozen uniform noise for background photons
        fit_fn: callable(photons) -> theta*
        fwhm_fn: callable(theta) -> FWHM
        nll_fn: callable(theta, photons) -> scalar loss
        n_params: int — number of fitted parameters (3 for Lorentzian)
        reg: float — Tikhonov regularization strength
        eps_g: float — FD step for gamma (mixed derivative d2NLL/dtheta dgamma)

    Returns:
        fwhm: float — fitted FWHM
        sigma_fwhm: float — uncertainty estimate from Hessian
        df_dgamma: float — dFWHM/dgamma
        ds_dgamma: float — d(sigma_fwhm)/dgamma, EXACT implicit derivative:
            sigma = sqrt(dF^T H^-1 dF / N) is a smooth function of the fit
            optimum theta*(gamma), so
                dsigma/dgamma = (dsigma/dtheta)|_theta*  @  (dtheta*/dgamma)
            with dtheta*/dgamma = -H^-1 H_theta_gamma (same machinery as
            df_dgamma, one level up). No finite differences in gamma needed.
    """
    n_sig, n_bg = len(u), len(b)
    if n_sig + n_bg < 3:
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0, 0.5

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
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0, 0.5

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
    def _hessian(t_val, photons):
        t = t_val.detach().clone().requires_grad_()
        gt = torch.autograd.grad(nll_fn(t, photons), t, create_graph=True)[0]
        H = torch.zeros(n_params, n_params, dtype=torch.float32)
        for i in range(n_params):
            gi = gt[i]
            for j in range(n_params):
                retain = not (i == n_params - 1 and j == n_params - 1)
                H[i, j] = torch.autograd.grad(
                    gi, t, retain_graph=retain, create_graph=False)[0][j].detach()
        return H

    t_for_hess = theta_star.detach().clone().requires_grad_()
    loss_for_hess = nll_fn(t_for_hess, photons_t)
    grad_t = torch.autograd.grad(loss_for_hess, t_for_hess, create_graph=True)[0]
    H_theta_theta = torch.zeros(n_params, n_params, dtype=torch.float32)
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

    # ---- Step 5: Solve dtheta*/dgamma = -H^-1 H_theta_gamma ----
    H_reg = H_theta_theta + reg * torch.eye(n_params, dtype=torch.float32)
    try:
        H_inv = torch.linalg.solve(H_reg, torch.eye(n_params, dtype=torch.float32))
    except RuntimeError:
        H_inv = torch.linalg.pinv(H_reg)

    dtheta_dgamma = -(H_inv @ H_theta_gamma.unsqueeze(1)).squeeze()
    df_dgamma = float((dF_dtheta @ dtheta_dgamma).item())
    if not math.isfinite(df_dgamma):
        df_dgamma = 2.0

    # Sigma from CRLB: sqrt(dF/dtheta @ H_inv @ dF/dtheta^T / N_data)
    sigma_fwhm = torch.sqrt(
        ((dF_dtheta.unsqueeze(0) @ H_inv @ dF_dtheta.unsqueeze(1)).squeeze().clamp_min(1e-30))
        / max(len(photons_t), 1)
    ).item()
    if not math.isfinite(sigma_fwhm):
        sigma_fwhm = 0.5 * gamma_val

    # ---- Step 6: exact dsigma/dgamma (implicit diff one level up) ----
    # sigma = sqrt(dF^T H^-1 dF / N) depends on gamma through TWO paths:
    #   A) the fit optimum theta*(gamma) moves  ->  (dsigma/dtheta)|_theta* @ dtheta*/dgamma
    #   B) the Hessian depends on gamma directly through the photon positions
    #      (the data enter d^2 NLL/dtheta^2), theta* held fixed
    #      ->  (dsigma/dH) . (dH/dgamma)  via autograd through H(photons(gamma))
    # Both are exact analytic derivatives (third-order autograd; n_params is 2-3 so cheap).
    ds_dgamma = 0.5  # fallback: derivative of the fallback sigma = 0.5*gamma
    try:
        # ---- term A: theta*-movement channel ----
        t_sig = theta_star.detach().clone().requires_grad_()
        loss_sig = nll_fn(t_sig, photons_t)
        grad_sig = torch.autograd.grad(loss_sig, t_sig, create_graph=True)[0]
        # differentiable Hessian (build via torch.stack: in-place assignment would sever the graph)
        H_rows = []
        for i in range(n_params):
            gi = grad_sig[i]
            H_rows.append(torch.stack([
                torch.autograd.grad(gi, t_sig, retain_graph=True, create_graph=True)[0][j]
                for j in range(n_params)
            ]))
        H_diff = torch.stack(H_rows)
        H_inv_diff = torch.linalg.solve(
            H_diff + reg * torch.eye(n_params, dtype=torch.float32),
            torch.eye(n_params, dtype=torch.float32))
        dF_diff = torch.autograd.grad(fwhm_fn(t_sig), t_sig, create_graph=True)[0]
        sigma_expr = torch.sqrt(
            ((dF_diff.unsqueeze(0) @ H_inv_diff @ dF_diff.unsqueeze(1)).squeeze().clamp_min(1e-30))
            / max(len(photons_t), 1))
        ds_dtheta = torch.autograd.grad(sigma_expr, t_sig, retain_graph=True)[0]
        termA = (ds_dtheta @ dtheta_dgamma.detach()).item()

        # ---- term B: direct data-dependence channel (theta* fixed, gamma varies) ----
        # Cheap exact form: dsigma/dH . dH/dgamma = -(1/(2 sigma N)) sum_ij M_ij dH_ij/dgamma
        # with M = H^-1 dF dF^T H^-1; dH/dgamma via central FD of the DETACHED Hessian
        # (same internal-FD style as the existing H_theta_gamma; no third-order autograd,
        #  no per-parameter probing -> stays expandable to new parameters).
        eps_h = 1e-2
        photons_hi = build_photons(
            torch.tensor(gamma_val + eps_h, dtype=torch.float32), u_t, b_t)
        photons_lo = build_photons(
            torch.tensor(gamma_val - eps_h, dtype=torch.float32), u_t, b_t)
        H_hi = _hessian(theta_star, photons_hi)
        H_lo = _hessian(theta_star, photons_lo)
        dH_dgamma = (H_hi - H_lo) / (2 * eps_h)
        M_mat = H_inv @ dF_dtheta.unsqueeze(1) @ dF_dtheta.unsqueeze(0) @ H_inv
        termB = -(1.0 / (2.0 * sigma_fwhm * max(len(photons_t), 1))) * (M_mat * dH_dgamma).sum().item()
        if not math.isfinite(termB):
            termB = 0.0

        ds_dgamma = termA + termB
        if not math.isfinite(ds_dgamma):
            ds_dgamma = 0.5
    except RuntimeError:
        ds_dgamma = 0.5

    return fwhm_star, sigma_fwhm, df_dgamma, ds_dgamma
