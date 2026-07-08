"""
Samplers — photon generation strategies for the differentiable MC pipeline.

Each sampler takes parameters (gamma, n, etc.) and returns a set of photon
detunings (positions). The key difference between methods is how randomness
is handled for differentiability.
"""

import torch
import numpy as np


def sample_cauchy_reparam(gamma, n_photons, rng=None):
    """
    Reparameterized Cauchy sampling.

    All randomness is isolated in uniform noise u. gamma enters as a
    smooth, differentiable multiplier — gradients flow through it.

    x = gamma · tan(pi · (u - 0.5)),  u ~ Uniform(0, 1)

    Parameters
    ----------
    gamma : float or tensor
        Half-width at half-maximum (HWHM) of the Cauchy.
    n_photons : int
        Number of photons to sample.
    rng : numpy random Generator, optional

    Returns
    -------
    detunings : tensor, shape (n_photons,)
    """
    if rng is None:
        rng = np.random.default_rng()
    u = torch.tensor(rng.uniform(0, 1, n_photons), dtype=torch.float32)
    return gamma * torch.tan(torch.pi * (u - 0.5))


def sample_cauchy_truncated(gamma, n_photons, window=150.0, rng=None):
    """
    Reparameterized Cauchy with truncation to a finite detection window.

    Samples are truncated to [-window/2, window/2]. The truncation is
    handled by rejection sampling in the inverse CDF domain: only uniform
    draws that map inside the window are accepted.

    Parameters
    ----------
    gamma : float or tensor
        HWHM of the Cauchy.
    n_photons : int
        Number of photons to sample.
    window : float
        Full width of the detection window (MHz).
    rng : numpy random Generator, optional

    Returns
    -------
    detunings : tensor, shape (n_photons,)
    """
    if rng is None:
        rng = np.random.default_rng()
    half = window / 2.0
    # CDF: F(x) = 0.5 + (1/pi) * arctan(x / gamma)
    # F(-half) and F(half) bound the uniform range
    u_min = 0.5 + torch.atan(torch.tensor(-half / gamma, dtype=torch.float32)) / torch.pi
    u_max = 0.5 + torch.atan(torch.tensor(half / gamma, dtype=torch.float32)) / torch.pi

    u = torch.tensor(rng.uniform(u_min.item(), u_max.item(), n_photons), dtype=torch.float32)
    return gamma * torch.tan(torch.pi * (u - 0.5))


def sample_cauchy_masked(gamma, n_photons, window=150.0, rng=None):
    """
    Reparameterized Cauchy with masking instead of truncation.

    Sample from the full Cauchy, then mask (zero out) photons that fall
    outside the detection window. The mask is non-differentiable but can
    be handled via STE or ignored for small rejection rates.

    Parameters
    ----------
    gamma : float or tensor
        HWHM of the Cauchy.
    n_photons : int
        Number of photons to attempt to sample.
    window : float
        Full width of the detection window (MHz).
    rng : numpy random Generator, optional

    Returns
    -------
    detunings : tensor, shape (n_photons,)
    mask : tensor, shape (n_photons,), bool — True for in-window photons
    """
    detunings = sample_cauchy_reparam(gamma, n_photons, rng)
    half = window / 2.0
    mask = (detunings >= -half) & (detunings <= half)
    return detunings, mask


def sample_normal(mu, sigma, n_samples, rng=None):
    """
    Simple normal sampler with reparameterization.

    x = mu + sigma * eps,  eps ~ N(0, 1)

    Parameters
    ----------
    mu : float or tensor
    sigma : float or tensor
    n_samples : int
    rng : numpy random Generator, optional

    Returns
    -------
    samples : tensor, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = torch.tensor(rng.normal(0, 1, n_samples), dtype=torch.float32)
    return mu + sigma * eps


def sample_poisson_reparam(lam, n_samples, rng=None):
    """
    Poisson sampling via Gaussian approximation + rounding (differentiable
    via STE or surrogate gradients).

    For large lambda, Poisson(lam) ≈ N(lam, sqrt(lam)). We use this
    approximation and round to integers.

    Parameters
    ----------
    lam : float or tensor
        Rate parameter of the Poisson.
    n_samples : int
    rng : numpy random Generator, optional

    Returns
    -------
    counts : tensor, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = torch.tensor(rng.normal(0, 1, n_samples), dtype=torch.float32)
    raw = lam + torch.sqrt(torch.abs(lam) + 1e-8) * eps
    return torch.round(torch.clamp(raw, min=0))
