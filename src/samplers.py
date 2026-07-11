"""
Samplers — photon generation strategies for the differentiable MC pipeline.

Each sampler takes parameters (gamma, n, etc.) and returns a set of photon
detunings (positions). The key difference between methods is how randomness
is handled for differentiability.
"""

import torch
import numpy as np


# --- Detection window constants ---
FREQ_MIN = -75.0
FREQ_MAX = 75.0
FREQ_RANGE = FREQ_MAX - FREQ_MIN   # 150 MHz


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


def sample_cauchy_truncated(gamma, n_photons, window=FREQ_RANGE, rng=None):
    """
    Reparameterized Cauchy with truncation to a finite detection window.

    Samples are truncated to [-window/2, window/2] using inverse transform
    sampling of the truncated Cauchy. All samples land inside the window
    and are a smooth function of gamma.

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
    gamma_t = torch.as_tensor(gamma, dtype=torch.float32)
    # Truncated Cauchy CDF bounds
    u_min = 0.5 + torch.atan(torch.tensor(-half / gamma_t)) / torch.pi
    u_max = 0.5 + torch.atan(torch.tensor(half / gamma_t)) / torch.pi
    u = torch.tensor(rng.uniform(float(u_min), float(u_max), n_photons), dtype=torch.float32)
    return gamma_t * torch.tan(torch.pi * (u - 0.5))


def sample_cauchy_masked(gamma, n_photons, window=FREQ_RANGE, rng=None):
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


# =========================================================================
# Truncated signal sampling (used by all differentiable notebooks)
# =========================================================================

def signal_detunings(gamma, u, window=FREQ_RANGE):
    """
    Map frozen quantiles `u` to window-truncated Cauchy signal detunings.

    Uses inverse transform of the Cauchy TRUNCATED to [-window/2, window/2].
    Keeps the photon set count fixed (= len(u)) — no rejection loop needed.
    All positions are smooth functions of gamma (differentiable).

    Parameters
    ----------
    gamma : scalar tensor
        Lorentzian HWHM (MHz).
    u : tensor (N,)
        Frozen uniform(0,1) quantiles.
    window : float
        Detection window full width.

    Returns
    -------
    detunings : tensor (N,)
    """
    half = 0.5 * window
    # Truncated Cauchy quantile function:
    # F_t^{-1}(u) = gamma * tan(atan(L/gamma) * (2*u - 1))
    return gamma * torch.tan(torch.atan(half / gamma) * (2.0 * u - 1.0))


def build_photons(gamma, u, b, window=FREQ_RANGE):
    """
    Build one run's photons: window-truncated signal at `gamma` + background.

    Parameters
    ----------
    gamma : scalar tensor
        Lorentzian HWHM.
    u : tensor (N,)
        Frozen signal quantiles.
    b : tensor (M,)
        Background detunings (already drawn uniformly).
    window : float
        Detection window full width.

    Returns
    -------
    photons : tensor (N + M,)
    """
    return torch.cat([signal_detunings(gamma, u, window), b])


def draw_fixed_noise(nbar, sigma, lambda_, rng):
    """
    Draw and FREEZE one run's parameter-free randomness.

    Returns (u, b, N) where:
        u: (N,) uniform quantiles for signal photons (frozen)
        b: (M,) background detunings (frozen)
        N: int — number of signal photons
    """
    n = int(max(round(nbar + sigma * rng.standard_normal()), 0))
    u = torch.tensor(rng.uniform(0.0, 1.0, size=n), dtype=torch.float32)
    b = torch.tensor(rng.uniform(FREQ_MIN, FREQ_MAX, size=int(rng.poisson(lambda_))),
                     dtype=torch.float32)
    return u, b, n
