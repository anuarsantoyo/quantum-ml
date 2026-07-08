"""
Fitting — pseudo-Voigt profile fitting for optical linewidth extraction.

Provides differentiable log-pdf calculations (with window-normalized PDF
via Z factors) and MLE fitting for PLE spectroscopy data.
"""

import torch
import math


# --- Physical constants (detection window, MHz) ---
FREQ_MIN = -75.0
FREQ_MAX = 75.0
FREQ_RANGE = FREQ_MAX - FREQ_MIN   # 150 MHz
UNIFORM_DENSITY = 1.0 / FREQ_RANGE

# --- Width mapping constants ---
WIDTH_MAX = FREQ_RANGE      # 150 MHz
WIDTH_EPS = 1e-2            # tiny floor so widths stay strictly positive

# --- Pseudo-Voigt mixing ---
FIT_ETA = 0.2               # fixed mixing between Gaussian (eta) and Lorentzian (1-eta)


# =========================================================================
# Width mapping (unconstrained -> valid range)
# =========================================================================

def _width(raw):
    """Map an unconstrained raw param to a width in (WIDTH_EPS, WIDTH_EPS + WIDTH_MAX)."""
    return WIDTH_EPS + WIDTH_MAX * torch.sigmoid(raw)


def _raw_from_width(width):
    """Inverse of _width: map a constrained width back to raw space."""
    frac = (width - WIDTH_EPS) / WIDTH_MAX
    frac = torch.clamp(frac, 1e-10, 1 - 1e-10)
    return torch.logit(frac)


# =========================================================================
# CDFs for window normalization
# =========================================================================

def _lorentz_cdf(x, center, gamma):
    """CDF of a Lorentzian (Cauchy) at x."""
    return torch.atan((x - center) / gamma) / math.pi + 0.5


def _normal_cdf(x, center, sigma):
    """CDF of a normal distribution at x (erf-based)."""
    return 0.5 * (1.0 + torch.erf((x - center) / (sigma * math.sqrt(2.0))))


# =========================================================================
# Log-pdf
# =========================================================================

def log_pdf(freqs, center, raw_gamma, raw_sigma, logit_w=None, uniform_bg=True):
    """
    Log-pdf of the fitted model.

    The signal components (pseudo-Voigt: eta * Gaussian + (1-eta) * Lorentzian)
    are always normalized by their mass inside [FREQ_MIN, FREQ_MAX] via Z_l and Z_g.
    This ensures the likelihood is correct for photons sampled within the window.

    When uniform_bg=True, the full model is:
        w * signal_normalized + (1-w) * Uniform(window)

    When uniform_bg=False, the model is just the normalized signal (pure pseudo-Voigt).

    Parameters
    ----------
    freqs : tensor (N,)
        Photon detunings (MHz).
    center : scalar tensor
        Line center.
    raw_gamma : scalar tensor
        Unconstrained Lorentzian-width parameter (mapped via _width).
    raw_sigma : scalar tensor
        Unconstrained Gaussian-width parameter (mapped via _width).
    logit_w : scalar tensor or None
        Logit of signal/uniform weight. Ignored if uniform_bg=False.
    uniform_bg : bool
        If True, include uniform background component.

    Returns
    -------
    log_pdf : tensor (N,)
    """
    gamma = _width(raw_gamma)
    sigma_g = _width(raw_sigma)

    # --- Pseudo-Voigt signal components ---
    lorentz = (gamma / math.pi) / ((freqs - center) ** 2 + gamma ** 2)
    gauss = torch.exp(-0.5 * ((freqs - center) / sigma_g) ** 2) / (sigma_g * math.sqrt(2.0 * math.pi))

    # Window normalization (Z factors)
    hi = torch.as_tensor(FREQ_MAX, dtype=freqs.dtype)
    lo = torch.as_tensor(FREQ_MIN, dtype=freqs.dtype)
    Z_l = _lorentz_cdf(hi, center, gamma) - _lorentz_cdf(lo, center, gamma)
    Z_g = _normal_cdf(hi, center, sigma_g) - _normal_cdf(lo, center, sigma_g)

    signal = FIT_ETA * gauss / Z_g + (1.0 - FIT_ETA) * lorentz / Z_l

    # --- Optional uniform background ---
    if uniform_bg:
        w = torch.sigmoid(logit_w)
        pdf = w * signal + (1.0 - w) * torch.as_tensor(UNIFORM_DENSITY, dtype=freqs.dtype)
    else:
        pdf = signal

    return torch.log(pdf + 1e-30)


# =========================================================================
# Negative log-likelihood
# =========================================================================

def nll(theta, photons, uniform_bg=True):
    """
    Negative log-likelihood of photons under the fitted model.

    theta = (center, raw_gamma, raw_sigma[, logit_w])
    - If uniform_bg=True: theta has 4 elements.
    - If uniform_bg=False: theta has 3 elements.

    Parameters
    ----------
    theta : tensor (3,) or (4,)
    photons : tensor (N,)
    uniform_bg : bool

    Returns
    -------
    loss : scalar tensor
    """
    center = theta[0]
    raw_gamma = theta[1]
    raw_sigma = theta[2]
    logit_w = theta[3] if uniform_bg and len(theta) >= 4 else None

    lp = log_pdf(photons, center, raw_gamma, raw_sigma, logit_w, uniform_bg=uniform_bg)
    return -torch.sum(lp)


# =========================================================================
# FWHM extraction
# =========================================================================

def fwhm_from_theta(theta):
    """
    Extract the FWHM of the Lorentzian component from fitted theta.

    FWHM = 2 * gamma, where gamma is the Lorentzian HWHM.
    """
    raw_gamma = theta[1]
    gamma = _width(raw_gamma)
    return 2.0 * gamma


# =========================================================================
# Fitting (inner optimization)
# =========================================================================

def fit_pseudo_voigt(photons, n_iters=100, lr=0.5, uniform_bg=True):
    """
    Fit a pseudo-Voigt profile to photon detunings using L-BFGS.

    Parameters
    ----------
    photons : tensor (N,)
    n_iters : int
        Number of L-BFGS iterations.
    lr : float
        L-BFGS step size.
    uniform_bg : bool
        If True, include uniform background component in the model.

    Returns
    -------
    theta : tensor (3,) or (4,)
        Optimized parameters (center, raw_gamma, raw_sigma[, logit_w]).
    """
    n_params = 4 if uniform_bg else 3
    theta = torch.zeros(n_params, requires_grad=True)
    with torch.no_grad():
        theta[0] = 0.0        # center
        theta[1] = 0.0        # raw_gamma
        theta[2] = 0.0        # raw_sigma
        if uniform_bg:
            theta[3] = 2.0    # logit_w -> w ~ 0.88

    opt = torch.optim.LBFGS([theta], max_iter=n_iters, lr=lr)

    def closure():
        opt.zero_grad()
        loss = nll(theta, photons.detach(), uniform_bg=uniform_bg)
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()
