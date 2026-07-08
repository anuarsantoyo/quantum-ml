"""
Fitting — pseudo-Voigt profile fitting with optional truncation and uniform background.

Provides differentiable log-pdf calculations and MLE fitting for optical
linewidth extraction from PLE spectroscopy data.
"""

import torch
import math


# --- Physical constants (detection window, MHz) ---
FREQ_MIN = -75.0
FREQ_MAX = 75.0
FREQ_RANGE = FREQ_MAX - FREQ_MIN   # 150 MHz
UNIFORM_DENSITY = 1.0 / FREQ_RANGE

# --- Width mapping constants ---
WIDTH_MAX = FREQ_RANGE      # 150 MHz — a width wider than the window is meaningless
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
# CDFs for truncation normalization
# =========================================================================

def _lorentz_cdf(x, center, gamma):
    """CDF of a Lorentzian (Cauchy) at x."""
    return torch.atan((x - center) / gamma) / math.pi + 0.5


def _normal_cdf(x, center, sigma):
    """CDF of a normal distribution at x (erf-based)."""
    return 0.5 * (1.0 + torch.erf((x - center) / (sigma * math.sqrt(2.0))))


# =========================================================================
# Log-pdf functions (choose truncation + uniform background on/off)
# =========================================================================

def _signal_pdf(freqs, center, gamma, sigma_g, truncated=True):
    """
    Pseudo-Voigt signal component: eta * Gaussian + (1-eta) * Lorentzian.

    Parameters
    ----------
    freqs : tensor
        Photon detunings (MHz).
    center : tensor
        Line center.
    gamma : tensor
        Lorentzian HWHM (already mapped to valid range).
    sigma_g : tensor
        Gaussian std (already mapped to valid range).
    truncated : bool
        If True, normalize each component by its mass inside [FREQ_MIN, FREQ_MAX].

    Returns
    -------
    signal_pdf : tensor, same shape as freqs
    """
    lorentz = (gamma / math.pi) / ((freqs - center) ** 2 + gamma ** 2)
    gauss = torch.exp(-0.5 * ((freqs - center) / sigma_g) ** 2) / (sigma_g * math.sqrt(2.0 * math.pi))

    if truncated:
        hi = torch.as_tensor(FREQ_MAX, dtype=freqs.dtype)
        lo = torch.as_tensor(FREQ_MIN, dtype=freqs.dtype)
        Z_l = _lorentz_cdf(hi, center, gamma) - _lorentz_cdf(lo, center, gamma)
        Z_g = _normal_cdf(hi, center, sigma_g) - _normal_cdf(lo, center, sigma_g)
        lorentz = lorentz / Z_l
        gauss = gauss / Z_g

    return FIT_ETA * gauss + (1.0 - FIT_ETA) * lorentz


def log_pdf(freqs, center, raw_gamma, raw_sigma, logit_w=None,
            truncated=True, uniform_bg=True):
    """
    Log-pdf of the fitted model.

    Three modes controlled by `uniform_bg` and `truncated`:

    | uniform_bg | truncated | Model |
    |------------|-----------|-------|
    | True       | True      | w * signal_truncated + (1-w) * Uniform(window) |
    | True       | False     | w * signal + (1-w) * Uniform(window) |
    | False      | True      | signal_truncated (pure pseudo-Voigt) |
    | False      | False     | signal (pure pseudo-Voigt, no truncation) |

    Parameters
    ----------
    freqs : tensor (N,)
        Photon detunings.
    center : scalar tensor
    raw_gamma : scalar tensor
        Unconstrained Lorentzian-width parameter (mapped internally).
    raw_sigma : scalar tensor
        Unconstrained Gaussian-width parameter (mapped internally).
    logit_w : scalar tensor or None
        Logit of signal/uniform weight. If None, uniform_bg is treated as False.
    truncated : bool
        Whether to truncate the signal components to the detection window.
    uniform_bg : bool
        Whether to include the uniform background component.

    Returns
    -------
    log_pdf : tensor (N,)
    """
    gamma = _width(raw_gamma)
    sigma_g = _width(raw_sigma)
    signal = _signal_pdf(freqs, center, gamma, sigma_g, truncated=truncated)

    if uniform_bg and logit_w is not None:
        w = torch.sigmoid(logit_w)
        pdf = w * signal + (1.0 - w) * torch.as_tensor(UNIFORM_DENSITY, dtype=freqs.dtype)
    else:
        pdf = signal

    return torch.log(pdf + 1e-30)


# =========================================================================
# Negative log-likelihood
# =========================================================================

def nll(theta, photons, truncated=True, uniform_bg=True):
    """
    Negative log-likelihood of photons under the fitted model.

    theta = (center, raw_gamma, raw_sigma[, logit_w])
    - if uniform_bg is True, theta must have 4 elements
    - if uniform_bg is False, theta must have 3 elements

    Parameters
    ----------
    theta : tensor (3,) or (4,)
    photons : tensor (N,)
        Photon detunings.
    truncated : bool
    uniform_bg : bool

    Returns
    -------
    loss : scalar tensor
    """
    center = theta[0]
    raw_gamma = theta[1]
    raw_sigma = theta[2]

    if uniform_bg and len(theta) >= 4:
        logit_w = theta[3]
    else:
        logit_w = None

    lp = log_pdf(photons, center, raw_gamma, raw_sigma, logit_w,
                 truncated=truncated, uniform_bg=uniform_bg)
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

def fit_pseudo_voigt(photons, n_iters=100, lr=0.5, truncated=True, uniform_bg=True):
    """
    Fit a pseudo-Voigt profile to photon detunings using L-BFGS.

    Parameters
    ----------
    photons : tensor (N,)
    n_iters : int
        Number of L-BFGS iterations.
    lr : float
        L-BFGS step size.
    truncated : bool
    uniform_bg : bool

    Returns
    -------
    theta : tensor (3,) or (4,)
        Optimized parameters (center, raw_gamma, raw_sigma[, logit_w]).
    """
    n_params = 4 if uniform_bg else 3
    theta = torch.zeros(n_params, requires_grad=True)
    # Initialize: center=0, raw_gamma=~0, raw_sigma=~0, logit_w=0
    with torch.no_grad():
        theta[0] = 0.0        # center
        theta[1] = 0.0        # raw_gamma (sigmoid -> ~WIDTH_EPS + WIDTH_MAX/2)
        theta[2] = 0.0        # raw_sigma
        if uniform_bg:
            theta[3] = 2.0    # logit_w -> w ~ 0.88 (mostly signal)

    opt = torch.optim.LBFGS([theta], max_iter=n_iters, lr=lr)

    def closure():
        opt.zero_grad()
        loss = nll(theta, photons.detach(), truncated=truncated, uniform_bg=uniform_bg)
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()
