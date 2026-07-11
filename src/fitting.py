"""
Fitting — pseudo-Voigt & Lorentzian profile fitting for optical linewidth extraction.

Provides differentiable log-pdf calculations (with window-normalized PDF
via Z factors) and MLE fitting for PLE spectroscopy data.

Supports two models:
    - 'pseudo-voigt' (4 params: center, raw_gamma, raw_sigma, logit_w)
    - 'lorentzian'   (3 params: center, raw_gamma, logit_w)
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
    if isinstance(frac, (int, float)):
        frac = max(1e-10, min(frac, 1 - 1e-10))
        return math.log(frac / (1.0 - frac))
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
# Voigt FWHM (Olivero-Longbothum approximation)
# =========================================================================

def voigt_fwhm(lorentzian_fwhm, gaussian_fwhm):
    """
    Voigt FWHM approximation, combining Lorentzian and Gaussian widths.
    
    f_V ≈ 0.5346 · f_L + sqrt(0.2166 · f_L² + f_G²)
    
    Parameters
    ----------
    lorentzian_fwhm : scalar tensor or float
        FWHM of the Lorentzian component (2 * gamma).
    gaussian_fwhm : scalar tensor or float
        FWHM of the Gaussian component (2 * sqrt(2*ln2) * sigma).
    
    Returns
    -------
    fwhm : scalar tensor or float
    """
    # Handle both tensor and float inputs
    if isinstance(lorentzian_fwhm, (int, float)) and isinstance(gaussian_fwhm, (int, float)):
        return 0.5346 * lorentzian_fwhm + math.sqrt(0.2166 * lorentzian_fwhm ** 2 + gaussian_fwhm ** 2)
    return 0.5346 * lorentzian_fwhm + torch.sqrt(0.2166 * lorentzian_fwhm ** 2 + gaussian_fwhm ** 2)


# =========================================================================
# Log-pdf
# =========================================================================

def log_pdf(freqs, center, raw_gamma, raw_sigma=None, logit_w=None,
            uniform_bg=True, model='pseudo-voigt'):
    """
    Log-pdf of the fitted model.

    Parameters
    ----------
    freqs : tensor (N,)
        Photon detunings (MHz).
    center : scalar tensor
        Line center.
    raw_gamma : scalar tensor
        Unconstrained Lorentzian-width parameter (mapped via _width).
    raw_sigma : scalar tensor or None
        Unconstrained Gaussian-width parameter (mapped via _width).
        Required for model='pseudo-voigt', ignored for model='lorentzian'.
    logit_w : scalar tensor or None
        Logit of signal/uniform weight. Ignored if uniform_bg=False.
    uniform_bg : bool
        If True, include uniform background component.
    model : str
        'pseudo-voigt' (4 params) or 'lorentzian' (3 params).

    Returns
    -------
    log_pdf : tensor (N,)
    """
    gamma = _width(raw_gamma)
    hi = torch.as_tensor(FREQ_MAX, dtype=freqs.dtype)
    lo = torch.as_tensor(FREQ_MIN, dtype=freqs.dtype)

    # Lorentzian component (always present)
    lorentz = (gamma / math.pi) / ((freqs - center) ** 2 + gamma ** 2)
    Z_l = _lorentz_cdf(hi, center, gamma) - _lorentz_cdf(lo, center, gamma)
    signal = lorentz / (Z_l + 1e-30)

    if model == 'pseudo-voigt':
        # Add Gaussian component (pseudo-Voigt mix)
        sigma_g = _width(raw_sigma)
        gauss = torch.exp(-0.5 * ((freqs - center) / sigma_g) ** 2) / (sigma_g * math.sqrt(2.0 * math.pi))
        Z_g = _normal_cdf(hi, center, sigma_g) - _normal_cdf(lo, center, sigma_g)
        signal = FIT_ETA * gauss / (Z_g + 1e-30) + (1.0 - FIT_ETA) * signal

    if uniform_bg:
        w = torch.sigmoid(logit_w)
        pdf = w * signal + (1.0 - w) * torch.as_tensor(UNIFORM_DENSITY, dtype=freqs.dtype)
    else:
        pdf = signal

    return torch.log(pdf + 1e-30)


# =========================================================================
# Negative log-likelihood
# =========================================================================

def nll(theta, photons, uniform_bg=True, model='pseudo-voigt'):
    """
    Negative log-likelihood of photons under the fitted model.
    
    theta = (center, raw_gamma[, raw_sigma[, logit_w]])
      - model='pseudo-voigt', uniform_bg=True:  theta[0..3] = center, raw_gamma, raw_sigma, logit_w
      - model='pseudo-voigt', uniform_bg=False: theta[0..2] = center, raw_gamma, raw_sigma
      - model='lorentzian',  uniform_bg=True:  theta[0..2] = center, raw_gamma, logit_w
      - model='lorentzian',  uniform_bg=False: theta[0..1] = center, raw_gamma

    Parameters
    ----------
    theta : tensor
    photons : tensor (N,)
    uniform_bg : bool
    model : str
    """
    center = theta[0]
    raw_gamma = theta[1]
    
    if model == 'pseudo-voigt':
        idx = 2
        raw_sigma = theta[2] if len(theta) > 2 else None
        logit_w = theta[3] if len(theta) > 3 and uniform_bg else None
        lp = log_pdf(photons, center, raw_gamma, raw_sigma, logit_w,
                     uniform_bg=uniform_bg, model=model)
    else:  # lorentzian
        raw_sigma = None
        logit_w = theta[2] if len(theta) > 2 and uniform_bg else None
        lp = log_pdf(photons, center, raw_gamma, raw_sigma, logit_w,
                     uniform_bg=uniform_bg, model=model)

    return -torch.sum(lp)


# =========================================================================
# FWHM extraction
# =========================================================================

def fwhm_from_theta(theta, model='pseudo-voigt'):
    """
    Extract the FWHM from fitted theta.

    For model='lorentzian': returns 2 * gamma (Lorentzian FWHM).
    For model='pseudo-voigt': returns Voigt FWHM via Olivero-Longbothum.

    Parameters
    ----------
    theta : tensor
    model : str

    Returns
    -------
    fwhm : scalar tensor
    """
    if model == 'lorentzian':
        return 2.0 * _width(theta[1])
    
    # pseudo-voigt: Voigt FWHM
    f_L = 2.0 * _width(theta[1])
    sigma_g = _width(theta[2])
    f_G = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_g
    return voigt_fwhm(f_L, f_G)


# =========================================================================
# Fitting (inner optimization)
# =========================================================================

def fit_profile(photons, n_iters=80, model='pseudo-voigt', uniform_bg=True):
    """
    Fit a profile (pseudo-Voigt or Lorentzian) to photon detunings using L-BFGS.

    Uses median-based initialization (more robust than zero-init).

    Parameters
    ----------
    photons : tensor (N,)
    n_iters : int
        Number of L-BFGS iterations.
    model : str
        'pseudo-voigt' or 'lorentzian'
    uniform_bg : bool
        If True, include uniform background component.

    Returns
    -------
    theta : tensor or None
        Optimized parameters, or None if convergence failed.
    """
    if len(photons) < 3:
        return None
    data = photons.detach()

    if model == 'pseudo-voigt':
        n_params = 4 if uniform_bg else 3
        theta = torch.zeros(n_params, requires_grad=True, dtype=data.dtype)
        with torch.no_grad():
            theta[0] = float(data.median())
            theta[1] = float(_raw_from_width(15.0)) if isinstance(data, torch.Tensor) else _raw_from_width(15.0)
            theta[2] = float(_raw_from_width(5.0)) if isinstance(data, torch.Tensor) else _raw_from_width(5.0)
            if uniform_bg:
                theta[3] = 0.0
    else:  # lorentzian
        n_params = 3 if uniform_bg else 2
        theta = torch.zeros(n_params, requires_grad=True, dtype=data.dtype)
        with torch.no_grad():
            theta[0] = float(data.median())
            theta[1] = float(_raw_from_width(15.0)) if isinstance(data, torch.Tensor) else _raw_from_width(15.0)
            if uniform_bg:
                theta[2] = 0.0

    opt = torch.optim.LBFGS([theta], max_iter=n_iters, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = nll(theta, data, model=model, uniform_bg=uniform_bg)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError:
        return None

    if not torch.isfinite(theta).all().item():
        return None
    return theta.detach()


# =========================================================================
# Full PLE scan helpers
# =========================================================================

def run_one_ple_scan(n_signal, gamma, lambda_, rng, model='pseudo-voigt', n_iters=80):
    """
    Run one complete PLE scan: sample photons, fit profile, return FWHM.

    Parameters
    ----------
    n_signal : int
        Number of signal photons for this scan.
    gamma : float
        Lorentzian HWHM (MHz) of the true line shape.
    lambda_ : float
        Mean number of background photons.
    rng : numpy Generator
    model : str
        Which fit model to use.
    n_iters : int
        L-BFGS iterations for the fit.

    Returns
    -------
    fwhm : float
        Extracted FWHM (MHz), or 2*gamma as fallback.
    """
    from src.samplers import build_photons

    u = torch.tensor(rng.uniform(0.0, 1.0, size=n_signal), dtype=torch.float32)
    bg_count = int(rng.poisson(lambda_))
    b = torch.tensor(rng.uniform(FREQ_MIN, FREQ_MAX, size=bg_count), dtype=torch.float32)

    if len(u) + len(b) < 3:
        return 2.0 * gamma

    photons = build_photons(torch.tensor(gamma, dtype=torch.float32), u, b,
                            window=FREQ_RANGE)
    theta = fit_profile(photons, n_iters=n_iters, model=model)
    if theta is None:
        return 2.0 * gamma
    return fwhm_from_theta(theta, model=model).item()
