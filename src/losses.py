"""
Loss functions for comparing distributions of FWHM values.

Each loss takes two sets of samples (simulated and target) and returns
a scalar distance that is differentiable w.r.t. the simulated samples.
"""

import torch


def mmd_loss(x, y, sigma=None):
    """
    Maximum Mean Discrepancy squared (MMD²) with a Gaussian kernel.

    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2 * E[k(x,y)]

    Uses the median heuristic to set sigma if not provided.

    Parameters
    ----------
    x : tensor (N, 1) or (N,)
        Simulated samples.
    y : tensor (M, 1) or (M,)
        Target samples.
    sigma : float or None
        Kernel bandwidth. If None, uses median heuristic.

    Returns
    -------
    loss : scalar tensor
    """
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    if sigma is None:
        # Median heuristic: sigma = median pairwise distance / sqrt(2)
        all_data = torch.cat([x, y])
        dists = torch.cdist(all_data, all_data)
        median_dist = torch.median(dists[dists > 0])
        sigma = median_dist / math.sqrt(2.0) if median_dist > 0 else 1.0

    def gaussian_gram(a, b, s):
        dists = torch.cdist(a, b)
        return torch.exp(-dists ** 2 / (2.0 * s ** 2))

    K_xx = gaussian_gram(x, x, sigma)
    K_yy = gaussian_gram(y, y, sigma)
    K_xy = gaussian_gram(x, y, sigma)

    n = x.shape[0]
    m = y.shape[0]

    # Unbiased estimator
    return (K_xx.sum() - torch.trace(K_xx)) / (n * (n - 1)) \
         + (K_yy.sum() - torch.trace(K_yy)) / (m * (m - 1)) \
         - 2.0 * K_xy.mean()


def mmd_loss_fixed_sigma(x, y, sigma):
    """
    MMD² with a user-specified kernel bandwidth (no median heuristic).

    Useful for comparing results across experiments with a fixed sigma.

    Parameters
    ----------
    x : tensor (N,)
    y : tensor (M,)
    sigma : float
        Fixed kernel bandwidth.

    Returns
    -------
    loss : scalar tensor
    """
    return mmd_loss(x, y, sigma=sigma)


def l2_loss(x, y):
    """
    L2 distance between KDE-smoothed histograms.

    Parameters
    ----------
    x : tensor (N,)
    y : tensor (M,)

    Returns
    -------
    loss : scalar tensor
    """
    # Simple L2 on histograms with shared bins
    all_data = torch.cat([x, y])
    lo = torch.min(all_data).item()
    hi = torch.max(all_data).item()
    n_bins = min(50, len(all_data) // 5)

    bins = torch.linspace(lo, hi, n_bins + 1, device=x.device)
    hx = torch.histc(x, bins=n_bins, min=lo, max=hi) / len(x)
    hy = torch.histc(y, bins=n_bins, min=lo, max=hi) / len(y)

    return torch.sum((hx - hy) ** 2)


def wasserstein_loss(x, y):
    """
    1D Wasserstein (Earth Mover's) distance.

    For 1D distributions, this is just the L1 distance between sorted samples.

    Parameters
    ----------
    x : tensor (N,)
    y : tensor (M,)

    Returns
    -------
    loss : scalar tensor
    """
    x_sorted = torch.sort(x)[0]
    y_sorted = torch.sort(y)[0]

    # Interpolate to same length
    n = min(len(x), len(y))
    x_sel = x_sorted[::max(1, len(x) // n)][:n]
    y_sel = y_sorted[::max(1, len(y) // n)][:n]

    return torch.mean(torch.abs(x_sel - y_sel))


import math  # needed by mmd_loss median heuristic
