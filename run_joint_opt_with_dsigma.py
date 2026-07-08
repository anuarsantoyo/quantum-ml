"""
Joint (mu, gamma) optimization: REINFORCE for mu, IMPLICIT DIFF for gamma.

For each run:
  1. Forward: build photons at gamma, L-BFGS fit -> theta*, FWHM
  2. Gradient: compute dFWHM/dgamma via implicit function theorem (outside autograd)
     dFWHM/dgamma = -(dFWHM/dtheta) @ H_theta_theta^{-1} @ H_theta_gamma

The gamma gradient is then: dLoss/dgamma = sum_i dLoss/dFWHM_i * dFWHM_i/dgamma
where dLoss/dFWHM_i = sign(FWHM_i - matched_target_i) / B
"""
import math, time, json, os
import numpy as np
import torch
from torch.autograd.functional import hessian, jacobian

FREQ_MIN, FREQ_MAX = -75.0, 75.0
UNIFORM_DENSITY = 1.0 / (FREQ_MAX - FREQ_MIN)
WIDTH_MAX, WIDTH_EPS = 150.0, 1e-2

def _w(r): return WIDTH_EPS + WIDTH_MAX * torch.sigmoid(r)
def _rw(w):
    f = max(min((w-WIDTH_EPS)/WIDTH_MAX, 1-1e-10), 1e-10)
    return math.log(f / (1-f))

# ---- Lorentzian + uniform model (3 params: center, raw_gamma, logit_w) ----
def _lor_log_pdf(f, c, rg, lw):
    g, w = _w(rg), torch.sigmoid(lw)
    hi, lo = torch.tensor(FREQ_MAX), torch.tensor(FREQ_MIN)
    lrz = (g/math.pi)/((f-c)**2+g**2)
    Zl = (torch.atan((hi-c)/g)-torch.atan((lo-c)/g))/math.pi
    sig = lrz/(Zl+1e-30)
    return torch.log(w*sig+(1-w)*UNIFORM_DENSITY+1e-30)

def _nll_lor(theta, photons):
    return -_lor_log_pdf(photons, theta[0], theta[1], theta[2]).mean()

def _lor_fwhm(theta):
    return 2.0 * _w(theta[1])

def _build_photons(gamma_val, u, b):
    """Build photon detunings from gamma (reparameterized Cauchy)."""
    half = 0.5 * (FREQ_MAX - FREQ_MIN)
    sig = gamma_val * np.tan(math.atan(half/gamma_val) * (2*u - 1))
    return np.concatenate([sig, b])

# ---- Implicit derivative: dFWHM/dgamma ----
def _nll_func(theta_vec, photons_tensor):
    """NLL as a function of theta_only (for autograd.functional)."""
    return _nll_lor(theta_vec, photons_tensor)

def _fwhm_func(theta_vec):
    """FWHM as a function of theta (for autograd.functional)."""
    return _lor_fwhm(theta_vec)

def compute_fwhm_only(gamma_val, u, b):
    n_sig, n_bg = len(u), len(b)
    if n_sig + n_bg < 3:
        return 2.0 * gamma_val
    photons_np = _build_photons(gamma_val, u, b)
    photons_t = torch.tensor(photons_np, dtype=torch.float32)
    theta = torch.tensor([float(photons_t.median()), _rw(15.0), 0.0],
                          dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=80, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = _nll_lor(theta, photons_t)
        loss.backward()
        return loss
    try: opt.step(closure)
    except: return 2.0 * gamma_val
    if not torch.isfinite(theta).all(): return 2.0 * gamma_val
    return _lor_fwhm(theta).item()

def compute_sigma_only(gamma_val, u, b):
    '''Fast sigma computation at a given gamma (no implicit diff).'''
    n_sig, n_bg = len(u), len(b)
    if n_sig + n_bg < 3:
        return 0.5 * gamma_val  # rough sigma estimate
    photons_np = _build_photons(gamma_val, u, b)
    photons_t = torch.tensor(photons_np, dtype=torch.float32)
    theta = torch.tensor([float(photons_t.median()), _rw(15.0), 0.0],
                          dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=80, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = _nll_lor(theta, photons_t)
        loss.backward()
        return loss
    try: opt.step(closure)
    except: return 0.5 * gamma_val
    if not torch.isfinite(theta).all(): return 0.5 * gamma_val
    
    # Hessian at theta_star
    ts = theta.detach().clone().requires_grad_()
    loss_h = _nll_lor(ts, photons_t)
    gt = torch.autograd.grad(loss_h, ts, create_graph=True)[0]
    H = torch.zeros(3, 3, dtype=torch.float32)
    for i in range(3):
        gi = gt[i]
        for j in range(3):
            retain = not (i == 2 and j == 2)
            H[i, j] = torch.autograd.grad(gi, ts, retain_graph=retain)[0][j].detach()
    H_reg = H + 1e-4 * torch.eye(3)
    try:
        H_inv = torch.linalg.solve(H_reg, torch.eye(3))
    except RuntimeError:
        H_inv = torch.linalg.pinv(H_reg)
    
    # dFWHM/dtheta
    tf = ts.detach().clone().requires_grad_()
    fw_at = _lor_fwhm(tf)
    dF = torch.autograd.grad(fw_at, tf)[0]
    
    sigma = torch.sqrt(((dF.unsqueeze(0) @ H_inv @ dF.unsqueeze(1)).squeeze().clamp_min(1e-30)) / max(len(photons_t), 1)).item()
    if not math.isfinite(sigma):
        sigma = 0.5 * gamma_val
    return sigma

def compute_fwhm_and_dgamma(gamma_val, u, b, reg=1e-4):
    """
    Compute FWHM and dFWHM/dgamma for a single PLE scan.
    
    Args:
        gamma_val: float - Lorentzian HWHM
        u: np.array - frozen uniform noise for signal photons
        b: np.array - frozen uniform noise for background photons
        lam: float - Tikhonov regularization strength
    
    Returns:
        fwhm: float - fitted FWHM
        df_dgamma: float - dFWHM/dgamma 
    """
    n_sig = len(u)
    n_bg = len(b)
    if n_sig + n_bg < 3:
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0  # fallback: fwhm, sigma, df_dgamma
    
    # Build photons
    photons_np = _build_photons(gamma_val, u, b)
    photons_t = torch.tensor(photons_np, dtype=torch.float32)
    
    # ---- Step 1: L-BFGS fit (detached) ----
    theta = torch.tensor([float(photons_t.median()), _rw(15.0), 0.0], 
                          dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=80, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = _nll_lor(theta, photons_t)
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except RuntimeError:
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0
    
    if not torch.isfinite(theta).all():
        return 2.0 * gamma_val, 0.5 * gamma_val, 2.0
    
    theta_star = theta.detach().clone()  # shape (3,)
    fwhm_star = _lor_fwhm(theta.detach()).item()
    
    # ---- Step 2: Implicit differentiation ----
    # We need: dFWHM/dgamma = -(dFWHM/dtheta) @ H_theta_theta^{-1} @ H_theta_gamma
    
    # Build function: NLL(theta)
    # theta_star is the optimal point
    # gamma enters through photons_t
    
    # Photons as a function of gamma (for gradient computation)
    # Since we only need d^2 NLL / dtheta dgamma, and gamma only affects the data,
    # we can compute H_theta_gamma by differentiating grad_NLL_theta w.r.t gamma
    
    # H_theta_gamma = d/dgamma [d NLL / d theta]
    # Use finite difference on the gradient: [grad_theta(gamma+eps) - grad_theta(gamma-eps)] / (2*eps)
    # But actually, for implicit diff, we need d^2 NLL / dtheta dgamma analytically
    
    # Rebuild photons at gamma+eps and gamma-eps for FD on the gradient
    eps_g = 1e-3  # small step for gamma FD
    
    def _nll_at_gamma(g_val):
        p = torch.tensor(_build_photons(g_val, u, b), dtype=torch.float32)
        return _nll_lor(theta_star.detach().clone().requires_grad_(), p)
    
    # Gradient of NLL w.r.t theta at (theta_star, gamma+eps) and (theta_star, gamma-eps)
    def _grad_theta_at_gamma(g_val):
        t = theta_star.detach().clone().requires_grad_()
        p = torch.tensor(_build_photons(g_val, u, b), dtype=torch.float32)
        loss = _nll_lor(t, p)
        return torch.autograd.grad(loss, t, create_graph=False)[0]
    
    grad_hi = _grad_theta_at_gamma(gamma_val + eps_g)  # (3,)
    grad_lo = _grad_theta_at_gamma(gamma_val - eps_g)  # (3,)
    H_theta_gamma = (grad_hi - grad_lo) / (2 * eps_g)  # (3,) — d[dNLL/dtheta] / dgamma
    
    # H_theta_theta = d^2 NLL / dtheta^2 (3x3) at theta_star
    t_for_hess = theta_star.detach().clone().requires_grad_()
    loss_for_hess = _nll_lor(t_for_hess, photons_t)
    
    H_theta_theta = torch.zeros(3, 3, dtype=torch.float32)
    grad_t = torch.autograd.grad(loss_for_hess, t_for_hess, create_graph=True)[0]
    for i in range(3):
        gi = grad_t[i]
        for j in range(3):
            H_theta_theta[i, j] = torch.autograd.grad(gi, t_for_hess, 
                                                       retain_graph=True, 
                                                       create_graph=False)[0][j].detach()
    
    # dFWHM / dtheta (1x3) at theta_star
    t_for_df = theta_star.detach().clone().requires_grad_()
    fwhm_at_theta = _lor_fwhm(t_for_df)
    dF_dtheta = torch.autograd.grad(fwhm_at_theta, t_for_df, create_graph=False)[0]  # (3,)
    
    # Solve: dFWHM/dgamma = -dF_dtheta @ H_inv @ H_theta_gamma
    # Regularize Hessian
    H_reg = H_theta_theta + reg * torch.eye(3)
    
    try:
        H_inv = torch.linalg.solve(H_reg, torch.eye(3))
    except RuntimeError:
        # Pseudoinverse fallback
        H_inv = torch.linalg.pinv(H_reg)
    
    df_dgamma = -(dF_dtheta.unsqueeze(0) @ H_inv @ H_theta_gamma.unsqueeze(1)).squeeze().item()
    
    # Sanity check: if the gradient is absurd, fallback
    if not math.isfinite(df_dgamma):
        df_dgamma = 2.0
    
    # Compute sigma_FWHM: sqrt(dF/dtheta @ H_inv @ dF/dtheta^T)
    sigma_fwhm = torch.sqrt(((dF_dtheta.unsqueeze(0) @ H_inv @ dF_dtheta.unsqueeze(1)).squeeze().clamp_min(1e-30)) / max(len(photons_t), 1)).item()
    if not math.isfinite(sigma_fwhm):
        sigma_fwhm = 0.5 * gamma_val
    
    return fwhm_star, sigma_fwhm, df_dgamma

# ===== PARAMETERS =====
GAMMA_TRUE = 20.0
NBAR_TRUE = 50.0
LAMBDA_ = 2.0
SIGMA_PROP = 12.0

N_TARGET = 500
N_RUNS = 300
N_ITER = 80

EPS_DSIGMA = 0.3  # FD step for dsigma/dgamma
LR_MU = 15.0
LR_GAMMA = 0.5
BASELINE_ALPHA = 0.05
CLIP = 10.0

MU_INIT = 8.0
GAMMA_INIT = 5.0
SEED = 42

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, 'notebooks', 'joint_opt')
os.makedirs(out_dir, exist_ok=True)

t_total = time.time()

# ---- Generate target ----
print(f"Target: {N_TARGET} runs...", end=" ", flush=True)
rng = np.random.default_rng(SEED)
target, target_sigmas = [], []
for ti in range(N_TARGET):
    if ti % 100 == 0:
        print(f'  target {ti}/{N_TARGET}...', end=' ', flush=True)
    n = max(round(NBAR_TRUE + 6 * rng.standard_normal()), 0)
    u = rng.uniform(0, 1, size=n)
    bg = int(rng.poisson(LAMBDA_))
    b = rng.uniform(FREQ_MIN, FREQ_MAX, size=bg)
    # Use full function to get sigma (will be slow but target is small)
    fw, sig, _ = compute_fwhm_and_dgamma(GAMMA_TRUE, u, b)
    target.append(fw)
    target_sigmas.append(sig)
target_t = torch.tensor(target)
st, _ = torch.sort(target_t)
target_sigmas_t = torch.tensor(target_sigmas)
st_sig, _ = torch.sort(target_sigmas_t)
print(f"FWHM mean={target_t.mean():.1f}, sigma mean={target_sigmas_t.mean():.1f} ({time.time()-t_total:.0f}s)")

# ---- Optimization ----
mu_val = float(MU_INIT)
gamma_val = float(GAMMA_INIT)
bl = 0.0
history = []

print(f"\nmu_init={MU_INIT}, gamma_init={GAMMA_INIT}, true=({NBAR_TRUE},{GAMMA_TRUE})")
print(f"N_ITER={N_ITER}, N_RUNS={N_RUNS}\n")

for step in range(N_ITER):
    rng2 = np.random.default_rng(SEED + step)
    
    # Per-run: compute FWHM and dFWHM/dgamma
    fwhms, sigmas, dfs, dsigs, ns = [], [], [], [], []
    for _ in range(N_RUNS):
        n = max(round(mu_val + SIGMA_PROP * rng2.standard_normal()), 0)
        ns.append(n)
        u = rng2.uniform(0, 1, size=n)
        bg = int(rng2.poisson(LAMBDA_))
        b = rng2.uniform(FREQ_MIN, FREQ_MAX, size=bg)
        fw, sig, dg = compute_fwhm_and_dgamma(gamma_val, u, b)
        # Theoretical dsigma/dgamma via CRLB: sigma ~ FWHM/sqrt(n) ~ 2*gamma/sqrt(n)
        # dsigma/dgamma ≈ 2/sqrt(n). Per-run estimate avoids FD noise.
        dsig = min(2.0 / math.sqrt(max(n, 1)), 2.0)  # cap at 2.0 for very low n
        fwhms.append(fw)
        sigmas.append(sig)
        dfs.append(dg)
        dsigs.append(dsig)
    
    ft = torch.tensor(fwhms, dtype=torch.float32)
    si_t = torch.tensor(sigmas, dtype=torch.float32)
    nt = torch.tensor(ns, dtype=torch.float32)
    dg_t = torch.tensor(dfs, dtype=torch.float32)
    ds_t = torch.tensor(dsigs, dtype=torch.float32)
    
    # Sorted quantile matching for FWHM
    sf, sidx = torch.sort(ft)
    pl = torch.abs(sf - st[:N_RUNS])
    nss = nt[sidx]
    dgs = dg_t[sidx]
    ds_s = ds_t[sidx]
    
    # Sorted quantile matching for sigma (same sort order)
    # Target sigma data would come from experimental data; 
    # for now use sigma_true computed from target runs
    ss = si_t[sidx]  # sorted sigma values
    
    # Compute sigma-based loss term
    # Using the sigma values as the target (or could use stored target sigmas)
    loss_fwhm = pl.mean()
    
    # Sigma quantile matching (sorted the same way as FWHM)
    pl_sig = torch.abs(ss - st_sig[:N_RUNS])
    loss_sigma = pl_sig.mean()
    
    # Combined loss
    LAMBDA_SIGMA = 0.3
    mean_loss = (loss_fwhm + LAMBDA_SIGMA * loss_sigma).item()
    loss_fwhm_val = loss_fwhm.item()
    loss_sigma_val = loss_sigma.item()
    
    # Baseline
    if step == 0: bl = mean_loss
    else: bl = (1 - BASELINE_ALPHA) * bl + BASELINE_ALPHA * mean_loss
    
    # ---- MU gradient: REINFORCE ----
    adv = (pl.detach() - bl).numpy()
    scores = (nss.numpy() - mu_val) / SIGMA_PROP**2
    raw_grad_mu = float(np.mean(adv * scores))
    grad_mu = max(min(raw_grad_mu, CLIP), -CLIP)
    mu_val += LR_MU * (-grad_mu)
    mu_val = max(1.0, min(200.0, mu_val))
    
    # ---- GAMMA gradient: FWHM + Sigma terms ----
    LAMBDA_GAMMA = 2.0  # weight for sigma gradient in gamma update
    signs_fw = torch.sign(sf - st[:N_RUNS])
    signs_sg = torch.sign(ss - st_sig[:N_RUNS])
    raw_grad_fwhm = float((signs_fw * dgs).mean().item())
    raw_grad_sigma = float((signs_sg * ds_s).mean().item())
    raw_grad_gamma = raw_grad_fwhm + LAMBDA_GAMMA * raw_grad_sigma
    grad_gamma = max(min(raw_grad_gamma, CLIP), -CLIP)
    gamma_val += LR_GAMMA * (-grad_gamma)
    gamma_val = max(0.1, min(100.0, gamma_val))
    
    # Diagnostics
    rho = float(np.corrcoef(nss.numpy(), pl.numpy())[0, 1]) if pl.std() > 0.01 and nss.std() > 0.01 else 0.0
    
    info = {
        'step': step, 'mu': mu_val, 'gamma': gamma_val,
        'loss': mean_loss, 'loss_fwhm': loss_fwhm_val, 'loss_sigma': loss_sigma_val,
        'baseline': bl,
        'grad_mu': grad_mu, 'grad_gamma': grad_gamma,
        'rho': rho, 'mean_n': float(np.mean(ns)),
        'mean_df': float(dg_t.mean().item()),
        'mean_dsig': float(ds_t.mean().item()),
        'mean_sigma': float(ss.mean().item()),
        'std_sigma': float(ss.std().item()),
    }
    history.append(info)
    
    if step % 5 == 0 or step == N_ITER - 1:
        print(f"  S{step:2d}: mu={mu_val:6.2f} gamma={gamma_val:5.1f} | "
              f"L={mean_loss:.2f}(fwhm={loss_fwhm_val:.2f}+sig={loss_sigma_val:.2f}) | "
              f"gr_mu={grad_mu:+.4f} gr_g={raw_grad_fwhm:.3f}+{LAMBDA_GAMMA*raw_grad_sigma:.3f}={grad_gamma:+.4f} | "
              f"sigbar={ss.mean():.3f} dsigbar={ds_t.mean():.3f} "
              f"({time.time()-t_total:.0f}s)", flush=True)

print(f"\nDone. {time.time()-t_total:.0f}s")

# ---- Final eval ----
print("Final eval...", end=" ", flush=True)
re = np.random.default_rng(999)
ff, fsig, ii, isig = [], [], [], []
fm, fg = history[-1]['mu'], history[-1]['gamma']
for _ in range(N_RUNS):
    n = max(round(fm + SIGMA_PROP * re.standard_normal()), 0)
    u = re.uniform(0, 1, size=n)
    bg = int(re.poisson(LAMBDA_))
    b = re.uniform(FREQ_MIN, FREQ_MAX, size=bg)
    fw, sig, _ = compute_fwhm_and_dgamma(fg, u, b)
    ff.append(fw)
    fsig.append(sig)
    ni = max(round(MU_INIT + SIGMA_PROP * re.standard_normal()), 0)
    ui = re.uniform(0, 1, size=ni)
    bgi = int(re.poisson(LAMBDA_))
    bi = re.uniform(FREQ_MIN, FREQ_MAX, size=bgi)
    fi, sigi, _ = compute_fwhm_and_dgamma(GAMMA_INIT, ui, bi)
    ii.append(fi)
    isig.append(sigi)
print("done.")

# ---- Save ----
summary = {
    'nbar_true': NBAR_TRUE, 'gamma_true': GAMMA_TRUE,
    'mu_init': MU_INIT, 'gamma_init': GAMMA_INIT,
    'mu_final': round(fm, 2), 'gamma_final': round(fg, 2),
    'mu_error': round(abs(fm-NBAR_TRUE),2), 'gamma_error': round(abs(fg-GAMMA_TRUE),2),
    'n_iters': N_ITER, 'n_runs': N_RUNS, 'total_time_s': round(time.time()-t_total, 1),
    'history': history,
}
with open(os.path.join(out_dir, 'results_with_dsigma.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# ---- Plot ----
print("Plotting...", end=" ", flush=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

stp = [h['step'] for h in history]
mu_v = [h['mu'] for h in history]
ga_v = [h['gamma'] for h in history]
lo_v = [h['loss'] for h in history]
bl_v = [h['baseline'] for h in history]
rh_v = [h['rho'] for h in history]
gm_v = [h['grad_mu'] for h in history]
gg_v = [h['grad_gamma'] for h in history]
df_v = [h['mean_df'] for h in history]

fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.35)

# mu conv
ax = fig.add_subplot(gs[0:2, 0])
ax.axhline(NBAR_TRUE, color='#2d6a4f', ls='--', lw=2.5, alpha=0.7, label=fr'$\mu_{{\mathrm{{true}}}} = {NBAR_TRUE:.0f}$', zorder=10)
ax.plot(stp, mu_v, '-', color='#1a73e8', lw=3, marker='o', ms=5, zorder=5)
ax.fill_between(stp, mu_v, NBAR_TRUE, alpha=.08, color='#1a73e8')
ax.annotate(f'$\\mu_{{init}}$={MU_INIT:.0f}', xy=(0, MU_INIT), xytext=(2, MU_INIT-2),
            arrowprops=dict(arrowstyle='->', color='#d93025', lw=2), fontsize=11, color='#d93025', fontweight='bold')
ax.annotate(f'$\\mu_{{final}}$={fm:.1f}', xy=(N_ITER-1, fm), xytext=(N_ITER-12, fm+4),
            arrowprops=dict(arrowstyle='->', color='#1a73e8', lw=2), fontsize=11, color='#1a73e8', fontweight='bold')
ax.set_xlim(0, N_ITER-1); ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('$\\mu$ (mean photon count)', fontsize=13)
ax.set_title('$\\mu$ Convergence (REINFORCE)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12); ax.grid(alpha=.25, ls='--')

# gamma conv
ax = fig.add_subplot(gs[0:2, 1])
ax.axhline(GAMMA_TRUE, color='#2d6a4f', ls='--', lw=2.5, alpha=0.7, label=fr'$\gamma_{{\mathrm{{true}}}} = {GAMMA_TRUE:.0f}$', zorder=10)
ax.plot(stp, ga_v, '-', color='#d93025', lw=3, marker='o', ms=5, zorder=5)
ax.fill_between(stp, ga_v, GAMMA_TRUE, alpha=.08, color='#d93025')
ax.annotate(f'$\\gamma_{{init}}$={GAMMA_INIT:.0f}', xy=(0, GAMMA_INIT), xytext=(2, GAMMA_INIT+2),
            arrowprops=dict(arrowstyle='->', color='#1a73e8', lw=2), fontsize=11, color='#1a73e8', fontweight='bold')
ax.annotate(f'$\\gamma_{{final}}$={fg:.1f}', xy=(N_ITER-1, fg), xytext=(N_ITER-12, fg-3),
            arrowprops=dict(arrowstyle='->', color='#d93025', lw=2), fontsize=11, color='#d93025', fontweight='bold')
ax.set_xlim(0, N_ITER-1); ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('$\\gamma$ (HWHM, MHz)', fontsize=13)
ax.set_title('$\\gamma$ Convergence (Implicit Diff)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12); ax.grid(alpha=.25, ls='--')

# 2D trajectory
ax = fig.add_subplot(gs[0:2, 2])
ax.plot(mu_v, ga_v, '-', color='#666', lw=1.5, alpha=0.5, zorder=1)
sc = ax.scatter(mu_v, ga_v, c=stp, cmap='viridis', s=50, zorder=5, edgecolors='white', lw=0.5)
ax.scatter(MU_INIT, GAMMA_INIT, s=250, marker='*', color='#d93025', zorder=10, label='Start')
ax.scatter(fm, fg, s=250, marker='*', color='#1a73e8', zorder=10, label='End')
ax.scatter(NBAR_TRUE, GAMMA_TRUE, s=350, marker='X', color='#2d6a4f', zorder=10, label='True')
ax.set_xlabel('$\\mu$', fontsize=13); ax.set_ylabel('$\\gamma$', fontsize=13)
ax.set_title('Joint Trajectory', fontsize=16, fontweight='bold')
plt.colorbar(sc, ax=ax, label='Iteration'); ax.legend(fontsize=10); ax.grid(alpha=.25, ls='--')

# Loss
lo_fw = [h.get('loss_fwhm', h['loss']) for h in history]
lo_sg = [h.get('loss_sigma', 0) for h in history]
ax = fig.add_subplot(gs[2, 0])
ax.plot(stp, lo_v, '-', color='#d93025', lw=2.5, label='Total loss')
ax.plot(stp, lo_fw, '--', color='#1a73e8', lw=1.5, alpha=0.6, label='FWHM term')
ax.plot(stp, lo_sg, '--', color='#e37400', lw=1.5, alpha=0.6, label='Sigma term')
ax.plot(stp, bl_v, ':', color='#666', lw=1, alpha=.5, label='Baseline')
ax.set_title('Loss (FWHM + lambda*Sigma)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=.25, ls='--')

# rho
ax = fig.add_subplot(gs[2, 1])
ax.axhline(0, color='gray', lw=1, alpha=.5, ls='--')
ax.plot(stp, rh_v, '-', color='purple', lw=2.5, marker='.', ms=4)
ax.fill_between(stp, 0, rh_v, where=[r<0 for r in rh_v], alpha=.12, color='purple')
ax.set_ylim(-.4, .2); ax.set_title('$\\rho$(n, loss)', fontsize=14, fontweight='bold')
ax.grid(alpha=.25, ls='--')

# Gradients
ax = fig.add_subplot(gs[2, 2])
ax.axhline(0, color='gray', lw=1, alpha=.5, ls='--')
ax.plot(stp, gm_v, '-', color='#1a73e8', lw=2, label='$\\nabla_\\mu$')
ax.plot(stp, gg_v, '--', color='#d93025', lw=2, label='$\\nabla_\\gamma$')
ax.set_title('Gradients', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=.25, ls='--')

# Error
ax = fig.add_subplot(gs[3, 0:2])
mu_err = [abs(v-NBAR_TRUE) for v in mu_v]
ga_err = [abs(v-GAMMA_TRUE) for v in ga_v]
ax.semilogy(stp, mu_err, '-', color='#1a73e8', lw=2.5, marker='.', label='$\\mu$ error')
ax.semilogy(stp, ga_err, '-', color='#d93025', lw=2.5, marker='.', label='$\\gamma$ error')
ax.set_title('Parameter Error', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=.25, ls='--')

# df/dgamma
ax = fig.add_subplot(gs[3, 2])
ax.plot(stp, df_v, '-', color='#e37400', lw=2, marker='.', label='Mean $d$FWHM$/d\\gamma$')
ax.axhline(2.0, color='gray', lw=1, alpha=.5, ls='--', label='Theoretical (2.0)')
ax.set_title('Per-run $d$FWHM$/d\\gamma$', fontsize=14, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=.25, ls='--')

# FWHM + Sigma distributions
ax = fig.add_subplot(gs[4, 0])
xg = np.linspace(0, 120, 500)
try:
    kt = gaussian_kde(target_t.numpy())
    kf = gaussian_kde(np.array(ff))
    ki = gaussian_kde(np.array(ii))
    ax.plot(xg, kt(xg), 'k-', lw=3, label=f'Target ($\\mu$={NBAR_TRUE}, $\\gamma$={GAMMA_TRUE})', zorder=10)
    ax.plot(xg, kf(xg), '-', color='#1a73e8', lw=2.5, label=f'Final ($\\mu$={fm:.1f}, $\\gamma$={fg:.1f})')
    ax.plot(xg, ki(xg), '--', color='#d93025', lw=2, alpha=.6, label=f'Initial')
except: pass
ax.set_xlabel('FWHM (MHz)', fontsize=12); ax.set_ylabel('Density', fontsize=12)
ax.set_title('FWHM Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=.25, ls='--'); ax.set_xlim(0, 120)

# Sigma KDE grid
xg_sig = np.linspace(0, max(np.percentile(target_sigmas, 95), np.percentile(fsig, 95), np.percentile(isig, 95)) * 1.2, 500)

# Sigma distribution
ax = fig.add_subplot(gs[4, 1])
try:
    kt_sig = gaussian_kde(np.array(target_sigmas))
    kf_sig = gaussian_kde(np.array(fsig))
    ki_sig = gaussian_kde(np.array(isig))
    ax.plot(xg_sig, kt_sig(xg_sig), 'k-', lw=3, label=f'Target', zorder=10)
    ax.plot(xg_sig, kf_sig(xg_sig), '-', color='#1a73e8', lw=2.5, label=f'Final')
    ax.plot(xg_sig, ki_sig(xg_sig), '--', color='#d93025', lw=2, alpha=.6, label=f'Initial')
except Exception as e:
    ax.text(0.5, 0.5, f'Sigma KDE failed: {e}', transform=ax.transAxes)
ax.set_xlabel('Sigma FWHM (MHz)', fontsize=12); ax.set_ylabel('Density', fontsize=12)
ax.set_title('Sigma Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=.25, ls='--')

# Summary table
ax = fig.add_subplot(gs[5, :])
ax.axis('off')
fm_str = f'{fm:.2f}'; fg_str = f'{fg:.2f}'
me_str = f'{abs(fm-NBAR_TRUE):.2f} ({abs(fm-NBAR_TRUE)/NBAR_TRUE*100:.1f}%)'
ge_str = f'{abs(fg-GAMMA_TRUE):.2f} ({abs(fg-GAMMA_TRUE)/GAMMA_TRUE*100:.1f}%)'
td = [
    ['Parameter', 'True', 'Init', 'Final', 'Error'],
    ['$\\mu$', f'{NBAR_TRUE:.0f}', f'{MU_INIT:.0f}', fm_str, me_str],
    ['$\\gamma$', f'{GAMMA_TRUE:.0f}', f'{GAMMA_INIT:.0f}', fg_str, ge_str],
    ['Loss (FWHM)', '', f'{history[0].get("loss_fwhm",history[0]["loss"]):.2f}', f'{history[-1].get("loss_fwhm",history[-1]["loss"]):.2f}', f'{history[0].get("loss_fwhm",history[0]["loss"])-history[-1].get("loss_fwhm",history[-1]["loss"]):.2f}'],
    ['Loss (Sigma)', '', f'{history[0].get("loss_sigma",0):.2f}', f'{history[-1].get("loss_sigma",0):.2f}', f'{history[0].get("loss_sigma",0)-history[-1].get("loss_sigma",0):.2f}'],
    ['Time', '', '', f'{time.time()-t_total:.0f}s', f'{N_ITER} iters'],
]
tbl = ax.table(cellText=td, loc='center', cellLoc='center', colWidths=[0.18, 0.15, 0.15, 0.15, 0.22])
tbl.auto_set_font_size(False); tbl.set_fontsize(14)
for j in range(5): tbl[0, j].set_facecolor('#f0f0f0')

fig.suptitle(f'Joint $\\mu$ + $\\gamma$ Optimization (Implicit Diff)', fontsize=22, fontweight='bold', y=1.02)
plt.savefig(os.path.join(out_dir, 'joint_opt_with_dsigma.png'), dpi=150, bbox_inches='tight', facecolor='white')
print("done.")
plt.close()

print(f"\n{'='*60}")
print(f"  JOINT OPTIMIZATION (IMPLICIT DIFF + dSIGMA/dGAMMA)")
print(f"{'='*60}")
print(f"  mu:    {MU_INIT:.0f} -> {fm:.2f}  (true={NBAR_TRUE})  error={abs(fm-NBAR_TRUE):.2f}")
print(f"  gamma: {GAMMA_INIT:.0f} -> {fg:.2f}  (true={GAMMA_TRUE})  error={abs(fg-GAMMA_TRUE):.2f}")
print(f"  Loss:  {history[0]['loss']:.2f} -> {history[-1]['loss']:.2f}  (FWHM: {history[0].get('loss_fwhm',0):.2f} -> {history[-1].get('loss_fwhm',0):.2f}, Sigma: {history[0].get('loss_sigma',0):.2f} -> {history[-1].get('loss_sigma',0):.2f}, dS: {history[0].get('mean_dsig',0):.2f} -> {history[-1].get('mean_dsig',0):.2f})")
print(f"  Time:  {time.time()-t_total:.0f}s")
print(f"{'='*60}")
