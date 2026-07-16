#!/usr/bin/env python3
"""12-joint-opt-with-sigma — FIXED: LR_MU=15 + independent sigma sorting."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.set_default_dtype(torch.float32)

from src.fitting import (
    _raw_from_width as _rw,
    log_pdf, nll, fwhm_from_theta, fit_profile
)
from src.samplers import draw_fixed_noise, build_photons
from src.implicit import compute_fwhm_and_dgamma

# ── Parameters ──
GAMMA_TRUE = 20.0; NBAR_TRUE = 50.0; LAMBDA_ = 2.0; SIGMA_PROP = 6.0; SEED = 42
N_TARGET = 200; N_RUNS = 200; N_ITER = 80
LR_MU = 15.0                # FIXED: was 3.0, now matching notebook 10
LR_GAMMA = 0.5
BASELINE_ALPHA = 0.05
CLIP = 10.0
LAMBDA_SIGMA = 0.3
LAMBDA_GAMMA_SIGMA = 2.0
MU_INIT = 8.0; GAMMA_INIT = 5.0

def _fit_fn(ph):
    return fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)
def _fwhm_fn(th):
    return fwhm_from_theta(th, model='lorentzian')
def _nll_fn(th, ph):
    return nll(th, ph, model='lorentzian', uniform_bg=False)

# ── Target data ──
print(f"Generating target ({N_TARGET} runs)...", end=" ", flush=True)
rng = np.random.default_rng(SEED)
target_fwhms, target_sigmas = [], []
for ti in range(N_TARGET):
    if ti % 100 == 0: print(f'{ti}...', end=' ', flush=True)
    u, b, n = draw_fixed_noise(NBAR_TRUE, SIGMA_PROP, LAMBDA_, rng)
    fw, sig, _ = compute_fwhm_and_dgamma(
        GAMMA_TRUE, u.numpy(), b.numpy(),
        _fit_fn, _fwhm_fn, _nll_fn, n_params=2)
    target_fwhms.append(fw); target_sigmas.append(sig)
target_t = torch.tensor(target_fwhms, dtype=torch.float32)
target_s_t = torch.tensor(target_sigmas, dtype=torch.float32)
st, _ = torch.sort(target_t)
st_sig, _ = torch.sort(target_s_t)
print(f"done. FWHM mean={target_t.mean():.1f}, σ mean={target_s_t.mean():.2f}")

# ── Optimization ──
mu_val = float(MU_INIT); gamma_val = float(GAMMA_INIT); bl = 0.0; history = []
print(f"\nμ_init={MU_INIT}, γ_init={GAMMA_INIT}, true=({NBAR_TRUE},{GAMMA_TRUE})")
print(f"N_ITER={N_ITER}, N_RUNS={N_RUNS}, LR_MU={LR_MU}, λ_sig={LAMBDA_SIGMA}, λ_γ_sig={LAMBDA_GAMMA_SIGMA}\n")
t_total = time.time()

for step in range(N_ITER):
    rng2 = np.random.default_rng(SEED + step)
    fwhms, sigmas, dfs, ns = [], [], [], []
    for _ in range(N_RUNS):
        u, b, n = draw_fixed_noise(mu_val, SIGMA_PROP, LAMBDA_, rng2)
        ns.append(n)
        fw, sig, dg = compute_fwhm_and_dgamma(
            gamma_val, u.numpy(), b.numpy(),
            _fit_fn, _fwhm_fn, _nll_fn, n_params=2)
        fwhms.append(fw); sigmas.append(sig); dfs.append(dg)

    ft = torch.tensor(fwhms, dtype=torch.float32)
    si_t = torch.tensor(sigmas, dtype=torch.float32)
    nt = torch.tensor(ns, dtype=torch.float32)
    dg_t = torch.tensor(dfs, dtype=torch.float32)

    # FWHM sorted
    sf, sidx = torch.sort(ft)
    pl = torch.abs(sf - st[:N_RUNS])
    nss = nt[sidx]
    dgs = dg_t[sidx]

    # FIXED: sigma sorted independently by sigma rank
    ss, ss_idx = torch.sort(si_t)
    pl_sig = torch.abs(ss - st_sig[:N_RUNS])

    # FWHM + sigma loss
    loss_fwhm = pl.mean(); loss_sigma = pl_sig.mean()
    loss_fwhm_val = loss_fwhm.item(); loss_sigma_val = loss_sigma.item()
    mean_loss = (loss_fwhm + LAMBDA_SIGMA * loss_sigma).item()

    if step == 0: bl = mean_loss
    else: bl = (1 - BASELINE_ALPHA) * bl + BASELINE_ALPHA * mean_loss

    # ── MU gradient: REINFORCE ──
    per_run_reward = -(pl.detach() + LAMBDA_SIGMA * pl_sig.detach())
    adv = (per_run_reward - bl).numpy()
    scores = (nss.numpy() - mu_val) / SIGMA_PROP**2
    raw_grad_mu = float(np.mean(adv * scores))
    grad_mu = max(min(raw_grad_mu, CLIP), -CLIP)
    mu_val += LR_MU * (-grad_mu)
    mu_val = max(1.0, min(200.0, mu_val))

    # ── GAMMA gradient: implicit diff (FWHM) + CRLB (sigma) ──
    dsigs = torch.tensor(
        [min(2.0 / math.sqrt(max(int(n), 1)), 2.0) for n in ns], dtype=torch.float32)
    ds_s = dsigs[ss_idx]    # FIXED: now sorted by sigma rank to match pl_sig
    signs_fw = torch.sign(sf - st[:N_RUNS])
    signs_sg = torch.sign(ss - st_sig[:N_RUNS])
    raw_grad_fwhm = float((signs_fw * dgs).mean().item())
    raw_grad_sigma = float((signs_sg * ds_s).mean().item())
    raw_grad_gamma = raw_grad_fwhm + LAMBDA_GAMMA_SIGMA * raw_grad_sigma
    grad_gamma = max(min(raw_grad_gamma, CLIP), -CLIP)
    gamma_val += LR_GAMMA * (-grad_gamma)
    gamma_val = max(0.1, min(100.0, gamma_val))

    rho = float(np.corrcoef(nss.numpy(), pl.numpy())[0, 1]) if pl.std()>0.01 and nss.std()>0.01 else 0.0

    info = {
        'step': step, 'mu': mu_val, 'gamma': gamma_val,
        'loss': mean_loss, 'loss_fwhm': loss_fwhm_val, 'loss_sigma': loss_sigma_val,
        'baseline': bl,
        'grad_mu': grad_mu, 'grad_gamma': grad_gamma,
        'rho': rho, 'mean_n': float(np.mean(ns)),
        'mean_fwhm': float(ft.mean().item()),
        'mean_sigma': float(ss.mean().item()),
    }
    history.append(info)

    if step % 5 == 0 or step == N_ITER - 1:
        print(f"  S{step:2d}: μ={mu_val:6.2f} γ={gamma_val:5.1f} | "
              f"L={mean_loss:.2f}(F={loss_fwhm_val:.2f}+S={loss_sigma_val:.2f}) | "
              f"∇μ={grad_mu:+.4f} ∇γ={grad_gamma:+.4f} | "
              f"n̄={info['mean_n']:4.1f} ρ={rho:+.3f} "
              f"({time.time()-t_total:.0f}s)", flush=True)

# ── Save ──
out = {
    'params': {
        'GAMMA_TRUE': GAMMA_TRUE, 'NBAR_TRUE': NBAR_TRUE,
        'N_TARGET': N_TARGET, 'N_RUNS': N_RUNS, 'N_ITER': N_ITER,
        'LR_MU': LR_MU, 'LR_GAMMA': LR_GAMMA,
        'LAMBDA_SIGMA': LAMBDA_SIGMA, 'LAMBDA_GAMMA_SIGMA': LAMBDA_GAMMA_SIGMA,
        'MU_INIT': MU_INIT, 'GAMMA_INIT': GAMMA_INIT,
    },
    'final': history[-1] if history else None,
    'first': history[0] if history else None,
    'history': history,
    'target_fwhm_mean': float(target_t.mean()),
    'total_seconds': time.time() - t_total,
}
out_path = os.path.join(os.path.dirname(__file__), '12_fixed_results.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  JOINT OPTIMIZATION — FWHM + SIGMA MATCHING (FIXED)")
print(f"{'='*65}")
if history:
    print(f"  μ:     {MU_INIT:.0f} → {history[-1]['mu']:.2f}  (true={NBAR_TRUE})  error={abs(history[-1]['mu']-NBAR_TRUE):.2f}")
    print(f"  γ:     {GAMMA_INIT:.0f} → {history[-1]['gamma']:.2f}  (true={GAMMA_TRUE})  error={abs(history[-1]['gamma']-GAMMA_TRUE):.2f}")
    print(f"  Loss:  {history[0]['loss']:.2f} → {history[-1]['loss']:.2f}")
    print(f"  FWHM:  {history[0]['loss_fwhm']:.2f} → {history[-1]['loss_fwhm']:.2f}")
    print(f"  Sigma: {history[0]['loss_sigma']:.2f} → {history[-1]['loss_sigma']:.2f}")
    print(f"  μ final grad: {history[-1]['grad_mu']:+.4f}")
print(f"  Time:  {time.time()-t_total:.0f}s")
print(f"{'='*65}")
print(f"Results → {out_path}")
