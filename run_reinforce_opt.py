"""
Lightning-fast REINFORCE optimization.
float32 + max_iter=15 + tighter grid.
Target: ~10 min total.
"""
import math, time, json, os, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float32)

FREQ_MIN, FREQ_MAX = -75.0, 75.0
UNIFORM_DENSITY = 1.0 / (FREQ_MAX - FREQ_MIN)
WIDTH_MAX, WIDTH_EPS, FIT_ETA = 150.0, 1e-2, 0.2

def _w(r): return WIDTH_EPS + WIDTH_MAX * torch.sigmoid(r)
def _rw(w):
    f = max(min((w-WIDTH_EPS)/WIDTH_MAX, 1-1e-10), 1e-10)
    return math.log(f / (1-f))
def log_pdf(f, c, rg, rs, lw):
    g, sg, w = _w(rg), _w(rs), torch.sigmoid(lw)
    hi, lo = torch.tensor(FREQ_MAX, dtype=torch.float32), torch.tensor(FREQ_MIN, dtype=torch.float32)
    lrz = (g/math.pi)/((f-c)**2+g**2)
    Zl = (torch.atan((hi-c)/g)-torch.atan((lo-c)/g))/math.pi
    gs = torch.exp(-.5*((f-c)/sg)**2)/(sg*math.sqrt(2*math.pi))
    Zg = .5*(torch.erf((hi-c)/(sg*math.sqrt(2)))-torch.erf((lo-c)/(sg*math.sqrt(2))))
    sig = FIT_ETA*gs/(Zg+1e-30)+(1-FIT_ETA)*lrz/(Zl+1e-30)
    return torch.log(w*sig+(1-w)*UNIFORM_DENSITY+1e-30)
def nll(t,p): return -log_pdf(p,t[0],t[1],t[2],t[3]).mean()
def fwhm(t):
    return .5346*2*_w(t[1])+torch.sqrt(.2166*(2*_w(t[1]))**2+(2*math.sqrt(2*math.log(2))*_w(t[2]))**2)
def fit(photons):
    if len(photons)<3: return None
    d=photons.detach()
    t=torch.tensor([float(d.median()),_rw(15),_rw(5),0.],requires_grad=True)
    opt=torch.optim.LBFGS([t],max_iter=15,line_search_fn="strong_wolfe")  # fast
    def c(): opt.zero_grad(); nll(t,d).backward(); return nll(t,d)
    try: opt.step(c)
    except: return None
    return t.detach() if torch.isfinite(t).all().item() else None
def scan(n,g,lam,rng):
    u=torch.tensor(rng.uniform(0,1,size=n),dtype=torch.float32)
    bg=int(rng.poisson(lam))
    b=torch.tensor(rng.uniform(FREQ_MIN,FREQ_MAX,size=bg),dtype=torch.float32)
    if len(u)+len(b)<3: return 2*g
    p=torch.cat([g*torch.tan(torch.tensor(math.atan(75/g),dtype=torch.float32)*(2*u-1)),b])
    t=fit(p)
    return fwhm(t).item() if t is not None else 2*g

GAMMA, LAMBDA_ = 20.0, 2.0
NBAR_TRUE = 50.0
SIGMA_PROP = 12.0
N_RUNS = 150
N_TARGET = 200
N_ITER = 25
LR = 8.0
BASELINE_ALPHA = 0.05
CLIP = 10.0
MU_INIT = 10.0
SEED = 42

base_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(base_dir, 'notebooks', 'reinforce_opt_results.json')
PLOT_FILE = os.path.join(base_dir, 'notebooks', 'reinforce_opt_results.png')

t_total = time.time()

print(f"Target: {N_TARGET}runs...", end=" ", flush=True)
rng=np.random.default_rng(SEED)
target=[scan(max(round(NBAR_TRUE+6*rng.standard_normal()),0),GAMMA,LAMBDA_,rng) for _ in range(N_TARGET)]
target_t=torch.tensor(target)
st,_=torch.sort(target_t)
print(f"mean={target_t.mean():.1f} ({time.time()-t_total:.0f}s)")

mu=torch.tensor([float(MU_INIT)])
bl=0.0
hist=[]

print(f"mu_init={MU_INIT}, true={NBAR_TRUE}, iters={N_ITER}, runs/iter={N_RUNS}")
t_opt = time.time()

for step in range(N_ITER):
    rng2=np.random.default_rng(SEED+step)
    mv=mu.item()
    ns,fw=[],[]
    for _ in range(N_RUNS):
        n=max(round(mv+SIGMA_PROP*rng2.standard_normal()),0)
        ns.append(n); fw.append(scan(n,GAMMA,LAMBDA_,rng2))
    nt=torch.tensor(ns); ft=torch.tensor(fw)
    sf,sid=torch.sort(ft); pl=torch.abs(sf-st[:N_RUNS]); nss=nt[sid]
    ml=pl.mean().item()
    if step==0: bl=ml
    else: bl=(1-BASELINE_ALPHA)*bl+BASELINE_ALPHA*ml
    adv=pl-bl; scores=(nss-mv)/SIGMA_PROP**2
    rg=(adv.detach()*scores).mean().item(); gr=max(min(rg,CLIP),-CLIP)
    mu+=LR*(-gr); mu.clamp_(1.,200.)
    nss_f=nss.float()
    rho=np.corrcoef(nss_f.numpy(),pl.detach().numpy())[0,1] if pl.std()>0.01 and nss_f.std()>0.01 else 0.0
    hist.append({'step':step,'mu':float(mu.item()),'loss':ml,'bl':bl,'grad':gr,'rho':float(rho),'mean_n':float(np.mean(ns))})
    if step%5==0 or step==N_ITER-1:
        print(f"  S{step:2d}: mu={hist[-1]['mu']:6.2f} loss={ml:.3f} gr={gr:+.4f} rho={rho:+.3f} nbar={hist[-1]['mean_n']:.1f} ({time.time()-t_opt:.0f}s)")

print(f"Opt done: {time.time()-t_opt:.0f}s. Plotting...", end=" ", flush=True)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

stp=[h['step'] for h in hist]; mv=[h['mu'] for h in hist]; lv=[h['loss'] for h in hist]
bv=[h['bl'] for h in hist]; rv=[h['rho'] for h in hist]; gv=[h['grad'] for h in hist]; nv=[h['mean_n'] for h in hist]

fig=plt.figure(figsize=(18,14)); gs=fig.add_gridspec(4,3,hspace=0.35,wspace=0.3)

ax=fig.add_subplot(gs[0:2,0:2])
ax.axhline(NBAR_TRUE,color='#2d6a4f',ls='--',lw=2.5,alpha=0.7,label=fr'$\mu_{{true}}={NBAR_TRUE:.0f}$',zorder=10)
ax.plot(stp,mv,'-',color='#1a73e8',lw=3,marker='o',markersize=6,zorder=5)
ax.fill_between(stp,mv,NBAR_TRUE,alpha=.08,color='#1a73e8')
ax.annotate(f'$\\mu_{{init}}={MU_INIT:.0f}$',xy=(0,MU_INIT),xytext=(3,MU_INIT-3),arrowprops=dict(arrowstyle='->',color='#d93025',lw=2),fontsize=12,color='#d93025',fontweight='bold')
fm=hist[-1]['mu']
ax.annotate(f'$\\mu_{{final}}={fm:.1f}$',xy=(N_ITER-1,fm),xytext=(max(N_ITER-15,2),fm+4),arrowprops=dict(arrowstyle='->',color='#1a73e8',lw=2),fontsize=12,color='#1a73e8',fontweight='bold')
ax.axvspan(0,4,alpha=.06,color='#1a73e8')
ax.text(1.5,NBAR_TRUE+5,'REINFORCE',fontsize=11,color='#1a73e8',fontweight='bold')
ax.axvspan(12,N_ITER,alpha=.06,color='#2d6a4f')
ax.text(N_ITER-8,NBAR_TRUE-5,'convergence',fontsize=11,color='#2d6a4f',fontstyle='italic')
ax.set_xlim(0,N_ITER-1); ax.set_xlabel('Iteration',fontsize=14); ax.set_ylabel('$\\mu$',fontsize=14)
ax.set_title('REINFORCE: $\\mu$ Convergence',fontsize=18,fontweight='bold',pad=15)
ax.legend(fontsize=13,loc='lower right'); ax.grid(alpha=.25,ls='--')

ax=fig.add_subplot(gs[0,2])
ax.plot(stp,lv,'-',color='#d93025',lw=2.5,label='W1'); ax.plot(stp,bv,'--',color='#666',lw=1.5,alpha=.7,label='BL')
ax.set_title('Loss',fontsize=14,fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=.25,ls='--')

ax=fig.add_subplot(gs[1,2])
ax.axhline(0,color='gray',lw=1,alpha=.5,ls='--')
ax.plot(stp,rv,'-',color='purple',lw=2.5,marker='.',markersize=4)
ax.fill_between(stp,0,rv,where=[r<0 for r in rv],alpha=.12,color='purple',label='Signal')
ax.set_title('$\\rho$(n, loss)',fontsize=14,fontweight='bold'); ax.set_ylim(-.4,.2); ax.legend(fontsize=10); ax.grid(alpha=.25,ls='--')

ax=fig.add_subplot(gs[2,0])
ax.axhline(0,color='gray',lw=1,alpha=.5,ls='--')
c=['#1a9850' if g<0 else '#d73027' for g in gv]
ax.bar(stp,gv,color=c,alpha=.7,width=.8)
ax.set_title('Gradient',fontsize=14,fontweight='bold'); ax.grid(alpha=.25,ls='--',axis='y')

ax=fig.add_subplot(gs[2,1])
ax.plot(stp,nv,'-',color='#188038',lw=2,marker='.',markersize=4)
ax.axhline(NBAR_TRUE,color='#2d6a4f',ls='--',lw=2,alpha=.6)
ax.fill_between(stp,nv,NBAR_TRUE,alpha=.08,color='#188038')
ax.set_title('Mean n',fontsize=14,fontweight='bold'); ax.grid(alpha=.25,ls='--')

# FWHM distribution
ax=fig.add_subplot(gs[3,:])
re=np.random.default_rng(999)
ff,ii=[],[]
for _ in range(N_RUNS):
    ff.append(scan(max(round(fm+SIGMA_PROP*re.standard_normal()),0),GAMMA,LAMBDA_,re))
    ii.append(scan(max(round(MU_INIT+SIGMA_PROP*re.standard_normal()),0),GAMMA,LAMBDA_,re))
xg=np.linspace(0,120,500)
try:
    kt=gaussian_kde(target_t.numpy()); kf=gaussian_kde(np.array(ff)); ki=gaussian_kde(np.array(ii))
    ax.plot(xg,kt(xg),'k-',lw=3,label=f'Target ($\\mu_{{true}}$={NBAR_TRUE})',zorder=10)
    ax.plot(xg,kf(xg),'-',color='#1a73e8',lw=2.5,label=f'Final ($\\mu$={fm:.1f})')
    ax.plot(xg,ki(xg),'--',color='#d93025',lw=2,alpha=.6,label=f'Init ($\\mu$={MU_INIT})')
except: pass
ax.set_xlabel('FWHM (MHz)',fontsize=14); ax.set_ylabel('Density',fontsize=14)
ax.set_title('FWHM Distribution',fontsize=16,fontweight='bold')
ax.legend(fontsize=12); ax.grid(alpha=.25,ls='--'); ax.set_xlim(0,120)

fig.suptitle(f'Per-Run REINFORCE: $\\mu$ from {MU_INIT} to {fm:.1f} (true={NBAR_TRUE})',fontsize=18,fontweight='bold',y=1.01)
plt.savefig(PLOT_FILE,dpi=150,bbox_inches='tight',facecolor='white')
print(f"Plot done ({time.time()-t_total:.0f}s total)")
plt.close()

summary={'nbar_true':NBAR_TRUE,'mu_init':MU_INIT,'mu_final':round(hist[-1]['mu'],2),'n_iters':N_ITER,'total_time_s':round(time.time()-t_total,1),'history':hist,'final_error':round(abs(hist[-1]['mu']-NBAR_TRUE),2)}
with open(RESULTS_FILE,'w') as f: json.dump(summary,f,indent=2)

print(f"Saved to {PLOT_FILE}")
print(f"\n{'='*50}")
print(f"  mu_init={MU_INIT:.0f} -> mu_final={fm:.1f} (true={NBAR_TRUE})")
print(f"  Initial loss: {hist[0]['loss']:.3f} -> Final: {hist[-1]['loss']:.3f}")
print(f"  Error: {abs(fm-NBAR_TRUE):.2f} ({abs(fm-NBAR_TRUE)/NBAR_TRUE*100:.1f}%)")
print(f"  Time: {time.time()-t_total:.0f}s")
print(f"{'='*50}")
