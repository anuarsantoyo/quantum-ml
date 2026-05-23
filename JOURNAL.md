# Project Journal

*Development of the project step by step, from ideas, tasks and conclusions*

---
# 13.05.2026

## Meeting Dr. Pieplow

Had the first meeting with Dr. Pieplow. He mentioned that I am free to try what I feel makes sense, as this is not a master thesis I can take my time if I manage to get something out of the project, then we all win.

He mentioned some topics that I can start to get familiar with. 

- 2 Level Quantum system with stochastic noise "Spectral shaped based on noise process"
- Description of PLE NV (how does the measurement work in detail)
- Big Goal: Parameter Estimation specifically line with in low signal regime
- Fischer Infomation
- Parameter uncertainty.

We agreed on having weekly meetings.


## Idea: Hyrid Model approach
I am curious to explore the different way you can model this process and if a hybrid approach of an enhanced physic informed model could be apply to achieve the desired goal. **Never the less** first understanding the current status in detail seems the most sensitive first step.
 
# 15.05.2026

## OpenClaw Configuration
Created and OpenClaw Agent called Pukky running in separate computer. I tested if it could acces the main branch and decided to create a specific github account for it and joined it as collaborator.

gmail account: pukky.struki@gmail.com
github account: pukky-struki


Also Added Deepseek Extension to VS Code, now my agent in VS Code and openclaw use the same model.

## Worked on the structure of the project.
Did some research on different project structures for projects based on research and coding.

We decided on a lean **research-first** structure:

```
qm-ml/
├── README.md          # Dashboard — overview, tasks, checklist
├── JOURNAL.md         # Running log — ideas, decisions, notes
├── docs/papers/       # PDFs of papers we read
├── notebooks/         # Jupyter notebooks for exploration & prototyping
├── requirements.txt
└── .gitignore
```

The idea: keep it simple during the research phase. When we move to writing actual library code, we'll expand with:

```
src/          # Production code (quantum/, models/, utils/)
tests/        # Unit tests
data/         # Datasets (gitignored)
``` 



# 22.05.2026

## Understanding paper

I started reading the papaer in detail, I am getting a much better idea of the whole problem and the approach to solve it.

Here is a summary of the Monte Carlo Method in General
![alt text](docs/journal/MCM.png)


Here is a Summary of one Monte Carlo Simulation
![alt text](docs/journal/MCSimulation.png)


## Idea: Optimizing $\sigma$
Try optimizing for more parameters, I understand that in the experiment they do a grid search for $\gamma$ and $\bar{n}$ (as they are only 2) but in the process of understanding the montecarlo simulation I though we can also try to optimize other parameters that are fixed like $\sigma = 6$ or $\lambda = 2$, for more parameters grid search might be better substituted by hyperopt or similar. I will first have a deeper understanding on each step of the MCM to see if this makes sense

## Idea: Hybrid approach
I still believe that we could create a hybbrid model where we exploit our knowledge of the physics behind the process. Maybe creating a different type of MonteCarlo simulation, maybe creating something using the real data and optimizing few parameters form the hybrid model.

## TODO

- ~~I have a much better understanding but would like to go deeper on the choice of a cauchy distribution and what physical meaning does $\gamma$ has in the whole story and in realtion to the Cauchy distribution.~~


## Notebook

I created a notebook `notebooks/Hyperopt-n-sigma-gamma.ipynb` where I command Pukky to test my idea of using hyper opt. This is more a test for pukkies capabilites and to have a start.

## Questions for Gregor

1. [ ] Can I get access to the real data
2. [ ] Why sigma 6? Is there more parameters that we could optimize?
3. [ ] Is the idea in general sinfull?


# 23.05.2026

## Optimization Algorithm
My Agent completed the notebook `notebooks/Hyperopt-n-sigma-gamma.ipynb` and I did the first test and it seems to work, never the less, I want to debugg it line by line to be sure of its functionality.

## Cauchy Interpretation
I finished understanting that the Cauchy distribution represent the absortion probabilty (which comes from the Fourier of the decay function). This made me realize that the whole physical process of absortion and emitions of photons is summarized in this curve. This gave rise to the next idea.

## Idea: Analyzing complexer models

Here are there models that go a step deeper and model the interactions to a more detailed level. Can we shift the parameter optimization to that model? This would be the beginning of the hybrid model approach.

Following is a first list of possible ideas obtained from a simple prompt (not analized in detail yet). This is a good start point to think of other strategies, by understanding other models I can start getting an idea of how to build a hybrid model. What I understand now, is that in the paper from Dr. Pieplow, a simplified 2-level system was used and therefore all that we want to find is the FWHM, for this we used MCM. My idea would be to create a model from scratch based on one of these complexer models, identify important parameters to optimize and use the existing real data to optimize them using PyTorch. I am aware of the fact that the real data had the issue of low signal-noise ration, this could make the simple suppervised learning approach difficult. So maybe some kind of MC must be included aswell, because now that i think about it, how would my loss function for the optimization look like? 


## Complexer Models

This models came from a simple prompt. There has been no analysis yet, this is simply a starting point.

>Prompt: can you research if there are more complex models that model this relaitonship. Right now we are modeling the whole absorbtion and emision simply with the absorbtion probaility, is ther a mathematical model that goes a bit deeper?

## 1. Weisskopf-Wigner (your paper's model)

**What it models:** Absorption + emission collapsed into a single Lorentzian lineshape.

**Key approximations:**
- Weak excitation (NV mostly in ground state)
- Zero temperature (no phonon absorption)
- No distinction between absorption and emission steps
- No quantum correlations between emitted photons

**Equation:**
$$ I_{\text{PL}}(\omega) \propto \frac{\gamma}{\pi[(\omega-\omega_0)^2 + \gamma^2]}, \quad \gamma = \frac{1}{4\pi\tau} $$

**When valid:** Weak excitation, low temperature, no cavity, only need emission probability (not photon statistics).

---

## 2. Master equation (Jaynes–Cummings + Lindblad)

**What it adds:** Explicit time evolution of the NV + laser field + environment. Treats absorption and emission as separate quantum jumps.

**Model details:**
The NV center is treated as a vibronic system with ground state sublevels $|g_i\rangle$, excited state $|e\rangle$, and metastable singlet state $^1A$. The Hamiltonian includes:

$$ \hat{H}_{\text{JC}} = \sum_i \omega_{g_i} |g_i\rangle\langle g_i| + \omega_e |e\rangle\langle e| + \omega_C \hat{a}^\dagger \hat{a} + \frac{1}{2}\sum_i \left( \Omega_i \hat{a}^\dagger |g_i\rangle\langle e| + \text{H.c.} \right) $$

The Lindblad master equation adds dissipation:

$$ \frac{d\rho}{dt} = -i[\hat{H},\rho] + \sum_j \gamma_{g_j e} \mathcal{L}[\sigma_{g_j e},\rho] + \kappa \mathcal{L}[\hat{a},\rho] + \gamma_{\text{deph}} \mathcal{L}[\sigma_{ee},\rho] $$

where $\mathcal{L}[\hat{O},\rho] = \hat{O}\rho\hat{O}^\dagger - \frac{1}{2}(\hat{O}^\dagger\hat{O}\rho + \rho\hat{O}^\dagger\hat{O})$ are Lindblad terms for spontaneous emission, cavity loss, and dephasing.

**Key physics captured:**
- **Purcell enhancement:** $\gamma_{\text{eff}} = \gamma_0(1 + F_p)$ with $F_p = \frac{3Q(\lambda/n)^3}{4\pi^2V}$
- **Power broadening:** $\Delta\nu = \Delta\nu_0 \sqrt{1 + I/I_{\text{sat}}}$
- **Mollow triplet:** at very high power, single Lorentzian splits into three peaks

**When to use:** Cavity-coupled emitters, high excitation power, when photon statistics matter.

---

## 3. Huang-Rhys (electron‑phonon coupling)

**What it adds:** Explicitly models how the NV's electronic transition couples to diamond lattice vibrations (phonons). This produces the phonon sideband (PSB) seen in NV spectra.

**Huang-Rhys model:**
Total luminescence lineshape:

$$ I(\omega) = I_0 \underbrace{e^{-S}\delta(\omega - \omega_0)}_{\text{ZPL}} + \underbrace{I_0 e^{-S} \sum_{n\ge1} \frac{S^n}{n!} \rho_n(\omega - \omega_0)}_{\text{Phonon sideband}} $$

where:
- $S$ = Huang-Rhys factor (electron-phonon coupling strength)
- For NV⁻, $S \approx 3.5$
- Debye-Waller factor (ZPL fraction) $DW = e^{-S} \approx 0.03$ → only ~3% of emission is in the ZPL

**When to use:** Understanding the low signal in PLE (most emitted light is in the PSB, not the ZPL).

---

## 4. Rapid dispersal (time‑dependent lineshape)

**What it adds:** The Lorentzian lineshape assumes detection after infinite time. At finite times after emission, the lineshape is not Lorentzian.

**Key idea:**
Immediately after emission, the photon wavepacket hasn't fully separated from the emitter. Measuring at early times gives a different spectral distribution than the asymptotic Lorentzian.

**When to use:** Ultrafast spectroscopy (picosecond time resolution), single-photon sources with sub‑ns timing.

---

## 5. Charge‑state switching (NV⁻ ↔ NV⁰)

**What it adds:** Under resonant excitation at 637 nm, NV⁻ can ionize to NV⁰ (dark under red excitation) and then recombine back. This causes blinking.

**Rate equation model:**

$$ \frac{dP_-}{dt} = -k_{\text{ion}} P_- + k_{\text{rec}} P_0 $$
$$ \frac{dP_0}{dt} = +k_{\text{ion}} P_- - k_{\text{rec}} P_0 $$

where $k_{\text{ion}}$ and $k_{\text{rec}}$ are power‑dependent. For NV⁻ in diamond, $k_{\text{ion}}$ is a two‑photon process (quadratic in power), $k_{\text{rec}}$ can be linear in power if donor impurities (e.g., phosphorus) provide electrons.

**When to use:** When you observe blinking or want to model the fraction of time the NV is in the bright (NV⁻) state.


## TODO

- Debugg Hyperopt-n-sigma-gamma.ipynb to be sure that the AI generated code works as expected (first test showed no errors but deeper analysis is needed)

- Read the complexer models more in detail




