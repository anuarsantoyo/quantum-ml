# AG-HYPOPT — Novelty Assessment & Prior-Art Map

*Written 2026-09-11 by Pukky at Anuar's request. Question: is AG-HYPOPT a publishable
novelty, or "a nice tool we built for ourselves"?*

> **Scope caveat.** The web-search APIs were bot-walled during this review; the survey is
> therefore **arXiv-centric** (via the arXiv API). There may be non-arXiv (journal/workshop)
> work missed. arXiv IDs are given so each claim is checkable.

---

## 1. What AG-HYPOPT actually is, decomposed

Three layers with very different novelty profiles:

### 1a. The optimizer — *not novel*
An uncertainty-aware **TPE**: good/bad split on an uncertainty-adjusted loss
(`ℓ̃ = ℓ + λs`), variable-bandwidth KDEs with a uniform prior, per-observation bandwidths
`h_i ∝ (u_i/u_med)^β`, and `explore_slots` reserved fully-random draws per batch. This is
textbook *Tree-structured Parzen Estimator* (Bergstra et al., 2011) plus minor variants.
Nothing here is a contribution on its own.

### 1b. The LLM's roles — *the interesting part*
1. **Selector.** A classical optimizer proposes ~10 candidates; the LLM reads the registry +
   candidate table + **domain (physics) context** and picks **exactly one**. It is *not* the
   proposer and *not* the surrogate — it is the **decision/acquisition policy**.
2. **Domain-prior injector.** The choice uses structure a statistical acquisition cannot
   (the μ-attractor, the Voigt-vs-Lorentzian spread gap, "this γ is a settled attractor, stop
   chasing it").
3. **Scribe.** Writes the structured per-trial record (summary + key insight); the notebook's
   last cell generates the next trial.

### 1c. The scaffold — *engineering, not science*
"Agent writes no code and creates no files"; self-contained, in-place executed notebooks;
deterministic runs (`SEED`/`SYNTH_SEED` fixed); frozen roles for every file; per-trial
registry (`trials.json`). This is an MLOps/reproducibility stance, not a scientific claim.

---

## 2. Prior art — where each layer already lives

**LLM as *proposer* in HPO / AutoML**
- *Using Large Language Models for Hyperparameter Optimization* — arXiv:2312.04528 (2023)
- *OPRO: Large Language Models as Optimizers* — arXiv:2309.03409 (2023)
- **AgentHPO: Large Language Model Agent for Hyper-Parameter Optimization** — arXiv:2402.01881 (2024) ⭐
- *AutoML-GPT* (2023), *AutoML-Agent* (2407.12899), *MLE-Dojo* (2502.13989), *AutoLLMResearch* (2026)

**LLM *inside* Bayesian optimization (surrogate / prior / acquisition)**
- **LLAMBO: Large Language Models to Enhance Bayesian Optimization** — arXiv:2402.03921 (2024) ⭐
- *PFNs4BO: In-Context Learning for Bayesian Optimization* — arXiv:2305.17535 (2023)
- *Position: Leverage Foundational Models for Black-Box Optimization* — arXiv:2402.10216 (2024)
- *Language Model Embeddings Can Be Sufficient for Bayesian Optimization* — arXiv:2404.08079 (2024)
- *SemanticOpt: LLM-Based Semantic Black-Box Optimization* — arXiv:2503.01707 (2025)
- *LILO: Bayesian Optimization with Natural Language Feedback* (2025)

**LLM as the *decision-maker / selector* in the loop — closest to AG-HYPOPT**
- **LMABO: Adaptive Acquisition Selection for Bayesian Optimization with LLMs** — arXiv:2602.07904 (2026) ⭐
  (LLM selects which *acquisition function* to use each iteration)
- **LGBO: Unleashing LLMs in BO — Preference-Guided Framework for Scientific Discovery** — arXiv:2605.17976 (2026) ⭐
- *Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization* (2026)
- *Evolve Cost-aware Acquisition Functions Using LLMs* — arXiv:2404.17564 (2024)
- *FunBO: Discovering Acquisition Functions with FunSearch* — arXiv:2404.17569 (2024)
- **OptiMindTune: A Multi-Agent Framework for Intelligent HPO** — arXiv:2505.19205 (2025) ⭐
  (Recommender + Evaluator + **Decision** agents)

**LLM agents for scientific discovery / experiment design**
- Coscientist (Boiko et al., *Nature* 2023 — autonomous chemical research)
- **AI-Mandel: Towards autonomous quantum physics research using LLM agents** — arXiv:2511.11752 (2025) ⭐
- *Auto Researching, not hyperparameter tuning* — arXiv:2603.15916 (2026)
- *From AI for Science to Agentic Science: A Survey* (2025); *Autonomous Agents for Scientific
  Discovery* (2025); *LLM and Simulation as Bilevel Optimizers* (2024)

**Physics-domain HPO agent**
- **NQS-Agent: Health-Aware Agentic HPO for Neural-Network Quantum States** — arXiv:2606.30464 (2026) ⭐
  (nearest domain neighbour: agentic HPO of a physics pipeline)

**Expert / human-in-the-loop BO**
- *Taking the Human Out of the Loop: A Review of Bayesian Optimization* (Shahriari et al. 2016)
- Human-in-the-loop BO (exosuit/wearables, 2017–2018); expert-in-the-loop acquisition design

> Note: the exact phrase **"agent-guided hyperparameter optimization" returns 0 arXiv hits** —
> the *name* is free, but the *territory* is dense.

---

## 3. The gap

AG-HYPOPT's distinctive combination — **LLM as the selector over a classical proposal batch,
steered by domain/physics context, inside a reproducible no-code notebook scaffold** — is not
published under a single name. But every *component* is established, and the nearest neighbours
(LMABO, LGBO, OptiMindTune, AgentHPO, NQS-Agent) cover most of the space. The "selector +
domain-context" configuration is only lightly explored → **thin, incremental novelty**.

**Verdict: as it stands, a nice tool, not a headline novelty.**

---

## 4. Routes to something publishable

1. **Empirical finding (best shot).** The claim nobody has nailed: *does an LLM selector beat
   the statistical acquisition function specifically because it injects domain context?*
   Design: AG-HYPOPT vs {TPE-only, random search, GP-EI, LLM-proposer (AgentHPO-style),
   **LLM-selector without context** (ablation), human}, across this pipeline + 1–2 other
   scientific pipelines; report best-so-far, regret, budget-to-threshold. If the context
   ablation shows the gain comes from *domain reasoning*, that is a real finding.
2. **The physics is the paper.** The differentiable-MC inversion of real NV PLE linewidths +
   the Voigt/Lorentzian spread gap (FM#6). If it resolves something, *that* is the headline;
   AG-HYPOPT is a tool in the methods section.
3. **Systems/scaffold paper.** Reproducible, self-documenting, no-code agent-driven HPO as an
   artifact (MLOSS / workshop). Honest, modest venue.

---

## 5. Must-read (novelty boundary)

| Paper | arXiv | Why |
|---|---|---|
| LLAMBO | 2402.03921 | LLM inside BO (propose + evaluate) |
| AgentHPO | 2402.01881 | LLM agent runs HPO end-to-end |
| LMABO | 2602.07904 | LLM selects the acquisition function per step |
| OptiMindTune | 2505.19205 | multi-agent HPO with an explicit Decision agent |
| NQS-Agent | 2606.30464 | agentic HPO of a physics pipeline |
| LGBO | 2605.17976 | LLM preference-guided BO for discovery |
| AI-Mandel | 2511.11752 | LLM agent for quantum-physics experiments |

---

## 6. Recommendation

- Do **not** pitch "LLM for HPO" — saturated. Pitch either the **domain-context ablation
  result** (§4.1) or the **physics** (§4.2).
- The current real-data campaign (`experiment_4`) is a *feasibility demo*, not a novelty
  experiment. A paper needs the controlled comparison + ablation in §4.1.
- Next concrete steps: (a) formalise the ablation spec; (b) related-work section with the
  citations above; (c) decide whether the target is a methods paper or the physics paper.
