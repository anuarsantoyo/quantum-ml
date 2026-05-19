# Project Journal

*Development of the project step by step, from ideas, tasks and conclusions*

---
## 2026-05-13

### Meeting Dr. Pieplow

Had the first meeting with Dr. Pieplow. He mentioned that I am free to try what I feel makes sense, as this is not a master thesis I can take my time if I manage to get something out of the project, then we all win.

He mentioned some topics that I can start to get familiar with. 

- 2 Level Quantum system with stochastic noise "Spectral shaped based on noise process"
- Description of PLE NV (how does the measurement work in detail)
- Big Goal: Parameter Estimation specifically line with in low signal regime
- Fischer Infomation
- Parameter uncertainty.

We agreed on having weekly meetings.


### Idea
I am curious to explore the different way you can model this process and if a hybrid approach of an enhanced physic informed model could be apply to achieve the desired goal. **Never the less** first understanding the current status in detail seems the most sensitive first step.
 
## 2026-05-16

### OpenClaw Configuration
Created and OpenClaw Agent called Pukky running in separate computer. I tested if it could acces the main branch and decided to create a specific github account for it and joined it as collaborator.

gmail account: pukky.struki@gmail.com
github account: pukky-struki

### Worked on the structure of the project.
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


