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




## 22.05.2026

### Understanding paper

I started reading the papaer in detail, I am getting a much better idea of the whole problem and the approach to solve it.

Here is a summary of the Monte Carlo Method in General
![alt text](docs/journal/MCM.png)


Here is a Summary of one Monte Carlo Simulation
![alt text](docs/journal/MCSimulation.png)


### Ideas:

#### Optimizing $\sigma$
Try optimizing for more parameters, I understand that in the experiment they do a grid search for $\gamma$ and $\bar{n}$ (as they are only 2) but in the process of understanding the montecarlo simulation I though we can also try to optimize other parameters that are fixed like $\sigma = 6$ or $\lambda = 2$, for more parameters grid search might be better substituted by hyperopt or similar. I will first have a deeper understanding on each step of the MCM to see if this makes sense

#### Hybrid approach
I still believe that we could create a hybbrid model where we exploit our knowledge of the physics behind the process. Maybe creating a different type of MonteCarlo simulation, maybe creating something using the real data and optimizing few parameters form the hybrid model.

### Todo

I have a much better understanding but would like to go deeper on the choice of a cauchy distribution and what physical meaning does $\gamma$ has in the whole story and in realtion to the Cauchy distribution.


### Notebook

I created a notebook where I command Pukky to test my idea of using hyper opt. This is more a test for pukkies capabilites and to have a start.

### Questions for Gregor

1. Can I get access to the real data
2. Why sigma 6? Is there more parameters that we could optimize?
3. Is the idea in general sinfull?





