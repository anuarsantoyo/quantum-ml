# Updates 03 Jul 2026
---



# Toy examples
Our goal is a differentiable Monte Carlo simulation: a program that takes `n` and `gamma` as inputs and returns a loss function that is optimizable with respect to both. To build intuition, I first worked through some toy examples of the optimization idea.

## For n

I created a simple program that samples `n` from a N(n, sigma) distribution, uses that sample to run a for loop that adds +1 to a prediction at each iteration, and then compares the prediction against a target (simulating the case where `n` only determines the number of runs).

```
sample   = normal(mean=n, std=sigma)
pred     = 0
for i in range(round(sample)):
    pred = pred + 1
loss     = (pred - target)^2
# backprop loss -> optimize n
```

![alt text](image.png)

![alt text](image-2.png)

## For sigma
![alt text](image-18.png)
![alt text](image-19.png)
![alt text](image-20.png)
![alt text](image-21.png)
![alt text](image-22.png)

# EDA

Once I had the data, I began analyzing it. I built a single DataFrame with all experiments: 2 lasers (1 nW and 3 nW), each measured across transmissions of [5, 10, 20, 40, 60, 80, 100].

![alt text](image-5.png)

![alt text](image-6.png)

## Amount of runs per PLE (with failed fwhm)
![alt text](image-23.png)

## Success rate vs. transmission

Higher transmission yields fewer NaN values (failed fits) in both cases.

![alt text](image-7.png)

## FWHM distribution per experiment
### Using log scale
The FWHM values are widely spread, so log scaling is appropriate.
![alt text](image-9.png)


### Distributions
At higher transmission the FWHM distribution is clean and well defined; at lower transmission it becomes much noisier.
![alt text](image-8.png)
![alt text](image-25.png)

### FWHM vs. fit error
The smaller the FWHM, the higher the fit error — this is especially pronounced for the extremely small values.
![alt text](image-10.png)

### Good news!
Key finding: the FWHM are continuous values, not binned as we had assumed. This simplifies the final loss.


# MMD²

I found a loss function that compares two lists of samples and tested whether it optimizes on real data, using `(df["transmission"] == 60) & (df["power_nW"] == 3)`.

MMD² (Maximum Mean Discrepancy) measures how different two sets of samples are, without needing to know their underlying distributions. It maps the samples into a feature space via a kernel (here a Gaussian/RBF kernel) and compares their mean embeddings:

MMD²(X, Y) = mean(k(x, x')) + mean(k(y, y')) − 2·mean(k(x, y))

It is 0 when the two sample sets come from the same distribution and grows as they diverge. Because it is built from a smooth kernel, it is differentiable — which is exactly what we need to backpropagate through and optimize.

## Development plot for test

We can follow the optimization step by step.
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)
![alt text](image-14.png)
![alt text](image-15.png)




## Sigma tuning

The choice of sigma in the loss affects the result. This warrants further experimentation.
![alt text](image-16.png)


## Conclusion
It works: dLoss/dFWHM is computed. The next step is dFWHM/dfrequencies.

# Gamma differentiable

We test it by creating a true gamma distribution and building the MMD² loss from it.

![alt text](image-17.png)
