# presentation of today

#  Current status

We were able to optimize the model and recover the true params using syntetic data generated from the true values and calulate fisher (must check wy confidence by 05 so small)

# Next Goal: trying to tunne the model to fit all experiments

Idea if our model is not able to recover the true values from syntetic data then we have a gap on the model which would never be able to recover from the real data. As the model already works this process is simply trial and error.

# Idea: Structural tunning with Agent loop

First attempts of agent tunning mechanism. The idea is to create a notebook that starts with a summary and context of what happend before, proposes a new change, does it test it and does a summary of the results and what went wrong and right and why, and iterate through many. 

These are broad changes that have a high impact in the results. Once the result was achieved, fine hyperparameter tunning was hard because agent was over enginering and not simply tunning hyperparameters and the amount of experiments needed where to many as the search space is very large.

# Idea: Take advantage of optimization algorithms space search strategy but use agent which is physics informed

the idea is there a way where I can give this agent not only the context of what has happened but an insight of which changes make an improvement more probable. Several optimization algorithms came to mind as they all generate a type of improvement probability distribution. So the question was how can I feed the agent the distribution information and the context so that It could make a better decision on what to change. 

# Ag-hypopt:

## Uncertanty Aware TPE
We explain first our uncertainty tpe and why we chose tpe (more flexible)

## AG-Hypopt
Then explain our ag-hypopt cycle

## Results
 Show current results.