# TODO

1. multi-stage CG
2. annealing (cool-down; LR scheduling) to avoid big jumps 
3. protein sims 
4. Water sims 
4. MD Langevin integrator


## Priorities
In pure LJ loop tests, we observe that direct GD actually gets close to good configuration, but becomes unstable near the end, shooting out some nodes. 
This happens despite the clamping of grads. 
We need to examine what is causing this and whether it can be avoided. 
We can try:
1. LR scheduling.
2. modify repulsion node pairs update (update may suddenly cause overlaps)
3. modify early stopping (use moving avg and recent drops)

To facilitate testing this, we need to modify the way experiments and logging work. 
Right now, the Minimizer class does part of logging, but the main experiment results are logged using the `ExperimentLogger` class. 
We should change how the logging and running of experiment is done:

1. Let the Minimizer use an external logger class. 
2. Have the experiment running, updating pairs, and logging be handled  steps. 
