# DendriticCorticalMicrocircuitThesis
This repository contains the code and experiments for a Bachelor's thesis exploring biologically plausible learning algorithms in neural networks, inspired by dendritic cortical microcircuits ([Sacramento et al., 2018](https://papers.nips.cc/paper_files/paper/2018/hash/1dc3a89d0d440ba31729b0ba74b93a33-Abstract.html)).


## Repository Structure

- `main.py` — Entry point for running experiments.
- `netClasses.py` — Neural network and teacher model definitions.
- `config.py` — Experiment and hyperparameter configuration.
- `training_and_eval.py` — Training and evaluation routines.
- `plotFunctions.py` — Plotting utilities.
- `requirements.txt` — Python dependencies.
- `graphics/`, `runs/`, `weights/` — Output directories for results and model checkpoints.

## Run an Experiment  

```
python main.py
```

   Modify `main.py` to select different experiments or configurations.

## View Results
- Plots and logs are saved in the `runs/` and `graphics/` directories.
- Model weights are saved in the `weights/` directory.


## Abstract

Deep learning has achieved remarkable success by using Backpropagation to solve
the credit assignment problem. 
However, Backpropagation relies on biologically implausible mechanisms, particularly the need for symmetry between feedforward and feedback weights, the computation and communication of error signals through a separate feedback pass that can access exact derivatives of activations from the forward pass and does not work in a spiking context, where activation functions are inherently non-differentiable and spiking patterns across time add a temporal axis to credit assignment.
Although exact Backpropagation is almost certainly not implemented in the brain, its core principles might still direct credit assignment in biological neural networks.

There have been a number of attempts to approximate Backpropagation within more biologically plausible learning frameworks. 
Feedback Alignment relaxes the requirement for weight symmetry and frameworks like Equilibrium Propagation approximate Backpropagation in the context of a mismatch energy, without the need for additional neural computations and propagating signals as part of continuous neural activity. 
Approaches like target learning or predictive coding minimize discrepancies between current states and local targets, and in the limit of small discrepancies, their updates approximate Backpropagation. 

I investigate the approximation of the Backpropagation algorithm in the context of a model of dendritic cortical microcircuits, which is a biologically-inspired neural networkthat captures key features of cortical organization and dendritic structures. 
While the attempt to replicate the original model’s performance was unsuccessful, I investigate the neuronal dynamics and robustness of the rules for synaptic plasticity
and explore the relationship between different hyperparameters and their effect on
learning. 
Finally I explore how the original learning algorithm relates to other methods that approximate Backpropagation like Target Propagation, Predictive Coding, and Equilibrium Propagation, suggesting common principles for achieving credit assignment through local neuronal dynamics. 

These findings emphasize the importance of testing learning algorithms in physiologically realistic models and review several directions for bridging the gap between the computational efficiency of deep learning and the biological constraints of neural circuits.

