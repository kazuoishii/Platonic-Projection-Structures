# Platonic Projection Structures (PPS)

Official code repository for:

**Platonic Projection Structures: Operator-Induced Observability in Representation Learning**

Kazuo Ishii, Bishnu Prasad Gautam, Jieling Wu, and Javaid Saher

Published in *Entropy* (2026), 28(7), 768.

## Paper

DOI: https://doi.org/10.3390/e28070768

arXiv: https://arxiv.org/abs/2607.05175

## Overview

Platonic Projection Structures (PPS) describe observability through self-adjoint positive semidefinite operators. The framework emphasizes that observable and interpretable structure is induced by an observation operator, its kernel, range, rank, and spectrum.

A system is represented as a triple

\[
(H, \Pi, O),
\]

where \(H\) is a latent space, \(\Pi\) is an observation operator, and

\[
O(v) = \langle v, \Pi v \rangle
\]

defines the observable quantity.

The experiments in this repository illustrate three central properties of PPS:

1. Kernel-invariant observability
2. Rank-controlled observable geometry
3. Operator-consistent representation transfer

## Included

- Core quadratic observable and operator utilities
- Kernel/range decomposition
- Effective numerical rank
- Kernel-invariance demonstration
- Rank-controlled spectral observability simulations
- PPS-aware knowledge distillation experiments
- Observability, attribution, and causal-influence experiments

## Repository Structure

```text
simulations/
├── figure2_pps_distillation_commutativity.py
├── figure5_observability_shap_causal_shift.py
├── figure6_kernel_observability_invariance.py
└── figure7_rank_controlled_observability.py
```

## Quick Start

Install the required packages:

```bash
pip install numpy matplotlib scikit-learn shap torch torchvision
```

Run individual experiments:

```bash
python simulations/figure5_observability_shap_causal_shift.py
python simulations/figure6_kernel_observability_invariance.py
python simulations/figure7_rank_controlled_observability.py
```

The CIFAR-10 distillation experiment can be run using:

```bash
python simulations/figure2_pps_distillation_commutativity.py
```

A GPU environment such as Google Colab is recommended for the CIFAR-10 experiment.

## Figure 2 — PPS-Aware Knowledge Distillation

`figure2_pps_distillation_commutativity.py`

Evaluates the PPS commutativity condition

\[
\Phi \Pi_T \approx \Pi_S \Phi
\]

using teacher-student knowledge distillation on CIFAR-10.

## Figure 5 — Observability, Attribution, and Causal Influence

`figure5_observability_shap_causal_shift.py`

Illustrates the distinction between attribution-based detectability and causal influence under restricted observation.

## Figure 6 — Kernel-Invariant Observability

`figure6_kernel_observability_invariance.py`

Directly verifies that perturbations confined to \(\ker(\Pi)\) do not change the observable \(O(v)\).

## Figure 7 — Rank-Controlled Observability

`figure7_rank_controlled_observability.py`

Examines how \(\operatorname{rank}(\Pi)\) controls the effective observable dimension, predictive accuracy, attribution stability, and spectral geometry.

## Requirements

The experiments use Python and the following packages:

- NumPy
- Matplotlib
- scikit-learn
- SHAP
- PyTorch
- torchvision

For Google Colab, SHAP can be installed using:

```bash
pip install shap
```

## Reproducibility

All synthetic experiments use fixed random seeds where applicable.

The CIFAR-10 dataset is downloaded automatically through torchvision.

Each script generates the corresponding figures used in the PPS experiments.

## Citation

If you use this code or the PPS framework, please cite:

```bibtex
@article{ishii2026pps,
  title={Platonic Projection Structures: Operator-Induced Observability in Representation Learning},
  author={Ishii, Kazuo and Gautam, Bishnu Prasad and Wu, Jieling and Saher, Javaid},
  journal={Entropy},
  volume={28},
  number={7},
  pages={768},
  year={2026},
  doi={10.3390/e28070768}
}
```

GitHub's "Cite this repository" function is enabled through `CITATION.cff`.
