# Platonic Projection Structures (PPS)

Official reference implementation and reproducibility materials for:

Kazuo Ishii, Bishnu Prasad Gautam, Jieling Wu, Javaid Saher.  
**Platonic Projection Structures: Operator-Induced Observability in Representation Learning.**  
*Entropy* 2026, 28(7), 768.  
DOI: https://doi.org/10.3390/e28070768  
arXiv: https://arxiv.org/abs/2607.05175

## Overview

Platonic Projection Structures (PPS) describe observability through self-adjoint positive semidefinite operators. The framework emphasizes that observable and interpretable structure is induced by an observation operator, its kernel, range, rank, and spectrum.

## Included

- Core quadratic observable and operator utilities
- Kernel/range decomposition
- Effective numerical rank
- Kernel-invariance demonstration
- Rank/spectral observability simulation

## Quick start

```bash
pip install -r requirements.txt
python examples/minimal_projection_demo.py
python simulations/kernel_invariance.py
python simulations/rank_spectral_observability.py
```

## Citation

Please cite the associated paper. GitHub's “Cite this repository” function is enabled through `CITATION.cff`.

## License

MIT License.
