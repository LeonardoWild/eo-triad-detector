# eo-triad-detector

**Existence-Oriented (EO) Triad Detector** – A Python framework for assessing resonance, prototypicality, and human-likeness using ternary triad logic.

## Overview

This package implements a novel multi-axis detection system based on three (or four) core triads:

- **Identity** – Python object model canonical strength
- **Membership** – Prototype theory centrality in sequences
- **Relatedness** – Statistical resonance in distributions (Mahalanobis or cosine)
- **Spatial (optional)** – Learned trusted regions via PyTorch (hyperrectangle or ellipsoid)

The triads are fused into a single affinity score useful for anomaly detection, bot/human distinction, behavioral authenticity, and trusted-signal modeling.

## Installation

```bash
pip install eo-triad-detector

(Once published to PyPI – for now, install from GitHub:)
pip install git+https://github.com/YOUR_USERNAME/eo-triad-detector.git

## Quick Example
from eo_triad import EO_identity, EO_membership, EO_relatedness, EO_affinity, EO, TriadLevel
import numpy as np
import torch

# Pure triad scoring
identity = EO_identity(42)
membership = EO_membership("apple", ["banana", "apple", "orange"])
vec = np.random.randn(8)
dist = np.random.randn(100, 8)
relatedness = EO_relatedness(vec, dist)

# PyTorch spatial triad
eo = EO(dim=8, mode="ellipsoid")
clean = torch.randn(1000, 8)
eo.calibrate_ellipsoid(clean)
test = torch.randn(1, 8)
spatial = TriadLevel(eo(test)[0].item())

score = EO_affinity(identity, membership, relatedness, spatial=spatial, mode="min")
print("EO Affinity:", score)

## Use Cases
	•	Bot vs. human detection on social platforms
	•	Anomaly detection in behavioral signals
	•	Authenticity scoring for AI-generated content
	•	Trusted-region modeling in embedding spaces

## How to Run the Tests
After uploading everything:
# Install with test dependencies
pip install -e ".[test]"

# Run tests
pytest

# With coverage
pytest --cov=eo_triad

## License
Apache License 2.0
## Author
Daniel Leonard Wild