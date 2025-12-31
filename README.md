# eo-triad-detector

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/LeonardoWild/eo-triad-detector/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/LeonardoWild/eo-triad-detector)

**Existence-Oriented (EO) Triad Detector** – A novel Python framework for assessing resonance, prototypicality, and human-likeness using **ternary triad logic** (Absent · Boundary · Central).

Unlike binary classifiers, EO-Triad measures *degrees of canonical existence* across multiple axes, enabling nuanced detection of anomalies, bots, AI-generated content, and behavioral drift.

## Why EO-Triad?

Traditional outlier detection asks "in or out?"  
EO-Triad asks a deeper, more human-like question:

> "Is this **expected** (deeply resonant), merely **on the edge** (technically compliant but unnatural), or clearly **divergent** (absent from the prototype)?"

This fractal ternary logic (inspired by prototype theory and object identity) produces richer, more explainable results.

Core axes:
- **Identity** – Canonical strength in the Python object model
- **Membership** – Centrality in sequences (prototype theory)
- **Relatedness** – Statistical/directional resonance in distributions
- **Temporal** *(optional)* – Behavioral stability over time
- **Spatial** – Learned trusted regions (PyTorch: hyperrectangle or ellipsoid)

## Installation

```bash
# From GitHub (until PyPI release)
pip install git+https://github.com/LeonardoWild/eo-triad-detector.git

# For development
pip install -e ".[test]"
Quick Start
python


from eo_triad import (
    EO_identity, EO_membership, EO_relatedness,
    EO_temporal_shift, EO_affinity, EO, TriadLevel
)
import numpy as np
import torch

# Pure triad functions
identity = EO_identity(42)                    # CENTRAL (small ints are canonical)
membership = EO_membership("apple", ["banana", "apple", "orange"])  # CENTRAL
vec = np.random.randn(8)
dist = np.random.randn(100, 8)
relatedness = EO_relatedness(vec, dist)       # Statistical resonance

# Temporal stability (optional)
time_series = np.random.randn(5, 8)            # 5 time points × 8 features
temporal = EO_temporal_shift(time_series)

# Combine into affinity score [0.0 – 2.0]
score = EO_affinity(identity, membership, relatedness, temporal=temporal, mode="weighted")
print(f"EO Affinity: {score:.2f}")
Full Example: Bot/Human Detection Pipeline
python


# Generate synthetic data
n_humans, n_bots, dim = 350, 150, 6
X_human = np.random.uniform(1, 8, (n_humans, dim))
X_bot = np.random.uniform(3, 15, (n_bots, dim))
X = np.vstack([X_human, X_bot])
X_tensor = torch.tensor(X, dtype=torch.float32)
labels = np.array([0] * n_humans + [1] * n_bots)

# Temporal series (optional)
temporal_human = np.stack([X_human + np.random.normal(0, 0.15, X_human.shape)] * 5)
temporal_bot = np.stack([X_bot + np.random.normal(0, 1.2, X_bot.shape)] * 5)
temporal_all = np.vstack([temporal_human, temporal_bot])

# Initialize & calibrate detector
detector = EO(dim=dim, mode="ellipsoid")
detector.calibrate_ellipsoid(torch.tensor(X_human))

# Optimize parameters for ternary separation
detector.optimize_parameters(X, labels, temporal_data=temporal_all)

print(f"Best center: {detector.best_center:.2f} ± {detector.best_bandwidth:.2f}")

# Ternary triad prediction
predictions = detector.predict(X_tensor, temporal_data=temporal_all)

# Results (Absent=0, Boundary=1, Central=2)
print("Humans in Central:", (predictions[labels==0] == TriadLevel.CENTRAL).float().mean().item())
print("Bots in Absent:   ", (predictions[labels==1] == TriadLevel.ABSENT).float().mean().item())
Use Cases
	•	Bot vs. human detection on X/Twitter and other platforms
	•	Authenticity scoring for AI-generated text/images
	•	Behavioral anomaly detection (e.g., account takeover)
	•	Trusted-region modeling in embedding spaces
	•	Content moderation and resonance-based ranking
Testing & Development
bash


# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# With coverage
pytest --cov=eo_triad
Contributing
Contributions are very welcome! This project follows a philosophy-first, clean-code approach.
## How to Contribute
	1.	Fork the repository
	2.	Create a feature branch (git checkout -b feature/amazing-triads)
	3.	Write tests for new functionality
	4.	Ensure all tests pass: pytest
	5.	Commit with clear messages
	6.	Open a Pull Request with:
	•	Description of changes
	•	Link to related issue (if any)
	•	Screenshots/results if applicable
## Guidelines
	•	Follow PEP 8 and type hints (mypy compliant)
	•	Add docstrings for public functions/classes
	•	Prefer clarity and philosophical alignment over micro-optimizations
	•	New features should respect the ternary/fractal logic when possible
## Ideas for Contributions
	•	Real-world feature extractors (e.g., X post patterns → identity/membership)
	•	Integration with Hugging Face embeddings
	•	Visualization tools for triad breakdowns
	•	Alternative optimization strategies (Bayesian, evolutionary)
	•	Soft probability output mode
We appreciate ideas, bug reports, documentation improvements, and philosophical discussions just as much as code!

## 1. Industrial & Manufacturing Anomaly DetectionPrototype-based methods already power time-series and image anomaly detection in factories (e.g., MVTec AD dataset for defect localization). EO-Triad adds nuanced ternary scoring: Central for canonical machine behavior, Boundary for early wear/degradation, Absent for clear faults.Temporal shift axis detects behavioral drift (e.g., vibration patterns changing over time).
Ellipsoid mode defines trusted operating regions in sensor embeddings (common with Mahalanobis distance for outlier detection in multivariate sensor data).

# Applications: Predictive maintenance, quality control in assembly lines, rotating machinery monitoring.

## 2. Cybersecurity & Fraud DetectionMulti-axis resonance fits intrusion detection and transaction monitoring, where anomalies often involve subtle behavioral shifts.Identity/membership axes score protocol conformance or user pattern prototypicality.
Temporal shift flags sudden changes (e.g., account takeover via altered activity rhythm).
Relatedness via Mahalanobis in network traffic embeddings detects deviations from normal flow distributions.

## Ternary output distinguishes clear attacks (Absent), suspicious but compliant activity (Boundary), and normal (Central)—superior to binary alerts for reducing false positives.Applications: Network intrusion detection, credit card fraud, ransomware behavioral profiling.

## 3. Healthcare & Medical Anomaly DetectionPrototype theory and uncertainty-aware prototypical learning apply to medical imaging and patient monitoring.Score lesions or vital signs against prototypical "healthy" distributions.
Temporal component detects behavioral shifts (e.g., irregular heart rhythms or mobility patterns in elderly care).
Fractal boundaries capture ambiguity: Boundary for uncertain/early-stage anomalies needing review.

# Applications: MRI/CT anomaly localization, wearable sensor monitoring for frailty/MCI risk, epidemic outbreak detection via population behavior shifts.

## 4. Financial Time-Series & Cloud Cost MonitoringTemporal shift analysis and multivariate ellipsoids suit detecting spend anomalies or market irregularities.Model normal trading/user behavior as prototypes.
Flag Boundary for unusual-but-plausible trades (potential insider activity) vs. Absent for clear fraud.

#Applications: Cloud spend anomaly detection (e.g., unexpected spikes), stock market manipulation detection, insurance claim fraud.

## 5. Geosciences & Climate MonitoringPrototype-anomaly scores help detect out-of-distribution changes in non-stationary data.Use EO-Triad to model "normal" climate patterns across spatial/temporal axes.
Detect subtle shifts (Boundary) vs. extreme events (Absent).

# Applications: Extreme weather event precursors, seismic anomaly detection.

## 6. Human Mobility & Smart CitiesBehavior-aware spatio-temporal frameworks align perfectly with EO-Triad's axes.Profile individual mobility prototypes.
Detect anomalies like sudden routine changes (potential health/security issues).

# Applications: Elderly monitoring, urban traffic anomaly detection, pandemic behavior shift tracking.

## Why EO-Triad Stands Out Across These AreasIts ternary/fractal logic provides richer interpretability than binary classifiers—Boundary regions highlight ambiguity for human review. Multi-axis design (including optional temporal) captures complex dependencies missed by single-metric methods. Open-source flexibility allows easy adaptation to domain-specific features.

# The framework's philosophical grounding in prototype theory makes it especially powerful for domains where "naturalness" or "expected behavior" is key. It's not just another anomaly detector—it's a resonance measurer

## License
Apache License 2.0 (LICENSE)
##Author
Daniel Leonard Wild

“Not everything that can be counted counts, and not everything that counts can be counted.” ——William Bruce Cameron.
Measuring resonance requires more than binary labels.


